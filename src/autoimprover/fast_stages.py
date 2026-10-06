"""Stages B, C and D of the fast pipeline and its second generation (SPEC R10, R16, R17, R24, R25;
ADR-002, ADR-006, ADR-011, ADR-012), with the state and the helpers every stage of a fast run
shares; `fast.py` adds stage A, stage E, the quick tier and the endings.

The free gates drop a rewrite that changes no meaning word of the original or of an earlier
rewrite (`meaning_words`), one over the length cap, one that lost a literal (SPEC R7, R9). Stage B
runs the original twice (samples 0 and 1: two calls, never one cached reply) and every rewrite on
the M scenarios, all in one wave. Stage C decides by pairwise preference (SPEC R25; ADR-012), in
one wave: one contract check of every rewrite (`contract.check_many`), the original's run 0
against its run 1 in both orders, the noise, and each rewrite's answers against the original's run
0 in both orders, each call holding every scenario of its pair (`fast_pairwise`). A rewrite wins
when it kept the contract and won more scenarios than it lost by more scenarios than the noise.

A plan with two generations then reflects (stage R): the reflection model reads the best one or
two first-generation rewrites that kept the contract, by their lead in scenarios, or the original
when none did, with the judge's reasons for each scenario they lost or tied, and writes K2
rewrites under distinct notes; they pass the same gates, run (stage B2) and are judged against the
same answers of the original with the same noise (stage C2, no new noise pair, one contract check
for all of them). Stage D picks over every winner of both generations: the largest lead, a tie to
the shorter rewrite, then the earlier. When the second generation does not fit the time left or
is cut, the first generation's result stands.

Before each stage the time and calls left are compared with its estimate (`fastplan`), by the
latency model fitted to the run's replies once there is one (after stage A, again after stage B,
`fast_calibrate`): what does not fit is shrunk, scenarios first, and a deadline or call limit
reached inside a stage ends it (`ended`). Stage B's calls end by the deadline less the cheapest
stage C (and E), so a task run that cannot finish by then is abandoned (ADR-011 decision 2); after
such a cut the scenarios that both runs of the original and a rewrite answered in time go on to
stage C, shrunk to the time left, or on one rewrite and one scenario when the time left covers
one pairwise call
(`fastplan.last_chance`). The deadline itself is never passed (SPEC R17, R25). A failed rewrite,
reflection, task or pairwise call drops what it was for (a pairwise order that failed makes its
scenarios ties); an original run left with no answer, a noise pair with no scenario both runs
answered or that the judge could not compare, or a failed contract check, ends the run as
BackendError (SPEC R24).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TextIO, cast

from autoimprover.backend import BudgetedBackend, Clock
from autoimprover.bench_judge import parse_pairwise_batch
from autoimprover.contract import Violation, _ask, check_many, literals_preserved
from autoimprover.fast_calibrate import Timed, fit
from autoimprover.fast_pairwise import (
    Preference,
    noise_count,
    pair_calls,
    pair_items,
    preference,
    prefers,
)
from autoimprover.fast_prompts import (
    REFLECT_NOTES,
    STRATEGY_NOTES,
    FastEvaluator,
    evidence,
    parse_rewrite,
    reflect_call,
    reflect_strategy,
    strategy,
)
from autoimprover.fastplan import (
    FastPlan,
    Latency,
    Stage,
    last_chance,
    misfit,
    scoring_stages,
    second_stages,
    shrink,
    tail,
)
from autoimprover.parallel import parallel_map
from autoimprover.runner import count_tokens, length_ok
from autoimprover.runstore import RunStore
from autoimprover.types import (
    Backend,
    BackendError,
    BudgetExhausted,
    Call,
    CallFailed,
    Contract,
    Plan,
    Scenario,
    StopCause,
)

# A word that carries meaning: a run of letters or digits. Single letters and articles do not.
_WORD = re.compile(r"[^\W_]+")
_ARTICLES = frozenset({"an", "the"})
# The first-generation rewrites that kept the contract, best first, a reflection reads.
_PARENTS = 2


class Rewrite(NamedTuple):
    """A rewrite that passed the free gates: its variant, its text, its length ratio and the line
    that says what it changed (SPEC R2)."""

    variant: int
    text: str
    ratio: float
    note: str


class Dropped(NamedTuple):
    """Why a call of a stage gave nothing, and the cause when the clock or the limit cut it."""

    why: str
    cut: StopCause | None = None


class Win(NamedTuple):
    """A rewrite's win, as shares of the M scenarios it was picked on: those the original won
    (`before`) and those it won (`after`), its lead (`gain`) and the noise (`bar`)."""

    rewrite: Rewrite
    before: float
    after: float
    gain: float
    bar: float


class Judged(NamedTuple):
    """A rewrite after stage C: its preference against the original and whether it kept the
    contract."""

    rewrite: Rewrite
    found: Preference
    keep: bool


@dataclass
class Stages:
    """One fast run: its inputs, the free gates, stages B to D and the second generation, and what
    every ending reports: `stop`, the first cause that shrank or cut a stage, and in `measured`
    the noise once known. `ended` is set when a stage was cut short, after which no stage runs.
    Every call goes through `timed`, and `latency` is the model fitted to its replies once there
    is one (`calibrate`), which every later estimate uses (SPEC R25)."""

    prompt: str
    plan: Plan
    fplan: FastPlan
    backend: Backend
    budgeted: BudgetedBackend
    clock: Clock
    store: RunStore
    log: TextIO
    workers: int
    stop: StopCause | None = None
    ended: bool = False
    measured: dict[str, Any] = field(default_factory=dict)
    latency: Latency | None = None
    timed: Timed = field(init=False)

    def __post_init__(self) -> None:
        self.timed = Timed(self.backend, count_tokens(self.prompt))
        self.backend = self.timed

    def gates(
        self, drafts: list[tuple[int, str]], earlier: Sequence[Rewrite] = (), second: bool = False
    ) -> list[Rewrite]:
        """The drafts that pass the free gates, in variant order: a change in meaning words from
        the original and every earlier rewrite, the length cap, every literal kept (SPEC R7, R9).
        `second` marks the reflections of a second generation."""
        name = "reflection" if second else "rewrite"
        kept: list[Rewrite] = []
        seen = {meaning_words(self.prompt): "no change in meaning words"}
        for rewrite in earlier:
            seen.setdefault(
                meaning_words(rewrite.text), f"the meaning words of rewrite {rewrite.variant} again"
            )
        for variant, text in drafts:
            fits, ratio = length_ok(self.prompt, text, self.plan.strictness, self.plan.allow_growth)
            if (words := meaning_words(text)) in seen:
                why = seen[words]
            elif not fits:
                why = "it is longer than the length cap"
            elif not literals_preserved(self.prompt, text):
                why = "it loses a literal of the original"
            else:
                notes = REFLECT_NOTES[reflect_strategy(variant)] if second else None
                note = notes or STRATEGY_NOTES[strategy(variant)]
                kept.append(Rewrite(variant, text, ratio, note))
                seen[words] = f"the meaning words of {name} {variant} again"
                continue
            self.note(f"{name} {variant} dropped: {why}")
        return kept

    def contest(
        self, contract: Contract, rewrites: list[Rewrite], pick: list[Scenario], holdout: int
    ) -> Win | None:
        """Stages B, C, the second generation and D: the best rewrite by the pick rule, or None."""
        w, p = self.workers, count_tokens(self.prompt)
        shape = self.shrunk(
            len(rewrites), len(pick), lambda k, m: tail(k, m, holdout, w, p, self.latency)
        )
        if shape is None:
            return None
        rewrites, pick = rewrites[: shape[0]], pick[: shape[1]]
        runs = [(self.prompt, 0), (self.prompt, 1), *((r.text, 0) for r in rewrites)]
        self.note(
            f"stage B: the original twice and {len(rewrites)} rewrite(s) on {shape[1]} scenarios"
        )

        def judging(k: int, m: int) -> tuple[Stage, ...]:  # stage C, then E
            lat = self.latency
            return (scoring_stages(k, m, w, p, lat)[1], *tail(0, 0, holdout, w, p, lat))

        outputs = self.stage_b(contract, runs, pick, judging(1, 1))
        self.calibrate("B")
        if self.ended:  # cut: the scenarios every run answered in time go on (ADR-011 decision 2)
            self.ended = False
            pick = [s for s in pick if answered(outputs, s.id)]
            if not pick:
                return None
        ran = [any(isinstance(outputs[n].get(s.id), str) for s in pick) for n in range(len(runs))]
        if not (ran[0] and ran[1]):
            raise BackendError("every task run of one of the original's two runs failed")
        alive = [n for n in range(2, len(runs)) if ran[n]]
        shape = self.shrunk(len(alive), len(pick), judging)
        if shape is None and alive and self.fits(last_chance(holdout, w, p, self.latency)):
            shape = (1, 1)  # stage B ended late: the time left covers one pairwise call
        if shape is None:
            return None
        alive, pick = alive[: shape[0]], pick[: shape[1]]
        chosen = [rewrites[n - 2] for n in alive]
        found = self.stage_c(
            contract, chosen, [outputs[n] for n in alive], outputs[0], pick, outputs[1]
        )
        if found is None:
            return None
        first, noise = found
        if not holdout:  # the checked tier's noise field is that of its held-out check
            self.measured["noise"] = noise / len(pick)
        wins = [win for j in first if (win := won(j, noise, len(pick))) is not None]
        if self.fplan.generations > 1 and not self.ended:
            wins += self.second(contract, pick, outputs[0], noise, first, holdout)
        if not wins:
            return None
        return min(wins, key=lambda w: (-round(w.gain, 9), count_tokens(w.rewrite.text)))

    def second(
        self,
        contract: Contract,
        pick: list[Scenario],
        original: dict[str, str | CallFailed],
        noise: int,
        first: list[Judged],
        holdout: int,
    ) -> list[Win]:
        """The second generation's winners (stages R, B2, C2), or none when it does not fit the
        time and calls left or is cut: then the first generation's result stands."""
        w, plan, k2, p = self.workers, self.plan, self.fplan.rewrites2, count_tokens(self.prompt)
        lat = self.latency
        later = (*second_stages(k2, len(pick), w, p, lat), *tail(0, 0, holdout, w, p, lat))
        if not self.fits(later):
            return []
        kept = sorted(
            (j for j in first if j.keep),
            key=lambda j: (j.found.losses - j.found.wins, count_tokens(j.rewrite.text)),
        )
        parents = [evidence(j.rewrite.text, pick, j.found.feedback) for j in kept[:_PARENTS]]
        parents = parents or [{"prompt": self.prompt, "scenarios": []}]
        self.note(f"stage R: {k2} reflection(s) on {len(parents)} candidate(s)")
        calls = [
            reflect_call(
                self.prompt,
                parents,
                contract,
                v,
                plan.models.reflect,
                plan.strictness,
                plan.allow_growth,
            )
            for v in range(k2)
        ]
        drafts = parallel_map(lambda call: self.parsed(self.ask(call)), calls, w)
        self.absorb([(f"reflection {v}", draft) for v, draft in enumerate(drafts)])
        texts = [(v, d) for v, d in enumerate(drafts) if isinstance(d, str)]
        rewrites = self.gates(texts, [j.rewrite for j in first], second=True)
        if self.ended or not rewrites or not self.fits(later[1:]):
            return []
        runs = [(r.text, 0) for r in rewrites]
        self.note(f"stage B2: {len(rewrites)} reflection(s) on {len(pick)} scenarios")
        outputs = self.stage_b(contract, runs, pick)
        if self.ended:
            return []
        found = self.stage_c(contract, rewrites, outputs, original, pick, None)
        if found is None or self.ended:
            return []
        return [win for j in found[0] if (win := won(j, noise, len(pick))) is not None]

    def stage_b(
        self,
        contract: Contract,
        runs: list[tuple[str, int]],
        pick: list[Scenario],
        then: Sequence[Stage] = (),
    ) -> list[dict[str, str | CallFailed]]:
        """One wave of task runs, each (prompt, sample) on every scenario: the evaluator's very
        calls with the fast tiers' suffix (`FastEvaluator`); each run's output or the CallFailed it
        gave per scenario (a cut is kept in `ended`). The calls end by the deadline less the
        seconds of `then`, the stages that must still fit after this one: until it ends, that is
        the deadline of the Budgeted and so the timeout of every call (ADR-011 decision 2)."""
        model = self.plan.models.task
        task = {n: FastEvaluator(self.backend, contract, model, "", n) for n in (0, 1)}
        calls = [task[sample]._task_call(text, s) for text, sample in runs for s in pick]
        budgeted, deadline = self.budgeted, self.budgeted.deadline
        budgeted.raise_limit(budgeted.limit, deadline - sum(stage.seconds for stage in then))
        try:
            results = parallel_map(self.ask, calls, self.workers)
        finally:
            budgeted.raise_limit(budgeted.limit, deadline)
        self.absorb([(f"task run {n}", result) for n, result in enumerate(results)])
        m = len(pick)
        return [
            {s.id: failed(r) for s, r in zip(pick, results[n * m : (n + 1) * m], strict=True)}
            for n in range(len(runs))
        ]

    def stage_c(
        self,
        contract: Contract,
        rewrites: list[Rewrite],
        answers: list[dict[str, str | CallFailed]],
        original: dict[str, str | CallFailed],
        pick: list[Scenario],
        noise_run: dict[str, str | CallFailed] | None,
    ) -> tuple[list[Judged], int] | None:
        """Stage C, or C2 without `noise_run`, one wave: the contract check of the rewrites (the
        longest call, first), the noise pair (the original's run 0 against `noise_run`) and each
        rewrite's `answers` against the original's run 0, each pair in two orders (SPEC R25;
        ADR-012). Each rewrite judged, and the noise (0 in stage C2, where the caller keeps stage
        C's); None when the clock or the call limit cut the noise pair. A rewrite whose order was
        cut has ties there, and a cut contract check vetoes every rewrite, so what was judged in
        time may still win (SPEC R25)."""
        model = self.plan.models.judge
        pairs = [pair_items(pick, original, mine) for mine in answers]
        if noise_run is not None:
            pairs.insert(0, items := pair_items(pick, original, noise_run))
            if not items:
                raise BackendError("the original's two runs left no scenario both answered")
        stage = "C" if noise_run is not None else "C2"
        self.note(f"stage {stage}: a contract check and {2 * len(pairs)} pairwise judge calls")
        texts = [r.text for r in rewrites]
        asks = [
            (n, call)
            for n, items in enumerate(pairs)
            if items
            for call in pair_calls(self.prompt, items, model)
        ]

        def job(index: int) -> object:
            try:
                if index < 0:
                    return check_many(self.backend, model, contract, self.prompt, texts)
                n, call = asks[index]
                names = [item[0] for item in pairs[n]]
                return _ask(self.backend, call, lambda text: parse_pairwise_batch(text, names))
            except CallFailed as error:
                if index < 0:
                    raise
                return dropped(error)
            except BudgetExhausted as error:
                return dropped(error)

        verdicts, *said = parallel_map(job, range(-1, len(asks)), self.workers)
        self.absorb(
            [("contract", verdicts), *((f"pairwise judge {i}", r) for i, r in enumerate(said))]
        )
        replies: dict[int, list[Any]] = {}
        for (n, _call), reply in zip(asks, said, strict=True):
            replies.setdefault(n, []).append(None if isinstance(reply, Dropped) else reply)
        orders = [replies.get(n, [None, None]) for n in range(len(pairs))]
        noise = 0
        if noise_run is not None:
            if None in orders[0]:
                if self.ended:
                    return None
                raise BackendError("the judge could not compare the original's two runs")
            noise = noise_count(pairs[0], *orders.pop(0))
            pairs.pop(0)
        vetoes = verdicts if isinstance(verdicts, list) else [verdicts] * len(rewrites)
        judged = [
            Judged(r, preference(items, *order), veto is None)
            for r, items, order, veto in zip(
                rewrites, pairs, orders, cast(list[Violation | None], vetoes), strict=True
            )
        ]
        return judged, noise

    def parsed(self, reply: str | Dropped) -> str | Dropped:
        """A rewrite's or a reflection's new prompt, or why there is none."""
        if isinstance(reply, Dropped):
            return reply
        try:
            return parse_rewrite(reply)
        except ValueError as error:
            return Dropped(str(error))

    # --- the clock, the calls and the log -----------------------------------------------------

    def left(self) -> tuple[float, int]:
        """The seconds to the deadline and the calls to the limit."""
        budgeted = self.budgeted
        return self.clock.remaining(budgeted.deadline), budgeted.limit - budgeted.used

    def calibrate(self, after: str) -> None:
        """The latency model fitted to every reply so far (`fast_calibrate.fit`), kept for the
        estimates that follow; nothing changes while too few replies carry a duration."""
        fitted, model = fit(self.timed.samples()), self.timed.model()
        if fitted is not None and model is not None:
            self.latency = model
            self.note(
                f"after stage {after}: a call takes {fitted.overhead_s:.1f} s plus its output "
                f"tokens at {fitted.tokens_per_s:.0f} per s, and replies are "
                f"{fitted.tokens_per_s / model.tokens_per_s:.1f} times the planned tokens"
            )

    def fits(self, stages: Sequence[Stage]) -> bool:
        """Whether `stages` fit in what is left; the cause of a misfit is kept in `stop`."""
        cause = misfit(stages, *self.left())
        self.stop = self.stop or cause
        return cause is None

    def shrunk(
        self, k: int, m: int, stages: Callable[[int, int], Sequence[Stage]]
    ) -> tuple[int, int] | None:
        """`fastplan.shrink` on what is left; the cause of a shrink is kept in `stop`."""
        shape, cause = shrink(k, m, stages, *self.left())
        self.stop = self.stop or cause
        return shape

    def ask(self, call: Call) -> str | Dropped:
        try:
            return self.backend.complete(call).text
        except (CallFailed, BudgetExhausted) as error:
            return dropped(error)

    def absorb(self, results: Sequence[tuple[str, object]]) -> None:
        """Log the calls of a wave that gave nothing, in order; a cut ends the run's stages."""
        for name, result in results:
            if isinstance(result, Dropped):
                self.note(f"{name} dropped: {result.why}")
                if result.cut:
                    self.stop, self.ended = self.stop or result.cut, True

    def note(self, line: str) -> None:
        self.log.write(f"{self.fplan.tier}: {line}\n")


def dropped(error: CallFailed | BudgetExhausted) -> Dropped:
    """A call that failed all its attempts, or one the clock or the call limit refused."""
    if isinstance(error, CallFailed):
        return Dropped(f"failed: {error}")
    return Dropped(f"cut: {error}", "clock" if error.cause == "clock" else "budget")


def failed[T](result: T | Dropped) -> T | CallFailed:
    """A call's result as the evaluator takes it: a dropped call is the CallFailed it stands for."""
    return CallFailed(result.why) if isinstance(result, Dropped) else result


def answered(outputs: Sequence[dict[str, str | CallFailed]], scenario: str) -> bool:
    """Whether the original's two runs (the first two) and a rewrite's answered `scenario`."""
    first, second, *rewrites = (isinstance(out.get(scenario), str) for out in outputs)
    return first and second and any(rewrites)


def meaning_words(text: str) -> tuple[str, ...]:
    """The words of `text` that carry its meaning, in order: lower-cased runs of letters or
    digits, without single letters and the articles a, an and the. Two prompts with the same
    meaning words differ only in case, punctuation, whitespace, single letters or articles."""
    words = _WORD.findall(text.lower())
    return tuple(w for w in words if not (len(w) == 1 and w.isalpha()) and w not in _ARTICLES)


def won(judged: Judged, noise: int, m: int) -> Win | None:
    """A judged rewrite's win over the original on `m` scenarios with `noise` of them noisy, or
    None: it kept the contract and won more scenarios than it lost by more than the noise."""
    found = judged.found
    if not judged.keep or not prefers(found, noise):
        return None
    return Win(
        judged.rewrite, found.losses / m, found.wins / m, (found.wins - found.losses) / m, noise / m
    )
