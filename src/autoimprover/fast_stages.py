"""Stages B, C and D of the fast pipeline and its second generation (SPEC R10, R16, R17, R24, R25,
with the noise rule of R12; ADR-002, ADR-006, ADR-011), with the state and the helpers every stage
of a fast run shares; `fast.py` adds stage A, stage E, the quick tier and the endings.

The free gates drop a rewrite that changes no meaning word of the original or of an earlier
rewrite (`meaning_words`), one over the length cap, one that lost a literal (SPEC R7, R9). Stage B
runs the original twice (samples 0 and 1: two calls, never one cached reply) and every rewrite on
the M scenarios, all in one wave; stage C asks the evaluator's judge call for each run, which sees
outputs only (ADR-002), and one contract check of every rewrite (`contract.check_many`), in one
wave. The two runs of the original measure the noise: the difference of their mean scores on the
scenarios both completed; their mean per scenario is the baseline. A rewrite wins when it kept the
contract and beats the baseline by MORE than max(FAST_MARGIN, 2 x noise) on the scenarios it
shares with the baseline, winning on more of them than it loses.

A plan with two generations then reflects (stage R): the reflection model reads the best one or
two first-generation rewrites that gained on the baseline, or the original when none did, with
their outputs and failed checks and the judge's quotes, and writes K2 rewrites under distinct
notes; they pass the same gates, run and are judged as above (stages B2 and C2, one contract check
for all of them) against the same baseline and bar. Stage D picks over every winner of both
generations: the highest gain, a tie to the shorter rewrite, then the earlier. When the second
generation does not fit the time left or is cut, the first generation's result stands.

Before each stage the time and calls left are compared with its estimate (`fastplan`): what does
not fit is shrunk, scenarios first, and a deadline or call limit reached inside a stage ends it
(`ended`). A failed rewrite, reflection, task or judge call drops what it was for; an original run
left with no scored scenario, or a failed contract check, ends the run as BackendError (SPEC R24).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TextIO, cast

from autoimprover.backend import BudgetedBackend, Clock
from autoimprover.contract import Violation, check_many, literals_preserved
from autoimprover.evaluator import Answers, Evaluator
from autoimprover.fast_prompts import (
    REFLECT_NOTES,
    STRATEGY_NOTES,
    FastEvaluator,
    gathered,
    parse_rewrite,
    reflect_call,
    reflect_strategy,
    strategy,
)
from autoimprover.fastplan import (
    FastPlan,
    Stage,
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

# The least a rewrite must gain on the baseline when the noise is smaller than half of it (R25).
FAST_MARGIN = 0.1
# Scores are shares of checks, so a gain equal to the bar may differ from it in the last bits.
_EPS = 1e-9
# A word that carries meaning: a run of letters or digits. Single letters and articles do not.
_WORD = re.compile(r"[^\W_]+")
_ARTICLES = frozenset({"an", "the"})
# The first-generation rewrites that gained, best first, a reflection reads (or the original).
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
    """A rewrite's win: the baseline's and its mean on their common scenarios, its gain, the bar."""

    rewrite: Rewrite
    before: float
    after: float
    gain: float
    bar: float


class Graded(NamedTuple):
    """A run after stage C: its score and side info per scenario it completed, the judge's quote
    per scenario and (check id, check text), and whether it kept the contract."""

    found: dict[str, tuple[float, dict[str, Any]]]
    quotes: dict[str, dict[tuple[str, str], str]]
    keep: bool

    @property
    def scores(self) -> dict[str, float]:
        return {sid: score for sid, (score, _info) in self.found.items()}


@dataclass
class Stages:
    """One fast run: its inputs, the free gates, stages B to D and the second generation, and what
    every ending reports: `stop`, the first cause that shrank or cut a stage, and in `measured`
    the original's scores and noise once known. `ended` is set when a stage was cut short, after
    which no stage runs."""

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
        w = self.workers
        shape = self.shrunk(len(rewrites), len(pick), lambda k, m: tail(k, m, holdout, w))
        if shape is None:
            return None
        rewrites, pick = rewrites[: shape[0]], pick[: shape[1]]
        runs = [(self.prompt, 0), (self.prompt, 1), *((r.text, 0) for r in rewrites)]
        self.note(
            f"stage B: the original twice and {len(rewrites)} rewrite(s) on {shape[1]} scenarios"
        )
        if (outputs := self.stage_b(contract, runs, pick)) is None:
            return None
        ran = [any(isinstance(out, str) for out in outputs[n].values()) for n in range(len(runs))]
        if not (ran[0] and ran[1]):
            raise BackendError("every task run of one of the original's two runs failed")
        alive = [n for n in range(2, len(runs)) if ran[n]]

        def judging(k: int, m: int) -> tuple[Stage, ...]:  # stage C, then E
            return (scoring_stages(k, m, w)[1], *tail(0, 0, holdout, w))

        if (shape := self.shrunk(len(alive), len(pick), judging)) is None:
            return None
        chosen, pick = [0, 1, *alive[: shape[0]]], pick[: shape[1]]
        graded = self.stage_c(
            contract, [runs[n] for n in chosen], [outputs[n] for n in chosen], pick, 2
        )
        if (found := baseline(graded[0].scores, graded[1].scores)) is None:
            if self.ended:
                return None
            raise BackendError("the original's two runs left no scenario both of them scored")
        base, noise = found
        bar = max(FAST_MARGIN, 2 * noise)
        before = math.fsum(base.values()) / len(base)
        if holdout:  # the checked tier's score fields are its held-out scores
            self.measured["search_score_before"] = before
        else:
            self.measured |= {"score_before": before, "noise": noise}
        first = [(rewrites[n - 2], g) for n, g in zip(chosen[2:], graded[2:], strict=True)]
        wins = [
            win for r, g in first if g.keep and (win := beats(r, base, g.scores, bar)) is not None
        ]
        if self.fplan.generations > 1 and not self.ended:
            wins += self.second(contract, pick, base, bar, graded[0], first, holdout)
        if not wins:
            return None
        return min(wins, key=lambda w: (-round(w.gain, 9), count_tokens(w.rewrite.text)))

    def second(
        self,
        contract: Contract,
        pick: list[Scenario],
        base: Mapping[str, float],
        bar: float,
        original: Graded,
        first: list[tuple[Rewrite, Graded]],
        holdout: int,
    ) -> list[Win]:
        """The second generation's winners (stages R, B2, C2), or none when it does not fit the
        time and calls left or is cut: then the first generation's result stands."""
        w, plan, k2 = self.workers, self.plan, self.fplan.rewrites2
        later = (
            *second_stages(k2, len(pick), w, count_tokens(self.prompt)),
            *tail(0, 0, holdout, w),
        )
        if not self.fits(later):
            return []
        gained = [(gain(base, g.scores), r, g) for r, g in first if g.keep]
        best = sorted(
            (x for x in gained if x[0] > 0), key=lambda x: (-x[0], count_tokens(x[1].text))
        )
        parents = [evidence(r.text, pick, g) for _, r, g in best[:_PARENTS]]
        parents = parents or [evidence(self.prompt, pick, original)]
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
        rewrites = self.gates(texts, [r for r, _ in first], second=True)
        if self.ended or not rewrites or not self.fits(later[1:]):
            return []
        runs = [(r.text, 0) for r in rewrites]
        self.note(f"stage B2: {len(rewrites)} reflection(s) on {len(pick)} scenarios")
        if (outputs := self.stage_b(contract, runs, pick)) is None:
            return []
        graded = self.stage_c(contract, runs, outputs, pick, 0)
        if self.ended:
            return []
        return [
            win
            for r, g in zip(rewrites, graded, strict=True)
            if g.keep and (win := beats(r, base, g.scores, bar)) is not None
        ]

    def stage_b(
        self, contract: Contract, runs: list[tuple[str, int]], pick: list[Scenario]
    ) -> list[dict[str, str | CallFailed]] | None:
        """One wave of task runs, each (prompt, sample) on every scenario: the evaluator's very
        calls with the fast tiers' suffix (`FastEvaluator`); each run's output or the CallFailed it
        gave per scenario, None when cut."""
        model = self.plan.models.task
        task = {n: FastEvaluator(self.backend, contract, model, "", n) for n in (0, 1)}
        calls = [task[sample]._task_call(text, s) for text, sample in runs for s in pick]
        results = parallel_map(self.ask, calls, self.workers)
        self.absorb([(f"task run {n}", result) for n, result in enumerate(results)])
        if self.ended:
            return None
        m = len(pick)
        return [
            {s.id: failed(r) for s, r in zip(pick, results[n * m : (n + 1) * m], strict=True)}
            for n in range(len(runs))
        ]

    def stage_c(
        self,
        contract: Contract,
        runs: list[tuple[str, int]],
        outputs: list[dict[str, str | CallFailed]],
        pick: list[Scenario],
        first_rewrite: int,
    ) -> list[Graded]:
        """Stage C, one wave: the contract check of the runs from `first_rewrite` on, the
        rewrites (first, the longest call), and the evaluator's judge call for each run, which
        sees outputs only (ADR-002); each run graded. A failed contract check ends the run (SPEC
        R24)."""
        judge_model = self.plan.models.judge
        judge = {n: Evaluator(self.backend, contract, "", judge_model, n) for n in (0, 1)}
        judged = {s.id: judge[0]._judged(s) for s in pick}
        rewrites = [text for text, _ in runs[first_rewrite:]]
        stage = "C" if first_rewrite else "C2"  # the second generation runs no original
        self.note(f"stage {stage}: a contract check and {len(runs)} judge calls")

        def job(n: int) -> list[Violation | None] | Answers | Dropped:
            try:
                if n < 0:
                    return check_many(self.backend, judge_model, contract, self.prompt, rewrites)
                ok = {sid: out for sid, out in outputs[n].items() if isinstance(out, str)}
                pending = [s for s in pick if s.id in ok and judged[s.id]]
                if not pending:
                    return {}
                asked = {s.id: {sent for sent, _ in judged[s.id]} for s in pending}
                evaluator = judge[runs[n][1]]
                return evaluator._ask_judge(evaluator._judge_call(pending, ok, judged), asked)
            except CallFailed as error:
                if n < 0:
                    raise
                return dropped(error)
            except BudgetExhausted as error:
                return dropped(error)

        verdicts, *answers = parallel_map(job, range(-1, len(runs)), self.workers)
        self.absorb([("contract", verdicts), *((f"judge {n}", a) for n, a in enumerate(answers))])
        vetoes = verdicts if isinstance(verdicts, list) else [verdicts] * len(rewrites)
        graded = []
        for n, ((text, _), got) in enumerate(zip(runs, answers, strict=True)):
            found = gathered(contract, text, pick, outputs[n], cast(Any, failed(got)))
            said: Mapping[str, Mapping[str, tuple[bool, str]]] = (
                got if isinstance(got, dict) else {}
            )
            quotes = {
                sid: {
                    (check.id, check.text): said.get(sid, {}).get(sent, (False, ""))[1]
                    for sent, check in judged[sid]
                }
                for sid in found
            }
            keep = n < first_rewrite or vetoes[n - first_rewrite] is None
            graded.append(Graded(found, quotes, keep))
        return graded

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


def meaning_words(text: str) -> tuple[str, ...]:
    """The words of `text` that carry its meaning, in order: lower-cased runs of letters or
    digits, without single letters and the articles a, an and the. Two prompts with the same
    meaning words differ only in case, punctuation, whitespace, single letters or articles."""
    words = _WORD.findall(text.lower())
    return tuple(w for w in words if not (len(w) == 1 and w.isalpha()) and w not in _ARTICLES)


def baseline(
    run0: Mapping[str, float], run1: Mapping[str, float]
) -> tuple[dict[str, float], float] | None:
    """The baseline (the two runs' mean per scenario both scored) and the noise (the difference
    of their mean scores there), or None when they scored no scenario in common."""
    both = [sid for sid in run0 if sid in run1]
    if not both:
        return None
    noise = abs(math.fsum(run0[sid] for sid in both) - math.fsum(run1[sid] for sid in both))
    return {sid: (run0[sid] + run1[sid]) / 2 for sid in both}, noise / len(both)


def gain(base: Mapping[str, float], scores: Mapping[str, float]) -> float:
    """How much `scores` gain on the baseline on the scenarios both have; 0 with none."""
    common = [sid for sid in base if sid in scores]
    if not common:
        return 0.0
    return math.fsum(scores[sid] - base[sid] for sid in common) / len(common)


def evidence(text: str, pick: Sequence[Scenario], graded: Graded) -> dict[str, Any]:
    """What a reflection reads of a candidate: its prompt and, per scenario it completed, the
    situation, an excerpt of its output and its failed checks with the judge's quote (the
    evaluator's side info and the judge's reply, nothing asked anew; SPEC R16)."""
    scenarios = []
    for scenario in pick:
        if scenario.id not in graded.found:
            continue
        _score, info = graded.found[scenario.id]
        quotes = graded.quotes.get(scenario.id, {})
        failures = []
        for check in info.get("failed", []):
            item = {"check": check["text"]}
            if quote := quotes.get((check["id"], check["text"])):
                item["judge_quote"] = quote
            failures.append(item)
        output = info.get("output_excerpt", "")
        scenarios.append({"input": scenario.input, "output": output, "failed": failures})
    return {"prompt": text, "scenarios": scenarios}


def beats(
    rewrite: Rewrite, base: Mapping[str, float], scores: Mapping[str, float], bar: float
) -> Win | None:
    """The rewrite's win over the baseline on the scenarios both have, or None: it gains MORE than
    `bar` on the mean and wins on more scenarios than it loses (SPEC R25)."""
    common = [sid for sid in base if sid in scores]
    if not common:
        return None
    before = math.fsum(base[sid] for sid in common) / len(common)
    after = math.fsum(scores[sid] for sid in common) / len(common)
    wins = sum(scores[sid] > base[sid] for sid in common)
    losses = sum(scores[sid] < base[sid] for sid in common)
    if after - before - bar <= _EPS or wins <= losses:
        return None
    return Win(rewrite, before, after, after - before, bar)
