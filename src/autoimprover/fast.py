"""The pipeline of the quick, fast and checked tiers (SPEC R25; ADR-011) and the gates every
rewrite it returns has passed (SPEC R6, R7, R9, R14a, R17, R24).

Stage A is one wave: the intake, the synthesis of the scenarios (the user's examples replace it)
and the K rewrites, none of which needs another's reply; a rewrite failing a free gate (length
cap, literals) is dropped there, before it costs a call. Stage B runs the original and every
rewrite on the M scenarios in one wave. Stage C is one wave too: the evaluator's judge call for
each prompt, which sees outputs only (ADR-002), and one contract check of every rewrite
(`contract.check_many`). Stage D returns a rewrite only when it kept the contract and beats the
original's mean judged score by at least FAST_MARGIN on the scenarios both completed, winning on
more of them than it loses; the highest gain wins, a tie goes to the shorter, then the earlier.
Stage E (checked tier) runs the winner against the original on the held-out scenarios on the
target model, decided as `runner` decides with MIN_THRESHOLD as the margin (SPEC R14a). The quick
tier only checks its one rewrite's contract.

Task and judge calls are the evaluator's, scored by its own code (SPEC R10). A stage's calls run
on `workers` threads, gathered by index, so the Outcome does not depend on the order they end in;
finished calls are cached by content (SPEC R22). Before stages B, C and E the time and calls left
are compared with their estimate (`fastplan`): what does not fit is shrunk, scenarios first, and
a deadline or call limit reached inside a stage ends it; a rewrite that has not passed every gate
is never returned (SPEC R17). A failed rewrite, task or judge call drops what it was for; a failed
intake, synthesis, contract check or held-out run, or an original with no scored scenario, ends
the run as BackendError (SPEC R24).

DEBT, private names used here and in `fast_prompts` until their owners add public seams:
`evaluator.Evaluator._task_call`, `._judged`, `._judge_call`, `._ask_judge` (and `._run`, `._judge`
overridden by `fast_prompts._Gathered`), `contract._ask`, `runner._token_cap`, `scenarios._loads`,
`scenarios._text_problem`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TextIO, cast

from autoimprover.backend import BudgetedBackend, Clock
from autoimprover.contract import (
    Violation,
    _ask,
    check,
    check_many,
    extract_contract,
    literals_preserved,
)
from autoimprover.evaluator import Answers, Evaluator
from autoimprover.fast_prompts import (
    STRATEGY_NOTES,
    gathered_scores,
    parse_rewrite,
    parse_synth,
    rewrite_call,
    strategy,
    synth_call,
)
from autoimprover.fastplan import (
    FastPlan,
    Stage,
    contract_stage,
    misfit,
    scoring_stages,
    shrink,
    tail,
)
from autoimprover.parallel import parallel_map
from autoimprover.runner import MIN_THRESHOLD, count_tokens, length_ok, score_holdout
from autoimprover.runstore import RunStore
from autoimprover.types import (
    Backend,
    BackendError,
    BudgetExhausted,
    Call,
    CallFailed,
    Contract,
    Kind,
    Outcome,
    Plan,
    Scenario,
    StopCause,
)

# The least a rewrite must gain on the original's mean judged score (SPEC R25).
FAST_MARGIN = 0.1
# Scores are shares of checks; a gain this close to the margin meets it (0.6 - 0.5 < 0.1 in floats).
_EPS = 1e-9
_FAST_LABEL = (
    "fast check: scored on the same few scenarios it was picked on, not verified on held-out "
    "scenarios, no noise measured"
)
_QUICK_LABEL = (
    "quick check: passed the contract check and the free gates, not scored on any scenario, not "
    "verified on held-out scenarios, no noise measured"
)
_NO_WIN = (
    f"no rewrite kept the contract and beat the original by at least {FAST_MARGIN} on more "
    "scenarios than it lost"
)
_CUT = {
    "clock": "the clock ran out before a rewrite passed every gate",
    "budget": "the call limit ran out before a rewrite passed every gate",
}


class _Rewrite(NamedTuple):
    """A rewrite that passed the free gates: its variant, its text and its length ratio."""

    variant: int
    text: str
    ratio: float


class _Dropped(NamedTuple):
    """Why a call of a stage gave nothing, and the cause when the clock or the limit cut it."""

    why: str
    cut: StopCause | None = None


class _Win(NamedTuple):
    rewrite: _Rewrite
    before: float
    after: float
    gain: float


def improve_fast(
    prompt: str,
    plan: Plan,
    fplan: FastPlan,
    *,
    backend: Backend,
    budgeted: BudgetedBackend,
    clock: Clock,
    store: RunStore,
    scenarios: Sequence[Scenario] | None,
    kind: Kind | None,
    log: TextIO,
    workers: int,
) -> Outcome:
    """Improve `prompt` by `fplan` (SPEC R25). `backend` is the run's stack (the cache, or a layer
    above it), `budgeted` the Budgeted inside it, whose limit and deadline bind the run; `clock`
    the run's clock, `store` its open run folder. `scenarios` are the user's examples, or None to
    synthesise them; the run folder's contract and scenarios win over a new intake, `kind` and
    `scenarios` (a resumed run). `log` gets a line per stage and per dropped call; `workers` is
    how many calls run at a time. The checked tier with nothing left to hold out keeps the
    original before any call."""
    run = _Fast(prompt, plan, fplan, backend, budgeted, clock, store, log, workers)
    given = store.scenarios() or (None if scenarios is None else list(scenarios))
    if fplan.tier == "checked" and given is not None and len(given) <= fplan.scenarios:
        return run.outcome(
            "no_holdout",
            why=f"the checked tier holds out the examples after the first {fplan.scenarios}, "
            "and there are none; give more examples, or a shorter --time",
        )
    try:
        return run.flow(given, kind)
    except BudgetExhausted as error:  # the intake, the synthesis, the contract check or stage E
        run.stop = run.stop or _dropped(error).cut
        return run.kept("")
    except CallFailed as error:
        raise BackendError(str(error)) from error


@dataclass
class _Fast:
    """One fast run: its inputs, its stages and what every ending reports: `stop`, the first cause
    that shrank or cut a stage, and in `measured` the original's scores once known. `ended` is set
    when a stage was cut short, after which no stage runs."""

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

    def flow(self, given: list[Scenario] | None, kind: Kind | None) -> Outcome:
        contract, scenarios, rewrites = self.stage_a(kind, given)
        if self.fplan.tier == "quick":
            return self.quick(contract, rewrites)
        if self.store.scenarios() is None:
            self.store.save_scenarios(scenarios)
        if not rewrites:
            return self.kept("no rewrite passed the free gates (length cap, literals)")
        m, h = self.fplan.scenarios, self.fplan.holdout
        win = self.contest(contract, rewrites, scenarios[:m], h)
        if win is None:
            return self.kept(_NO_WIN)
        if self.fplan.tier == "checked":
            return self.confirm(contract, win, scenarios[m : m + h])
        return self.outcome(
            "improved",
            win.rewrite,
            why=_FAST_LABEL,
            score_before=win.before,
            score_after=win.after,
            margin=win.gain - FAST_MARGIN,
        )

    # --- stage A -----------------------------------------------------------------------------

    def stage_a(
        self, kind: Kind | None, given: list[Scenario] | None
    ) -> tuple[Contract, list[Scenario], list[_Rewrite]]:
        """In one wave: the contract (the run folder's, or a new intake), the scenarios (`given`,
        else one synthesis call; none in the quick tier) and the rewrites that pass the free
        gates."""
        saved, fp, reflect = self.store.contract(), self.fplan, self.plan.models.reflect
        k, count = fp.rewrites, 0 if given else fp.scenarios + fp.holdout  # quick: 0 + 0
        jobs = (["intake"] if saved is None else []) + (["synth"] if count else []) + [*range(k)]
        self.note(f"stage A: {', '.join(map(str, jobs))}")

        def job(name: str | int) -> Contract | list[Scenario] | str | _Dropped:
            if name == "intake":
                return extract_contract(self.backend, reflect, self.prompt, kind)
            if name == "synth":
                call = synth_call(self.prompt, count, reflect)
                return _ask(self.backend, call, lambda text: parse_synth(text, count))
            return self.draft(cast(int, name))

        results = dict(zip(jobs, parallel_map(job, jobs, self.workers), strict=True))
        contract = saved or cast(Contract, results["intake"])
        if saved is None:
            self.store.save_contract(contract)
        drafts = [cast(str | _Dropped, results[v]) for v in range(k)]
        self.absorb([(f"rewrite {v}", draft) for v, draft in enumerate(drafts)])
        rewrites = self.gates([(v, d) for v, d in enumerate(drafts) if isinstance(d, str)])
        return contract, given or cast(list[Scenario], results.get("synth", [])), rewrites

    def draft(self, variant: int) -> str | _Dropped:
        plan = self.plan
        call = rewrite_call(
            self.prompt, variant, plan.models.reflect, plan.strictness, plan.allow_growth
        )
        reply = self.ask(call)
        if isinstance(reply, _Dropped):
            return reply
        try:
            return parse_rewrite(reply)
        except ValueError as error:
            return _Dropped(str(error))

    def gates(self, drafts: list[tuple[int, str]]) -> list[_Rewrite]:
        """The rewrites that pass the free gates, in variant order: neither the original nor an
        earlier rewrite again, within the length cap, every literal kept (SPEC R7, R9)."""
        kept: list[_Rewrite] = []
        seen = {self.prompt}
        for variant, text in drafts:
            fits, ratio = length_ok(self.prompt, text, self.plan.strictness, self.plan.allow_growth)
            if text in seen:
                why = "repeats the original or an earlier rewrite"
            elif not fits:
                why = "is longer than the length cap"
            elif not literals_preserved(self.prompt, text):
                why = "loses a literal of the original"
            else:
                kept.append(_Rewrite(variant, text, ratio))
                seen.add(text)
                continue
            self.note(f"rewrite {variant} dropped: it {why}")
        return kept

    # --- stages B, C and D -------------------------------------------------------------------

    def contest(
        self, contract: Contract, rewrites: list[_Rewrite], pick: list[Scenario], holdout: int
    ) -> _Win | None:
        """Stages B, C and D: the best rewrite by the pick rule, or None."""
        w = self.workers
        shape = self.shrunk(len(rewrites), len(pick), lambda k, m: tail(k, m, holdout, w))
        if shape is None:
            return None
        rewrites, pick = rewrites[: shape[0]], pick[: shape[1]]
        texts, m = [self.prompt, *(r.text for r in rewrites)], len(pick)
        self.note(f"stage B: {len(texts)} prompts on {m} scenarios")
        evaluator = Evaluator(self.backend, contract, self.plan.models.task, "")
        calls = [evaluator._task_call(text, s) for text in texts for s in pick]  # its very calls
        results = parallel_map(self.ask, calls, self.workers)
        self.absorb([(f"task run {n}", result) for n, result in enumerate(results)])
        if self.ended:
            return None
        outputs = [
            {s.id: _failed(r) for s, r in zip(pick, results[n * m : (n + 1) * m], strict=True)}
            for n in range(len(texts))
        ]
        ran = [any(isinstance(out, str) for out in outputs[n].values()) for n in range(len(texts))]
        if not ran[0]:
            raise BackendError("every task run of the original failed")
        alive = [n for n in range(1, len(texts)) if ran[n]]

        def judging(k: int, m: int) -> tuple[Stage, ...]:  # stage C, then E
            return (scoring_stages(k, m, w)[1], *tail(0, 0, holdout, w))

        if (shape := self.shrunk(len(alive), m, judging)) is None:
            return None
        chosen, pick = [0, *alive[: shape[0]]], pick[: shape[1]]
        scores, keeps = self.stage_c(
            contract, [texts[n] for n in chosen], [outputs[n] for n in chosen], pick
        )
        if not scores[0] and self.ended:
            return None
        if not scores[0]:
            raise BackendError("no scenario of the original was scored")
        on_pick = "search_score_before" if holdout else "score_before"  # score_*: the holdout's
        self.measured[on_pick] = math.fsum(scores[0].values()) / len(scores[0])
        wins = [
            win
            for n, mine, keep in zip(chosen[1:], scores[1:], keeps[1:], strict=True)
            if keep and (win := _beats(rewrites[n - 1], scores[0], mine)) is not None
        ]
        if not wins:
            return None
        return min(wins, key=lambda w: (-round(w.gain, 9), count_tokens(w.rewrite.text)))

    def stage_c(
        self,
        contract: Contract,
        texts: list[str],
        outputs: list[dict[str, str | CallFailed]],
        pick: list[Scenario],
    ) -> tuple[list[dict[str, float]], list[bool]]:
        """Stage C, one wave: the contract check of every rewrite (first, the longest call) and the
        evaluator's judge call for each prompt, which sees outputs only (ADR-002); each prompt's
        score per scenario it completed, and whether it kept the contract (the original does). A
        failed contract check ends the run (SPEC R24)."""
        judge_model = self.plan.models.judge
        evaluator = Evaluator(self.backend, contract, "", judge_model)
        judged = {s.id: evaluator._judged(s) for s in pick}
        self.note(f"stage C: a contract check and {len(texts)} judge calls")

        def job(n: int) -> list[Violation | None] | Answers | _Dropped:
            try:
                if n < 0:
                    return check_many(self.backend, judge_model, contract, self.prompt, texts[1:])
                ok = {sid: out for sid, out in outputs[n].items() if isinstance(out, str)}
                pending = [s for s in pick if s.id in ok and judged[s.id]]
                if not pending:
                    return {}
                asked = {s.id: {sent for sent, _ in judged[s.id]} for s in pending}
                return evaluator._ask_judge(evaluator._judge_call(pending, ok, judged), asked)
            except CallFailed as error:
                if n < 0:
                    raise
                return _dropped(error)
            except BudgetExhausted as error:
                return _dropped(error)

        verdicts, *answers = parallel_map(job, range(-1, len(texts)), self.workers)
        self.absorb([("contract", verdicts), *((f"judge {n}", a) for n, a in enumerate(answers))])
        vetoes = verdicts if isinstance(verdicts, list) else [verdicts] * (len(texts) - 1)
        scores = [
            gathered_scores(contract, text, pick, outputs[n], cast(Any, _failed(got)))
            for n, (text, got) in enumerate(zip(texts, answers, strict=True))
        ]
        return scores, [True, *(veto is None for veto in vetoes)]

    # --- stage E and the quick tier ----------------------------------------------------------

    def confirm(self, contract: Contract, win: _Win, holdout: Sequence[Scenario]) -> Outcome:
        """Stage E: the winner is returned, verified, only when it beats the original on the
        held-out scenarios on the target model by more than MIN_THRESHOLD (SPEC R3, R14a)."""
        self.measured["search_score_before"] = win.before
        if not self.fits(tail(0, 0, len(holdout), self.workers)):
            return self.kept("")
        self.note(f"stage E: the winner and the original on {len(holdout)} held-out scenarios")
        models, inner = self.plan.models, max(1, self.workers // 2)

        def on_target(text: str) -> float:
            evaluator = Evaluator(self.backend, contract, models.target, models.judge, 0, inner)
            return score_holdout(evaluator, text, holdout)

        before, after = parallel_map(on_target, [self.prompt, win.rewrite.text], self.workers)
        self.measured["score_before"] = before
        margin = after - before - MIN_THRESHOLD
        held = f"{len(holdout)} held-out scenarios, on the target model, by more than"
        if margin <= 0:
            return self.kept(f"the winner did not beat the original on {held} {MIN_THRESHOLD}")
        return self.outcome(
            "improved",
            win.rewrite,
            why=f"beats the original on {held} {MIN_THRESHOLD}; no noise measured",
            verified=True,
            score_after=after,
            search_score_after=win.after,
            margin=margin,
        )

    def quick(self, contract: Contract, rewrites: list[_Rewrite]) -> Outcome:
        """The quick tier: its rewrite is returned when it passes the contract check (SPEC R6)."""
        if not rewrites:
            return self.kept("the rewrite did not pass the free gates (length cap, literals)")
        if not self.fits((contract_stage(),)):
            return self.kept("")
        self.note("the contract check of the rewrite")
        if check(self.backend, self.plan.models.judge, contract, self.prompt, rewrites[0].text):
            return self.kept("the rewrite did not pass the contract check")
        return self.outcome("improved", rewrites[0], why=_QUICK_LABEL)

    # --- the clock, the calls and the endings -------------------------------------------------

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

    def ask(self, call: Call) -> str | _Dropped:
        try:
            return self.backend.complete(call).text
        except (CallFailed, BudgetExhausted) as error:
            return _dropped(error)

    def absorb(self, results: Sequence[tuple[str, object]]) -> None:
        """Log the calls of a wave that gave nothing, in order; a cut ends the run's stages."""
        for name, result in results:
            if isinstance(result, _Dropped):
                self.note(f"{name} dropped: {result.why}")
                if result.cut:
                    self.stop, self.ended = self.stop or result.cut, True

    def note(self, line: str) -> None:
        self.log.write(f"{self.fplan.tier}: {line}\n")

    def kept(self, why: str) -> Outcome:
        """The original, unchanged: for `why`, or out of time or calls when a stage was shrunk
        or cut and no rewrite passed every gate (SPEC R17)."""
        if self.stop:
            return self.outcome("unconfirmed_out_of_budget", why=_CUT[self.stop])
        return self.outcome("no_reliable_improvement", why=why)

    def outcome(
        self, code: str, rewrite: _Rewrite | None = None, why: str = "", **fields: Any
    ) -> Outcome:
        fields = {**self.measured, **fields}
        if rewrite is not None:
            fields["changes"] = (STRATEGY_NOTES[strategy(rewrite.variant)],)
            fields["length_ratio"] = rewrite.ratio
        return Outcome(
            status="unchanged" if rewrite is None else "improved",
            prompt=self.prompt if rewrite is None else rewrite.text,
            reason=f"tier {self.fplan.tier}: {why}",
            reason_code=code,
            stop=self.stop,
            calls_used=self.budgeted.used,
            run_dir=str(self.store.path),
            mode=self.fplan.tier,
            **fields,
        )


def _dropped(error: CallFailed | BudgetExhausted) -> _Dropped:
    """A call that failed all its attempts, or one the clock or the call limit refused."""
    if isinstance(error, CallFailed):
        return _Dropped(f"failed: {error}")
    return _Dropped(f"cut: {error}", "clock" if error.cause == "clock" else "budget")


def _failed[T](result: T | _Dropped) -> T | CallFailed:
    """A call's result as the evaluator takes it: a dropped call is the CallFailed it stands for."""
    return CallFailed(result.why) if isinstance(result, _Dropped) else result


def _beats(
    rewrite: _Rewrite, original: Mapping[str, float], scores: Mapping[str, float]
) -> _Win | None:
    """The rewrite's win over the original on the scenarios both completed, or None: it gains at
    least FAST_MARGIN on the mean and wins on more scenarios than it loses (SPEC R25)."""
    common = [sid for sid in original if sid in scores]
    if not common:
        return None
    before = math.fsum(original[sid] for sid in common) / len(common)
    after = math.fsum(scores[sid] for sid in common) / len(common)
    wins = sum(scores[sid] > original[sid] for sid in common)
    losses = sum(scores[sid] < original[sid] for sid in common)
    if after - before < FAST_MARGIN - _EPS or wins <= losses:
        return None
    return _Win(rewrite, before, after, after - before)
