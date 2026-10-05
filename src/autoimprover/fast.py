"""The pipeline of the quick, fast and checked tiers (SPEC R25; ADR-011) and the gates every
rewrite it returns has passed (SPEC R6, R7, R9, R14a, R17, R24).

Stage A runs the intake beside the K rewrites, then the synthesis, which needs the intake's
contract (the user's examples replace it); a rewrite failing a free gate (length cap, literals) is
dropped there, before it costs a call. Stage B runs the original and every rewrite on the M
scenarios in one wave; stage C asks one judge call per prompt in one wave, a rewrite's contract
check riding in it (ADR-011 decision 3). Stage D returns a rewrite only when it kept the contract
and beats the original's mean judged score by at least FAST_MARGIN on the scenarios both
completed, winning on more of them than it loses; the highest gain wins, a tie goes to the
shorter, then the earlier. Stage E (checked tier) runs the winner against the original on the
held-out scenarios on the target model, decided as `runner` decides with MIN_THRESHOLD as the
margin (SPEC R14a). The quick tier only checks its one rewrite's contract.

Task and judge calls are the evaluator's, scored by its own code (SPEC R10). A stage's calls run
on `workers` threads, gathered by index, so the Outcome does not depend on the order they end in;
finished calls are cached by content (SPEC R22). Before each stage the time and calls left are
compared with its estimate (`fastplan`): what does not fit is shrunk, scenarios first, and a
deadline or call limit reached inside a stage ends it; a rewrite that has not passed every gate is
never returned (SPEC R17). A failed rewrite, task or judge call drops what it was for; a failed
intake, synthesis, quick contract check or held-out run, or an original with no scored scenario,
ends the run as BackendError (SPEC R24).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TextIO, cast

from autoimprover.backend import BudgetedBackend, Clock
from autoimprover.contract import _contract_checks, check, extract_contract, literals_preserved
from autoimprover.evaluator import Answers, Evaluator
from autoimprover.fast_prompts import (
    CONTRACT_SCENARIO,
    STRATEGY_NOTES,
    gathered_scores,
    keeps_contract,
    parse_rewrite,
    rewrite_call,
    stage_c_call,
    strategy,
)
from autoimprover.fastplan import (
    FastPlan,
    Stage,
    contract_stage,
    misfit,
    scoring_stages,
    shrink,
    synthesis_stage,
    tail,
)
from autoimprover.parallel import parallel_map
from autoimprover.runner import MIN_THRESHOLD, count_tokens, length_ok, score_holdout
from autoimprover.runstore import RunStore
from autoimprover.scenarios import synthesize
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
        contract, rewrites = self.stage_a(kind)
        if self.fplan.tier == "quick":
            return self.quick(contract, rewrites)
        if not rewrites:
            return self.kept("no rewrite passed the free gates (length cap, literals)")
        if given is None and (given := self.synthesis(contract)) is None:
            return self.kept("")
        if self.store.scenarios() is None:
            self.store.save_scenarios(given)
        m, h = self.fplan.scenarios, self.fplan.holdout
        pick = [s for s in given[:m] if s.id != CONTRACT_SCENARIO]  # that id is the judge's own
        if not pick:
            return self.kept("no scenario left to compare the rewrites on")
        win = self.contest(contract, rewrites, pick, h)
        if win is None:
            return self.kept(_NO_WIN)
        if self.fplan.tier == "checked":
            return self.confirm(contract, win, given[m : m + h])
        return self.outcome(
            "improved",
            win.rewrite,
            why=_FAST_LABEL,
            score_before=win.before,
            score_after=win.after,
            margin=win.gain - FAST_MARGIN,
        )

    # --- stage A -----------------------------------------------------------------------------

    def stage_a(self, kind: Kind | None) -> tuple[Contract, list[_Rewrite]]:
        """The contract (the run folder's, or a new intake) and the rewrites that pass the free
        gates; the intake and the rewrite calls in one wave."""
        saved, k = self.store.contract(), self.fplan.rewrites
        jobs: list[int | None] = ([] if saved else [None]) + list(range(k))
        self.note(f"stage A: {'intake and ' if saved is None else ''}{k} rewrite call(s)")

        def job(variant: int | None) -> Contract | str | _Dropped:
            if variant is None:
                return extract_contract(self.backend, self.plan.models.reflect, self.prompt, kind)
            return self.draft(variant)

        results = parallel_map(job, jobs, self.workers)
        contract = saved or cast(Contract, results[0])
        if saved is None:
            self.store.save_contract(contract)
        drafts = cast(list[str | _Dropped], results[len(jobs) - k :])
        self.absorb([(f"rewrite {v}", draft) for v, draft in enumerate(drafts)])
        return contract, self.gates([(v, d) for v, d in enumerate(drafts) if isinstance(d, str)])

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

    def synthesis(self, contract: Contract) -> list[Scenario] | None:
        """The scenarios to pick on, then those to hold out, from one synthesis call; None when
        not even the smallest rest of the run fits after it."""
        count = self.fplan.scenarios + self.fplan.holdout
        if not self.fits((synthesis_stage(count), *tail(1, 1, self.fplan.holdout, self.workers))):
            return None
        self.note(f"stage A: synthesis of {count} scenarios")
        return synthesize(self.backend, self.plan.models.reflect, self.prompt, contract, count)

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
            {s.id: _output(r) for s, r in zip(pick, results[n * m : (n + 1) * m], strict=True)}
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
        """Stage C: one judge call per prompt, in one wave (the original's first); each prompt's
        score per scenario it completed, and whether it kept the contract."""
        evaluator = Evaluator(self.backend, contract, "", self.plan.models.judge)
        judged = {s.id: evaluator._judged(s) for s in pick}
        checks = _contract_checks(contract)
        self.note(f"stage C: {len(texts)} judge calls")

        def judge(n: int) -> Answers | _Dropped:
            ok = {sid: out for sid, out in outputs[n].items() if isinstance(out, str)}
            pending = [s for s in pick if s.id in ok and judged[s.id]]
            asked = {s.id: {sent for sent, _ in judged[s.id]} for s in pending}
            if n:
                asked[CONTRACT_SCENARIO] = {check_id for check_id, _ in checks}
            elif not pending:
                return {}
            base = evaluator._judge_call(pending, ok, judged)
            call = stage_c_call(base, self.prompt, texts[n] if n else None, checks)
            try:
                return evaluator._ask_judge(call, asked)
            except (CallFailed, BudgetExhausted) as error:
                return _dropped(error)

        answers = parallel_map(judge, range(len(texts)), self.workers)
        self.absorb([(f"judge call {n}", answer) for n, answer in enumerate(answers)])
        scores, keeps = [], []
        for n, (text, got) in enumerate(zip(texts, answers, strict=True)):
            found = {} if isinstance(got, _Dropped) else dict(got)
            verdicts = found.pop(CONTRACT_SCENARIO, {})
            judged_by = CallFailed(got.why) if isinstance(got, _Dropped) else found
            scores.append(gathered_scores(contract, text, pick, outputs[n], judged_by))
            keeps.append(n == 0 or keeps_contract(verdicts, checks, text))
        return scores, keeps

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
        return self.clock.remaining(
            self.budgeted.deadline
        ), self.budgeted.limit - self.budgeted.used

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
            **fields,
        )


def _dropped(error: CallFailed | BudgetExhausted) -> _Dropped:
    """A call that failed all its attempts, or one the clock or the call limit refused."""
    if isinstance(error, CallFailed):
        return _Dropped(f"failed: {error}")
    return _Dropped(f"cut: {error}", "clock" if error.cause == "clock" else "budget")


def _output(result: str | _Dropped) -> str | CallFailed:
    """A task run's result as the evaluator takes it: a dropped run is the CallFailed it stands
    for."""
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
