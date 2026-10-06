"""Stages B, C and D of the fast pipeline (SPEC R10, R17, R24, R25, with the noise rule of R12;
ADR-002, ADR-011), with the state and the helpers every stage of a fast run shares; `fast.py` adds
stage A, stage E, the quick tier and the endings.

Stage B runs the original twice (samples 0 and 1: two calls, never one cached reply) and every
rewrite on the M scenarios, all in one wave; stage C asks the evaluator's judge call for each run,
which sees outputs only (ADR-002), and one contract check of every rewrite (`contract.check_many`),
in one wave. The two runs of the original measure the noise: the difference of their mean scores
on the scenarios both completed; their mean per scenario is the baseline. Stage D returns a
rewrite only when it kept the contract and beats the baseline by MORE than max(FAST_MARGIN,
2 x noise) on the scenarios it shares with the baseline, winning on more of them than it loses;
the highest gain wins, a tie goes to the shorter rewrite, then the earlier.

Before stages B and C the time and calls left are compared with their estimate (`fastplan`): what
does not fit is shrunk, scenarios first, and a deadline or call limit reached inside a stage ends
it (`ended`). A failed task or judge call drops what it was for; an original run left with no
scored scenario, or a failed contract check, ends the run as BackendError (SPEC R24).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TextIO, cast

from autoimprover.backend import BudgetedBackend, Clock
from autoimprover.contract import Violation, check_many
from autoimprover.evaluator import Answers, Evaluator
from autoimprover.fast_prompts import gathered_scores
from autoimprover.fastplan import FastPlan, Stage, misfit, scoring_stages, shrink, tail
from autoimprover.parallel import parallel_map
from autoimprover.runner import count_tokens
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


class Rewrite(NamedTuple):
    """A rewrite that passed the free gates: its variant, its text and its length ratio."""

    variant: int
    text: str
    ratio: float


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


@dataclass
class Stages:
    """One fast run: its inputs, stages B to D and what every ending reports: `stop`, the first
    cause that shrank or cut a stage, and in `measured` the original's scores and noise once
    known. `ended` is set when a stage was cut short, after which no stage runs."""

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

    def contest(
        self, contract: Contract, rewrites: list[Rewrite], pick: list[Scenario], holdout: int
    ) -> Win | None:
        """Stages B, C and D: the best rewrite by the pick rule, or None."""
        w = self.workers
        shape = self.shrunk(len(rewrites), len(pick), lambda k, m: tail(k, m, holdout, w))
        if shape is None:
            return None
        rewrites, pick, m = rewrites[: shape[0]], pick[: shape[1]], shape[1]
        runs = [(self.prompt, 0), (self.prompt, 1), *((r.text, 0) for r in rewrites)]
        self.note(f"stage B: the original twice and {len(rewrites)} rewrite(s) on {m} scenarios")
        task = {n: Evaluator(self.backend, contract, self.plan.models.task, "", n) for n in (0, 1)}
        calls = [task[sample]._task_call(text, s) for text, sample in runs for s in pick]
        results = parallel_map(self.ask, calls, w)
        self.absorb([(f"task run {n}", result) for n, result in enumerate(results)])
        if self.ended:
            return None
        outputs = [
            {s.id: failed(r) for s, r in zip(pick, results[n * m : (n + 1) * m], strict=True)}
            for n in range(len(runs))
        ]
        ran = [any(isinstance(out, str) for out in outputs[n].values()) for n in range(len(runs))]
        if not (ran[0] and ran[1]):
            raise BackendError("every task run of one of the original's two runs failed")
        alive = [n for n in range(2, len(runs)) if ran[n]]

        def judging(k: int, m: int) -> tuple[Stage, ...]:  # stage C, then E
            return (scoring_stages(k, m, w)[1], *tail(0, 0, holdout, w))

        if (shape := self.shrunk(len(alive), m, judging)) is None:
            return None
        chosen, pick = [0, 1, *alive[: shape[0]]], pick[: shape[1]]
        scores, keeps = self.stage_c(
            contract, [runs[n] for n in chosen], [outputs[n] for n in chosen], pick
        )
        if (found := baseline(scores[0], scores[1])) is None:
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
        wins = [
            win
            for n, mine, keep in zip(chosen[2:], scores[2:], keeps[2:], strict=True)
            if keep and (win := beats(rewrites[n - 2], base, mine, bar)) is not None
        ]
        if not wins:
            return None
        return min(wins, key=lambda w: (-round(w.gain, 9), count_tokens(w.rewrite.text)))

    def stage_c(
        self,
        contract: Contract,
        runs: list[tuple[str, int]],
        outputs: list[dict[str, str | CallFailed]],
        pick: list[Scenario],
    ) -> tuple[list[dict[str, float]], list[bool]]:
        """Stage C, one wave: the contract check of every rewrite (first, the longest call) and the
        evaluator's judge call for each run, which sees outputs only (ADR-002); each run's score
        per scenario it completed, and whether it kept the contract (the original does). A failed
        contract check ends the run (SPEC R24)."""
        judge_model = self.plan.models.judge
        judge = {n: Evaluator(self.backend, contract, "", judge_model, n) for n in (0, 1)}
        judged = {s.id: judge[0]._judged(s) for s in pick}
        rewrites = [text for text, _ in runs[2:]]
        self.note(f"stage C: a contract check and {len(runs)} judge calls")

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
        scores = [
            gathered_scores(contract, text, pick, outputs[n], cast(Any, failed(got)))
            for n, ((text, _), got) in enumerate(zip(runs, answers, strict=True))
        ]
        return scores, [True, True, *(veto is None for veto in vetoes)]

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
