"""Stages C, C2, D and E of the fast and checked tiers when every example carries a reference (SPEC
R10, R10b, R11, R16, R24, R25; ADR-002, WP21): absolute agreement with the user's references
decides in place of the pairwise preference (`fast_stages`, `fast_pairwise`, ADR-012).

Stage C is one wave: the contract check of every rewrite (`contract.check_many`, as in the pairwise
stage C) and one judge call per run over the M scenarios, the original's run 0 and run 1 and each
rewrite's run, by the evaluator's own machinery (`Evaluator.score`: the judge sees the input and the
output and never the prompt, ADR-002; the quote rule of SPEC R10b), with only the references'
checks: `expected` is "the output agrees with the reference answer in substance" and each
`criteria` string one check (SPEC R11); the contract's own checks are not asked. A scenario's score
is the share of its checks passed. A rewrite wins by `reference_score.beats` against the original's
two runs on the scenarios all three scored in full; stage D picks the largest mean gain, a tie to
the shorter rewrite, then the earlier; with `--ungated` and no winner, the rewrite that kept the
contract with the largest mean gain, then the most scenarios improved, then the fewest tokens, a
rewrite with no score last. Stage C2 judges the reflections' runs only: the original's scores of
stage C stand. The reflection reads, per parent (the best one or two rewrites that kept the
contract, by mean gain, else the original), the examples where it failed a check, with the
reference, the start of the output and the failed checks (`fast_prompts.reference_evidence`).
Stage E (checked tier) runs the original twice and the winner on the held-out examples on the
target model and compares them the same way; a failed call there ends the run (SPEC R24).

A judge call that fails all its attempts leaves its run's scenarios unscored (a rewrite then has
no score and cannot win); when the original's two runs have no scenario both scored, the run ends
as BackendError unless the clock or the call limit cut the stage. With no reference (`ref` None)
every method here is the pairwise one of `Stages`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from autoimprover.contract import check_many
from autoimprover.evaluator import Evaluator
from autoimprover.fast_pairwise import Judged, Preference, Rewrite, Win
from autoimprover.fast_prompts import FastEvaluator, reference_evidence
from autoimprover.fast_stages import PARENTS, Dropped, Stages, dropped
from autoimprover.parallel import parallel_map
from autoimprover.reference_score import Margin, beats, compare, scored
from autoimprover.runner import count_tokens
from autoimprover.types import (
    BackendError,
    BudgetExhausted,
    CallFailed,
    Contract,
    Scenario,
)

Entries = list[tuple[float, dict[str, Any]]]


def shares(rewrite: Rewrite, margin: Margin | None, gated: bool = True) -> Win:
    """A rewrite's margin as a Win in means over the scenarios compared: the original's mean
    score (`before`), the rewrite's (`after`), the gain and the noise (`bar`); all 0 without a
    scenario compared."""
    if margin is None or not margin.scenarios:
        return Win(rewrite, 0.0, 0.0, 0.0, 0.0, gated)
    m = margin.scenarios
    gain = (margin.after - margin.before) / m
    return Win(rewrite, margin.before / m, margin.after / m, gain, margin.noise / m, gated)


@dataclass
class ReferenceStages(Stages):
    """`Stages` deciding by the references when `ref` is set: `base` holds the scores of the
    original's two runs from stage C, `graded` the evaluator entries of every prompt judged (the
    reflection's evidence)."""

    base: tuple[dict[str, float], dict[str, float]] | None = field(default=None, init=False)
    graded: dict[str, Entries] = field(default_factory=dict, init=False)

    def stage_c(
        self,
        contract: Contract,
        rewrites: list[Rewrite],
        answers: list[dict[str, str | CallFailed]],
        original: dict[str, str | CallFailed],
        pick: list[Scenario],
        noise_run: dict[str, str | CallFailed] | None,
    ) -> tuple[list[Judged], float] | None:
        """Stage C, or C2 without `noise_run`: the contract check and a reference judge call per
        run (the original's two in stage C), each judged rewrite with its margin, and the summed
        noise of the original's two runs (0 in C2); None when a cut left the original unscored."""
        if self.ref is None:
            return super().stage_c(contract, rewrites, answers, original, pick, noise_run)
        model = self.plan.models.judge
        runs = [] if noise_run is None else [(original, 0), (noise_run, 1)]
        runs += [(mine, 0) for mine in answers]
        stage = "C" if noise_run is not None else "C2"
        self.note(f"stage {stage}: a contract check and {len(runs)} reference judge calls")
        texts, bare = [r.text for r in rewrites], dataclasses.replace(contract, checks=())

        def job(index: int) -> object:
            try:
                if index < 0:
                    return check_many(self.backend, model, contract, self.prompt, texts)
                outputs, sample = runs[index]
                return Evaluator(self.backend, bare, "", model, sample).score(pick, outputs)
            except BudgetExhausted as error:
                return dropped(error)

        verdicts, *said = parallel_map(job, range(-1, len(runs)), self.workers)
        self.absorb(
            [("contract", verdicts), *((f"reference judge {i}", g) for i, g in enumerate(said))]
        )
        graded = cast(list[Entries | Dropped], said)
        noise = 0.0
        if noise_run is not None:
            first, second, *graded = graded
            if (found := self.originals(first, second)) is None:
                return None
            noise = found
        vetoes = verdicts if isinstance(verdicts, list) else [verdicts] * len(rewrites)
        judged = []
        for rewrite, entries, veto in zip(rewrites, graded, vetoes, strict=True):
            margin = None
            if not isinstance(entries, Dropped) and self.base is not None:
                self.graded[rewrite.text] = entries
                margin = compare(*self.base, scored(entries))
            counts = (0, 0, 0) if margin is None else margin[3:6]
            judged.append(Judged(rewrite, Preference(*counts, {}), veto is None, margin))
        return judged, noise

    def originals(self, first: Entries | Dropped, second: Entries | Dropped) -> float | None:
        """Keep the scores of the original's two runs from their entries and return their summed
        noise; None when a cut left them unscored; BackendError when the judge could not score
        both on any scenario."""
        if isinstance(first, Dropped) or isinstance(second, Dropped):
            return None
        a, b = scored(first), scored(second)
        common = [name for name in a if name in b]
        if not common:
            if self.ended:
                return None
            raise BackendError(
                "the judge could not score the original's two runs against the references"
            )
        self.base, self.graded[self.prompt] = (a, b), first
        return abs(math.fsum(a[n] for n in common) - math.fsum(b[n] for n in common))

    def won(self, judged: Judged, noise: float, m: int) -> Win | None:
        if self.ref is None:
            return super().won(judged, noise, m)
        if not judged.keep or judged.score is None or not beats(judged.score):
            return None
        return shares(judged.rewrite, judged.score)

    def best(self, judged: Sequence[Judged], noise: float, m: int) -> Win | None:
        if self.ref is None:
            return super().best(judged, noise, m)
        kept = [j for j in judged if j.keep]
        if not kept:
            return None

        def rank(j: Judged) -> tuple[bool, float, int, int]:
            gain = shares(j.rewrite, j.score).gain
            improved = 0 if j.score is None else j.score.improved
            return (j.score is None, -round(gain, 9), -improved, count_tokens(j.rewrite.text))

        chosen = min(kept, key=rank)
        return shares(chosen.rewrite, chosen.score, gated=False)

    def parents(self, first: list[Judged], pick: list[Scenario]) -> list[dict[str, Any]]:
        if self.ref is None:
            return super().parents(first, pick)
        kept = sorted(
            (j for j in first if j.keep and j.score is not None),
            key=lambda j: (
                -round(shares(j.rewrite, j.score).gain, 9),
                count_tokens(j.rewrite.text),
            ),
        )
        texts = [j.rewrite.text for j in kept[:PARENTS]] or [self.prompt]
        return [reference_evidence(text, pick, self.graded.get(text, [])) for text in texts]

    def held_out(self, contract: Contract, text: str, holdout: Sequence[Scenario]) -> Margin:
        """Stage E: the original twice (samples 0 and 1) and `text` on `holdout` on the target
        model, each judged against the references, and `text`'s margin; a failed call is
        BackendError (SPEC R24)."""
        models, inner = self.plan.models, max(1, self.workers // 3)
        bare = dataclasses.replace(contract, checks=())

        def on_target(job: tuple[str, int]) -> list[tuple[float, dict[str, Any]]]:
            prompt, sample = job
            judge = FastEvaluator(self.backend, bare, models.target, models.judge, sample, inner)
            return judge(prompt, holdout)

        jobs = [(self.prompt, 0), (self.prompt, 1), (text, 0)]
        runs = parallel_map(on_target, jobs, self.workers)
        for entries in runs:
            for _score, info in entries:
                if info.get("incomplete"):
                    why = info.get("error", "no reason given")
                    raise BackendError(f"a call failed while scoring on the holdout: {why}")
        first, second, mine = (scored(entries) for entries in runs)
        return compare(first, second, mine)
