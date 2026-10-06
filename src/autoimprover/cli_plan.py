"""What `--dry` prints, and a real run shows before its first call, for every tier (SPEC R4, R17,
R25): `PlanView` for the deep tier (the models, the efforts, the budget and its fixed costs, the
scenarios and their split, the estimated GEPA iterations, the clock and the share kept for the
final steps) and `FastView` for the quick, fast and checked tiers (the tier, the time, the workers,
the models and efforts, K rewrites, M scenarios to pick on and H held out, the stages with their
calls and estimated seconds, the estimate of calls and seconds). Both say why a real run would
refuse, or keep the original without a call. Their `--json` objects share one key set, DRY_KEYS,
in one order; a key that does not apply to a tier is null. Neither makes a call or writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from autoimprover.fastplan import FastPlan
from autoimprover.report import one_line
from autoimprover.runner import FixedCosts
from autoimprover.types import (
    BUDGET_CEILING,
    LENGTH_CAP,
    LENGTH_FLOOR_TOKENS,
    SEARCH_CLOCK_SHARE,
    Plan,
)

DRY_KEYS = (
    "status",
    "plan",
    "scenarios",
    "synthesised",
    "holdout",
    "valset",
    "dataset",
    "calls_before_search",
    "calls_after_search",
    "search_calls",
    "iteration_cost",
    "iterations",
    "iterations_best",
    "search_clock_s",
    "final_clock_s",
    "refusal",
    "keeps_original",
    "tier",
    "workers",
    "efforts",
    "rewrites",
    "stages",
    "est_calls",
    "est_seconds",
)
# What a result of each fast tier is worth (SPEC R25).
EVIDENCE = {
    "quick": "a quick check: the contract check and the free gates only, no scenario scored, not "
    "verified on held-out scenarios",
    "fast": "a fast check: preferred over the original by a pairwise judge on the scenarios it is "
    "picked on, noise measured by comparing the original with itself, not verified on held-out "
    "scenarios",
    "checked": "verified on held-out scenarios on the target model, no noise measured",
}


def dry_object(**values: Any) -> dict[str, Any]:
    """A `--dry --json` object: `values` under DRY_KEYS, in their order, null for the rest."""
    unknown = sorted(set(values) - set(DRY_KEYS))
    if unknown:
        raise ValueError(f"not a key of the dry object: {unknown}")
    return {key: values.get(key) for key in DRY_KEYS}


def duration(seconds: int) -> str:
    minutes, rest = divmod(seconds, 60)
    if not rest:
        return f"{minutes} min"
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def models_line(plan: Plan) -> str:
    models = plan.models
    return (
        f"models: task {models.task}, judge {models.judge}, reflection {models.reflect}, "
        f"target {models.target}"
    )


def efforts_line(plan: Plan) -> str:
    """The `--effort` of each role; `default` leaves the model's own (SPEC R25)."""
    efforts = plan.efforts
    return (
        f"effort: task {efforts.task or 'default'}, judge {efforts.judge or 'default'}, "
        f"reflection {efforts.reflect or 'default'}"
    )


def strictness_line(plan: Plan) -> str:
    cap = (
        "no length cap (--allow-growth)"
        if plan.allow_growth
        else f"length cap {LENGTH_CAP[plan.strictness]}x the original's tokens (at least the "
        f"original plus {LENGTH_FLOOR_TOKENS})"
    )
    return f"strictness: {plan.strictness}, {cap}" + ("; GEPA merge on" if plan.merge else "")


def verdict_lines(refusal: str | None, keeps_original: str | None) -> list[str]:
    """Why a real run would refuse, or keep the original without a call (SPEC R4)."""
    lines = [] if refusal is None else [f"a real run would refuse: {one_line(refusal)}"]
    if keeps_original is not None:
        lines.append(
            f"a real run would keep the original without a model call: {one_line(keeps_original)}"
        )
    return lines


def _lines(lines: list[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


# --- the deep tier: the GEPA search (SPEC R4, R15, R17) -------------------------------------------


@dataclass(frozen=True)
class PlanView:
    """The plan of a deep run: the plan, the scenario count and whether they are synthesised, the
    fixed costs, why a real run would refuse, and why it would keep the original without a call
    (no holdout)."""

    plan: Plan
    scenarios: int
    synthesised: bool
    costs: FixedCosts
    refusal: str | None = None
    keeps_original: str | None = None

    @property
    def dataset(self) -> int:
        """Below 8 scenarios every scenario is in the dataset too (SPEC R15)."""
        if not self.costs.holdout:
            return self.scenarios
        return self.scenarios - self.costs.holdout - self.costs.valset

    @property
    def search_clock_s(self) -> int:
        return round(SEARCH_CLOCK_SHARE * self.plan.wall_clock_s)

    def text(self) -> str:
        return plan_text(self)

    def object(self) -> dict[str, Any]:
        return plan_object(self)


def plan_text(view: PlanView) -> str:
    plan, costs = view.plan, view.costs
    source = "synthesised by one call" if view.synthesised else "from --examples"
    split = (
        f"holdout {costs.holdout}, valset {costs.valset}, dataset {view.dataset}"
        if costs.holdout
        else "no holdout (fewer than 8): every scenario is both dataset and valset"
    )
    final_s = plan.wall_clock_s - view.search_clock_s
    return _lines(
        [
            f"tier: deep (--time {duration(plan.wall_clock_s)}): the GEPA search, one call at a "
            "time",
            models_line(plan),
            efforts_line(plan),
            strictness_line(plan),
            f"budget: {plan.budget} calls (ceiling {BUDGET_CEILING}); fixed costs: {costs.pre} "
            f"before the search, {costs.final} after it, {max(0, costs.search_calls)} left for "
            "the search",
            f"scenarios: {view.scenarios}, {source}; {split}",
            f"iterations: about {costs.iterations} GEPA iterations (worst case, "
            f"{costs.iter_cost} calls each) to {costs.iterations_best} (best case); an estimate: "
            "the clock may end the search sooner (live calls take 5 to 45 s)",
            f"clock: {duration(plan.wall_clock_s)}; the search may use "
            f"{duration(view.search_clock_s)}, the final steps keep {duration(final_s)}",
            *verdict_lines(view.refusal, view.keeps_original),
        ]
    )


def plan_object(view: PlanView) -> dict[str, Any]:
    costs, plan = view.costs, view.plan
    return dry_object(
        status="dry",
        plan=asdict(plan),
        scenarios=view.scenarios,
        synthesised=view.synthesised,
        holdout=costs.holdout,
        valset=costs.valset,
        dataset=view.dataset,
        calls_before_search=costs.pre,
        calls_after_search=costs.final,
        search_calls=costs.search_calls,
        iteration_cost=costs.iter_cost,
        iterations=costs.iterations,
        iterations_best=costs.iterations_best,
        search_clock_s=view.search_clock_s,
        final_clock_s=plan.wall_clock_s - view.search_clock_s,
        refusal=view.refusal,
        keeps_original=view.keeps_original,
        tier=plan.tier,
        workers=plan.workers,
        efforts=asdict(plan.efforts),
    )


# --- the quick, fast and checked tiers: the stages (SPEC R25; ADR-011) ----------------------------


@dataclass(frozen=True)
class FastView:
    """The plan of a quick, fast or checked run: the plan, its fast plan (`fastplan.fast_plan`),
    whether the scenarios are synthesised, why a real run would refuse, and why it would keep the
    original without a call."""

    plan: Plan
    fast: FastPlan
    synthesised: bool
    refusal: str | None = None
    keeps_original: str | None = None

    @property
    def scenarios(self) -> int:
        return self.fast.scenarios + self.fast.holdout

    def text(self) -> str:
        plan, fast = self.plan, self.fast
        if fast.tier == "quick":
            scenarios = "scenarios: none, the quick tier scores no scenario"
        else:
            source = "synthesised by one call" if self.synthesised else "from --examples"
            scenarios = (
                f"scenarios: {self.scenarios}, {source} ({fast.scenarios} to pick on, "
                f"{fast.holdout} held out)"
            )
        return _lines(
            [
                f"tier: {fast.tier} (--time {duration(fast.time_s)}), {fast.workers} calls at a "
                "time",
                models_line(plan),
                efforts_line(plan),
                strictness_line(plan),
                f"rewrites: {fast.rewrites}; {scenarios}",
                "stages:",
                *(
                    f"  {stage.name}: {stage.calls} calls, about {stage.seconds:.1f} s"
                    for stage in fast.stages
                ),
                f"estimate: {fast.est_calls} calls in about {fast.est_seconds:.0f} s of "
                f"{duration(fast.time_s)}; budget: {plan.budget} calls (ceiling "
                f"{BUDGET_CEILING})",
                f"evidence: {EVIDENCE[fast.tier]}",
                *verdict_lines(self.refusal, self.keeps_original),
            ]
        )

    def object(self) -> dict[str, Any]:
        plan, fast = self.plan, self.fast
        return dry_object(
            status="dry",
            plan=asdict(plan),
            scenarios=self.scenarios,
            synthesised=self.synthesised and self.scenarios > 0,
            holdout=fast.holdout,
            refusal=self.refusal,
            keeps_original=self.keeps_original,
            tier=fast.tier,
            workers=fast.workers,
            efforts=asdict(plan.efforts),
            rewrites=fast.rewrites,
            stages=[
                {"name": stage.name, "calls": stage.calls, "seconds": stage.seconds}
                for stage in fast.stages
            ],
            est_calls=fast.est_calls,
            est_seconds=fast.est_seconds,
        )
