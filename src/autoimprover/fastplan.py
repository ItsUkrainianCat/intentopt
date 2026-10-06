"""The time tiers and the plan of a fast run (SPEC R25; ADR-011): `--time` picks the tier, and
`fast_plan` picks how many rewrites (K) and scenarios (M) the tier's stages can afford within the
time, from the measured latency model of ADR-011: a call costs OVERHEAD_S plus its output tokens
at TOKENS_PER_S, and a stage of parallel calls costs one slowest call per wave of `workers` calls.

Stages of the quick tier: the intake and one rewrite side by side, then the contract check (no
scoring). Stages of the fast and checked tiers: A one wave of the intake, the synthesis of the
scenarios (none when the user gives examples) and the K rewrites; B the original and every
rewrite run on the M scenarios; C one judge call per prompt, which sees outputs only (ADR-002),
and one contract check of every rewrite, K + 2 calls; D the free gates and the pick; E (checked
only) the winner against the original on the held-out scenarios, on the target model.

Pure and deterministic: no clock, no call. The fast runner (`fast.py`) uses the same estimates at
run time (`tail`, `misfit`, `shrink`) to shrink what the time or the calls left cannot cover
(SPEC R17).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

from autoimprover.types import StopCause

Tier = Literal["quick", "fast", "checked", "deep"]

# Where each tier starts, in seconds of `--time` (SPEC R25); below MIN_TIME_S a run is refused.
MIN_TIME_S = 15
FAST_FROM_S = 25
CHECKED_FROM_S = 60
DEEP_FROM_S = 600

# The latency model (ADR-011, the user's timing probe): a call's fixed cost and its output speed.
OVERHEAD_S = 2.4
TOKENS_PER_S = 70
# A plan may fill this share of `--time`; the rest absorbs the model's error.
PLAN_SHARE = 0.85

# Output tokens assumed per call (ADR-011 decision 2: every fast-tier reply is kept short).
INTAKE_TOKENS = 380
SYNTH_TOKENS_PER_SCENARIO = 45
REWRITE_MIN_TOKENS = 60
REWRITE_GROWTH = 1.2
REWRITE_MAX_TOKENS = 600
TASK_TOKENS = 200
JUDGE_TOKENS_PER_CHECK = 25  # a pass/fail and a quote of at most 8 words
JUDGED_CHECKS_PER_SCENARIO = 3
CONTRACT_CHECKS = 3  # the contract check's three fixed questions (contract.check)

# The shapes a tier may take: rewrites from the most down to 1, scenarios from the most down to
# the fewest, and the held-out scenarios of the checked tier.
MAX_REWRITES: dict[Tier, int] = {"quick": 1, "fast": 3, "checked": 6}
MAX_SCENARIOS = 4
MIN_SCENARIOS = 2
CHECKED_HOLDOUT = 4


class Stage(NamedTuple):
    """One stage of a plan: its name, its calls (retries not counted) and its estimated seconds."""

    name: str
    calls: int
    seconds: float


@dataclass(frozen=True)
class FastPlan:
    """What a quick, fast or checked run will do; `--dry` prints it (SPEC R4, R25). `rewrites` is
    K, `scenarios` M (the scenarios the rewrites are picked on, 0 in the quick tier), `holdout`
    the held-out scenarios of the checked tier (else 0)."""

    tier: Tier
    time_s: int
    workers: int
    rewrites: int
    scenarios: int
    holdout: int
    stages: tuple[Stage, ...]
    est_seconds: float
    est_calls: int


def tier_for(time_s: int) -> Tier:
    """The tier `--time` selects: quick from 15 s, fast from 25 s, checked from 60 s, deep from
    600 s; below 15 s ValueError (SPEC R25)."""
    if time_s < MIN_TIME_S:
        raise ValueError(f"--time must be at least {MIN_TIME_S} s, not {time_s} s")
    if time_s < FAST_FROM_S:
        return "quick"
    if time_s < CHECKED_FROM_S:
        return "fast"
    if time_s < DEEP_FROM_S:
        return "checked"
    return "deep"


def call_seconds(tokens: float) -> float:
    """The estimated seconds of one call that writes `tokens` output tokens (ADR-011)."""
    return OVERHEAD_S + tokens / TOKENS_PER_S


def wave_seconds(calls: int, workers: int, slowest: float) -> float:
    """Seconds of `calls` parallel calls on `workers` threads: one slowest call per wave."""
    return math.ceil(calls / workers) * slowest if calls else 0.0


def rewrite_tokens(prompt_tokens: int) -> float:
    """A rewrite is about as long as the prompt (ADR-011 decision 2), within fixed bounds."""
    return min(REWRITE_MAX_TOKENS, max(REWRITE_MIN_TOKENS, REWRITE_GROWTH * prompt_tokens))


def judge_seconds(scenarios: int) -> float:
    """One judge call over the outputs of `scenarios` scenarios."""
    return call_seconds(JUDGE_TOKENS_PER_CHECK * JUDGED_CHECKS_PER_SCENARIO * scenarios)


def contract_seconds(candidates: int) -> float:
    """One contract check of `candidates` rewrites (`contract.check_many`)."""
    return call_seconds(JUDGE_TOKENS_PER_CHECK * CONTRACT_CHECKS * candidates)


def stage_a(rewrites: int, synthesis: int, workers: int, prompt_tokens: int) -> Stage:
    """Stage A, one wave: the intake, the synthesis of `synthesis` scenarios (none for 0) and the
    rewrites; none of them needs another's reply."""
    slowest = max(
        call_seconds(INTAKE_TOKENS),
        call_seconds(rewrite_tokens(prompt_tokens)),
        call_seconds(SYNTH_TOKENS_PER_SCENARIO * synthesis),
    )
    calls = 1 + rewrites + (1 if synthesis else 0)
    name = "A: intake, synthesis and " if synthesis else "A: intake and "
    name += "rewrite" if rewrites == 1 else "rewrites"
    return Stage(name, calls, wave_seconds(calls, workers, slowest))


def scoring_stages(rewrites: int, scenarios: int, workers: int) -> tuple[Stage, ...]:
    """Stages B, C and D for the original and `rewrites` rewrites on `scenarios` scenarios; stage
    C is a judge call per prompt and one contract check of every rewrite."""
    prompts = rewrites + 1
    judging = max(judge_seconds(scenarios), contract_seconds(rewrites))
    return (
        Stage(
            "B: task runs",
            prompts * scenarios,
            wave_seconds(prompts * scenarios, workers, call_seconds(TASK_TOKENS)),
        ),
        Stage(
            "C: judge and contract checks", prompts + 1, wave_seconds(prompts + 1, workers, judging)
        ),
        Stage("D: free gates and pick", 0, 0.0),
    )


def holdout_stage(holdout: int, workers: int) -> Stage:
    """Stage E: the winner and the original on `holdout` held-out scenarios on the target model,
    a wave of task runs, then a judge call each."""
    seconds = wave_seconds(2 * holdout, workers, call_seconds(TASK_TOKENS))
    seconds += wave_seconds(2, workers, judge_seconds(holdout))
    return Stage("E: held-out check on the target model", 2 * holdout + 2, seconds)


def contract_stage() -> Stage:
    """The quick tier's contract check of its one rewrite."""
    return Stage("contract check", 1, contract_seconds(1))


def fast_plan(time_s: int, workers: int, prompt_tokens: int, have_examples: bool) -> FastPlan:
    """The plan of a quick, fast or checked run of `time_s` seconds on `workers` threads for a
    prompt of `prompt_tokens` tokens (`runner.count_tokens`), with the user's examples or with a
    synthesis call. Fast and checked take the most rewrites, then the most scenarios, whose
    estimate fits PLAN_SHARE of the time; when even 1 rewrite on MIN_SCENARIOS does not fit, that
    smallest plan is returned and the runner shrinks it at run time. The deep tier is the search
    (`runner.improve`), not a fast plan: ValueError."""
    tier = tier_for(time_s)
    if tier == "deep":
        raise ValueError(f"{time_s} s is the deep tier: the search of runner.improve")
    if workers < 1 or prompt_tokens < 0:
        raise ValueError("workers must be at least 1 and prompt_tokens not negative")
    if tier == "quick":
        stages = (stage_a(1, 0, workers, prompt_tokens), contract_stage())
        return _plan(tier, time_s, workers, 1, 0, 0, stages)
    holdout = CHECKED_HOLDOUT if tier == "checked" else 0
    plans = []
    for rewrites in range(MAX_REWRITES[tier], 0, -1):
        for scenarios in range(MAX_SCENARIOS, MIN_SCENARIOS - 1, -1):
            synthesis = 0 if have_examples else scenarios + holdout
            stages = (stage_a(rewrites, synthesis, workers, prompt_tokens),)
            stages += scoring_stages(rewrites, scenarios, workers)
            if holdout:
                stages += (holdout_stage(holdout, workers),)
            plans.append(_plan(tier, time_s, workers, rewrites, scenarios, holdout, stages))
    return next((p for p in plans if p.est_seconds <= PLAN_SHARE * time_s), plans[-1])


def _plan(
    tier: Tier,
    time_s: int,
    workers: int,
    rewrites: int,
    scenarios: int,
    holdout: int,
    stages: tuple[Stage, ...],
) -> FastPlan:
    return FastPlan(
        tier=tier,
        time_s=time_s,
        workers=workers,
        rewrites=rewrites,
        scenarios=scenarios,
        holdout=holdout,
        stages=stages,
        est_seconds=sum(stage.seconds for stage in stages),
        est_calls=sum(stage.calls for stage in stages),
    )


# --- what is left at run time ---------------------------------------------------------------------


def tail(rewrites: int, scenarios: int, holdout: int, workers: int) -> tuple[Stage, ...]:
    """The stages after stage A for a shape: B to D (none for 0 rewrites), then E for a holdout."""
    stages = scoring_stages(rewrites, scenarios, workers) if rewrites else ()
    return (*stages, *((holdout_stage(holdout, workers),) if holdout else ()))


def misfit(stages: Sequence[Stage], seconds_left: float, calls_left: int) -> StopCause | None:
    """Why `stages` cannot run in what is left: "clock" when their seconds pass `seconds_left`,
    else "budget" when their calls pass `calls_left`; None when they fit."""
    if sum(stage.seconds for stage in stages) > seconds_left:
        return "clock"
    if sum(stage.calls for stage in stages) > calls_left:
        return "budget"
    return None


def shrink(
    rewrites: int,
    scenarios: int,
    stages: Callable[[int, int], Sequence[Stage]],
    seconds_left: float,
    calls_left: int,
) -> tuple[tuple[int, int] | None, StopCause | None]:
    """The largest (rewrites, scenarios) up to the given ones whose `stages` fit in what is left,
    scenarios shrunk first, with the cause that shrank it (None when nothing was shrunk); None
    for the shape when not even (1, 1) fits."""
    cause: StopCause | None = None
    for k in range(rewrites, 0, -1):
        for m in range(scenarios, 0, -1):
            why = misfit(stages(k, m), seconds_left, calls_left)
            if why is None:
                return (k, m), cause
            cause = cause or why
    return None, cause
