"""The time tiers and the plan of a fast run (SPEC R25; ADR-011): `--time` picks the tier, and
`fast_plan` picks how many rewrites (K) and scenarios (M) the tier's stages can afford within the
time, from the measured latency model of ADR-011: a call costs OVERHEAD_S plus its output tokens
at TOKENS_PER_S, and a stage of parallel calls costs one slowest call per wave of `workers` calls.
A rewrite's output and a scoring run's grow with the prompt (`rewrite_tokens`, `task_tokens`).

Stages of the quick tier: the intake and one rewrite side by side, then the contract check (no
scoring). Stages of the fast and checked tiers: A one wave of the intake, the synthesis of the
scenarios (none when the user gives examples) and the K rewrites; B the original twice (its two
runs measure the noise) and every rewrite on the M scenarios, (K + 2) M task runs; C two
pairwise judge calls (both orders) per rewrite and for the original's two runs, each over the M
scenarios and seeing answers only (ADR-002, ADR-012), and one contract check of every rewrite,
2 K + 3 calls; from 45 s, when the whole plan fits, a second generation: R the reflection model
writes K2 rewrites from the judge's reasons on the first, B2 and C2 run and judge them;
D the free gates and the pick over every candidate; E (checked only) the winner against the
original on the held-out scenarios, on the target model.

When every example of the user carries a reference (`Reference`), stage C is one absolute judge
call per run over its M scenarios (the original's two and each rewrite's, ADR-002) and the
contract check, K + 3 calls, each longer than a pairwise call as it quotes per check
(`reference_seconds`); C2 is K2 + 1 calls; E runs the original twice and the winner, 3 H task runs
and 3 judge calls (WP21). Such a plan picks on up to REFERENCE_MAX_SCENARIOS of the examples (and,
in the checked tier, holds out REFERENCE_HOLDOUT of them) before any shape of the ordinary split,
as long as the examples last.

Pure and deterministic: no clock, no call. The fast runner (`fast.py`) uses the same estimates at
run time (`tail`, `misfit`, `shrink`) to shrink what the time or the calls left cannot cover
(SPEC R17), each with the latency model fitted to the run's own replies once it has one
(`Latency`, `fast_calibrate`), and to grow the pick scenarios into what that model leaves.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from autoimprover.types import HOLDOUT_MAX, JUDGE_BATCH_MAX, StopCause, Tier

# Where each tier starts, in seconds of `--time` (SPEC R25); below MIN_TIME_S a run is refused.
MIN_TIME_S = 15
FAST_FROM_S = 25
CHECKED_FROM_S = 60
DEEP_FROM_S = 600

# The latency model (ADR-011, the user's timing probe): a call's fixed cost and its output speed.
# The fixed cost is the start-up measured on the user's loaded machine with the trims of ADR-009's
# amendment of 2026-10-06 (median 3.4 s per trivial call).
OVERHEAD_S = 3.4
TOKENS_PER_S = 70
# A plan may fill this share of `--time`; the rest absorbs the model's error.
PLAN_SHARE = 0.85

# Output tokens assumed per call (ADR-011 decision 2: every fast-tier reply is kept short).
INTAKE_TOKENS = 380
SYNTH_TOKENS_PER_SCENARIO = 45
REWRITE_MIN_TOKENS = 60
REWRITE_GROWTH = 1.2
REWRITE_MAX_TOKENS = 600
# A scoring run asks for at most 120 words (fast_prompts.FAST_TASK_SUFFIX), which binds for a short
# prompt but not for one that asks for a full document: task_tokens(p) = 150 for p <= 50 prompt
# tokens (`runner.count_tokens`), 150 + 3 (p - 50) above, at most 600 (from p = 200). From the
# fast scoring runs of the live benches of 2026-10-06, read as (seconds - OVERHEAD_S) x
# TOKENS_PER_S: run medians 84 to 273 for prompts of 18 to 46 tokens, 357 to 602 for prompts of 135
# and 148 (formula: 405 and 444); the slowest run of the 148-token prompt took 13.9 s, 735.
TASK_MIN_TOKENS = 150
TASK_FROM_PROMPT_TOKENS = 50
TASK_GROWTH = 3
TASK_MAX_TOKENS = 600
JUDGE_TOKENS_PER_CHECK = 25  # a pass/fail and a quote of at most 8 words
JUDGED_CHECKS_PER_SCENARIO = 3
PAIRWISE_TOKENS_PER_SCENARIO = 40  # a winner and one short reason (ADR-012)
# A reference judge call (WP21): per scenario its name and list, then a pass and a quote per check.
REFERENCE_TOKENS_PER_SCENARIO = 20
CONTRACT_CHECKS = 3  # the contract check's three fixed questions (contract.check)

# The shapes a tier may take: rewrites from the most down to 1, scenarios from the most down to
# the fewest, and the held-out scenarios of the checked tier.
MAX_REWRITES: dict[Tier, int] = {"quick": 1, "fast": 3, "checked": 6}
MAX_SCENARIOS = 4
MIN_SCENARIOS = 2
CHECKED_HOLDOUT = 4
# The checked tier's split: when the time allows, it picks on MAX_SCENARIOS scenarios and holds
# out CHECKED_HOLDOUT down to CHECKED_MIN_HOLDOUT, and only when no such plan fits does it pick on
# fewer, with CHECKED_MIN_HOLDOUT held out (the bench of 2026-10-06 picked among six rewrites on 2
# scenarios with 4 held out: a pick by noise). Each tier's groups of (scenarios, holdout), best
# group first; a plan of an earlier group always comes before one of a later group.
CHECKED_MIN_HOLDOUT = 2
_SPLITS: dict[Tier, tuple[tuple[tuple[int, int], ...], ...]] = {
    "fast": (tuple((m, 0) for m in range(MAX_SCENARIOS, MIN_SCENARIOS - 1, -1)),),
    "checked": (
        tuple((MAX_SCENARIOS, h) for h in range(CHECKED_HOLDOUT, CHECKED_MIN_HOLDOUT - 1, -1)),
        tuple((m, CHECKED_MIN_HOLDOUT) for m in range(MAX_SCENARIOS - 1, MIN_SCENARIOS - 1, -1)),
    ),
}
# With references (WP21) a plan picks on up to one judge batch of the examples and holds out from
# HOLDOUT_MAX down to CHECKED_HOLDOUT of them, before the ordinary split, as the examples last.
REFERENCE_MAX_SCENARIOS = JUDGE_BATCH_MAX
REFERENCE_HOLDOUT = range(HOLDOUT_MAX, CHECKED_HOLDOUT - 1, -1)
# From this `--time` a run has a second generation of up to MAX_REWRITES2 rewrites, written by
# reflecting on the first one's failed checks, when the whole plan fits (SPEC R25; ADR-011).
TWO_GENERATIONS_FROM_S = 45
MAX_REWRITES2 = 2


class Stage(NamedTuple):
    """One stage of a plan: its name, its calls (retries not counted) and its estimated seconds."""

    name: str
    calls: int
    seconds: float


class Reference(NamedTuple):
    """The user's examples when every one carries a reference (`expected` or `criteria`): how
    many there are, and the most judged checks one of them has (SPEC R11). A plan with one
    decides by agreement with the references, not by pairwise preference (SPEC R25, WP21)."""

    examples: int
    checks: int


@dataclass(frozen=True)
class FastPlan:
    """What a quick, fast or checked run will do; `--dry` prints it (SPEC R4, R25). `rewrites` is
    K, `scenarios` M (the scenarios the rewrites are picked on, 0 in the quick tier), `holdout`
    the held-out scenarios of the checked tier (else 0), `generations` 2 when a second
    generation of `rewrites2` rewrites follows the first (else 1 and 0), `reference` the user's
    examples when the plan decides by their references (else None, the pairwise decision)."""

    tier: Tier
    time_s: int
    workers: int
    rewrites: int
    scenarios: int
    holdout: int
    stages: tuple[Stage, ...]
    est_seconds: float
    est_calls: int
    generations: int = 1
    rewrites2: int = 0
    reference: Reference | None = None


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


class Latency(NamedTuple):
    """The latency model of a call (ADR-011): its fixed seconds and its output tokens per second.
    The planner uses the constants (`latency`); a fast run fits its own to the replies it has had
    (`fast_calibrate.fit`) and passes it to every estimate of what is left (SPEC R25)."""

    overhead_s: float
    tokens_per_s: float


def latency() -> Latency:
    """The constants of the latency model, OVERHEAD_S and TOKENS_PER_S."""
    return Latency(OVERHEAD_S, TOKENS_PER_S)


def call_seconds(tokens: float, model: Latency | None = None) -> float:
    """The estimated seconds of one call that writes `tokens` output tokens (ADR-011), by `model`
    (None: the constants)."""
    overhead_s, tokens_per_s = model or latency()
    return overhead_s + tokens / tokens_per_s


def wave_seconds(calls: int, workers: int, slowest: float) -> float:
    """Seconds of `calls` parallel calls on `workers` threads: one slowest call per wave."""
    return math.ceil(calls / workers) * slowest if calls else 0.0


def rewrite_tokens(prompt_tokens: int) -> float:
    """A rewrite is about as long as the prompt (ADR-011 decision 2), within fixed bounds."""
    return min(REWRITE_MAX_TOKENS, max(REWRITE_MIN_TOKENS, REWRITE_GROWTH * prompt_tokens))


def task_tokens(prompt_tokens: int) -> float:
    """The output of a scoring run grows with the prompt it runs, within fixed bounds (the
    formula and its evidence are at TASK_MIN_TOKENS)."""
    grown = TASK_MIN_TOKENS + TASK_GROWTH * (prompt_tokens - TASK_FROM_PROMPT_TOKENS)
    return min(TASK_MAX_TOKENS, max(TASK_MIN_TOKENS, grown))


# Every estimate below takes the latency model as `model`, None for the constants.


def judge_seconds(scenarios: int, model: Latency | None = None) -> float:
    """One judge call over the outputs of `scenarios` scenarios."""
    return call_seconds(JUDGE_TOKENS_PER_CHECK * JUDGED_CHECKS_PER_SCENARIO * scenarios, model)


def pair_seconds(scenarios: int, model: Latency | None = None) -> float:
    """One pairwise judge call over the two answers on each of `scenarios` scenarios."""
    return call_seconds(PAIRWISE_TOKENS_PER_SCENARIO * scenarios, model)


def contract_seconds(candidates: int, model: Latency | None = None) -> float:
    """One contract check of `candidates` rewrites (`contract.check_many`)."""
    return call_seconds(JUDGE_TOKENS_PER_CHECK * CONTRACT_CHECKS * candidates, model)


def reference_tokens(checks: int) -> float:
    """The output tokens of one scenario of `checks` checks in a reference judge call."""
    return REFERENCE_TOKENS_PER_SCENARIO + JUDGE_TOKENS_PER_CHECK * checks


def reference_seconds(scenarios: int, checks: int, model: Latency | None = None) -> float:
    """One reference judge call over the outputs of `scenarios` scenarios of `checks` checks."""
    return call_seconds(reference_tokens(checks) * scenarios, model)


def _judges(
    pairs: int, runs: int, scenarios: int, ref: Reference | None, model: Latency | None
) -> tuple[int, float, str]:
    """The judge calls of a stage C on `scenarios` scenarios, the seconds of one and their kind:
    the two orders of each of `pairs` pairs, or with references one call per each of `runs`
    runs."""
    if ref is None:
        return 2 * pairs, pair_seconds(scenarios, model), "pairwise"
    return runs, reference_seconds(scenarios, ref.checks, model), "reference"


def stage_a(
    rewrites: int, synthesis: int, workers: int, prompt_tokens: int, model: Latency | None = None
) -> Stage:
    """Stage A, one wave: the intake, the synthesis of `synthesis` scenarios (none for 0) and the
    rewrites; none of them needs another's reply."""
    slowest = max(
        call_seconds(INTAKE_TOKENS, model),
        call_seconds(rewrite_tokens(prompt_tokens), model),
        call_seconds(SYNTH_TOKENS_PER_SCENARIO * synthesis, model),
    )
    calls = 1 + rewrites + (1 if synthesis else 0)
    name = "A: intake, synthesis and " if synthesis else "A: intake and "
    name += "rewrite" if rewrites == 1 else "rewrites"
    return Stage(name, calls, wave_seconds(calls, workers, slowest))


def synthesis_stage(count: int, model: Latency | None = None) -> Stage:
    """A second synthesis wave of a fast run, before stage B: one call for `count` more
    scenarios, when the calibrated model has time for them (SPEC R25)."""
    return Stage("A2: more scenarios", 1, call_seconds(SYNTH_TOKENS_PER_SCENARIO * count, model))


def scoring_stages(
    rewrites: int,
    scenarios: int,
    workers: int,
    prompt_tokens: int,
    model: Latency | None = None,
    ref: Reference | None = None,
) -> tuple[Stage, ...]:
    """Stages B, C and D for the original, run twice, and `rewrites` rewrites on `scenarios`
    scenarios; stage C is two pairwise calls (both orders) per rewrite and for the original's two
    runs (ADR-012), or with `ref` one reference judge call per run, and one contract check of
    every rewrite."""
    runs = rewrites + 2
    judges, seconds, kind = _judges(rewrites + 1, runs, scenarios, ref, model)
    judging = max(seconds, contract_seconds(rewrites, model))
    calls = judges + 1
    task = call_seconds(task_tokens(prompt_tokens), model)
    return (
        Stage("B: task runs", runs * scenarios, wave_seconds(runs * scenarios, workers, task)),
        Stage(f"C: {kind} judge and contract checks", calls, wave_seconds(calls, workers, judging)),
        Stage("D: free gates and pick", 0, 0.0),
    )


def second_stages(
    rewrites2: int,
    scenarios: int,
    workers: int,
    prompt_tokens: int,
    model: Latency | None = None,
    ref: Reference | None = None,
) -> tuple[Stage, ...]:
    """The second generation: `rewrites2` reflections (each a rewrite's length), their task runs
    on the `scenarios` scenarios, then two pairwise calls each against the original's answers (or
    with `ref` one reference judge call each) and one contract check of all."""
    reflection = call_seconds(rewrite_tokens(prompt_tokens), model)
    judges, seconds, kind = _judges(rewrites2, rewrites2, scenarios, ref, model)
    judging = max(seconds, contract_seconds(rewrites2, model))
    runs = rewrites2 * scenarios
    calls = judges + 1
    return (
        Stage(
            "R: reflection on the first generation",
            rewrites2,
            wave_seconds(rewrites2, workers, reflection),
        ),
        Stage(
            "B2: task runs of the second generation",
            runs,
            wave_seconds(runs, workers, call_seconds(task_tokens(prompt_tokens), model)),
        ),
        Stage(
            f"C2: {kind} judge and contract checks of the second generation",
            calls,
            wave_seconds(calls, workers, judging),
        ),
    )


def holdout_stage(
    holdout: int,
    workers: int,
    prompt_tokens: int,
    model: Latency | None = None,
    ref: Reference | None = None,
) -> Stage:
    """Stage E: the winner and the original (with `ref`, the original twice) on `holdout`
    held-out scenarios on the target model, a wave of task runs, then a judge call each."""
    runs = 2 if ref is None else 3
    task = call_seconds(task_tokens(prompt_tokens), model)
    judge = (
        judge_seconds(holdout, model)
        if ref is None
        else reference_seconds(holdout, ref.checks, model)
    )
    seconds = wave_seconds(runs * holdout, workers, task) + wave_seconds(runs, workers, judge)
    return Stage("E: held-out check on the target model", runs * holdout + runs, seconds)


def contract_stage() -> Stage:
    """The quick tier's contract check of its one rewrite."""
    return Stage("contract check", 1, contract_seconds(1))


def fast_plan(
    time_s: int,
    workers: int,
    prompt_tokens: int,
    have_examples: bool,
    reference: Reference | None = None,
) -> FastPlan:
    """The plan of a quick, fast or checked run of `time_s` seconds on `workers` threads for a
    prompt of `prompt_tokens` tokens (`runner.count_tokens`), with the user's examples or with a
    synthesis call, and with `reference` when every example carries one. Fast and checked take
    the most rewrites, then the most scenarios (the checked tier: the first group of its split
    that has a plan that fits, then the most held out; with `reference` first the group of the
    larger picks the examples allow), whose estimate fits PLAN_SHARE of the time; from
    TWO_GENERATIONS_FROM_S the most rewrites, then scenarios, then second-generation rewrites of
    a plan with two generations come first, and one generation only when none fits. When even 1
    rewrite on MIN_SCENARIOS does not fit, that smallest plan is returned and the runner shrinks
    it at run time. The deep tier is the search (`runner.improve`), not a fast plan:
    ValueError."""
    tier = tier_for(time_s)
    if tier == "deep":
        raise ValueError(f"{time_s} s is the deep tier: the search of runner.improve")
    if workers < 1 or prompt_tokens < 0:
        raise ValueError("workers must be at least 1 and prompt_tokens not negative")
    if tier == "quick":
        stages = (stage_a(1, 0, workers, prompt_tokens), contract_stage())
        return _plan(tier, time_s, workers, 1, 0, 0, stages)
    ordered: list[FastPlan] = []
    second = range(MAX_REWRITES2, 0, -1) if time_s >= TWO_GENERATIONS_FROM_S else range(0)
    w, p, ref = workers, prompt_tokens, reference
    for group in _splits(tier, ref):
        one: list[FastPlan] = []
        two: list[FastPlan] = []
        for rewrites in range(MAX_REWRITES[tier], 0, -1):
            for scenarios, holdout in group:
                synthesis = 0 if have_examples else scenarios + holdout
                *scoring, pick = scoring_stages(rewrites, scenarios, w, p, ref=ref)
                first = (stage_a(rewrites, synthesis, w, p), *scoring)
                last = (pick, *tail(0, 0, holdout, w, p, ref=ref))
                shape = (tier, time_s, w, rewrites, scenarios, holdout)
                one.append(_plan(*shape, (*first, *last), reference=ref))
                for rewrites2 in second:
                    stages = (*first, *second_stages(rewrites2, scenarios, w, p, ref=ref))
                    two.append(_plan(*shape, (*stages, *last), rewrites2, ref))
        ordered += (*two, *one)
    fitting = (p for p in ordered if p.est_seconds <= PLAN_SHARE * time_s)
    return next(fitting, ordered[-1])


def _splits(tier: Tier, ref: Reference | None) -> tuple[tuple[tuple[int, int], ...], ...]:
    """The groups of (scenarios, holdout) of `tier`, best first: with `ref`, first the larger
    picks of REFERENCE_MAX_SCENARIOS (and in the checked tier REFERENCE_HOLDOUT held out) that its
    examples cover, then the ordinary split."""
    if ref is None or tier not in _SPLITS:
        return _SPLITS[tier]
    if tier == "checked":
        wide = [(REFERENCE_MAX_SCENARIOS, holdout) for holdout in REFERENCE_HOLDOUT]
    else:
        wide = [(m, 0) for m in range(REFERENCE_MAX_SCENARIOS, MAX_SCENARIOS, -1)]
    covered = tuple((m, h) for m, h in wide if m + h <= ref.examples)
    return ((covered,) if covered else ()) + _SPLITS[tier]


def _plan(
    tier: Tier,
    time_s: int,
    workers: int,
    rewrites: int,
    scenarios: int,
    holdout: int,
    stages: tuple[Stage, ...],
    rewrites2: int = 0,
    reference: Reference | None = None,
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
        generations=2 if rewrites2 else 1,
        rewrites2=rewrites2,
        reference=reference,
    )


# --- what is left at run time ---------------------------------------------------------------------


def tail(
    rewrites: int,
    scenarios: int,
    holdout: int,
    workers: int,
    prompt_tokens: int,
    model: Latency | None = None,
    ref: Reference | None = None,
) -> tuple[Stage, ...]:
    """The stages after stage A for a shape of a prompt of `prompt_tokens` tokens: B to D (none
    for 0 rewrites), then E for a holdout; with `ref` judged by the references."""
    w, p = workers, prompt_tokens
    stages = scoring_stages(rewrites, scenarios, w, p, model, ref) if rewrites else ()
    held = (holdout_stage(holdout, w, p, model, ref),) if holdout else ()
    return (*stages, *held)


def last_chance(
    holdout: int,
    workers: int,
    prompt_tokens: int,
    model: Latency | None = None,
    ref: Reference | None = None,
) -> tuple[Stage, ...]:
    """What a stage B that ended late still needs to decide (SPEC R25): stage C's calls for one
    rewrite on one scenario, in the time of one pairwise call (with `ref` one reference call),
    then E for a holdout."""
    judging = scoring_stages(1, 1, workers, prompt_tokens, model, ref)[1]
    held = tail(0, 0, holdout, workers, prompt_tokens, model, ref)
    one = pair_seconds(1, model) if ref is None else reference_seconds(1, ref.checks, model)
    return (judging._replace(seconds=one), *held)


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
