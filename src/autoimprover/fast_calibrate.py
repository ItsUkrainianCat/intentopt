"""The in-run calibration of the fast tiers (SPEC R25; ADR-011): the latency model fitted to the
replies a run has had, and the re-plan that spends what it leaves on more pick scenarios.

The planner's constants (`fastplan.OVERHEAD_S`, `fastplan.TOKENS_PER_S`) were measured on a loaded
machine; the bench of 2026-10-06 (checked tier, `--time 2m`) planned 98 s and used 33 to 75 s.
So a fast run measures its own calls: every reply that comes back with a duration is a sample of
its output tokens and seconds, beside the output tokens the planner assumed for that call
(`Timed`, each distinct call once, so a cache hit of the same call adds nothing; a cached reply
carries the duration of its live call, so a resumed run fits what the original did). After stage
A, and again after stage B, `Timed.model` turns the samples into a model: `fit` finds the fixed
seconds and the speed of the real tokens, and since the planner applies the model to its own token
estimates, which real replies exceed (the bench: scoring runs of 280 to 617 tokens where 150 were
planned, pairwise calls of 322 to 478 where 80 were), the speed is divided by the median ratio of
real to planned tokens (`token_ratio`). Every later estimate of the run uses that model. After
stage A `grow` re-plans the rest: the most pick
scenarios, up to `fastplan.MAX_SCENARIOS`, whose remaining stages fit the time the run has left
within PLAN_SHARE of its clock (never past the deadline) and the calls it has left; from the
scenarios at hand (the user's examples) first, else with one more synthesis call.

Pure apart from `Timed`'s lock: no clock, no call.
"""

from __future__ import annotations

import itertools
import json
import statistics
import threading
from collections.abc import Sequence
from typing import NamedTuple

from autoimprover.evaluator import SCENARIO_PREFIX
from autoimprover.fastplan import (
    CONTRACT_CHECKS,
    INTAKE_TOKENS,
    JUDGE_TOKENS_PER_CHECK,
    JUDGED_CHECKS_PER_SCENARIO,
    MAX_SCENARIOS,
    PAIRWISE_TOKENS_PER_SCENARIO,
    REFERENCE_MAX_SCENARIOS,
    SYNTH_TOKENS_PER_SCENARIO,
    Latency,
    Reference,
    latency,
    misfit,
    reference_tokens,
    rewrite_tokens,
    second_stages,
    synthesis_stage,
    tail,
    task_tokens,
)
from autoimprover.pairwise_text import PAIRWISE_BATCH_SYSTEM
from autoimprover.runstore import cache_key
from autoimprover.types import Backend, Call, Reply, Tier

# Below this many timed replies nothing is fitted and the constants stand.
MIN_SAMPLES = 3
# A fitted value stays within these multiples of its constant, so one slow or one cached call
# cannot carry the model far from what was measured on the user's machine.
CLAMP = (0.5, 1.5)
# The ratio of real to planned output tokens is at least 1, so replies shorter than planned never
# make a plan more hopeful than the planner's own estimates, and at most 4: the bench's runs had a
# median of 1.5 to 2.2 over all their calls, and above 4 the estimates would leave a run no stage
# it could still afford, where the deadline already cuts what overruns (ADR-011 decision 2).
RATIO = (1.0, 4.0)


class Timed:
    """The backend a fast run's stages see: it passes every call to `inner` and keeps, for each
    distinct call (by cache key), the output tokens and seconds of its reply when the reply has a
    duration (a scripted reply has none), with the tokens the planner assumed for the call of a
    prompt of `prompt_tokens` tokens (`planned_tokens`). Errors pass through and are not
    samples."""

    def __init__(self, inner: Backend, prompt_tokens: int) -> None:
        self._inner = inner
        self._prompt_tokens = prompt_tokens
        self._samples: dict[str, tuple[int, float, float | None]] = {}
        self._lock = threading.Lock()

    def complete(self, call: Call) -> Reply:
        reply = self._inner.complete(call)
        if reply.duration_s > 0:
            planned = planned_tokens(call, self._prompt_tokens)
            with self._lock:
                self._samples.setdefault(
                    cache_key(call), (reply.tokens_out, reply.duration_s, planned)
                )
        return reply

    def samples(self) -> list[tuple[int, float]]:
        """The (output tokens, seconds) of every distinct call answered so far."""
        with self._lock:
            return [(tokens, seconds) for tokens, seconds, _planned in self._samples.values()]

    def model(self) -> Latency | None:
        """The latency model the planner's estimates take after these replies: `fit` of their
        real tokens and seconds, the speed divided by `token_ratio`; None below MIN_SAMPLES."""
        with self._lock:
            samples = list(self._samples.values())
        fitted = fit([(tokens, seconds) for tokens, seconds, _planned in samples])
        if fitted is None:
            return None
        ratio = token_ratio([(t, planned) for t, _s, planned in samples if planned])
        return Latency(fitted.overhead_s, fitted.tokens_per_s / ratio)


def planned_tokens(call: Call, prompt_tokens: int) -> float | None:
    """The output tokens `fastplan` assumes for `call` in a run of a prompt of `prompt_tokens`
    tokens, by its role and what its user message holds: the intake, a synthesis of `count`
    scenarios, a rewrite or reflection, a scoring run, a pairwise call or a judge call (the
    contract check of each `contract` scenario, a reference judge call of scenario checks only,
    by its checks, or stage E's checks of each output) over its scenarios; None for a call it
    cannot read."""
    if call.role == "intake":
        return INTAKE_TOKENS
    if call.role == "reflect":
        return rewrite_tokens(prompt_tokens)
    if call.role == "task":
        return task_tokens(prompt_tokens)
    try:
        user = json.loads(call.user)
        items = user["count"] if call.role == "synth" else len(user["scenarios"])
        names = [] if call.role == "synth" else [item["scenario"] for item in user["scenarios"]]
        asked = [] if call.role == "synth" else [_ids(item) for item in user["scenarios"]]
    except (ValueError, TypeError, KeyError):
        return None
    if call.role == "synth":
        return SYNTH_TOKENS_PER_SCENARIO * items
    if call.system == PAIRWISE_BATCH_SYSTEM:
        return PAIRWISE_TOKENS_PER_SCENARIO * items
    if asked and all(ids and all(i.startswith(SCENARIO_PREFIX) for i in ids) for ids in asked):
        return sum(reference_tokens(len(ids)) for ids in asked)  # a reference judge call
    contract = all(str(name).startswith("contract") for name in names)
    checks = CONTRACT_CHECKS if contract else JUDGED_CHECKS_PER_SCENARIO
    return JUDGE_TOKENS_PER_CHECK * checks * items


def _ids(item: object) -> list[str]:
    """The check ids of one scenario of a judge call's user JSON; TypeError when it has none."""
    checks = item.get("checks", []) if isinstance(item, dict) else None
    if not isinstance(checks, list):
        raise TypeError("a scenario without a list of checks")
    return [str(check["id"]) if isinstance(check, dict) else "" for check in checks]


def token_ratio(pairs: Sequence[tuple[float, float]]) -> float:
    """The median of real over planned output tokens of `pairs` (real, planned), within RATIO;
    1 when there is none."""
    low, high = RATIO
    ratios = [real / planned for real, planned in pairs if planned > 0]
    return min(high, max(low, statistics.median(ratios))) if ratios else low


def fit(samples: Sequence[tuple[float, float]]) -> Latency | None:
    """The latency model of `samples`, (output tokens, seconds) per reply, or None for fewer than
    MIN_SAMPLES. The speed is one over the median slope between every two replies of different
    lengths (Theil and Sen): a median slope of 0 or less is the top of the clamp, and replies all
    of one length keep the constant speed; the fixed seconds are the median of each reply's
    seconds less its tokens at that speed. Each is clamped to CLAMP times its constant. The
    result does not depend on the order of the samples."""
    if len(samples) < MIN_SAMPLES:
        return None
    constant = latency()
    low, high = CLAMP
    slopes = [
        (s2 - s1) / (t2 - t1)
        for (t1, s1), (t2, s2) in itertools.combinations(samples, 2)
        if t1 != t2
    ]
    speed = constant.tokens_per_s
    if slopes:
        slope = statistics.median(slopes)
        speed = high * speed if slope <= 0 else 1 / slope
    speed = min(high * constant.tokens_per_s, max(low * constant.tokens_per_s, speed))
    fixed = statistics.median(seconds - tokens / speed for tokens, seconds in samples)
    fixed = min(high * constant.overhead_s, max(low * constant.overhead_s, fixed))
    return Latency(fixed, speed)


class Growth(NamedTuple):
    """The re-plan after stage A: the pick `scenarios`, the `holdout` (the plan's) and how many
    scenarios one more synthesis call must add first (0: none)."""

    scenarios: int
    holdout: int
    synthesise: int


def grow(
    *,
    tier: Tier,
    rewrites: int,
    scenarios: int,
    holdout: int,
    have: int,
    can_synthesise: bool,
    rewrites2: int,
    workers: int,
    prompt_tokens: int,
    model: Latency,
    seconds_left: float,
    calls_left: int,
    ref: Reference | None = None,
) -> Growth | None:
    """More pick scenarios than the plan's `scenarios` for `rewrites` rewrites (and `rewrites2`
    reflections, `holdout` held out) when their remaining stages fit `seconds_left` and
    `calls_left` by `model` (SPEC R25): the most, up to MAX_SCENARIOS (with `ref`, the user's
    examples with references, REFERENCE_MAX_SCENARIOS and the reference stages' prices), from the
    `have` scenarios at hand (pick and held out), or with one synthesis call for the missing ones
    when `can_synthesise`; None when no larger count fits. The `tier` names the run in the plan's
    terms; the checked tier keeps its holdout."""
    if tier not in ("fast", "checked"):
        return None
    top = MAX_SCENARIOS if ref is None else REFERENCE_MAX_SCENARIOS
    for count in range(top, scenarios, -1):
        extra = max(0, count + holdout - have)
        if extra and not can_synthesise:
            continue
        stages = (synthesis_stage(extra, model),) if extra else ()
        stages += tail(rewrites, count, holdout, workers, prompt_tokens, model, ref)
        if rewrites2:
            stages += second_stages(rewrites2, count, workers, prompt_tokens, model, ref)
        if misfit(stages, seconds_left, calls_left) is None:
            return Growth(count, holdout, extra)
    return None
