"""The in-run calibration of the fast tiers (SPEC R25; ADR-011): the latency model fitted to the
replies a run has had, and the re-plan that spends what it leaves on more pick scenarios.

The planner's constants (`fastplan.OVERHEAD_S`, `fastplan.TOKENS_PER_S`) were measured on a loaded
machine; the bench of 2026-10-06 (checked tier, `--time 2m`) planned 98 s and used 33 to 75 s.
So a fast run measures its own calls: every reply that comes back with a duration is a sample of
its output tokens and seconds (`Timed`, each distinct call once, so a cache hit of the same call
adds nothing; a cached reply carries the duration of its live call, so a resumed run fits what
the original did). After stage A, and again after stage B, `fit` turns the samples into a model,
and every later estimate of the run uses it. After stage A `grow` re-plans the rest: the most pick
scenarios, up to `fastplan.MAX_SCENARIOS`, whose remaining stages fit the time the run has left
within PLAN_SHARE of its clock (never past the deadline) and the calls it has left; from the
scenarios at hand (the user's examples) first, else with one more synthesis call.

Pure apart from `Timed`'s lock: no clock, no call.
"""

from __future__ import annotations

import itertools
import statistics
import threading
from collections.abc import Sequence
from typing import NamedTuple

from autoimprover.fastplan import (
    MAX_SCENARIOS,
    Latency,
    latency,
    misfit,
    second_stages,
    synthesis_stage,
    tail,
)
from autoimprover.runstore import cache_key
from autoimprover.types import Backend, Call, Reply, Tier

# Below this many timed replies nothing is fitted and the constants stand.
MIN_SAMPLES = 3
# A fitted value stays within these multiples of its constant, so one slow or one cached call
# cannot carry the model far from what was measured on the user's machine.
CLAMP = (0.5, 1.5)


class Timed:
    """The backend a fast run's stages see: it passes every call to `inner` and keeps, for each
    distinct call (by cache key), the output tokens and seconds of its reply when the reply has a
    duration (a scripted reply has none). Errors pass through and are not samples."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        self._samples: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def complete(self, call: Call) -> Reply:
        reply = self._inner.complete(call)
        if reply.duration_s > 0:
            with self._lock:
                self._samples.setdefault(cache_key(call), (reply.tokens_out, reply.duration_s))
        return reply

    def samples(self) -> list[tuple[int, float]]:
        """The (output tokens, seconds) of every distinct call answered so far."""
        with self._lock:
            return list(self._samples.values())


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
) -> Growth | None:
    """More pick scenarios than the plan's `scenarios` for `rewrites` rewrites (and `rewrites2`
    reflections, `holdout` held out) when their remaining stages fit `seconds_left` and
    `calls_left` by `model` (SPEC R25): the most, up to MAX_SCENARIOS, from the `have` scenarios
    at hand (pick and held out), or with one synthesis call for the missing ones when
    `can_synthesise`; None when no larger count fits. The `tier` names the run in the plan's
    terms; the checked tier keeps its holdout."""
    if tier not in ("fast", "checked"):
        return None
    for count in range(MAX_SCENARIOS, scenarios, -1):
        extra = max(0, count + holdout - have)
        if extra and not can_synthesise:
            continue
        stages = (synthesis_stage(extra, model),) if extra else ()
        stages += tail(rewrites, count, holdout, workers, prompt_tokens, model)
        if rewrites2:
            stages += second_stages(rewrites2, count, workers, prompt_tokens, model)
        if misfit(stages, seconds_left, calls_left) is None:
            return Growth(count, holdout, extra)
    return None
