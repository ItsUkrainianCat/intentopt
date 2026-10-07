"""Whether the stages of a fast plan fit in what a run has left (SPEC R17, R25; ADR-011): `misfit`
says why some do not, `shrink` finds the largest shape that does. Split from `fastplan`, which
re-exports both and whose stages they take. Pure: no clock, no call."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from autoimprover.types import StopCause

if TYPE_CHECKING:
    from autoimprover.fastplan import Stage


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
