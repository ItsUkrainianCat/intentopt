"""The decision on reference scores (SPEC R11, R25): the pure part of the fast and checked picks
when every example of the user carries a reference, and of the bench's hidden examples (SPEC R26).

An example's `expected` answer is one judged check, "the output agrees with the reference answer in
substance", and each of its `criteria` one more (SPEC R11, `evaluator`); a scenario's score is the
share of its checks passed. The original runs twice; a candidate is compared on the scenarios all
three runs scored in full (`scored`: no failed call, no unknown checks). It wins iff its summed
score exceeds the mean summed score of the original's two runs by more than their difference (the
noise), it improves more scenarios than it worsens (against the mean of the original's two runs on
that scenario), and improves at least one; anything else, ties on all included, goes to the
original (`beats`). Sums are compared rounded to 9 decimals, so the last bit of a float sum never
makes a win.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from autoimprover.fastplan import Reference
from autoimprover.types import Scenario

# Float sums are compared at this many decimals: shares of checks are short fractions.
_DIGITS = 9


class Margin(NamedTuple):
    """A candidate against the original's two runs on the `scenarios` all three scored: the sum of
    the original's per-scenario means (`before`), the candidate's sum (`after`), the difference of
    the original's two sums (`noise`), and the scenarios the candidate improved, kept and
    worsened."""

    before: float
    after: float
    noise: float
    improved: int
    same: int
    worsened: int
    scenarios: int


def checks_of(scenario: Scenario) -> int:
    """The judged checks a scenario's reference adds: one per criteria string, one for
    `expected`."""
    return len(scenario.criteria) + (scenario.expected is not None)


def reference_of(scenarios: Sequence[Scenario]) -> Reference | None:
    """The plan's Reference when there are scenarios and every one carries a reference, else
    None: mixed or absent references keep the pairwise decision."""
    counts = [checks_of(scenario) for scenario in scenarios]
    if not counts or not all(counts):
        return None
    return Reference(len(counts), max(counts))


def scored(entries: Sequence[tuple[float, Mapping[str, Any]]]) -> dict[str, float]:
    """Scenario id -> score of the evaluator's entries judged in full: an incomplete entry (a call
    failed) or one scored 0 for a reason (too many unknown checks, no check) is left out."""
    return {
        str(info["scenario"]): score
        for score, info in entries
        if not info.get("incomplete") and info.get("reason") is None
    }


def compare(
    first: Mapping[str, float], second: Mapping[str, float], mine: Mapping[str, float]
) -> Margin:
    """`mine` against the original's runs `first` and `second`, on the scenarios all three
    scored, in the order of `first`."""
    common = [name for name in first if name in second and name in mine]
    means = {name: (first[name] + second[name]) / 2 for name in common}
    signs = [_sign(mine[name] - means[name]) for name in common]
    return Margin(
        before=math.fsum(means.values()),
        after=math.fsum(mine[name] for name in common),
        noise=abs(math.fsum(first[name] for name in common) - math.fsum(second[n] for n in common)),
        improved=signs.count(1),
        same=signs.count(0),
        worsened=signs.count(-1),
        scenarios=len(common),
    )


def beats(margin: Margin) -> bool:
    """Whether the candidate wins: a gain above the noise, more scenarios improved than worsened,
    and at least one improved."""
    gain = round(margin.after - margin.before - margin.noise, _DIGITS)
    return gain > 0 and margin.improved > margin.worsened and margin.improved >= 1


def _sign(difference: float) -> int:
    rounded = round(difference, _DIGITS)
    return (rounded > 0) - (rounded < 0)
