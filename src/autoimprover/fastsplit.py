"""The shapes a fast plan may take (SPEC R25; ADR-011, WP21, WP23): how many scenarios a tier picks
on and holds out, in the order `fastplan.fast_plan` tries them, and `Reference`, the user's
examples when every one carries a reference. Split from `fastplan`, which re-exports what it took.

A tier's shapes come in groups, best group first, and a plan of an earlier group always comes
before one of a later group (two generations before one within a group). A group is made of rows:
within a group the plan takes a row, then the most rewrites, then a shape of that row, so a row
of the ordinary split is the whole group (the most rewrites first) and a row of a reference split
is one pick size (the largest pick first). With references (WP23) a plan picks on the largest
number of the examples it can price, up to REFERENCE_MAX_SCENARIOS, and in the checked tier holds
out from HOLDOUT_MAX down to REFERENCE_MIN_HOLDOUT of them: first a group of the picks above
MAX_SCENARIOS, then one of the smaller picks, as long as the examples last; when they cover no
such shape, the ordinary split. Pure: no clock, no call."""

from __future__ import annotations

from typing import NamedTuple

from autoimprover.types import HOLDOUT_MAX, Tier

Shape = tuple[int, int]  # (pick scenarios, held out)
Group = tuple[tuple[Shape, ...], ...]  # rows of shapes

# Scenarios from the most down to the fewest, and the held-out scenarios of the checked tier.
MAX_SCENARIOS = 4
MIN_SCENARIOS = 2
CHECKED_HOLDOUT = 4
# The checked tier's split: when the time allows, it picks on MAX_SCENARIOS scenarios and holds
# out CHECKED_HOLDOUT down to CHECKED_MIN_HOLDOUT, and only when no such plan fits does it pick on
# fewer, with CHECKED_MIN_HOLDOUT held out (the bench of 2026-10-06 picked among six rewrites on 2
# scenarios with 4 held out: a pick by noise).
CHECKED_MIN_HOLDOUT = 2
_SPLITS: dict[Tier, tuple[Group, ...]] = {
    "fast": (((tuple((m, 0) for m in range(MAX_SCENARIOS, MIN_SCENARIOS - 1, -1))),),),
    "checked": (
        (tuple((MAX_SCENARIOS, h) for h in range(CHECKED_HOLDOUT, CHECKED_MIN_HOLDOUT - 1, -1)),),
        (tuple((m, CHECKED_MIN_HOLDOUT) for m in range(MAX_SCENARIOS - 1, MIN_SCENARIOS - 1, -1)),),
    ),
}
# With references a plan picks on up to this many of the examples (each failed one is a lesson
# for the reflection, WP23; a judge call holds at most JUDGE_BATCH_MAX, so 7 or 8 take two) and
# the checked tier holds out at least REFERENCE_MIN_HOLDOUT of them for its confirmation.
REFERENCE_MAX_SCENARIOS = 8
REFERENCE_MIN_HOLDOUT = 3
# The pick sizes of the two groups of a reference split, from the most to the fewest.
_REFERENCE_PICKS = ((REFERENCE_MAX_SCENARIOS, MAX_SCENARIOS + 1), (MAX_SCENARIOS, MIN_SCENARIOS))


class Reference(NamedTuple):
    """The user's examples when every one carries a reference (`expected` or `criteria`): how
    many there are, and the most judged checks one of them has (SPEC R11). A plan with one
    decides by agreement with the references, not by pairwise preference (SPEC R25, WP21)."""

    examples: int
    checks: int


def splits(tier: Tier, ref: Reference | None) -> tuple[Group, ...]:
    """The groups of shapes of `tier` (fast or checked), best first: with `ref`, those of the
    reference split its examples cover (the picks above MAX_SCENARIOS, then the others), else the
    ordinary split."""
    if ref is None:
        return _SPLITS[tier]
    low = REFERENCE_MIN_HOLDOUT if tier == "checked" else 0
    groups: list[Group] = []
    for top, bottom in _REFERENCE_PICKS:
        rows = []
        for m in range(top, bottom - 1, -1):
            high = min(HOLDOUT_MAX, ref.examples - m) if low else 0
            if m + low <= ref.examples:
                rows.append(tuple((m, h) for h in range(high, low - 1, -1)))
        if rows:
            groups.append(tuple(rows))
    return tuple(groups) or _SPLITS[tier]
