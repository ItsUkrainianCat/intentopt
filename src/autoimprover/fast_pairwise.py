"""The pairwise decision of the fast tiers (SPEC R25 "Decision by pairwise preference"; ADR-012):
the pure part of stages C and D. A pair is the original's answers (always its run 0) against
another run's answers on the scenarios both completed; it is judged in two batched calls, the
original as A, then as B (`bench_judge.pairwise_batch_call`, the judge sees the original prompt as
the request and never a candidate prompt). A scenario counts for a side only when both orders pick
it (`bench_judge.scenario_vote`); a disagreement, a tie, a scenario the judge skipped or an order
whose call failed is a tie, so a judge that prefers a position never makes a win. The original's
run 0 against its run 1 gives the noise: the scenarios with an agreed winner. A rewrite wins when
it kept the contract and wins more scenarios than it loses, at least one, and by more scenarios
than the noise (`won`).

With `--ungated` (fast and checked tiers, a measuring aid), when no rewrite wins, the pick is the
best-ranked rewrite that kept the contract, whatever its lead (`best_ungated`): the largest lead in
scenarios (wins - losses), then the most wins, then the fewest tokens, then the earliest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from autoimprover.bench_judge import pairwise_batch_call, scenario_vote
from autoimprover.runner import count_tokens
from autoimprover.types import Call, CallFailed, Scenario

# (scenario id, situation, the original's answer, the other side's answer)
Item = tuple[str, str, str, str]
# Scenario id -> (winner, reason) of one order's reply; None when that call gave nothing.
Said = Mapping[str, tuple[str, str]] | None


class Preference(NamedTuple):
    """A rewrite against the original: the scenarios it won, tied and lost (both orders agreeing),
    and per scenario it lost or tied, "lost" or "tie" and the judge's reasons, the feedback of the
    second generation (SPEC R16, R25)."""

    wins: int
    ties: int
    losses: int
    feedback: dict[str, tuple[str, tuple[str, ...]]]


class Rewrite(NamedTuple):
    """A rewrite that passed the free gates: its variant, its text, its length ratio and the line
    that says what it changed (SPEC R2)."""

    variant: int
    text: str
    ratio: float
    note: str


class Judged(NamedTuple):
    """A rewrite after stage C: its preference against the original and whether it kept the
    contract."""

    rewrite: Rewrite
    found: Preference
    keep: bool


class Win(NamedTuple):
    """A rewrite's win, as shares of the M scenarios it was picked on: those the original won
    (`before`) and those it won (`after`), its lead (`gain`) and the noise (`bar`); `gated` False
    for the ungated pick, which need not lead at all."""

    rewrite: Rewrite
    before: float
    after: float
    gain: float
    bar: float
    gated: bool = True


def pair_items(
    pick: Sequence[Scenario],
    mine: Mapping[str, str | CallFailed],
    theirs: Mapping[str, str | CallFailed],
) -> list[Item]:
    """The scenarios of `pick` both sides answered, in order, with the two answers."""
    items = []
    for scenario in pick:
        a, b = mine.get(scenario.id), theirs.get(scenario.id)
        if isinstance(a, str) and isinstance(b, str):
            items.append((scenario.id, scenario.input, a, b))
    return items


def pair_calls(request: str, items: Sequence[Item], model: str) -> tuple[Call, Call]:
    """The pair's two judge calls: the original's answers as A, then as B."""
    first = [(name, situation, mine, theirs) for name, situation, mine, theirs in items]
    second = [(name, situation, theirs, mine) for name, situation, mine, theirs in items]
    return pairwise_batch_call(request, first, model, 0), pairwise_batch_call(
        request, second, model, 0
    )


def preference(items: Sequence[Item], first: Said, second: Said) -> Preference:
    """The verdicts of a pair from its two orders' replies (the original as A, then as B)."""
    counts = {"candidate": 0, "tie": 0, "original": 0}
    feedback: dict[str, tuple[str, tuple[str, ...]]] = {}
    for name, *_ in items:
        one, two = (first or {}).get(name), (second or {}).get(name)
        vote = scenario_vote(one and one[0], two and two[0]) or "tie"
        counts[vote] += 1
        if vote != "candidate":
            reasons = tuple(dict.fromkeys(got[1] for got in (one, two) if got is not None))
            feedback[name] = ("lost" if vote == "original" else "tie", reasons)
    return Preference(counts["candidate"], counts["tie"], counts["original"], feedback)


def noise_count(items: Sequence[Item], first: Said, second: Said) -> int:
    """The scenarios where the original's two runs have an agreed winner: its noise."""
    found = preference(items, first, second)
    return found.wins + found.losses


def prefers(found: Preference, noise: int) -> bool:
    """Whether the rewrite wins: more wins than losses, at least one, and a lead of more than
    `noise` scenarios (with noise at least 0 the last implies the other two)."""
    return found.wins - found.losses > noise


def won(judged: Judged, noise: int, m: int) -> Win | None:
    """A judged rewrite's win over the original on `m` scenarios with `noise` of them noisy, or
    None: it kept the contract and won more scenarios than it lost by more than the noise."""
    if not judged.keep or not prefers(judged.found, noise):
        return None
    return shares(judged, noise, m)


def best_ungated(judged: Sequence[Judged], noise: int, m: int) -> Win | None:
    """The ungated pick among `judged` on `m` scenarios with `noise` of them noisy: the rewrite
    that kept the contract with the largest lead, then the most wins, then the fewest tokens,
    then the earliest; None when none kept it."""
    kept = [j for j in judged if j.keep]
    if not kept:
        return None
    best = min(
        kept,
        key=lambda j: (
            j.found.losses - j.found.wins,
            -j.found.wins,
            count_tokens(j.rewrite.text),
        ),
    )
    return shares(best, noise, m)._replace(gated=False)


def shares(judged: Judged, noise: int, m: int) -> Win:
    found = judged.found
    return Win(
        judged.rewrite, found.losses / m, found.wins / m, (found.wins - found.losses) / m, noise / m
    )
