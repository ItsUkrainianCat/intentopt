"""The pairwise decision of the fast tiers, the pure part (SPEC R25 "Decision by pairwise
preference"; ADR-012): a scenario's verdict counts for a side only when both orders agree on it,
the original's two runs compared the same way give the noise, and a rewrite wins when it wins more
scenarios than it loses, at least one, and by more than the noise. The calls and their place in
stage C are tested in `test_fast_preference.py`."""

import json

import pytest

from autoimprover.bench_judge import PAIRWISE_BATCH_SYSTEM
from autoimprover.fast_pairwise import (
    Preference,
    noise_count,
    pair_calls,
    pair_items,
    preference,
    prefers,
)
from autoimprover.types import CallFailed, Scenario

PICK = [Scenario(id=f"s{n}", input=f"situation {n}") for n in (1, 2, 3)]
MINE = {"s1": "o1", "s2": "o2", "s3": "o3"}
THEIRS = {"s1": "r1", "s2": CallFailed("down"), "s3": "r3"}


def test_a_pair_holds_the_scenarios_both_sides_answered_in_pick_order():
    assert pair_items(PICK, MINE, THEIRS) == [
        ("s1", "situation 1", "o1", "r1"),
        ("s3", "situation 3", "o3", "r3"),
    ]
    assert pair_items(PICK, MINE, {"s2": CallFailed("down")}) == []


def test_the_two_calls_of_a_pair_show_the_original_as_a_then_as_b_with_the_original_request():
    items = pair_items(PICK, MINE, THEIRS)
    first, second = pair_calls("Summarise the notes.", items, "judge-model")
    assert first.system == second.system == PAIRWISE_BATCH_SYSTEM
    one, two = json.loads(first.user), json.loads(second.user)
    assert one["request"] == two["request"] == "Summarise the notes."
    assert [(s["answer_A"], s["answer_B"]) for s in one["scenarios"]] == [
        ("o1", "r1"),
        ("o3", "r3"),
    ]
    assert [(s["answer_A"], s["answer_B"]) for s in two["scenarios"]] == [
        ("r1", "o1"),
        ("r3", "o3"),
    ]
    assert (first.model, first.sample, second.sample) == ("judge-model", 0, 0)


ITEMS = [(f"s{n}", f"situation {n}", f"o{n}", f"r{n}") for n in (1, 2, 3, 4)]


def said(*winners: str) -> dict[str, tuple[str, str]]:
    return {f"s{n}": (w, f"reason {w} {n}") for n, w in enumerate(winners, start=1) if w != "-"}


def test_a_scenario_counts_for_a_side_only_when_both_orders_pick_it():
    """First order: the original is A; second: it is B. s1 both pick the rewrite, s2 both the
    original, s3 the orders disagree (a judge that prefers position A), s4 a tie."""
    found = preference(ITEMS, said("B", "A", "A", "tie"), said("A", "B", "A", "tie"))
    assert (found.wins, found.ties, found.losses) == (1, 2, 1)
    assert found.feedback == {  # the scenarios it lost or tied; one reason when both said it
        "s2": ("lost", ("reason A 2", "reason B 2")),
        "s3": ("tie", ("reason A 3",)),
        "s4": ("tie", ("reason tie 4",)),
    }


def test_a_judge_that_always_says_a_yields_only_ties():
    found = preference(ITEMS, said("A", "A", "A", "A"), said("A", "A", "A", "A"))
    assert (found.wins, found.ties, found.losses) == (0, 4, 0)


@pytest.mark.parametrize(
    ("first", "counts"), [(None, (0, 4, 0)), (said("B", "-", "B", "B"), (3, 1, 0))]
)
def test_a_failed_call_or_a_scenario_it_skipped_is_a_tie(first, counts):
    """The second order picks the rewrite everywhere; the first failed, or skipped s2."""
    found = preference(ITEMS, first, said("A", "A", "A", "A"))
    assert (found.wins, found.ties, found.losses) == counts


def test_the_noise_is_the_scenarios_where_the_originals_two_runs_differ_in_both_orders():
    assert noise_count(ITEMS, said("A", "B", "A", "tie"), said("B", "A", "A", "tie")) == 2
    assert noise_count(ITEMS, said("A", "A", "A", "A"), said("A", "A", "A", "A")) == 0
    assert noise_count(ITEMS, None, said("B", "B", "B", "B")) == 0


@pytest.mark.parametrize(
    ("wins", "losses", "noise", "wins_it"),
    [
        (1, 0, 0, True),
        (0, 0, 0, False),
        (2, 1, 0, True),
        (1, 1, 0, False),
        (1, 0, 1, False),  # with noise 1 a rewrite needs two more wins than losses
        (2, 0, 1, True),
        (3, 1, 1, True),
        (2, 1, 1, False),
        (3, 0, 3, False),
    ],
)
def test_a_rewrite_wins_by_more_scenarios_than_the_noise(wins, losses, noise, wins_it):
    assert prefers(Preference(wins, 0, losses, {}), noise) is wins_it
