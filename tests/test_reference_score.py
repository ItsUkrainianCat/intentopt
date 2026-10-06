"""The decision on reference scores (SPEC R11, R25; WP21): when every example carries a reference
(`expected` or `criteria`), a candidate wins iff its summed score exceeds the mean summed score of
the original's two runs by more than their difference (the noise), it improves more scenarios than
it worsens, and improves at least one; ties on all go to the original. Only the scenarios all three
runs had scored in full count."""

import pytest

from autoimprover.fastplan import Reference
from autoimprover.reference_score import Margin, beats, compare, reference_of, scored
from autoimprover.types import Scenario


def ids(*scores: float) -> dict[str, float]:
    return {f"e{n}": score for n, score in enumerate(scores, start=1)}


@pytest.mark.parametrize(
    ("first", "second", "mine", "wins"),
    [
        ((0, 0, 0), (0, 0, 0), (1, 0, 0), True),  # one scenario improved, no noise
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), False),  # ties on all go to the original
        ((1, 0, 0), (0, 0, 0), (1, 0, 0), False),  # gain 0.5 is not more than the noise 1
        ((1, 0, 0), (0, 0, 0), (1, 1, 0), True),  # gain 1.5 against the noise 1
        ((0, 1, 0), (0, 1, 0), (1, 0, 1), True),  # gains 2, loses 1: more improved than worsened
        ((0, 1, 1), (0, 1, 1), (1, 0, 1), False),  # one improved, one worsened, gain 0
        ((0, 1, 1), (0, 1, 1), (0.5, 1, 1), True),  # half the checks of e1: improved
    ],
)
def test_the_win_rule_at_its_boundaries(first, second, mine, wins):
    assert beats(compare(ids(*first), ids(*second), ids(*mine))) is wins


def test_more_improved_than_worsened_is_needed_whatever_the_gain():
    """The candidate gains 1.0 in sum on e1, e2 but worsens e3 and e4 by a little each."""
    margin = compare(ids(0, 0, 1, 1), ids(0, 0, 1, 1), ids(1, 1, 0.75, 0.75))
    assert (margin.improved, margin.worsened) == (2, 2)
    assert margin.after - margin.before == pytest.approx(1.5)
    assert not beats(margin)


def test_the_sums_and_counts_of_a_margin():
    margin = compare(ids(1, 0.5, 0), ids(0, 0.5, 0), ids(1, 1, 0))
    assert margin == Margin(
        before=pytest.approx(1.0),
        after=pytest.approx(2.0),
        noise=pytest.approx(1.0),
        improved=2,
        same=1,
        worsened=0,
        scenarios=3,
    )


def test_only_the_scenarios_all_three_runs_scored_are_compared():
    first, second = ids(0, 0, 0), {"e1": 0.0, "e3": 0.0}
    mine = {"e1": 1.0, "e2": 1.0}
    margin = compare(first, second, mine)
    assert (margin.scenarios, margin.improved, margin.after) == (1, 1, 1.0)


def test_nothing_compared_never_wins():
    margin = compare(ids(0), ids(0), {})
    assert margin.scenarios == 0 and not beats(margin)


def test_a_float_sum_equal_to_the_noise_is_not_more_than_it():
    """0.1 + 0.2 is 0.30000000000000004 in floating point: a gain of exactly the noise (0.3 -
    0.1 against 0.2) must not win by that last bit."""
    margin = compare(ids(0.2, 0, 0), ids(0, 0, 0), ids(0.1, 0.2, 0))
    assert margin.after - margin.before - margin.noise > 0  # the float sums alone would win
    assert (margin.improved, margin.worsened) == (1, 0) and not beats(margin)


def test_scored_keeps_the_scenarios_judged_in_full():
    entries = [
        (1.0, {"scenario": "e1", "scores": {}}),
        (0.0, {"scenario": "e2", "incomplete": True, "error": "down"}),
        (0.0, {"scenario": "e3", "reason": "unknown_checks"}),
        (0.0, {"scenario": "e4", "reason": "no_checks"}),
        (0.5, {"scenario": "e5"}),
    ]
    assert scored(entries) == {"e1": 1.0, "e5": 0.5}


@pytest.mark.parametrize(
    ("scenarios", "found"),
    [
        ([], None),
        ([Scenario("e1", "i")], None),  # no reference
        ([Scenario("e1", "i", expected="x"), Scenario("e2", "i")], None),  # mixed
        ([Scenario("e1", "i", expected="x")], Reference(1, 1)),
        (
            [
                Scenario("e1", "i", expected="x", criteria=("a", "b")),
                Scenario("e2", "i", criteria=("c",)),
            ],
            Reference(2, 3),  # the most judged checks one example has
        ),
    ],
)
def test_reference_of_every_example_or_none(scenarios, found):
    assert reference_of(scenarios) == found
