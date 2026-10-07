"""The order of the user's examples before a run splits them (SPEC R11, R25; WP25): when every
example's reference is a short label and there are 2 to 12 of them, a round robin over the labels,
so every head of the order covers the labels as evenly as their counts allow; otherwise the file
order. Property tests use a hand-written seeded generator (no new dependency, as SPEC R9's)."""

import random
from collections import Counter

import pytest

from autoimprover.strata import LABEL_MAX_CHARS, LABELS_MAX, coverage, label, ordered
from autoimprover.types import Scenario


def labelled(*labels: str) -> list[Scenario]:
    return [Scenario(f"s{n}", f"input {n}", expected=value) for n, value in enumerate(labels)]


def names(examples: list[Scenario]) -> list[str]:
    return [example.id for example in examples]


def random_set(rng: random.Random) -> list[Scenario]:
    """2 to 12 labels, 1 to 6 examples each, in a shuffled file order, the labels written with
    random case and spaces."""
    labels = [f"label {n}" for n in range(rng.randint(2, LABELS_MAX))]
    values = [value for value in labels for _ in range(rng.randint(1, 6))]
    rng.shuffle(values)
    spelled = [rng.choice([v, v.upper(), f"  {v} ", v.replace(" ", "\t ")]) for v in values]
    return labelled(*spelled)


SETS = [random_set(random.Random(seed)) for seed in range(200)]


# --- the round robin --------------------------------------------------------------------------


def test_a_label_heavy_head_is_spread_over_the_labels():
    """The ex-escalate shape: the file starts with 'no' answers only."""
    given = labelled("no", "no", "no", "no", "yes", "no", "yes", "yes")
    assert names(ordered(given)) == ["s0", "s4", "s1", "s6", "s2", "s7", "s3", "s5"]


def test_the_labels_come_in_order_of_first_appearance_each_in_file_order():
    given = labelled("b", "a", "b", "c", "a", "b")
    assert names(ordered(given)) == ["s0", "s1", "s3", "s2", "s4", "s5"]


def test_every_head_covers_every_label_once_it_is_as_long_as_the_labels():
    for given in SETS:
        order = ordered(given)
        assert sorted(names(order)) == sorted(names(given))
        every = {label(example) for example in given}
        for n in range(len(every), len(order) + 1):
            assert {label(example) for example in order[:n]} == every, (names(given), n)


def test_every_head_is_as_even_as_the_counts_allow():
    """In every head, a label has at most one example more than another, unless the other has
    no example left."""
    for given in SETS:
        total = Counter(label(example) for example in given)
        order = ordered(given)
        for n in range(1, len(order) + 1):
            head = Counter(label(example) for example in order[:n])
            for one in total:
                for other in total:
                    even = head[one] <= head[other] + 1 or head[other] == total[other]
                    assert even, (names(given), n, one, other)


def test_within_a_label_the_file_order_is_kept():
    for given in SETS:
        order = ordered(given)
        for value in {label(example) for example in given}:
            mine = [example.id for example in given if label(example) == value]
            assert [example.id for example in order if label(example) == value] == mine


def test_the_order_is_a_pure_function_and_ordering_it_again_changes_nothing():
    """A resumed run reads the run folder's examples, in the file order or in this one, and
    must split them as the run did (SPEC R22)."""
    for given in SETS:
        assert ordered(given) == ordered(list(given))
        assert ordered(ordered(given)) == ordered(given)


def test_case_and_whitespace_runs_do_not_make_another_label():
    given = labelled("Yes", " no", "YES ", "yes\n", "No")
    assert {label(example) for example in given} == {"yes", "no"}
    assert names(ordered(given)) == ["s0", "s1", "s2", "s4", "s3"]


# --- the file order stays ---------------------------------------------------------------------

FILE_ORDER = labelled("b", "b", "a", "a")


def test_a_label_longer_than_the_limit_keeps_the_file_order():
    long = "x" * (LABEL_MAX_CHARS + 1)
    at_limit = "y" * LABEL_MAX_CHARS
    assert names(ordered(labelled("b", "b", at_limit, at_limit))) == ["s0", "s2", "s1", "s3"]
    assert names(ordered(labelled("b", "b", long, long))) == ["s0", "s1", "s2", "s3"]


def test_more_than_twelve_labels_keep_the_file_order():
    twelve = [f"l{n}" for n in range(LABELS_MAX)]
    assert names(ordered(labelled("l0", *twelve))) != names(labelled("l0", *twelve))
    thirteen = [f"l{n}" for n in range(LABELS_MAX + 1)]
    given = labelled("l0", *thirteen)
    assert names(ordered(given)) == names(given)


def test_a_single_label_keeps_the_file_order():
    given = labelled("yes", "YES", " yes", "yes")
    assert ordered(given) == given


@pytest.mark.parametrize(
    "given",
    [
        [Scenario("s0", "a", criteria=("short",)), Scenario("s1", "b", criteria=("short",))],
        [Scenario("s0", "a", expected="b"), Scenario("s1", "b", criteria=("c",))],
        [Scenario("s0", "a", expected="b"), Scenario("s1", "b"), Scenario("s2", "c", expected="a")],
        [Scenario("s0", "a"), Scenario("s1", "b")],
        [],
    ],
    ids=["criteria only", "one without expected", "one without reference", "none", "empty"],
)
def test_examples_without_a_label_each_keep_the_file_order(given):
    assert ordered(given) == given
    assert coverage(given, 1) is None


# --- the warning ------------------------------------------------------------------------------


def test_a_pick_shorter_than_the_labels_says_how_many_it_covers():
    given = ordered(labelled("a", "b", "c", "a", "b", "c"))
    assert coverage(given, 2) == (2, 3)
    assert coverage(given, 3) is None
    assert coverage(given, 6) is None


def test_the_coverage_is_of_the_head_it_is_given():
    """The run asks about the split it made, so the list is read as it comes, not ordered
    again."""
    assert coverage(labelled("a", "a", "a", "b"), 3) == (1, 2)
