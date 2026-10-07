"""The free gates of a rewrite with the user's references (SPEC R7, R9; ADR-013, WP22): the length
cap is the larger of the strictness cap and the original plus GROWTH_TOKENS, unless the strictness
is conservative, and the reported ratio stays the true one; a rewrite holding a pick example's
input verbatim (an input of COPY_MIN_CHARS characters or more, case and runs of whitespace aside)
copies it, unless the original itself holds that input."""

import pytest

from autoimprover import fast_gates
from autoimprover.runner import length_ok
from autoimprover.types import Scenario

PROMPT = "Assign a priority to this support ticket."  # 8 tokens
EXAMPLES = (
    Scenario("e1", "Nobody at our company can log in since 9am.", expected="P1"),
    Scenario("e2", "Please send me last month's invoice again.", criteria=("names P4",)),
)
LEVELS = ("conservative", "balanced", "bold")


def test_the_cap_with_references_is_the_larger_of_the_strictness_cap_and_plus_150():
    assert fast_gates.GROWTH_TOKENS == 150
    assert [fast_gates.token_cap(8, s, True) for s in LEVELS] == [48, 158, 158]
    assert [fast_gates.token_cap(8, s, False) for s in LEVELS] == [48, 48, 48]
    assert fast_gates.token_cap(400, "bold", True) == 1000  # 2.5 x 400 is the larger
    assert fast_gates.token_cap(400, "balanced", True) == 600


def test_length_fits_takes_the_larger_cap_and_reports_the_true_ratio():
    long_rule = PROMPT + " " + " ".join(["word"] * 140)  # 148 tokens
    fits, ratio = fast_gates.length_fits(PROMPT, long_rule, "balanced", False, True)
    assert fits and ratio == length_ok(PROMPT, long_rule, "balanced", False)[1] == 148 / 8
    assert not fast_gates.length_fits(PROMPT, long_rule, "balanced", False, False)[0]
    assert not fast_gates.length_fits(PROMPT, long_rule, "conservative", False, True)[0]
    over = PROMPT + " " + " ".join(["word"] * 151)  # 159 tokens
    assert not fast_gates.length_fits(PROMPT, over, "balanced", False, True)[0]
    assert fast_gates.length_fits(PROMPT, over, "balanced", True, True)[0]  # --allow-growth


def test_without_references_length_fits_is_length_ok():
    for text in (PROMPT, PROMPT + " more" * 30, PROMPT + " more" * 50):
        for level in LEVELS:
            for growth in (False, True):
                assert fast_gates.length_fits(PROMPT, text, level, growth, False) == length_ok(
                    PROMPT, text, level, growth
                )


@pytest.mark.parametrize(
    ("text", "copies"),
    [
        (f"{PROMPT} For example: Nobody at our company can log in since 9am. is P1.", True),
        (f"{PROMPT} e.g. NOBODY at our company   can log in\nsince 9am.", True),
        (f"{PROMPT} Use P1 when nobody at a company can log in.", False),
        (PROMPT, False),
    ],
)
def test_a_rewrite_holding_a_pick_examples_input_copies_it(text, copies):
    assert fast_gates.copies_example(text, EXAMPLES, PROMPT) is copies


def test_a_short_input_or_one_the_original_holds_is_not_a_copy():
    assert fast_gates.COPY_MIN_CHARS == 20
    short = (Scenario("s", "site down, P1?", expected="P1"),)  # 14 characters
    assert not fast_gates.copies_example(f"{PROMPT} site down, P1?", short, PROMPT)
    quoted = 'Assign a priority to "Nobody at our company can log in since 9am." and others.'
    assert not fast_gates.copies_example(quoted, EXAMPLES, quoted)
    exactly = (Scenario("x", "a" * 20, expected="P1"),)
    assert fast_gates.copies_example(f"{PROMPT} {'A' * 20}", exactly, PROMPT)
    assert not fast_gates.copies_example(f"{PROMPT} {'A' * 20}", (), PROMPT)
