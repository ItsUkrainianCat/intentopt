"""The reflection of a run whose examples all carry a reference (SPEC R16, R25; ADR-013, WP23): it
reads the best candidate's text and, for the pick examples it failed, the input, the reference, the
start of its output and the failed checks (at most MAX_FAILURES, each field cut to FIELD_CHARS),
with the rules the contract learned from the examples, and is told that the prompt gets these
cases wrong; each of its K2 rewrites fixes them its own way (a boundary or threshold, a missing
condition), and each round has samples of its own, so no two rounds share a cache key."""

import json

import pytest

from autoimprover import fast_calibrate, fast_prompts, refine
from autoimprover.fastplan import rewrite_tokens
from autoimprover.runner import count_tokens
from autoimprover.types import Contract, Scenario

PROMPT = "Decide what to do with this refund request."
MODEL = "claude-sonnet-5-5"
LEARNED = "Escalate a purchase of about $1,000 or more."
CONTRACT = Contract(goal="decide a refund", kind="task", from_examples=(LEARNED,))
AGREES = "the output agrees with the reference answer in substance: "


def picks(n: int, long: bool = False) -> list[Scenario]:
    """`n` pick examples; the odd ones have a reference answer, and with `long` all of them, each
    field of more than 300 characters."""
    tail = " and more" * 60 if long else ""
    return [
        Scenario(f"e{i}", f"request {i}{tail}", expected=f"deny{tail}" if i % 2 or long else None)
        for i in range(1, n + 1)
    ]


def entries(pick: list[Scenario], failing: set[str], output: str = "approve") -> list:
    found = []
    for scenario in pick:
        failed = [{"id": "expected", "text": f"{AGREES}{scenario.expected}"}]
        info = {"scenario": scenario.id, "output_excerpt": output, "failed": []}
        if scenario.id in failing:
            info["failed"] = failed
        found.append((0.0 if scenario.id in failing else 1.0, info))
    return found


def reflection(variant: int = 0, candidates=None, round_number: int = 1):
    best = candidates or [refine.failures(PROMPT, picks(2), entries(picks(2), {"e1"}))]
    return fast_prompts.reflect_call(
        PROMPT,
        best,
        CONTRACT,
        variant,
        MODEL,
        "balanced",
        False,
        reference=True,
        round_number=round_number,
    )


# --- what the reflection reads --------------------------------------------------------------------


def test_the_evidence_is_the_failed_examples_with_input_reference_output_and_check():
    pick = picks(4)
    found = refine.failures(f"{PROMPT} {LEARNED}", pick, entries(pick, {"e1", "e2"}, "approve"))
    assert found == {
        "prompt": f"{PROMPT} {LEARNED}",
        "scenarios": [
            {
                "input": "request 1",
                "expected": "deny",
                "output": "approve",
                "failed": [f"{AGREES}deny"],
            },
            {
                "input": "request 2",
                "expected": None,
                "output": "approve",
                "failed": [f"{AGREES}None"],
            },
        ],
    }


def test_at_most_six_failures_are_read_each_field_cut_to_300_characters():
    pick = picks(8, long=True)
    long_output = "x" * 1000
    found = refine.failures(PROMPT, pick, entries(pick, {s.id for s in pick}, long_output))
    assert (refine.MAX_FAILURES, refine.FIELD_CHARS) == (6, 300)
    assert [s["input"][:9] for s in found["scenarios"]] == [f"request {i}" for i in range(1, 7)]
    for item in found["scenarios"]:
        assert len(item["input"]) == len(item["output"]) == len(item["expected"]) == 300
        assert [len(text) for text in item["failed"]] == [300]
    assert found["prompt"] == PROMPT  # the candidate's text is never cut


def test_an_incomplete_or_passed_example_is_no_failure():
    pick = picks(3)
    found = entries(pick, {"e1", "e2"})
    found[1][1]["incomplete"] = True
    assert [s["input"] for s in refine.failures(PROMPT, pick, found)["scenarios"]] == ["request 1"]


# --- the call ------------------------------------------------------------------------------------


def test_the_reflection_is_told_the_prompt_gets_these_cases_wrong_and_to_keep_what_works():
    made = reflection()
    assert refine.LESSON in made.system
    assert (
        "the prompt gets these cases wrong; state the rule that makes the reference answers "
        "right, as an instruction, keeping what already works; do not copy an input"
    ) in made.system
    sent = json.loads(made.user)
    assert sent["contract"]["from_examples"] == [LEARNED]
    (best,) = sent["candidates"]
    assert best["scenarios"][0] == {
        "input": "request 1",
        "expected": "deny",
        "output": "approve",
        "failed": [f"{AGREES}deny"],
    }
    assert PROMPT not in made.system and "request 1" not in made.system


@pytest.mark.parametrize("variant", [0, 1, 2, 3])
def test_each_reflection_of_a_round_fixes_the_failures_its_own_way(variant):
    made = reflection(variant)
    name = list(refine.REFINE_VARIANTS)[variant % 2]
    assert refine.refine_strategy(variant) == name
    assert refine.REFINE_VARIANTS[name] in made.system
    assert set(refine.REFINE_NOTES) == set(refine.REFINE_VARIANTS) == {"boundary", "condition"}
    assert "threshold" in refine.REFINE_VARIANTS["boundary"]
    assert "condition" in refine.REFINE_VARIANTS["condition"]
    other = reflection(variant + 1)
    assert other.system != made.system


@pytest.mark.parametrize(
    ("round_number", "samples"), [(1, (100, 101)), (2, (110, 111)), (3, (120, 121))]
)
def test_each_round_has_samples_of_its_own(round_number, samples):
    made = [reflection(v, round_number=round_number) for v in (0, 1)]
    assert tuple(call.sample for call in made) == samples
    assert refine.MAX_ROUNDS == 3


def test_a_round_number_below_one_is_refused():
    with pytest.raises(ValueError, match="round"):
        reflection(round_number=0)


def test_a_reflection_of_any_round_is_planned_as_a_rewrite():
    p = count_tokens(PROMPT)
    for number in (1, 2, 3):
        planned = fast_calibrate.planned_tokens(reflection(round_number=number), p)
        assert planned == rewrite_tokens(p)


def test_a_reflection_without_references_keeps_the_pairwise_texts():
    plain = fast_prompts.reflect_call(
        PROMPT, [{"prompt": PROMPT, "scenarios": []}], CONTRACT, 0, MODEL, "balanced", False
    )
    assert refine.LESSON not in plain.system
    assert fast_prompts.REFLECT_VARIANTS["repair"] in plain.system and plain.sample == 100
