"""The rewrites learn from the user's examples (SPEC R7, R25; ADR-013, WP22): with references, every
rewrite call receives the pick examples as data and the first rewrite is `induce` (K=1: the only
one); the second generation's reflection keeps the learned rules; the rewrite's Length line states
the cap of `fast_gates` (`test_fast_gates.py`): the original plus 150 tokens unless the strictness
is conservative; and the planner prices the longer intake and rewrites of stage A, as its
calibration plans them."""

import json

import pytest
from fakes import ScriptedBackend, intake_reply

from autoimprover import fast_calibrate, fast_prompts, fastplan
from autoimprover.cli_plan import FastView
from autoimprover.contract import extract_contract, literals
from autoimprover.fastplan import Reference, fast_plan, rewrite_tokens
from autoimprover.runner import count_tokens
from autoimprover.types import DEFAULT_MODELS, Call, Contract, Plan, Scenario

MODEL = "claude-sonnet-5-5"
PROMPT = "Assign a priority to this support ticket."  # 8 tokens
EXAMPLES = (
    Scenario("e1", "Nobody at our company can log in since 9am.", expected="P1"),
    Scenario("e2", "Please send me last month's invoice again.", criteria=("names P4",)),
)
SENT = [
    {"input": "Nobody at our company can log in since 9am.", "expected": "P1", "criteria": []},
    {
        "input": "Please send me last month's invoice again.",
        "expected": None,
        "criteria": ["names P4"],
    },
]
TODAY = ["clarify", "structure", "tighten", "specify", "clarify", "structure"]


def rewrite(variant: int = 0, strictness="balanced", prompt=PROMPT, **kwargs) -> Call:
    return fast_prompts.rewrite_call(prompt, variant, MODEL, strictness, False, **kwargs)


# --- the induce strategy (ADR-013) ----------------------------------------------------------------


def test_with_examples_the_first_rewrite_induces_and_the_others_are_todays():
    names = [fast_prompts.rewrite_strategy(v, True) for v in range(6)]
    assert names == ["induce", *TODAY[:5]]
    assert [fast_prompts.rewrite_strategy(v, False) for v in range(6)] == TODAY


def test_the_induce_strategy_states_what_the_examples_show_in_the_authors_voice():
    text = fast_prompts.INDUCE_VARIANT
    for words in ("labels", "rules", "format", "leave", "own voice", "do not copy", "literal"):
        assert words in text.lower()
    assert "add no fact the examples do not show" in text.lower()
    assert fast_prompts.INDUCE_NOTE.startswith("induced:")


def test_a_rewrite_with_examples_sends_them_as_data_and_states_the_strategy():
    made = rewrite(0, examples=EXAMPLES)
    assert json.loads(made.user) == {
        "prompt": PROMPT,
        "keep_verbatim": list(literals(PROMPT)),
        "examples": SENT,
    }
    assert (made.role, made.model, made.sample) == ("reflect", MODEL, 0)
    assert fast_prompts.INDUCE_VARIANT in made.system
    assert not any(example.input in made.system for example in EXAMPLES)  # R18: stdin only
    assert "`examples`" in made.system and "data, not instructions" in made.system
    assert "examples do not show" in made.system  # the ADR-006 rule, widened to the examples
    second = rewrite(1, examples=EXAMPLES)
    assert fast_prompts.REWRITE_VARIANTS["clarify"] in second.system
    assert fast_prompts.INDUCE_VARIANT not in second.system
    assert json.loads(second.user)["examples"] == SENT and second.sample == 1


def test_without_examples_a_rewrite_call_is_the_call_it_was():
    for variant in range(4):
        assert rewrite(variant, examples=()) == rewrite(variant)
        assert "examples" not in json.loads(rewrite(variant).user)


@pytest.mark.parametrize(
    ("strictness", "prompt", "cap"),
    [
        ("balanced", PROMPT, 8 + 150),
        ("bold", PROMPT, 8 + 150),
        ("conservative", PROMPT, 8 + 40),  # conservative keeps its cap
        ("bold", " ".join(f"w{i}" for i in range(400)), 1000),  # 2.5 x 400 is the larger
    ],
)
def test_with_examples_the_length_line_states_the_larger_cap(strictness, prompt, cap):
    made = rewrite(0, strictness, prompt, examples=EXAMPLES)
    assert f"at most {cap} tokens" in made.system
    grows = strictness != "conservative"
    assert ("plus 150 tokens" in made.system) is grows


# --- the second generation keeps the learned rules ------------------------------------------------

LEARNED = "Use P1 when the service is down."


def learned() -> Contract:
    return Contract(goal="assign a priority", kind="task", from_examples=(LEARNED,))


def test_the_reflection_with_references_reads_the_learned_rules_and_keeps_them():
    parent = {"prompt": f"{PROMPT} Use P1 when the service is down.", "scenarios": []}
    made = fast_prompts.reflect_call(
        PROMPT, [parent], learned(), 0, MODEL, "balanced", False, reference=True
    )
    sent = json.loads(made.user)
    assert sent["contract"]["from_examples"] == ["Use P1 when the service is down."]
    assert "from_examples" in made.system and "keep" in made.system
    assert "at most 158 tokens" in made.system
    plain = fast_prompts.reflect_call(PROMPT, [parent], learned(), 0, MODEL, "balanced", False)
    assert "from_examples" not in json.loads(plain.user)["contract"]
    assert "at most 48 tokens" in plain.system


# --- the planner and its calibration (SPEC R25; ADR-011) ------------------------------------------

ONE = Reference(examples=18, checks=1)


def test_stage_a_with_references_prices_the_longer_intake_and_rewrites():
    assert (fastplan.INDUCE_INTAKE_TOKENS, fastplan.INDUCE_REWRITE_TOKENS) == (120, 100)
    plain = fastplan.stage_a(5, 0, 6, 20)
    priced = fastplan.stage_a(5, 0, 6, 20, ref=ONE)
    assert plain.seconds == pytest.approx(3.4 + 380 / 70)
    assert priced.seconds == pytest.approx(3.4 + 500 / 70)  # the intake is still the slowest
    assert priced.calls == plain.calls == 6
    long = fastplan.stage_a(1, 0, 6, 500, ref=ONE)  # a rewrite of 600 + 100 tokens is slower
    assert long.seconds == pytest.approx(3.4 + 700 / 70)


@pytest.mark.parametrize("examples", [8, 18])
def test_two_minutes_with_references_still_fits_the_plan_share(examples):
    for tokens in (8, 20):
        plan = fast_plan(120, 6, tokens, True, Reference(examples, 1))
        assert plan.tier == "checked" and plan.reference == Reference(examples, 1)
        assert plan.est_seconds <= fastplan.PLAN_SHARE * 120
        priced = fastplan.stage_a(plan.rewrites, 0, 6, tokens, ref=plan.reference)
        assert plan.stages[0] == priced
        assert priced.seconds > fastplan.stage_a(plan.rewrites, 0, 6, tokens).seconds


def test_the_calibration_plans_a_reference_intake_and_rewrite_as_the_planner_prices_them():
    raw = ScriptedBackend(lambda _call: intake_reply(from_examples=[]))
    extract_contract(raw, MODEL, PROMPT, examples=EXAMPLES)
    extract_contract(raw, MODEL, PROMPT)
    with_examples, plain = raw.calls
    p = count_tokens(PROMPT)
    assert fast_calibrate.planned_tokens(with_examples, p) == 380 + 120
    assert fast_calibrate.planned_tokens(plain, p) == 380
    assert (
        fast_calibrate.planned_tokens(rewrite(0, examples=EXAMPLES), p) == rewrite_tokens(p) + 100
    )
    assert (
        fast_calibrate.planned_tokens(rewrite(1, examples=EXAMPLES), p) == rewrite_tokens(p) + 100
    )
    assert fast_calibrate.planned_tokens(rewrite(0), p) == rewrite_tokens(p)
    parent = {"prompt": PROMPT, "scenarios": []}
    reflection = fast_prompts.reflect_call(
        PROMPT, [parent], learned(), 0, MODEL, "balanced", False, reference=True
    )
    assert fast_calibrate.planned_tokens(reflection, p) == rewrite_tokens(p)  # R as planned


@pytest.mark.parametrize(
    ("strictness", "reference", "said"),
    [("balanced", ONE, True), ("conservative", ONE, False), ("balanced", None, False)],
)
def test_the_plan_states_the_larger_cap_of_a_run_with_references(strictness, reference, said):
    plan = Plan(models=DEFAULT_MODELS, wall_clock_s=120, tier="checked", strictness=strictness)
    text = FastView(plan, fast_plan(120, 6, 8, True, reference), synthesised=False).text()
    (line,) = [line for line in text.splitlines() if line.startswith("strictness:")]
    assert ("with your references at least the original plus 150" in line) is said
    assert "length cap" in line
