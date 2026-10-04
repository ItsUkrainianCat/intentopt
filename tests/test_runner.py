"""The runner's pure parts (SPEC R4, R7, R8, R9, R12, R17, R24; ADR-006, ADR-008): the fixed costs
and the iteration estimate of a plan, the refusal of a budget that cannot pay for a search, the
token count and the length cap, the holdout score, and the reflection prompt handed to GEPA (one
text per strictness level, with the intent contract, the fidelity rules, the length cap, the reply
format and GEPA's two placeholders exactly once each). What `improve` decides on a given search
result is tested in `test_runner_prompts.py`, the whole flow on GEPA in `test_runner_flow.py`."""

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

import pytest

from autoimprover import runner
from autoimprover.types import (
    DEFAULT_MODELS,
    INSTRUCTION_BEGIN,
    INSTRUCTION_END,
    SYSTEM_PROMPT_MAX_BYTES,
    BackendError,
    Contract,
    Plan,
    Scenario,
    Strictness,
)


def plan(budget: int = 100) -> Plan:
    return Plan(models=DEFAULT_MODELS, budget=budget)


# --- fixed costs and iterations (SPEC R4, R17; ARCHITECTURE section 1) ----------------------------


def test_twelve_synthesised_scenarios_at_budget_100_leave_5_to_7_iterations():
    costs = runner.fixed_costs(plan(100), 12, synthesising=True)
    assert costs == runner.FixedCosts(
        holdout=4,
        valset=3,
        minibatch=3,
        pre=16,  # intake, synthesis, two seed runs of 4 + 1 judge, seed valset pass of 3 + 1
        final=18,  # three finalist runs of 4 + 1 judge, three contract checks
        iter_cost=13,  # reflect, parent and child minibatches of 3 + 1 judge, valset 3 + 1
        search_calls=66,
        iterations=5,
        iterations_best=7,  # a rejected child costs 1 + 2 x (3 + 1) = 9
    )


@pytest.mark.parametrize(
    ("n", "sizes", "pre", "final", "iter_cost", "iterations"),
    [
        (8, (3, 2), 12, 15, 12, 6),
        (12, (4, 3), 15, 18, 13, 5),
        (40, (6, 4), 20, 24, 14, 4),
    ],
)
def test_the_users_examples_cost_no_synthesis_call(
    n: int, sizes: tuple[int, int], pre: int, final: int, iter_cost: int, iterations: int
):
    costs = runner.fixed_costs(plan(100), n, synthesising=False)
    assert (costs.holdout, costs.valset, costs.minibatch) == (*sizes, 3)
    assert (costs.pre, costs.final, costs.iter_cost) == (pre, final, iter_cost)
    assert (costs.search_calls, costs.iterations) == (100 - pre - final, iterations)


def test_seven_examples_have_no_holdout_runs_and_two_judge_calls_per_valset_pass():
    costs = runner.fixed_costs(plan(100), 7, synthesising=False)
    assert (costs.holdout, costs.valset, costs.minibatch) == (0, 7, 3)
    assert costs.pre == 1 + 7 + 2  # intake and GEPA's seed valset pass; no seed holdout runs
    assert costs.final == 3  # the contract checks only; no finalist holdout runs
    assert costs.iter_cost == 1 + 2 * (3 + 1) + 7 + 2
    assert (costs.search_calls, costs.iterations, costs.iterations_best) == (87, 4, 9)


@pytest.mark.parametrize(("n", "minibatch", "iter_cost"), [(1, 1, 1 + 4 + 2), (2, 2, 1 + 6 + 3)])
def test_a_dataset_smaller_than_a_minibatch_makes_the_minibatch_smaller(
    n: int, minibatch: int, iter_cost: int
):
    costs = runner.fixed_costs(plan(100), n, synthesising=False)
    assert (costs.minibatch, costs.iter_cost) == (minibatch, iter_cost)


def test_no_scenario_at_all_has_no_costs():
    with pytest.raises(ValueError, match="scenario"):
        runner.fixed_costs(plan(100), 0, synthesising=False)


def test_fixed_costs_above_the_budget_leave_a_negative_search_share_and_no_iteration():
    costs = runner.fixed_costs(plan(30), 12, synthesising=True)
    assert (costs.search_calls, costs.iterations, costs.iterations_best) == (-4, 0, 0)


def test_the_costs_add_up_to_the_budget_for_every_plan():
    for n in range(1, 201):
        for budget in range(1, 301):
            for synthesising in (n == 12, False):
                costs = runner.fixed_costs(plan(budget), n, synthesising)
                assert costs.pre + costs.final + costs.search_calls == budget, (n, budget)
                share = max(0, costs.search_calls)
                assert costs.iterations * costs.iter_cost <= share
                assert share < (costs.iterations + 1) * costs.iter_cost
                assert costs.iterations <= costs.iterations_best
                assert costs.holdout <= 6 and costs.valset <= max(4, min(n, 7))


# --- refusal before any paid call (SPEC R4, R17) --------------------------------------------------


def test_budget_66_with_12_scenarios_affords_2_iterations_and_is_refused():
    costs = runner.fixed_costs(plan(66), 12, synthesising=True)
    assert costs.iterations == 2
    message = runner.refusal(costs, force_low_budget=False)
    assert message is not None
    assert "2 search iterations" in message and "fewer than 4" in message
    assert "--budget 86" in message and "--force-low-budget" in message
    assert runner.refusal(costs, force_low_budget=True) is None


def test_four_iterations_are_enough_and_three_are_not():
    four = runner.fixed_costs(plan(34 + 4 * 13), 12, synthesising=True)
    three = runner.fixed_costs(plan(34 + 4 * 13 - 1), 12, synthesising=True)
    assert (four.iterations, three.iterations) == (4, 3)
    assert runner.refusal(four, force_low_budget=False) is None
    assert runner.refusal(three, force_low_budget=False) is not None


def test_fixed_costs_above_the_budget_are_refused_even_with_force_low_budget():
    for budget in (1, 33):
        costs = runner.fixed_costs(plan(budget), 12, synthesising=True)
        for force in (False, True):
            message = runner.refusal(costs, force_low_budget=force)
            assert message is not None and "fixed costs" in message, (budget, force)
            assert f"exceed the budget of {budget};" in message
            assert "--budget 86" in message


def test_fixed_costs_equal_to_the_budget_are_not_refused_when_forced():
    costs = runner.fixed_costs(plan(34), 12, synthesising=True)
    assert costs.search_calls == 0
    assert runner.refusal(costs, force_low_budget=True) is None
    assert "fewer than 4" in (runner.refusal(costs, force_low_budget=False) or "")


# --- token count and length cap (SPEC R7, R8; ADR-004) --------------------------------------------


def test_a_token_is_a_run_of_word_characters_or_one_other_visible_character():
    assert runner.count_tokens("") == 0
    assert runner.count_tokens("  \n\t ") == 0
    assert runner.count_tokens("Answer the user's request.") == 7
    assert runner.count_tokens("x_1 = f(y);") == 7
    assert runner.count_tokens("naïve café 42") == 3


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize(
    ("strictness", "cap"), [("conservative", 250), ("balanced", 300), ("bold", 500)]
)
def test_a_candidate_may_grow_to_the_strictness_cap_and_not_one_token_more(
    strictness: Strictness, cap: int
):
    original = words(200)
    assert runner.length_ok(original, words(cap), strictness, False) == (True, cap / 200)
    assert runner.length_ok(original, words(cap + 1), strictness, False)[0] is False


def test_bold_allows_two_and_a_half_times_the_original():
    assert runner.length_ok(words(100), words(250), "bold", False) == (True, 2.5)
    assert runner.length_ok(words(100), words(251), "bold", False) == (False, 2.51)


def test_a_candidate_26_percent_longer_is_rejected_at_conservative():
    fits, ratio = runner.length_ok(words(200), words(252), "conservative", False)
    assert (fits, ratio) == (False, 1.26)
    assert runner.length_ok(words(200), words(252), "balanced", False)[0] is True


def test_a_short_prompt_may_always_gain_40_tokens():
    assert runner.length_ok(words(10), words(50), "conservative", False) == (True, 5.0)
    assert runner.length_ok(words(10), words(51), "conservative", False)[0] is False
    assert runner.length_ok(words(160), words(200), "conservative", False)[0] is True
    assert runner.length_ok(words(160), words(201), "conservative", False)[0] is False


def test_allow_growth_removes_the_cap_but_not_the_byte_limit_of_a_system_prompt():
    assert runner.length_ok(words(10), words(5000), "conservative", True) == (True, 500.0)
    huge = "x" * (SYSTEM_PROMPT_MAX_BYTES + 1)
    assert runner.length_ok(words(10), huge, "bold", True) == (False, 0.1)
    assert runner.length_ok(words(10), huge[:-1], "bold", True)[0] is True
    wide = "é" * (SYSTEM_PROMPT_MAX_BYTES // 2 + 1)  # two bytes each, one token
    assert runner.length_ok(words(10), wide, "bold", True)[0] is False


def test_a_shorter_candidate_fits_and_reports_its_ratio():
    assert runner.length_ok(words(40), words(30), "conservative", False) == (True, 0.75)


def test_an_original_without_tokens_counts_as_one_token_for_the_ratio():
    assert runner.length_ok("   ", words(41), "conservative", False) == (False, 41.0)
    assert runner.length_ok("   ", words(40), "conservative", False) == (True, 40.0)


# --- the holdout score (SPEC R3, R12, R24) --------------------------------------------------------

HOLDOUT = [Scenario(id=f"h{i}", input=f"input {i}") for i in range(1, 5)]
Entry = tuple[float, dict[str, Any]]


class Recorder:
    """A batch evaluator that returns the given entries and records what it was asked."""

    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        self.asked: list[tuple[str, tuple[Scenario, ...]]] = []

    def __call__(self, candidate: str, scenarios: Sequence[Scenario]) -> list[Entry]:
        self.asked.append((candidate, tuple(scenarios)))
        return self.entries


def test_the_holdout_score_is_the_mean_of_the_scenario_scores():
    evaluator = Recorder([(1.0, {}), (0.5, {}), (0.0, {}), (0.75, {})])
    assert runner.score_holdout(evaluator, "the prompt", HOLDOUT) == 0.5625
    assert evaluator.asked == [("the prompt", tuple(HOLDOUT))]


def test_a_failed_call_on_the_holdout_is_never_a_score():
    entries = [(1.0, {}), (0.0, {"incomplete": True, "error": "task call failed"})]
    with pytest.raises(BackendError, match="task call failed"):
        runner.score_holdout(Recorder([*entries, (1.0, {}), (1.0, {})]), "p", HOLDOUT)


def test_an_empty_holdout_has_no_score():
    with pytest.raises(ValueError, match="holdout"):
        runner.score_holdout(Recorder([]), "p", [])


# --- the reflection prompt (SPEC R8, R9; ADR-006, ADR-008) ------------------------------------

PLACEHOLDERS = ("<curr_param>", "<side_info>")
LEVELS: tuple[Strictness, ...] = ("conservative", "balanced", "bold")
CONTRACT = Contract(
    goal="Summarise a pull request for reviewers",
    kind="template",
    keep=("the repository is called autoimprover", "mention the ticket number"),
    constraints=("at most five bullet points", "never paste secrets"),
    output_format="a Markdown bullet list",
    language="English",
    tone="dry and factual",
)
RULES = (
    "do not add facts, names, numbers or requirements that appear only in the examples",
    "prefer deleting or tightening over adding",
    "preserve the author's voice, language, structure and every literal",
    "change only what a failed check points to",
)


def template(
    strictness: Strictness = "conservative",
    tokens: int = 200,
    allow_growth: bool = False,
    contract: Contract = CONTRACT,
) -> str:
    return runner.reflection_template(contract, strictness, tokens, allow_growth)


@pytest.mark.parametrize("strictness", LEVELS)
@pytest.mark.parametrize("allow_growth", [False, True])
def test_each_placeholder_appears_exactly_once(strictness: Strictness, allow_growth: bool):
    text = template(strictness, allow_growth=allow_growth)
    assert [text.count(token) for token in PLACEHOLDERS] == [1, 1]


def test_gepa_s_plain_replacement_puts_the_prompt_before_the_feedback_and_leaves_no_token():
    rendered = template().replace("<curr_param>", "THE PROMPT").replace("<side_info>", "FEEDBACK")
    assert rendered.count("THE PROMPT") == 1 and rendered.count("FEEDBACK") == 1
    assert rendered.index("THE PROMPT") < rendered.index("FEEDBACK")
    assert not any(token in rendered for token in PLACEHOLDERS)


@pytest.mark.parametrize("strictness", LEVELS)
def test_the_intent_contract_is_carried_in_full(strictness: Strictness):
    text = template(strictness)
    fields = (CONTRACT.goal, *CONTRACT.keep, *CONTRACT.constraints, CONTRACT.output_format)
    for value in (*fields, CONTRACT.language, CONTRACT.tone):
        assert value in text, value
    assert "keep verbatim" in text.lower()


def test_an_empty_contract_field_is_named_as_empty_rather_than_left_blank():
    bare = Contract(goal="Answer the question", kind="task")
    text = template(contract=bare)
    assert "Answer the question" in text
    for label in ("Keep verbatim: nothing listed", "Constraints: none listed"):
        assert label in text
    assert "Output format: none required" in text


@pytest.mark.parametrize("strictness", LEVELS)
def test_the_fidelity_rules_of_adr_006_are_in_every_level(strictness: Strictness):
    text = template(strictness).lower()
    for rule in RULES:
        assert rule in text, rule


def test_the_three_levels_ask_for_different_edits():
    conservative, balanced, bold = (template(level) for level in LEVELS)
    assert len({conservative, balanced, bold}) == 3
    assert "smallest" in conservative.lower() and "smallest" not in bold.lower()
    assert "restructure" in bold.lower() and "restructure" not in conservative.lower()


@pytest.mark.parametrize(
    ("strictness", "tokens", "cap"),
    [
        ("conservative", 200, 250),
        ("balanced", 200, 300),
        ("bold", 200, 500),
        ("conservative", 10, 50),  # the floor: the original plus 40 tokens
        ("conservative", 201, 251),  # 251.25 rounds down: the gate counts whole tokens
    ],
)
def test_the_length_cap_is_stated_in_tokens(strictness: Strictness, tokens: int, cap: int):
    text = template(strictness, tokens)
    assert f"at most {cap} tokens" in text and f"the user's original has {tokens}" in text


def test_allow_growth_states_no_cap():
    text = template("conservative", 200, allow_growth=True)
    assert "no length cap" in text and not re.search(r"at most \d+ tokens", text)


def test_the_reply_format_is_the_delimiter_lines_then_three_bullet_lines():
    lines = template().splitlines()
    begin, end = lines.index(INSTRUCTION_BEGIN), lines.index(INSTRUCTION_END)
    assert begin < end and lines.count(INSTRUCTION_BEGIN) == lines.count(INSTRUCTION_END) == 1
    assert [line.startswith("- ") for line in lines[end + 1 : end + 4]] == [True] * 3
    assert "three lines" in template().lower()


@pytest.mark.parametrize("token", PLACEHOLDERS)
@pytest.mark.parametrize(
    "field", ["goal", "keep", "constraints", "output_format", "language", "tone"]
)
def test_a_contract_holding_a_gepa_placeholder_is_refused(token: str, field: str):
    value = f"text with {token} inside"
    if field in ("keep", "constraints"):
        changed = dataclasses.replace(CONTRACT, **{field: ("fine", value)})
    else:
        changed = dataclasses.replace(CONTRACT, **{field: value})
    with pytest.raises(ValueError, match=token):
        template(contract=changed)


def test_the_strictness_must_be_one_of_the_three_levels():
    with pytest.raises(ValueError, match="strictness"):
        template("reckless")  # type: ignore[arg-type]
