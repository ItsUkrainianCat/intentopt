"""The in-run calibration when every example carries a reference (SPEC R25; WP21): a reference
judge call's planned tokens are what the planner priced it at (`fastplan.reference_tokens` per
scenario and its checks), and the re-plan after stage A grows the pick into the examples at hand,
up to REFERENCE_MAX_SCENARIOS (8 since WP23), with the reference stages' prices and never a
synthesis."""

import json

import pytest
from fakes import ScriptedBackend, judge_reply

from autoimprover.contract import check_many
from autoimprover.evaluator import Evaluator
from autoimprover.fast_calibrate import grow, planned_tokens
from autoimprover.fastplan import (
    CONTRACT_CHECKS,
    JUDGE_TOKENS_PER_CHECK,
    JUDGED_CHECKS_PER_SCENARIO,
    Latency,
    Reference,
    latency,
    reference_tokens,
)
from autoimprover.types import Check, Contract, Scenario

EXAMPLES = [
    Scenario("e1", "one", expected="P1"),
    Scenario("e2", "two", expected="P2", criteria=("short", "names a priority")),
]


def judge_calls(contract: Contract) -> list:
    raw = ScriptedBackend(lambda call: judge_reply(call))
    Evaluator(raw, contract, "", "judge-m").score(EXAMPLES, {"e1": "out 1", "e2": "out 2"})
    return raw.calls


def test_a_reference_judge_call_is_planned_per_scenario_and_check():
    (call,) = judge_calls(Contract(goal="g", kind="task"))
    assert planned_tokens(call, 20) == reference_tokens(1) + reference_tokens(3)


def test_a_judge_call_with_contract_checks_keeps_its_plan():
    contract = Contract(goal="g", kind="task", checks=(Check("c1", "content", "on topic"),))
    (call,) = judge_calls(contract)
    assert planned_tokens(call, 20) == JUDGE_TOKENS_PER_CHECK * JUDGED_CHECKS_PER_SCENARIO * 2


def test_a_contract_check_shown_the_examples_is_planned_as_one_without():
    """WP24: the examples add input, which the latency model does not price (ADR-011), and the
    quote of `no-new-goal` becomes an example's input, no longer than the quote it replaces (the
    live refund bench of 2026-10-07: its quotes 6 to 50 words and marks, median 20; the shipped
    inputs 7 to 25, median 14), so the call keeps its plan of CONTRACT_CHECKS per candidate."""
    contract = Contract(goal="g", kind="task", from_examples=("P1 for an outage",))
    calls = []
    for shown in ((), EXAMPLES):
        raw = ScriptedBackend(lambda call: judge_reply(call))
        check_many(raw, "judge-m", contract, "Rate it.", ["Rate it now.", "Rate it, P1."], shown)
        calls += raw.calls
    bare, shown = calls
    assert "examples" in json.loads(shown.user) and "examples" not in json.loads(bare.user)
    assert planned_tokens(shown, 20) == planned_tokens(bare, 20) == 2 * CONTRACT_CHECKS * 25


def grown(have: int, ref: Reference | None, scenarios: int = 3, model: Latency | None = None):
    return grow(
        tier="fast",
        rewrites=1,
        scenarios=scenarios,
        holdout=0,
        have=have,
        can_synthesise=False,
        rewrites2=0,
        workers=6,
        prompt_tokens=20,
        model=model or latency(),
        seconds_left=1000.0,
        calls_left=1000,
        ref=ref,
    )


@pytest.mark.parametrize(("have", "count"), [(12, 8), (8, 8), (7, 7), (6, 6), (5, 5), (4, 4)])
def test_the_pick_grows_into_the_examples_up_to_eight(have, count):
    """WP23: up to REFERENCE_MAX_SCENARIOS, as the plan picks."""
    found = grown(have, Reference(have, 1))
    assert found is not None and (found.scenarios, found.synthesise) == (count, 0)


def test_without_references_the_pick_grows_to_four_as_before():
    found = grown(8, None)
    assert found is not None and found.scenarios == 4


def test_the_reference_stages_are_what_must_fit():
    """K=1 on 6 scenarios with references: B 3 waves of T, C one wave of F6 = 3.4 + 270/70; it
    fits 24 s, but not with 2 checks per scenario (F6 = 3.4 + 420/70)."""
    t = 3.4 + 150 / 70
    need = 3 * t + 3.4 + 270 / 70
    fits = grow(
        tier="fast",
        rewrites=1,
        scenarios=3,
        holdout=0,
        have=8,
        can_synthesise=False,
        rewrites2=0,
        workers=6,
        prompt_tokens=20,
        model=latency(),
        seconds_left=need,
        calls_left=1000,
        ref=Reference(8, 1),
    )
    assert fits is not None and fits.scenarios == 6
    two_checks = grow(
        tier="fast",
        rewrites=1,
        scenarios=3,
        holdout=0,
        have=8,
        can_synthesise=False,
        rewrites2=0,
        workers=6,
        prompt_tokens=20,
        model=latency(),
        seconds_left=need,
        calls_left=1000,
        ref=Reference(8, 2),
    )
    assert two_checks is None or two_checks.scenarios < 6
