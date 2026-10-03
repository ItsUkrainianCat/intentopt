"""The shared types enforce the rules the spec puts on them."""

import dataclasses

import pytest

from autoimprover.types import (
    BUDGET_CEILING,
    LENGTH_CAP,
    Call,
    Contract,
    Models,
    Plan,
)


def models(**kw: str) -> Models:
    base = {"task": "haiku", "judge": "sonnet", "reflect": "opus", "target": "sonnet"}
    return Models(**{**base, **kw})


def test_judge_must_differ_from_task_model():
    with pytest.raises(ValueError, match="R14"):
        models(task="sonnet", judge="sonnet")


def test_target_may_equal_judge_or_task():
    assert models(target="haiku").target == "haiku"
    assert models(target="sonnet").target == "sonnet"


@pytest.mark.parametrize("budget", [0, -1, BUDGET_CEILING + 1])
def test_budget_out_of_range_is_rejected(budget: int):
    with pytest.raises(ValueError, match="budget"):
        Plan(models=models(), budget=budget)


def test_plan_defaults_are_the_spec_defaults():
    plan = Plan(models=models())
    assert (plan.budget, plan.strictness, plan.merge, plan.allow_growth) == (
        100,
        "conservative",
        False,
        False,
    )


def test_length_caps_match_spec_r8():
    assert LENGTH_CAP == {"conservative": 1.25, "balanced": 1.5, "bold": 2.5}


def test_types_are_frozen():
    call = Call(role="task", model="haiku", user="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        call.user = "changed"  # type: ignore[misc]
    contract = Contract(goal="g", kind="task")
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.goal = "changed"  # type: ignore[misc]
