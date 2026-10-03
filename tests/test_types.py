"""The shared types enforce the rules the spec puts on them."""

import dataclasses

import pytest

from autoimprover.types import (
    BUDGET_CEILING,
    CALL_RETRIES,
    CALL_TIMEOUT_S,
    DEFAULT_MODELS,
    EXIT_BACKEND,
    EXIT_INTERRUPTED,
    EXIT_NOT_LOCKED_DOWN,
    EXIT_OK,
    EXIT_USAGE,
    LENGTH_CAP,
    MAX_CONSECUTIVE_FAILURES,
    PROGRAMMATIC_RULES,
    SYSTEM_PROMPT_MAX_BYTES,
    Call,
    Check,
    Contract,
    Models,
    Outcome,
    Plan,
)


def models(**kw: str) -> Models:
    base = {"task": "haiku", "judge": "opus", "reflect": "opus", "target": "sonnet"}
    return Models(**{**base, **kw})


def test_judge_must_differ_from_task_model():
    with pytest.raises(ValueError, match="R14"):
        models(task="sonnet", judge="sonnet")


def test_judge_must_differ_from_target_model():
    # Sonnet judging a Sonnet-produced holdout run would grade its own work (SPEC R14).
    with pytest.raises(ValueError, match="R14"):
        models(judge="sonnet", target="sonnet")


def test_target_may_equal_task_or_reflection_model():
    assert models(target="haiku").target == "haiku"
    assert models(target="opus", judge="sonnet", reflect="opus").target == "opus"


def test_default_models_never_grade_their_own_work():
    assert DEFAULT_MODELS.judge not in (DEFAULT_MODELS.task, DEFAULT_MODELS.target)


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


def test_call_fields_are_fixed_so_the_cache_key_is_a_decision():
    # The disk cache is keyed by every field of Call (ADR-004). Adding a field must be deliberate.
    assert [f.name for f in dataclasses.fields(Call)] == [
        "role",
        "model",
        "user",
        "system",
        "json_schema",
        "sample",
    ]


def test_second_seed_run_is_a_different_call():
    # SPEC R12: run 2 must not be served from the cache, or the noise estimate is always 0.
    first = Call(role="task", model="m", user="u")
    second = dataclasses.replace(first, sample=1)
    assert first != second
    assert len({first, second}) == 2


def test_regex_is_not_a_programmatic_rule():
    # A pattern written by a model would run in this process (SPEC R19).
    assert "regex" not in PROGRAMMATIC_RULES
    with pytest.raises(ValueError, match="rule"):
        Check(id="c1", group="format", text="t", rule="regex", arg="(a+)+$")


def test_programmatic_check_needs_an_argument_and_a_judged_check_does_not():
    with pytest.raises(ValueError, match="argument"):
        Check(id="c1", group="content", text="t", rule="contains")
    assert Check(id="c2", group="content", text="t").rule is None
    assert Check(id="c3", group="format", text="t", rule="max_chars", arg="500").arg == "500"


def test_outcome_is_unverified_unless_the_holdout_decided():
    out = Outcome(status="improved", prompt="p", reason="r")
    assert out.verified is False
    assert out.stop == "finished"


def test_exit_codes_match_spec_r2():
    assert (EXIT_OK, EXIT_USAGE, EXIT_BACKEND, EXIT_NOT_LOCKED_DOWN, EXIT_INTERRUPTED) == (
        0,
        2,
        3,
        4,
        130,
    )


def test_call_limits_match_spec_r17_r18_r24():
    assert (CALL_TIMEOUT_S, CALL_RETRIES, MAX_CONSECUTIVE_FAILURES) == (300, 2, 3)
    assert SYSTEM_PROMPT_MAX_BYTES < 131_072
