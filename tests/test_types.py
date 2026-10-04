"""The shared types enforce the rules the spec puts on them."""

import dataclasses

import pytest

from autoimprover.types import (
    BUDGET_CEILING,
    CALL_RETRIES,
    CALL_TIMEOUT_S,
    DEFAULT_MODELS,
    EXIT_BACKEND,
    EXIT_INTERNAL,
    EXIT_INTERRUPTED,
    EXIT_NOT_LOCKED_DOWN,
    EXIT_OK,
    EXIT_USAGE,
    HOLDOUT_MAX,
    JUDGE_BATCH_MAX,
    LENGTH_CAP,
    LENGTH_FLOOR_TOKENS,
    MAX_CONSECUTIVE_FAILURES,
    MODEL_ALIASES,
    PROGRAMMATIC_RULES,
    PROMPT_MAX_CHARS,
    SEARCH_CLOCK_SHARE,
    SYSTEM_PROMPT_MAX_BYTES,
    WALL_CLOCK_DEFAULT_S,
    Call,
    Check,
    Contract,
    Models,
    Outcome,
    Plan,
    Reply,
    canonical_model,
    default_models,
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


@pytest.mark.parametrize("target", ["opus", "OPUS", " claude-opus-5-5 ", "claude-opus-5-5[1m]"])
def test_judge_and_target_are_compared_after_aliases_are_resolved(target: str):
    with pytest.raises(ValueError, match="R14"):
        models(judge="claude-opus-5-5", target=target)


def test_models_are_stored_as_full_ids():
    m = models()
    assert (m.task, m.judge, m.target) == tuple(
        MODEL_ALIASES[k] for k in ("haiku", "opus", "sonnet")
    )


def test_target_may_equal_task_or_reflection_model():
    assert models(target="haiku").target == MODEL_ALIASES["haiku"]
    assert models(target="opus", judge="sonnet", reflect="opus").target == MODEL_ALIASES["opus"]


def test_unknown_model_names_pass_through_and_empty_ones_are_rejected():
    assert canonical_model("Some-Future-Model") == "some-future-model"
    with pytest.raises(ValueError, match="empty"):
        canonical_model("  ")


def test_default_models_never_grade_their_own_work():
    assert DEFAULT_MODELS.judge not in (DEFAULT_MODELS.task, DEFAULT_MODELS.target)
    assert DEFAULT_MODELS.judge == MODEL_ALIASES["opus"]


@pytest.mark.parametrize("target", ["opus", "claude-opus-5-5", "Opus"])
def test_default_judge_falls_back_when_the_target_is_the_default_judge(target: str):
    # `/improve` passes the session model as the target; in an Opus session it must still work.
    m = default_models(target)
    assert m.target == MODEL_ALIASES["opus"]
    assert m.judge == MODEL_ALIASES["sonnet"]


def test_default_models_for_other_targets_keep_the_default_judge():
    for target in ("haiku", "sonnet", None):
        assert default_models(target).judge == MODEL_ALIASES["opus"]


@pytest.mark.parametrize("budget", [0, -1, BUDGET_CEILING + 1, 2.5, True, "100"])
def test_budget_out_of_range_or_not_a_whole_number_is_rejected(budget: object):
    with pytest.raises(ValueError, match="budget"):
        Plan(models=models(), budget=budget)  # type: ignore[arg-type]


def test_plan_rejects_an_unknown_strictness_and_a_bad_clock():
    with pytest.raises(ValueError, match="strictness"):
        Plan(models=models(), strictness="extreme")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wall_clock_s"):
        Plan(models=models(), wall_clock_s=0)


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


@pytest.mark.parametrize("system", ["has\0nul", "x" * (SYSTEM_PROMPT_MAX_BYTES + 1), "é" * 60_000])
def test_a_system_prompt_that_cannot_travel_as_one_argument_is_rejected(system: str):
    with pytest.raises(ValueError, match="R18"):
        Call(role="task", model="m", user="u", system=system)


def test_a_system_prompt_at_the_limit_is_accepted():
    assert Call(role="task", model="m", user="u", system="x" * SYSTEM_PROMPT_MAX_BYTES)


def test_regex_is_not_a_programmatic_rule():
    # A pattern written by a model would run in this process (SPEC R19).
    assert "regex" not in PROGRAMMATIC_RULES
    with pytest.raises(ValueError, match="rule"):
        Check(id="c1", group="format", text="t", rule="regex", arg="(a+)+$")


def test_programmatic_check_needs_an_argument_and_a_judged_check_does_not():
    with pytest.raises(ValueError, match="argument"):
        Check(id="c1", group="content", text="t", rule="contains")
    with pytest.raises(ValueError, match="no argument"):
        Check(id="c2", group="content", text="t", arg="x")
    assert Check(id="c3", group="content", text="t").rule is None
    assert Check(id="c4", group="format", text="t", rule="max_chars", arg="500").arg == "500"


@pytest.mark.parametrize("arg", ["ten", "-5", "2.5", "٣"])
def test_length_checks_need_a_whole_number(arg: str):
    with pytest.raises(ValueError, match="whole number"):
        Check(id="c1", group="format", text="t", rule="min_chars", arg=arg)


def test_check_group_and_contract_kind_are_validated():
    with pytest.raises(ValueError, match="group"):
        Check(id="c1", group="style", text="t")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        Contract(goal="g", kind="agent")  # type: ignore[arg-type]


def test_outcome_is_unverified_unless_the_holdout_decided():
    out = Outcome(status="improved", prompt="p", reason="r")
    assert out.verified is False
    assert out.stop == "finished"


def test_outcome_rejects_an_unknown_status_and_an_improvement_that_is_not_one():
    with pytest.raises(ValueError, match="status"):
        Outcome(status="error", prompt="p", reason="r")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="higher"):
        Outcome(status="improved", prompt="p", reason="r", score_before=0.6, score_after=0.5)
    assert Outcome(status="improved", prompt="p", reason="r", score_before=0.5, score_after=0.6)


def test_exit_codes_match_spec_r2():
    codes = (EXIT_OK, EXIT_INTERNAL, EXIT_USAGE, EXIT_BACKEND, EXIT_NOT_LOCKED_DOWN)
    assert (*codes, EXIT_INTERRUPTED) == (0, 1, 2, 3, 4, 130)


def test_spec_constants_are_pinned():
    # Each is a number the SPEC states (R7, R15, R17, R18, R24); a change must be deliberate.
    assert (CALL_TIMEOUT_S, CALL_RETRIES, MAX_CONSECUTIVE_FAILURES) == (300, 2, 3)
    assert SYSTEM_PROMPT_MAX_BYTES < 131_072
    assert (WALL_CLOCK_DEFAULT_S, SEARCH_CLOCK_SHARE) == (45 * 60, 0.75)
    assert (LENGTH_FLOOR_TOKENS, PROMPT_MAX_CHARS) == (40, 20_000)
    assert (HOLDOUT_MAX, JUDGE_BATCH_MAX) == (6, 6)
    assert BUDGET_CEILING == 300


def test_a_reply_carries_the_duration_the_search_meter_needs():
    assert Reply(text="x").duration_s == 0.0
    assert Reply(text="x", cached=True, duration_s=4.5).duration_s == 4.5


def test_outcome_carries_the_report_fields_of_r2_and_r14a():
    out = Outcome(
        status="improved",
        prompt="p",
        reason="r",
        verified=True,
        changes=("tightened the opening", "kept the output format"),
        score_before=0.5,
        score_after=0.7,
        search_score_before=0.4,
        search_score_after=0.8,
    )
    assert len(out.changes) == 2
    assert (out.search_score_before, out.search_score_after) == (0.4, 0.8)
    assert Outcome(status="unchanged", prompt="p", reason="r").changes == ()


def test_the_synthesis_schema_fixes_the_scenario_count_the_budget_assumes():
    from autoimprover.types import SYNTH_COUNT, SYNTH_SCHEMA

    items = SYNTH_SCHEMA["properties"]["scenarios"]
    assert SYNTH_COUNT == 12
    assert (items["minItems"], items["maxItems"]) == (SYNTH_COUNT, SYNTH_COUNT)


def test_a_judge_quote_cannot_be_empty():
    from autoimprover.types import JUDGE_SCHEMA

    check = JUDGE_SCHEMA["properties"]["results"]["items"]["properties"]["checks"]["items"]
    assert check["properties"]["quote"]["minLength"] == 1
