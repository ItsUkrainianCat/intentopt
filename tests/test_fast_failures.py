"""Failed calls in the fast pipeline (SPEC R24, R25): a failed call about a rewrite drops that
rewrite and the others go on; a failed call that leaves the original without a score, the quick
tier's contract check or a held-out run on the target model ends the run as BackendError, as a
failed call outside the search does. And the scenario id the judge call keeps for the contract
check is never a scenario to compare on (ADR-008). Each "not returned" test has a twin that is.
"""

import json

import pytest
from fakes import MARKER
from test_fast_world import (  # noqa: F401  (no_disk_flush is an autouse fixture)
    BETTER,
    CHECKED,
    K1M2_EXAMPLES,
    K2M2,
    MODELS,
    PROMPT,
    QUICK,
    World,
    judged_scenarios,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
)

from autoimprover.types import BackendError, Call, CallError, Scenario

CLEAR = f"{BETTER} Be clear."


def judges_of(result, text: str) -> list[Call]:
    """The judge calls whose contract scenario holds `text` as the rewrite."""
    return [
        c
        for c in result.calls("judge")
        if any(
            item["scenario"] == "contract" and item["output"] == text
            for item in json.loads(c.user)["scenarios"]
        )
    ]


@pytest.mark.parametrize("fails", [False, True])
def test_a_rewrite_whose_judge_call_fails_is_dropped_and_another_wins(tmp_path, fails):
    def hook(call: Call) -> None:
        if fails and call.role == "judge" and f'"output": "{BETTER}"' in call.user:
            raise CallError("down")

    result = run(tmp_path, World(rewrites=(BETTER, CLEAR), hook=hook), K2M2)
    assert result.outcome.prompt == (CLEAR if fails else BETTER)
    assert len(judges_of(result, BETTER)) == (3 if fails else 1)


@pytest.mark.parametrize("fails", [False, True])
def test_a_rewrite_whose_task_runs_all_fail_is_never_judged(tmp_path, fails):
    def task(call: Call) -> str:
        if fails and prompt_of(call) == BETTER:
            raise CallError("down")
        return "GOOD answer" if MARKER in prompt_of(call) else "BAD answer"

    result = run(tmp_path, World(rewrites=(BETTER, CLEAR), task=task), K2M2)
    assert result.outcome.prompt == (CLEAR if fails else BETTER)
    assert len(judges_of(result, BETTER)) == int(not fails)


@pytest.mark.parametrize("fails", [False, True])
def test_an_original_whose_judge_call_fails_ends_the_run(tmp_path, fails):
    def hook(call: Call) -> None:
        if fails and call.role == "judge" and "contract" not in judged_scenarios(call):
            raise CallError("down")

    if not fails:
        assert run(tmp_path, World(hook=hook), K2M2).outcome.prompt == BETTER
        return
    with pytest.raises(BackendError, match="original"):
        run(tmp_path, World(hook=hook), K2M2)


@pytest.mark.parametrize("fails", [False, True])
def test_a_failed_quick_contract_check_ends_the_run(tmp_path, fails):
    def hook(call: Call) -> None:
        if fails and call.role == "judge":
            raise CallError("down")

    if not fails:
        assert run(tmp_path, World(hook=hook), QUICK).outcome.prompt == BETTER
        return
    with pytest.raises(BackendError, match="judge"):
        run(tmp_path, World(hook=hook), QUICK)


@pytest.mark.parametrize(("rewrite", "returned"), [(BETTER + " more" * 50, False), (BETTER, True)])
def test_a_quick_rewrite_over_the_cap_is_never_checked_or_returned(tmp_path, rewrite, returned):
    result = run(tmp_path, World(rewrites=(rewrite,)), QUICK)
    assert (result.outcome.prompt == rewrite, len(result.calls("judge"))) == (returned, returned)


@pytest.mark.parametrize("fails", [False, True])
def test_a_failed_held_out_run_on_the_target_ends_the_run(tmp_path, fails):
    def hook(call: Call) -> None:
        on_target = call.role == "task" and call.model == MODELS.target
        if fails and on_target and scenario_of(call) == "situation 3" and MARKER in call.user:
            raise CallError("down")

    if not fails:
        assert run(tmp_path, World(hook=hook), CHECKED).outcome.verified
        return
    with pytest.raises(BackendError, match="holdout"):
        run(tmp_path, World(hook=hook), CHECKED)


NAMED = [Scenario(id="contract", input="example 0"), Scenario(id="e1", input="example 1")]


def test_a_scenario_named_like_the_contract_check_is_never_compared_on(tmp_path):
    result = run(tmp_path, World(), K1M2_EXAMPLES, examples=NAMED)
    assert {scenario_of(c) for c in result.calls("task")} == {"example 1"}
    assert [judged_scenarios(c) for c in result.calls("judge")] == [["e1"], ["e1", "contract"]]
    assert result.outcome.prompt == BETTER


def test_with_only_such_a_scenario_the_original_is_kept_before_any_scoring(tmp_path):
    result = run(tmp_path, World(), K1M2_EXAMPLES, examples=NAMED[:1])
    assert (result.outcome.prompt, result.outcome.reason_code) == (
        PROMPT,
        "no_reliable_improvement",
    )
    assert result.calls("task") == [] and "no scenario left" in result.outcome.reason
