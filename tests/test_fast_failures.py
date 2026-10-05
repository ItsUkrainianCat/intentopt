"""Failed calls in the fast pipeline (SPEC R24, R25): a failed task or judge call about a rewrite
drops that rewrite and the others go on; a failed call that leaves the original without a score,
the contract check, or a held-out run on the target model ends the run as BackendError, as a
failed call outside the search does. Each "not returned" test has a twin that is.
"""

import json

import pytest
from fakes import MARKER
from test_fast_world import (  # noqa: F401  (no_disk_flush is an autouse fixture)
    BETTER,
    CHECKED,
    K2M2,
    MODELS,
    QUICK,
    World,
    is_contract_check,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
    scoring_judges,
    tagged,
)

from autoimprover.types import BackendError, Call, CallError

CLEAR = f"{BETTER} Be clear."


def graded(result, text: str) -> list[Call]:
    """The scoring judge calls that grade the outputs of prompt `text` (`tagged` outputs)."""
    return [c for c in scoring_judges(result) if f"answer {len(text)}" in c.user]


def checked_rewrites(result) -> list[str]:
    (call,) = [c for c in result.calls("judge") if is_contract_check(c)]
    return [item["output"] for item in json.loads(call.user)["scenarios"]]


@pytest.mark.parametrize("fails", [False, True])
def test_a_rewrite_whose_judge_call_fails_is_dropped_and_another_wins(tmp_path, fails):
    def hook(call: Call) -> None:
        if fails and call.role == "judge" and f"answer {len(BETTER)}" in call.user:
            raise CallError("down")

    result = run(tmp_path, World(rewrites=(BETTER, CLEAR), task=tagged, hook=hook), K2M2)
    assert result.outcome.prompt == (CLEAR if fails else BETTER)
    assert len(graded(result, BETTER)) == (3 if fails else 1)  # three attempts, then dropped


@pytest.mark.parametrize("fails", [False, True])
def test_a_rewrite_whose_task_runs_all_fail_is_never_judged(tmp_path, fails):
    def task(call: Call) -> str:
        if fails and prompt_of(call) == BETTER:
            raise CallError("down")
        return tagged(call)

    result = run(tmp_path, World(rewrites=(BETTER, CLEAR), task=task), K2M2)
    assert result.outcome.prompt == (CLEAR if fails else BETTER)
    assert len(graded(result, BETTER)) == int(not fails)
    assert checked_rewrites(result) == ([CLEAR] if fails else [BETTER, CLEAR])


@pytest.mark.parametrize("fails", [False, True])
def test_an_original_whose_judge_call_fails_ends_the_run(tmp_path, fails):
    def hook(call: Call) -> None:
        if fails and call.role == "judge" and '"BAD answer"' in call.user:
            raise CallError("down")

    if not fails:
        assert run(tmp_path, World(hook=hook), K2M2).outcome.prompt == BETTER
        return
    with pytest.raises(BackendError, match="original"):
        run(tmp_path, World(hook=hook), K2M2)


@pytest.mark.parametrize("tier", ["quick", "fast"])
@pytest.mark.parametrize("fails", [False, True])
def test_a_failed_contract_check_ends_the_run(tmp_path, tier, fails):
    """It is the one veto every rewrite needs, as in SPEC R24 ("contract checks")."""

    def hook(call: Call) -> None:
        if fails and is_contract_check(call):
            raise CallError("down")

    fplan = QUICK if tier == "quick" else K2M2
    if not fails:
        assert run(tmp_path, World(hook=hook), fplan).outcome.prompt == BETTER
        return
    with pytest.raises(BackendError, match="judge"):
        run(tmp_path, World(hook=hook), fplan)


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
