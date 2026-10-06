"""The clock and the call limit in the fast pipeline (SPEC R17, R25; ADR-011): before each stage the
time and the calls left are compared with the stage's estimate (`fastplan`, one call at a time
here: workers 1) and what does not fit is shrunk, scenarios first, then rewrites; a deadline or a
call limit reached inside a stage ends it; a rewrite that has not passed every gate is never
returned. Estimates at workers 1: a task call 5.26 s; a judge call on n scenarios, or a contract
check of n rewrites, 3.47 s (n = 1), 4.54 s (2), 5.61 s (3); stage C is K + 2 calls, each as long
as the longest; stage E (4 held out) 55.4 s.
"""

import dataclasses

import pytest
from fakes import FakeClock
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2,
    K1M2_EXAMPLES,
    K2M2,
    K3M3,
    MODELS,
    PLAN,
    PROMPT,
    QUICK,
    World,
    is_contract_check,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
    scoring_judges,
    tagged,
)

from autoimprover.types import Call, CallError, Scenario

CLEAR = f"{BETTER} Be clear."
VERY_CLEAR = f"{BETTER} Be very clear."


def kept_unconfirmed(outcome, stop="clock"):
    assert (outcome.status, outcome.prompt, outcome.reason_code, outcome.stop) == (
        "unchanged",
        PROMPT,
        "unconfirmed_out_of_budget",
        stop,
    )
    assert not outcome.verified


def is_original_judge(call: Call) -> bool:
    return call.role == "judge" and not is_contract_check(call) and "BAD answer" in call.user


def test_stage_b_that_does_not_fit_runs_on_fewer_scenarios(tmp_path):
    """40 s left: 3 scenarios need 48.4 s for stages B and C, 2 need 34.7 s."""
    result = run(tmp_path, World(), K3M3, deadline=40)
    assert {scenario_of(c) for c in result.calls("task")} == {"situation 1", "situation 2"}
    outcome = result.outcome
    assert (outcome.prompt, outcome.reason_code, outcome.stop) == (BETTER, "improved", "clock")


def test_fewer_rewrites_when_one_scenario_is_not_enough(tmp_path):
    """35 s left: 3 rewrites on 1 scenario need 49.1 s, 2 rewrites 33.9 s."""
    result = run(tmp_path, World(rewrites=(BETTER, CLEAR, VERY_CLEAR)), K3M3, deadline=35)
    tasks = result.calls("task")
    assert [(prompt_of(c), scenario_of(c)) for c in tasks] == [
        (text, "situation 1") for text in (PROMPT, BETTER, CLEAR)
    ]
    assert (result.outcome.prompt, result.outcome.stop) == (BETTER, "clock")


EXAMPLES = [Scenario(id=f"e{n}", input=f"example {n}") for n in (1, 2)]


@pytest.mark.parametrize(
    ("examples", "roles"),
    [
        (None, ["intake", "synth", "reflect", "reflect", "reflect"]),
        (EXAMPLES, ["intake", "reflect"]),
    ],
)
def test_nothing_fits_keeps_the_original_unconfirmed(tmp_path, examples, roles):
    """19 s left: even 1 rewrite on 1 scenario needs 20.9 s for stages B and C."""
    fplan = K3M3 if examples is None else K1M2_EXAMPLES
    result = run(tmp_path, World(), fplan, deadline=19, examples=examples)
    kept_unconfirmed(result.outcome)
    assert [c.role for c in result.raw.calls] == roles


def test_a_deadline_already_passed_spends_nothing(tmp_path):
    result = run(tmp_path, World(), K3M3, deadline=0)
    kept_unconfirmed(result.outcome)
    assert result.raw.calls == []


def jump(clock: FakeClock, when, fail: bool = False) -> World:
    """A world that moves the clock past the deadline during the first call `when` picks; with
    `fail` that call fails once, so its retry and every later call are refused."""
    done: list[Call] = []

    def hook(call: Call) -> None:
        if when(call) and not done:
            done.append(call)
            clock.advance(1000)
            if fail:
                raise CallError("down")

    return World(hook=hook)


def is_original_task(call: Call) -> bool:
    return call.role == "task" and prompt_of(call) == PROMPT


@pytest.mark.parametrize("cut", [False, True])
def test_a_deadline_inside_stage_b_ends_it_and_keeps_the_original(tmp_path, cut):
    clock = FakeClock()
    world = jump(clock, is_original_task) if cut else World()
    result = run(tmp_path, world, K3M3, clock=clock, deadline=1000)
    if not cut:
        assert result.outcome.prompt == BETTER
        return
    kept_unconfirmed(result.outcome)
    assert len(result.calls("task")) == 1 and result.calls("judge") == []


@pytest.mark.parametrize("cut", [False, True])
def test_a_deadline_inside_stage_c_ends_it_and_keeps_the_original(tmp_path, cut):
    """Stage C asks the contract check first, then the judge calls; the rewrite's is refused."""
    clock = FakeClock()
    world = jump(clock, is_original_judge) if cut else World()
    result = run(tmp_path, world, K3M3, clock=clock, deadline=1000)
    if not cut:
        assert result.outcome.prompt == BETTER
        return
    kept_unconfirmed(result.outcome)
    assert len(result.calls("judge")) == 2


def test_the_call_limit_shrinks_a_stage_like_the_clock(tmp_path):
    """12 calls: 5 in stage A, then 3 scenarios need 9, 2 need 7."""
    result = run(tmp_path, World(), K3M3, plan=dataclasses.replace(PLAN, budget=12))
    assert {scenario_of(c) for c in result.calls("task")} == {"situation 1", "situation 2"}
    outcome = result.outcome
    assert (outcome.prompt, outcome.stop, outcome.calls_used) == (BETTER, "budget", 12)


@dataclasses.dataclass
class InvalidFirstOriginalJudgeReply(World):
    def __call__(self, call: Call) -> str | Exception:
        if is_original_judge(call) and call.sample == 0:
            return "not json"
        return super().__call__(call)


@pytest.mark.parametrize(("budget", "returned"), [(14, False), (15, True)])
def test_the_call_limit_reached_inside_stage_c_keeps_the_original(tmp_path, budget, returned):
    """14 calls cover the plan, but the original's judge reply is asked again, so the rewrite's
    judge call finds the limit spent."""
    plan = dataclasses.replace(PLAN, budget=budget)
    result = run(tmp_path, InvalidFirstOriginalJudgeReply(), K3M3, plan=plan)
    if returned:
        assert result.outcome.prompt == BETTER
    else:
        kept_unconfirmed(result.outcome, stop="budget")
        assert len(result.calls("judge")) == 3


@pytest.mark.parametrize(("deadline", "returned"), [(3, False), (4, True)])
def test_the_quick_contract_check_runs_only_when_it_fits(tmp_path, deadline, returned):
    """The contract check is estimated at 3.47 s."""
    result = run(tmp_path, World(), QUICK, deadline=deadline)
    assert (result.outcome.prompt == BETTER) is returned
    assert len(result.calls("judge")) == int(returned)
    if not returned:
        kept_unconfirmed(result.outcome)


@pytest.mark.parametrize("cut", [False, True])
def test_a_checked_run_without_time_for_stage_e_keeps_the_original(tmp_path, cut):
    """Stage E is estimated at 55.4 s; the rewrite's judge call leaves 40 s."""
    clock = FakeClock()

    def hook(call: Call) -> None:
        if cut and call.role == "judge" and "GOOD answer" in call.user:
            clock.advance(960)

    result = run(tmp_path, World(hook=hook), CHECKED, clock=clock, deadline=1000)
    on_target = [c for c in result.calls("task") if c.model == MODELS.target]
    if not cut:
        assert (result.outcome.prompt, result.outcome.verified) == (BETTER, True)
        return
    kept_unconfirmed(result.outcome)
    assert on_target == [] and result.outcome.search_score_before == 0.0
    assert result.outcome.score_before is None  # no held-out score: the pick score is not one


def test_a_rewrite_judged_before_the_deadline_may_win_when_a_later_one_is_cut(tmp_path):
    """The contract check passes both rewrites; the deadline passes during the first rewrite's
    judge call and the second's is refused. The first passed every gate in time (SPEC R25)."""
    clock = FakeClock()
    world = jump(clock, lambda call: call.role == "judge" and f"answer {len(BETTER)}" in call.user)
    world.rewrites, world.task = (BETTER, CLEAR), tagged
    result = run(tmp_path, world, K2M2, clock=clock, deadline=1000)
    assert (result.outcome.prompt, result.outcome.stop) == (BETTER, "clock")
    assert len(result.calls("judge")) == 3  # the contract check, the original's, the first's


def test_a_deadline_that_refuses_every_run_of_the_original_is_no_backend_failure(tmp_path):
    clock = FakeClock()
    world = jump(clock, lambda call: call.role == "task", fail=True)
    result = run(tmp_path, world, K3M3, clock=clock, deadline=1000)
    kept_unconfirmed(result.outcome)
    assert len(result.calls("task")) == 1 and result.calls("judge") == []


def test_a_deadline_that_refuses_the_originals_judge_call_is_no_backend_failure(tmp_path):
    clock = FakeClock()
    result = run(
        tmp_path, jump(clock, is_original_judge, fail=True), K3M3, clock=clock, deadline=1000
    )
    kept_unconfirmed(result.outcome)
    assert len(result.calls("judge")) == 2  # the contract check, the original's one attempt


def test_the_checked_tier_runs_no_task_without_time_for_stage_e(tmp_path):
    """70 s left: 1 scenario needs 20.9 s for stages B and C, 76.4 s with stage E."""
    result = run(tmp_path, World(), CHECKED, deadline=70)
    kept_unconfirmed(result.outcome)
    assert result.calls("task") == []


def test_the_checked_tier_shrinks_stage_b_to_keep_time_for_stage_e(tmp_path):
    """82 s left: 2 scenarios need 90.1 s with stage E, 1 needs 76.4 s."""
    result = run(tmp_path, World(), CHECKED, deadline=82)
    picked = [c for c in result.calls("task") if c.model == MODELS.task]
    assert {scenario_of(c) for c in picked} == {"situation 1"}
    assert (result.outcome.prompt, result.outcome.verified) == (BETTER, True)


def test_a_checked_run_skips_stage_c_when_stage_e_would_not_fit_after_it(tmp_path):
    """After stage B 20 s are left: stage C needs 13.6 s, but stage E 55.4 s more."""
    clock = FakeClock()

    def hook(call: Call) -> None:
        if call.role == "task" and prompt_of(call) == BETTER and scenario_of(call) == "situation 2":
            clock.advance(980)

    result = run(tmp_path, World(hook=hook), CHECKED, clock=clock, deadline=1000)
    kept_unconfirmed(result.outcome)
    assert result.calls("judge") == []


def test_stage_c_shrinks_to_the_time_stage_b_left(tmp_path):
    """Stage B overruns and leaves 12 s: stage C's three calls on 2 scenarios need 13.6 s, on 1
    scenario 10.4 s."""
    clock = FakeClock()

    def hook(call: Call) -> None:
        if call.role == "task" and prompt_of(call) == BETTER and scenario_of(call) == "situation 2":
            clock.advance(988)

    result = run(tmp_path, World(hook=hook), K1M2, clock=clock, deadline=1000)
    assert [judged_scenarios(c) for c in scoring_judges(result)] == [["s1"], ["s1"]]
    assert (result.outcome.prompt, result.outcome.stop) == (BETTER, "clock")
