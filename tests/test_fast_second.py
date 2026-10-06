"""The second generation of the fast tiers (SPEC R25 "Quality of the rewrites", R16; ADR-011
amendment of 2026-10-06): from 45 s, when the plan has room, the reflection model reads the best
one or two first-generation rewrites that gained on the baseline (or the original when none did),
their outputs and their failed checks with the judge's quotes, and writes K2 rewrites under
distinct notes; they pass the same gates and are scored by the same noise rule, one contract check
for all of them, and the pick is over every candidate. When the second generation does not fit or
is cut, the first generation's result stands. Every "not returned" test has a twin that is.
"""

import dataclasses
import json

import pytest
from fakes import MARKER, FakeClock
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2,
    MODELS,
    PROMPT,
    TWO_K1,
    TWO_K3,
    Verbatim,
    World,
    is_contract_check,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    tagged,
)

from autoimprover.fast_prompts import REFLECT_NOTES, STRATEGY_NOTES
from autoimprover.types import Call, CallError

PLAIN = "Answer the user's request plainly."  # a rewrite that runs no better
SECOND = f"Answer the user's request {MARKER} now."
SHORT_SECOND = f"Answer {MARKER}."


def reflections(result) -> list[Call]:
    return [c for c in result.calls("reflect") if c.sample >= 100]


def candidates_of(call: Call) -> list[dict]:
    return json.loads(call.user)["candidates"]


def test_the_plans_have_the_shapes_these_tests_assume():
    assert (TWO_K1.rewrites, TWO_K1.scenarios, TWO_K1.generations, TWO_K1.rewrites2) == (1, 2, 2, 2)
    assert (TWO_K3.rewrites, TWO_K3.scenarios, TWO_K3.generations, TWO_K3.rewrites2) == (3, 2, 2, 2)
    assert K1M2.generations == 1


def test_when_no_rewrite_gains_the_reflection_reads_the_original_and_its_failures(tmp_path):
    result = run(tmp_path, World(rewrites=(PLAIN,), reflections=(SECOND,)), TWO_K1)
    calls = reflections(result)
    assert [c.sample for c in calls] == [100, 101] and len({c.system for c in calls}) == 2
    assert {c.model for c in calls} == {MODELS.reflect}
    [parent] = candidates_of(calls[0])
    assert parent["prompt"] == PROMPT
    assert [s["input"] for s in parent["scenarios"]] == ["situation 1", "situation 2"]
    scenario = parent["scenarios"][0]
    assert scenario["output"] == "BAD answer"
    assert scenario["failed"] == [{"check": "answers the request", "judge_quote": "BAD answer"}]
    outcome = result.outcome
    assert (outcome.prompt, outcome.changes) == (SECOND, (REFLECT_NOTES["repair"],))


def test_the_reflection_reads_the_best_two_rewrites_that_gained(tmp_path):
    """Equal gains: the shorter first, whatever order the variants came in."""
    world = World(rewrites=(f"{BETTER} Be clear.", BETTER, PLAIN), reflections=(PLAIN,))
    result = run(tmp_path, world, TWO_K3)
    calls = reflections(result)
    assert [p["prompt"] for p in candidates_of(calls[0])] == [BETTER, f"{BETTER} Be clear."]
    assert result.outcome.prompt == BETTER  # the reflections gained nothing: the first stands


def test_of_three_rewrites_that_gained_the_reflection_reads_two(tmp_path):
    texts = (f"{BETTER} Be clear.", BETTER, f"{BETTER} Be brief and clear.")
    calls = reflections(run(tmp_path, World(rewrites=texts, reflections=(PLAIN,)), TWO_K3))
    assert [p["prompt"] for p in candidates_of(calls[0])] == list(texts[1::-1])


def test_the_rewrite_that_gained_more_is_read_first_even_when_longer(tmp_path):
    half = f"Answer {MARKER}."  # good on situation 1 only: it gains half of what BETTER gains

    def task(call: Call) -> str:
        text = prompt_of(call)
        good = MARKER in text and (text != half or "situation 1" in call.user)
        return "GOOD answer" if good else "BAD answer"

    world = World(rewrites=(half, BETTER, PLAIN), reflections=(PLAIN,), task=task)
    calls = reflections(run(tmp_path, world, TWO_K3))
    assert [p["prompt"] for p in candidates_of(calls[0])] == [BETTER, half]


def test_a_gain_is_measured_on_the_scenarios_the_rewrite_completed(tmp_path):
    """`gapped` fails on situation 2 and is good on situation 1: a gain of 1.0 on the one it
    completed, more than the 0.5 of `half`, good on situation 1 of both, though it is longer."""
    half, gapped = f"Answer {MARKER}.", f"Answer the user's request {MARKER} in full."

    def task(call: Call) -> str:
        text, first = prompt_of(call), "situation 1" in call.user
        if text == gapped and not first:
            raise CallError("down")
        return "GOOD answer" if MARKER in text and first else "BAD answer"

    world = World(rewrites=(half, gapped, PLAIN), reflections=(PLAIN,), task=task)
    calls = reflections(run(tmp_path, world, TWO_K3))
    assert [p["prompt"] for p in candidates_of(calls[0])] == [gapped, half]


@pytest.mark.parametrize("keeps", [False, True])
def test_a_rewrite_that_gained_but_broke_the_contract_is_not_what_the_reflection_reads(
    tmp_path, keeps
):
    world = World(
        rewrites=(BETTER,), reflections=(PLAIN,), contract_ok=lambda text: keeps or text != BETTER
    )
    [parent] = candidates_of(reflections(run(tmp_path, world, TWO_K1))[0])
    assert parent["prompt"] == (BETTER if keeps else PROMPT)


def test_one_rewrite_that_gained_is_the_only_one_the_reflection_reads(tmp_path):
    result = run(tmp_path, World(rewrites=(BETTER, PLAIN), reflections=(PLAIN,)), TWO_K3)
    assert [p["prompt"] for p in candidates_of(reflections(result)[0])] == [BETTER]


@pytest.mark.parametrize(("second", "winner"), [(SHORT_SECOND, SHORT_SECOND), (PLAIN, BETTER)])
def test_the_pick_is_over_every_candidate_of_both_generations(tmp_path, second, winner):
    """A second-generation rewrite that ties with the first's on gain and is shorter wins; one
    that does not gain loses to it."""
    result = run(tmp_path, World(rewrites=(BETTER,), reflections=(second,)), TWO_K1)
    assert result.outcome.prompt == winner
    note = REFLECT_NOTES["repair"] if winner == second else STRATEGY_NOTES["clarify"]
    assert result.outcome.changes == (note,)


def test_the_second_generation_has_its_own_runs_judges_and_one_contract_check(tmp_path):
    world = World(rewrites=(PLAIN,), reflections=(SECOND, SHORT_SECOND))
    result = run(tmp_path, world, TWO_K1)
    tasks = [prompt_of(c) for c in result.calls("task")]
    assert tasks.count(SECOND) == 2 and tasks.count(SHORT_SECOND) == 2
    contracts = [c for c in result.calls("judge") if is_contract_check(c)]
    assert [judged_scenarios(c) for c in contracts] == [
        ["contract-1"],
        ["contract-1", "contract-2"],
    ]
    assert result.outcome.prompt == SHORT_SECOND
    assert "fast: stage C2: a contract check and 2 judge calls" in result.log


def test_one_generation_below_45_s(tmp_path):
    result = run(tmp_path, World(rewrites=(PLAIN,), reflections=(SECOND,)), K1M2)
    assert reflections(result) == [] and result.outcome.prompt == PROMPT
    assert "stage R" not in result.log


@pytest.mark.parametrize(
    ("second", "log"),
    [
        (PROMPT + "?", "reflection 0 dropped: no change in meaning words"),
        (
            "answer the user's request plainly!",
            "reflection 0 dropped: the meaning words of rewrite 0",
        ),
    ],
)
def test_a_reflection_that_repeats_a_candidate_in_meaning_words_is_dropped(tmp_path, second, log):
    result = run(tmp_path, World(rewrites=(PLAIN,), reflections=(second, SECOND)), TWO_K1)
    assert log in result.log
    assert second not in [prompt_of(c) for c in result.calls("task")]
    assert result.outcome.prompt == SECOND  # the other reflection still runs


def test_a_failed_reflection_is_dropped_and_the_other_goes_on(tmp_path):
    world = World(rewrites=(PLAIN,), reflections=(CallError("down"), SECOND))
    result = run(tmp_path, world, TWO_K1)
    assert len(reflections(result)) == 3 + 1 and result.outcome.prompt == SECOND
    assert "reflection 0 dropped: failed" in result.log


def test_a_reflection_without_a_delimited_prompt_is_dropped(tmp_path):
    world = World(rewrites=(PLAIN,), reflections=(Verbatim("just text"), SECOND))
    assert run(tmp_path, world, TWO_K1).outcome.prompt == SECOND


@pytest.mark.parametrize("late", [False, True])
def test_a_second_generation_that_does_not_fit_leaves_the_first_ones_result(tmp_path, late):
    """The first generation's last judge call leaves 20 s; the second needs 2 Rs + 4 T + 3 J2 =
    38.3 s at workers 1."""
    clock = FakeClock()

    def hook(call: Call) -> None:
        if late and call.role == "judge" and '"output": "GOOD answer"' in call.user:
            clock.advance(980)

    world = World(rewrites=(BETTER,), reflections=(SHORT_SECOND,), hook=hook)
    result = run(tmp_path, world, TWO_K1, clock=clock, deadline=1000)
    assert (reflections(result) == []) is late
    outcome = result.outcome
    assert outcome.prompt == (BETTER if late else SHORT_SECOND)
    assert outcome.stop == ("clock" if late else None)


def test_a_second_generation_cut_by_the_clock_leaves_the_first_ones_result(tmp_path):
    clock = FakeClock()

    def hook(call: Call) -> None:
        if call.role == "task" and prompt_of(call) == SHORT_SECOND:
            clock.advance(1000)

    world = World(rewrites=(BETTER,), reflections=(SHORT_SECOND,), hook=hook)
    outcome = run(tmp_path, world, TWO_K1, clock=clock, deadline=1000).outcome
    assert (outcome.prompt, outcome.stop) == (BETTER, "clock")


@pytest.mark.parametrize("cut", [False, True])
def test_a_second_generation_cut_in_its_judge_calls_leaves_the_first_ones_result(tmp_path, cut):
    """The first judge call that sees a good answer is the second generation's first; the clock
    then reaches the deadline, so its second judge call is refused (`tagged` outputs differ, so
    the cache cannot answer it)."""
    clock, once = FakeClock(), []

    def hook(call: Call) -> None:
        if cut and call.role == "judge" and '"output": "GOOD' in call.user and not once:
            once.append(call)
            clock.advance(1000)

    world = World(rewrites=(PLAIN,), reflections=(SECOND, SHORT_SECOND), task=tagged, hook=hook)
    outcome = run(tmp_path, world, TWO_K1, clock=clock, deadline=1000).outcome
    expected = (PROMPT, "clock") if cut else (SHORT_SECOND, None)
    assert (outcome.prompt, outcome.stop) == expected


@pytest.mark.parametrize("noisy", [False, True])
def test_the_second_generation_must_clear_the_first_ones_bar(tmp_path, noisy):
    """The original's first run is good on situation 1 when noisy: baseline 0.5 and 0, noise 0.5,
    bar 1.0; the reflection is good on both, a gain of 0.75, which clears 0.1 but not 1.0."""

    def task(call: Call) -> str:
        text = prompt_of(call)
        lucky = noisy and text == PROMPT and call.sample == 0 and "situation 1" in call.user
        return "GOOD answer" if MARKER in text or lucky else "BAD answer"

    world = World(rewrites=(PLAIN,), reflections=(SECOND,), task=task)
    outcome = run(tmp_path, world, TWO_K1).outcome
    assert (outcome.prompt, outcome.noise) == ((PROMPT, 0.5) if noisy else (SECOND, 0.0))


def test_the_checked_tier_confirms_a_second_generation_winner_on_the_target(tmp_path):
    two = dataclasses.replace(CHECKED, generations=2, rewrites2=1)
    result = run(tmp_path, World(rewrites=(PLAIN,), reflections=(SECOND,)), two)
    assert (result.outcome.prompt, result.outcome.verified) == (SECOND, True)
    on_target = {prompt_of(c) for c in result.calls("task") if c.model == MODELS.target}
    assert on_target == {PROMPT, SECOND}


@pytest.mark.parametrize("left", [60, 80])
def test_the_checked_tier_keeps_the_time_of_its_held_out_check_from_the_second_generation(
    tmp_path, left
):
    """At workers 1 the second generation of one reflection needs Rs + 2 T + 2 J2 = 21.4 s and
    stage E 8 T + 2 J4 = 49.7 s, 71.1 s in all: with 60 s left after the first generation's judge
    calls, E checks the first's winner; with 80 s, the second's."""
    clock, once = FakeClock(), []

    def hook(call: Call) -> None:
        if call.role == "judge" and '"output": "GOOD answer"' in call.user and not once:
            once.append(call)
            clock.advance(1000 - left)

    two = dataclasses.replace(CHECKED, generations=2, rewrites2=1)
    world = World(rewrites=(BETTER,), reflections=(SHORT_SECOND,), hook=hook)
    outcome = run(tmp_path, world, two, clock=clock, deadline=1000).outcome
    winner = BETTER if left == 60 else SHORT_SECOND
    assert (outcome.prompt, outcome.verified) == (winner, True)


def test_a_scenario_a_candidate_did_not_complete_is_left_out_of_what_the_reflection_reads(tmp_path):
    def task(call: Call) -> str:
        if prompt_of(call) == PROMPT and call.sample == 0 and "situation 2" in call.user:
            raise CallError("down")
        return "GOOD answer" if MARKER in prompt_of(call) else "BAD answer"

    result = run(tmp_path, World(rewrites=(PLAIN,), reflections=(SECOND,), task=task), TWO_K1)
    [parent] = candidates_of(reflections(result)[0])
    assert [s["input"] for s in parent["scenarios"]] == ["situation 1"]


@pytest.mark.parametrize("late", [False, True])
def test_no_second_task_run_when_the_reflections_leave_no_time_for_it(tmp_path, late):
    """A reflection that overruns leaves 20 s; the second generation's runs and judge calls
    need 4 T + 3 J2 = 31.8 s at workers 1."""
    clock = FakeClock()

    def hook(call: Call) -> None:
        if late and call.role == "reflect" and call.sample == 101:
            clock.advance(980)

    world = World(rewrites=(BETTER,), reflections=(SHORT_SECOND,), hook=hook)
    result = run(tmp_path, world, TWO_K1, clock=clock, deadline=1000)
    ran = SHORT_SECOND in [prompt_of(c) for c in result.calls("task")]
    assert (ran, result.outcome.prompt) == ((False, BETTER) if late else (True, SHORT_SECOND))


@pytest.mark.parametrize("keeps", [False, True])
def test_a_second_generation_rewrite_must_keep_the_contract(tmp_path, keeps):
    world = World(
        rewrites=(PLAIN,),
        reflections=(SECOND,),
        contract_ok=lambda text: keeps or text != SECOND,
    )
    assert (run(tmp_path, world, TWO_K1).outcome.prompt == SECOND) is keeps
