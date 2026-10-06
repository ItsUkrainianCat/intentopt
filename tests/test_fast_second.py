"""The second generation of the fast tiers (SPEC R25 "Quality of the rewrites" and "Decision by
pairwise preference", R16; ADR-011 amendment of 2026-10-06, ADR-012): from 45 s, when the plan has
room, the reflection model reads the best one or two first-generation rewrites that kept the
contract, by their lead in scenarios (or the original when none did), with the pairwise judge's
reasons for the scenarios each lost or tied, and writes K2 rewrites under distinct notes; they pass
the same gates and are judged against the same answers of the original with the same noise, one
contract check for all of them, and the pick is over every candidate. When the second generation
does not fit or is cut, the first generation's result stands. Every "not returned" test has a twin
that is.
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
    is_pairwise,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    tagged,
)

from autoimprover.fast_prompts import REFLECT_NOTES, STRATEGY_NOTES
from autoimprover.runner import count_tokens
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


def test_when_no_rewrite_wins_the_reflection_reads_the_best_one_and_the_judges_reasons(tmp_path):
    """PLAIN ties the original everywhere: the reflection reads it with the reasons for each
    tie, from both orders (the world's judge says the same in both)."""
    result = run(tmp_path, World(rewrites=(PLAIN,), reflections=(SECOND,)), TWO_K1)
    calls = reflections(result)
    assert [c.sample for c in calls] == [100, 101] and len({c.system for c in calls}) == 2
    assert {c.model for c in calls} == {MODELS.reflect}
    [parent] = candidates_of(calls[0])
    assert parent == {
        "prompt": PLAIN,
        "scenarios": [
            {"input": "situation 1", "verdict": "tie", "reasons": ["tie for s1"]},
            {"input": "situation 2", "verdict": "tie", "reasons": ["tie for s2"]},
        ],
    }
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


def test_a_lead_counts_scenarios_so_completing_fewer_earns_nothing(tmp_path):
    """`gapped` fails on situation 2 and wins situation 1; `half` wins situation 1 and ties
    situation 2: both lead by one scenario, so the shorter is read first."""
    half, gapped = f"Answer {MARKER}.", f"Answer the user's request {MARKER} in full."

    def task(call: Call) -> str:
        text, first = prompt_of(call), "situation 1" in call.user
        if text == gapped and not first:
            raise CallError("down")
        return "GOOD answer" if MARKER in text and first else "BAD answer"

    world = World(rewrites=(half, gapped, PLAIN), reflections=(PLAIN,), task=task)
    calls = reflections(run(tmp_path, world, TWO_K3))
    assert [p["prompt"] for p in candidates_of(calls[0])] == [half, gapped]


@pytest.mark.parametrize("keeps", [False, True])
def test_a_rewrite_that_gained_but_broke_the_contract_is_not_what_the_reflection_reads(
    tmp_path, keeps
):
    world = World(
        rewrites=(BETTER,), reflections=(PLAIN,), contract_ok=lambda text: keeps or text != BETTER
    )
    [parent] = candidates_of(reflections(run(tmp_path, world, TWO_K1))[0])
    assert parent["prompt"] == (BETTER if keeps else PROMPT)
    if not keeps:  # the original has no reasons
        assert parent["scenarios"] == []


def test_a_rewrite_that_did_not_win_is_read_too_after_the_better_one(tmp_path):
    """Its reasons say what to fix; BETTER won everywhere, so it carries none."""
    result = run(tmp_path, World(rewrites=(BETTER, PLAIN), reflections=(PLAIN,)), TWO_K3)
    parents = candidates_of(reflections(result)[0])
    assert [(p["prompt"], len(p["scenarios"])) for p in parents] == [(BETTER, 0), (PLAIN, 2)]


@pytest.mark.parametrize(("second", "winner"), [(SHORT_SECOND, SHORT_SECOND), (PLAIN, BETTER)])
def test_the_pick_is_over_every_candidate_of_both_generations(tmp_path, second, winner):
    """A second-generation rewrite that ties with the first's on lead and is shorter wins; one
    that does not lead loses to it."""
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
    assert "fast: stage C2: a contract check and 4 pairwise judge calls" in result.log


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


# PROMPT has 7 tokens, so the cap is 7 + 40 = 47; NEAR has 39 tokens, at least three quarters of it.
NEAR = f"{BETTER} " + "Answer it in full, in plain words." * 3
OVER = f"{SECOND} " + "Say why, in plain words." * 5  # 48 tokens
WITHIN = f"{SECOND} Say why."


def test_a_reflection_over_the_cap_is_dropped_whole_and_one_within_it_runs(tmp_path):
    """The reflections read a best candidate near the cap, so they are asked for a shorter text;
    one that is longer than the cap anyway is never cut down to it: it is dropped, never run, and
    its twin within the cap runs and wins (the same lead as NEAR, and shorter)."""
    assert (count_tokens(PROMPT), count_tokens(NEAR), count_tokens(OVER)) == (7, 39, 48)
    result = run(tmp_path, World(rewrites=(NEAR,), reflections=(OVER, WITHIN)), TWO_K1)
    calls = reflections(result)
    assert [p["prompt"] for p in candidates_of(calls[0])] == [NEAR]
    assert all("already has 39 tokens" in c.system and "shorter than it" in c.system for c in calls)
    assert "reflection 0 dropped: it is longer than the length cap" in result.log
    ran = [prompt_of(c) for c in result.calls("task")]
    assert OVER not in ran and ran.count(WITHIN) == 2
    assert result.outcome.prompt == WITHIN


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
    """The rewrite's first pairwise call leaves 20 s; the second generation needs 2 Rs + 4 T + 5
    J2 = 47.4 s at workers 1."""
    clock, once = FakeClock(), []

    def hook(call: Call) -> None:
        if late and is_pairwise(call) and "GOOD answer" in call.user and not once:
            once.append(call)
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
    """The first pairwise call that sees a good answer is the second generation's first; the
    clock then reaches the deadline, so its other pairwise calls are refused (`tagged` outputs
    differ, so the cache cannot answer them) and its rewrites have ties only."""
    clock, once = FakeClock(), []

    def hook(call: Call) -> None:
        if cut and is_pairwise(call) and "GOOD" in call.user and not once:
            once.append(call)
            clock.advance(1000)

    world = World(rewrites=(PLAIN,), reflections=(SECOND, SHORT_SECOND), task=tagged, hook=hook)
    outcome = run(tmp_path, world, TWO_K1, clock=clock, deadline=1000).outcome
    expected = (PROMPT, "clock") if cut else (SHORT_SECOND, None)
    assert (outcome.prompt, outcome.stop) == expected


@pytest.mark.parametrize("noisy", [False, True])
def test_the_second_generation_must_clear_the_first_ones_bar(tmp_path, noisy):
    """The original's first run is good on situation 1 when noisy: its two runs disagree there,
    a noise of 1 of 2 scenarios; the reflection wins situation 2 and ties situation 1, a lead of
    1, which is more than no noise but not more than 1."""

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
    """At workers 1 the second generation of one reflection needs Rs + 2 T + 3 P2 = 23.0 s and
    stage E 8 T + 2 J4 = 49.7 s, 72.7 s in all: with 60 s left after the rewrite's first pairwise
    call, E checks the first's winner; with 80 s, the second's."""
    clock, once = FakeClock(), []

    def hook(call: Call) -> None:
        if is_pairwise(call) and "GOOD answer" in call.user and not once:
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
    need 4 T + 5 J2 = 40.9 s at workers 1."""
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
