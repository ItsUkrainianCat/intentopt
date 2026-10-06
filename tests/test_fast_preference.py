"""Stage C of the fast and checked tiers decides by pairwise preference (SPEC R25 "Decision by
pairwise preference"; ADR-012): per rewrite two batched judge calls against the original's run 0
answers, two for the original's run 0 against its run 1 (the noise), and one contract check of
every rewrite, in one wave. A rewrite is returned only when it kept the contract and its lead in
scenarios won over lost is above the noise; its scores are the win rates of the two sides, its
noise the share of scenarios the original's runs disagree on. Every "not returned" test has a
twin that is.
"""

import json

import pytest
from fakes import MARKER, FakeClock
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    K1M2,
    PROMPT,
    TWO_K1,
    World,
    is_contract_check,
    is_pairwise,
    mechanics_latency_model,
    no_disk_flush,
    position_a,
    prompt_of,
    run,
    tagged,
)

from autoimprover.types import BackendError, BudgetExhausted, Call, CallError

HALF = f"Answer {MARKER} briefly."  # good on situation 1 only (see `half_good`)


def half_good(call: Call) -> str:
    text = prompt_of(call)
    good = MARKER in text and (text != HALF or "situation 1" in call.user)
    return "GOOD answer" if good else "BAD answer"


def pairwise_requests(result) -> list[dict]:
    return [json.loads(c.user) for c in result.calls("judge") if is_pairwise(c)]


@pytest.mark.parametrize("judge", ["position", "content"])
def test_a_judge_that_prefers_a_position_never_makes_a_win(tmp_path, judge):
    world = World(pairwise=position_a) if judge == "position" else World()
    outcome = run(tmp_path, world, K1M2).outcome
    assert outcome.prompt == (PROMPT if judge == "position" else BETTER)


def by_run(call: Call) -> str:
    """As the world's default, with the run's sample in the answer, so no two pairwise calls of
    the run are the same call (the cache would answer the second)."""
    return f"{'GOOD' if MARKER in prompt_of(call) else 'BAD'} answer {call.sample}"


def test_the_judge_sees_the_original_prompt_as_the_request_and_anonymous_answers(tmp_path):
    result = run(tmp_path, World(task=by_run), K1M2)
    requests = pairwise_requests(result)
    assert len(requests) == 2 * 1 + 2  # two orders for the rewrite, two for the noise pair
    assert {r["request"] for r in requests} == {PROMPT}
    assert not any(BETTER in json.dumps(r) for r in requests)
    assert all([s["scenario"] for s in r["scenarios"]] == ["s1", "s2"] for r in requests)


def test_a_returned_rewrite_reports_win_rates_noise_and_margin(tmp_path):
    outcome = run(tmp_path, World(), K1M2).outcome
    assert (outcome.score_before, outcome.score_after, outcome.noise) == (0.0, 1.0, 0.0)
    assert outcome.margin == pytest.approx(1.0)  # (2 wins - 0 losses) / 2 - 0


@pytest.mark.parametrize(("rewrite", "returned"), [(HALF, False), (BETTER, True)])
def test_with_noise_on_one_scenario_a_rewrite_needs_a_lead_of_two(tmp_path, rewrite, returned):
    """The original's run 1 is good on situation 1, its run 0 is not: noise 1 of 2. BETTER wins
    both scenarios against run 0 (2 - 0 > 1); HALF wins one (1 - 0, not more than 1)."""

    def task(call: Call) -> str:
        lucky = prompt_of(call) == PROMPT and call.sample == 1 and "situation 1" in call.user
        return "GOOD answer" if lucky else half_good(call)

    outcome = run(tmp_path, World(rewrites=(rewrite,), task=task), K1M2).outcome
    assert outcome.noise == 0.5
    assert outcome.prompt == (rewrite if returned else PROMPT)


@pytest.mark.parametrize(("task", "returned"), [("noisy", False), ("calm", True)])
def test_a_lead_of_one_wins_only_without_noise(tmp_path, task, returned):
    def noisy(call: Call) -> str:
        lucky = prompt_of(call) == PROMPT and call.sample == 1 and "situation 2" in call.user
        return "GOOD answer" if lucky else half_good(call)

    world = World(rewrites=(HALF,), task=noisy if task == "noisy" else half_good)
    assert (run(tmp_path, world, K1M2).outcome.prompt == HALF) is returned


@pytest.mark.parametrize("keeps", [False, True])
def test_a_rewrite_preferred_in_every_scenario_that_breaks_the_contract_is_never_returned(
    tmp_path, keeps
):
    world = World(contract_ok=lambda text: keeps or text != BETTER)
    outcome = run(tmp_path, world, K1M2).outcome
    assert outcome.prompt == (BETTER if keeps else PROMPT)
    assert outcome.reason_code == ("improved" if keeps else "no_reliable_improvement")


@pytest.mark.parametrize(
    ("rewrite", "returned"), [("Answer the request well.", False), (BETTER, True)]
)
def test_ties_leave_the_original(tmp_path, rewrite, returned):
    outcome = run(tmp_path, World(rewrites=(rewrite,)), K1M2).outcome
    assert (outcome.prompt == rewrite) is returned and (outcome.prompt == PROMPT) is not returned


def test_one_contract_check_holds_every_rewrite_and_no_absolute_judge_call_is_made(tmp_path):
    result = run(tmp_path, World(), K1M2)
    judges = result.calls("judge")
    assert sum(is_contract_check(c) for c in judges) == 1
    assert all(is_pairwise(c) or is_contract_check(c) for c in judges)


def test_the_reflection_reads_the_judges_reasons_for_the_scenarios_a_rewrite_lost_or_tied(
    tmp_path,
):
    """HALF wins situation 1 and ties situation 2: the reflection reads the tie's reasons."""
    world = World(rewrites=(HALF,), reflections=(BETTER,), task=half_good)
    result = run(tmp_path, world, TWO_K1)
    reflection = next(c for c in result.calls("reflect") if c.sample >= 100)
    [parent] = json.loads(reflection.user)["candidates"]
    assert parent["prompt"] == HALF
    assert parent["scenarios"] == [
        {"input": "situation 2", "verdict": "tie", "reasons": ["tie for s2"]}
    ]
    assert result.outcome.prompt == BETTER  # the second generation's rewrite wins 2 - 0


@pytest.mark.parametrize("cut", [False, True])
def test_a_contract_check_the_clock_cut_vetoes_every_rewrite(tmp_path, cut):
    """The pairwise calls finish and prefer the rewrite, but no contract check answered: no
    rewrite is returned unvetted (SPEC R6, R17)."""

    def hook(call: Call) -> None:
        if cut and is_contract_check(call):
            raise BudgetExhausted("the deadline passed", "clock")

    outcome = run(tmp_path, World(hook=hook), K1M2).outcome
    assert (outcome.prompt, outcome.stop) == ((PROMPT, "clock") if cut else (BETTER, None))


SECOND = f"Answer the user's request {MARKER} now."
SHORT_SECOND = f"Answer {MARKER}."


@pytest.mark.parametrize("cut", [False, True])
def test_a_second_generation_cut_after_one_reflection_was_judged_leaves_the_first_ones_result(
    tmp_path, cut
):
    """SECOND's two orders finish before the clock passes the deadline, then SHORT_SECOND's are
    refused: the second generation was cut, so its judged rewrite does not count either."""
    clock, once = FakeClock(), []
    second_order = f"GOOD answer {len(SECOND)}"  # SECOND's answers shown as A: its second order

    def hook(call: Call) -> None:
        shown = json.loads(call.user)["scenarios"][0]["answer_A"] if is_pairwise(call) else ""
        if cut and shown == second_order and not once:
            once.append(call)
            clock.advance(1000)

    world = World(
        rewrites=("Answer the request well.",),
        reflections=(SECOND, SHORT_SECOND),
        task=tagged,
        hook=hook,
    )
    outcome = run(tmp_path, world, TWO_K1, clock=clock, deadline=1000).outcome
    assert (outcome.prompt, outcome.stop) == ((PROMPT, "clock") if cut else (SHORT_SECOND, None))


@pytest.mark.parametrize("disjoint", [False, True])
def test_original_runs_that_answered_no_scenario_in_common_end_the_run(tmp_path, disjoint):
    """Run 0 fails situation 2 and run 1 situation 1: there is no noise pair to judge."""

    def task(call: Call) -> str:
        lost = {0: "situation 2", 1: "situation 1"}[call.sample]
        if disjoint and prompt_of(call) == PROMPT and lost in call.user:
            raise CallError("down")
        return half_good(call)

    if not disjoint:
        assert run(tmp_path, World(task=task), K1M2).outcome.prompt == BETTER
        return
    with pytest.raises(BackendError, match="left no scenario both answered"):
        run(tmp_path, World(task=task), K1M2)
