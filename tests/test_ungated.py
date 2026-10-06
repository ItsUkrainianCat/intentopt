"""`--ungated`, the measuring mode of the fast and checked picks (SPEC R25, R26; WP20): stages A to
C and the second generation run as always, and a rewrite that wins is returned as always; when no
rewrite wins at the pick, the best-ranked rewrite that passed the free gates and the contract check
is returned, ranked by its lead in scenarios (wins - losses), then its wins, then fewer tokens,
without the noise rule, the rule of one win and stage E. It is labelled, never verified, and a run
without the flag is unchanged. Every test that shows a rewrite is NOT returned has a twin showing
one IS. The command line, the run folder and the bench are in `test_ungated_cli.py`."""

import dataclasses

import pytest
from fakes import FakeClock
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2,
    K2M2,
    MODELS,
    PROMPT,
    TWO_K1,
    World,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
)

from autoimprover.fast_pairwise import Judged, Preference, Rewrite, Win, best_ungated
from autoimprover.types import Call

PLAIN = "Answer the request well."  # no marker: its answers tie the original's
SECOND = "Answer the user's request in plain words."
LABEL = "ungated: the best-ranked candidate; no win over the original was shown"
LONG = "Answer the user's request in full, step by step."
SHORT = "Answer it."


def judged(text: str, wins: int, ties: int, losses: int, keep=True, variant=0) -> Judged:
    return Judged(Rewrite(variant, text, 1.0, "a note"), Preference(wins, ties, losses, {}), keep)


def original_wins(call: Call) -> str:
    """The original answers well, every rewrite badly (SECOND as well as the original)."""
    return "GOOD answer" if prompt_of(call) in (PROMPT, SECOND) else "BAD answer"


# --- the ranking (pure) -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        # the lead first: 1 - 0 above 2 - 2, though that one has more wins and fewer tokens
        ((judged(SHORT, 2, 0, 2), judged(LONG, 1, 3, 0)), LONG),
        # then the wins: 1 - 1 above 0 - 0 at the same lead, though longer
        ((judged(SHORT, 0, 4, 0), judged(LONG, 1, 2, 1)), LONG),
        # then fewer tokens
        ((judged(LONG, 1, 2, 1), judged(SHORT, 1, 2, 1)), SHORT),
    ],
)
def test_the_ungated_pick_ranks_by_lead_then_wins_then_fewer_tokens(candidates, expected):
    best = best_ungated(list(candidates), 4, 4)
    assert best is not None and best.rewrite.text == expected


def test_at_an_equal_rank_the_earlier_rewrite_is_picked():
    first, second = judged("Answer it now.", 0, 4, 0), judged("Now answer it.", 0, 4, 0, variant=1)
    for order in ([first, second], [second, first]):
        best = best_ungated(order, 0, 4)
        assert best is not None and best.rewrite == order[0].rewrite


def test_a_vetoed_rewrite_is_never_the_ungated_pick_however_it_ranks():
    best = best_ungated([judged(LONG, 4, 0, 0, keep=False), judged(SHORT, 0, 0, 4)], 4, 4)
    assert best is not None and best.rewrite.text == SHORT
    assert best_ungated([judged(LONG, 4, 0, 0, keep=False)], 4, 4) is None
    assert best_ungated([], 0, 4) is None


def test_the_ungated_pick_carries_its_shares_and_says_it_is_not_gated():
    best = best_ungated([judged(SHORT, 1, 1, 2)], 1, 4)
    assert best is not None
    assert best == Win(best.rewrite, 0.5, 0.25, -0.25, 0.25, gated=False)
    assert Win(best.rewrite, 0.5, 0.25, -0.25, 0.25).gated  # a win is gated unless it says not


# --- the pipeline ---------------------------------------------------------------------------------


@pytest.mark.parametrize("ungated", [False, True])
def test_with_no_winner_an_ungated_run_returns_the_best_ranked_rewrite_labelled(tmp_path, ungated):
    outcome = run(tmp_path, World(rewrites=(PLAIN,)), K1M2, ungated=ungated).outcome
    if not ungated:
        assert (outcome.status, outcome.prompt) == ("unchanged", PROMPT)
        assert outcome.reason_code == "no_reliable_improvement"
        return
    assert (outcome.status, outcome.prompt) == ("improved", PLAIN)
    assert (outcome.reason_code, outcome.reason) == (
        "ungated_best_candidate",
        f"tier fast: {LABEL}",
    )
    assert (outcome.verified, outcome.mode, outcome.stop) == (False, "fast", None)
    assert (outcome.search_score_before, outcome.search_score_after) == (0.0, 0.0)
    assert (outcome.score_before, outcome.score_after) == (None, None)
    assert (outcome.noise, outcome.margin) == (0.0, 0.0)
    assert outcome.changes and outcome.length_ratio is not None


def test_a_rewrite_that_loses_every_scenario_is_still_the_ungated_pick(tmp_path):
    world = World(rewrites=(PLAIN,), task=original_wins)
    outcome = run(tmp_path, world, K1M2, ungated=True).outcome
    assert (outcome.prompt, outcome.reason_code) == (PLAIN, "ungated_best_candidate")
    assert (outcome.search_score_before, outcome.search_score_after) == (1.0, 0.0)
    assert outcome.margin == -1.0


@pytest.mark.parametrize("vetoed", [False, True])
def test_a_rewrite_that_broke_the_contract_is_never_returned_ungated(tmp_path, vetoed):
    world = World(rewrites=(PLAIN,), contract_ok=lambda _text: not vetoed)
    outcome = run(tmp_path, world, K1M2, ungated=True).outcome
    assert outcome.prompt == (PROMPT if vetoed else PLAIN)
    assert outcome.reason_code == (
        "no_reliable_improvement" if vetoed else "ungated_best_candidate"
    )


def test_the_best_ranked_rewrite_that_kept_the_contract_is_returned_over_a_vetoed_winner(
    tmp_path,
):
    world = World(rewrites=(BETTER, PLAIN), contract_ok=lambda text: text != BETTER)
    outcome = run(tmp_path, world, K2M2, ungated=True).outcome
    assert (outcome.prompt, outcome.reason_code) == (PLAIN, "ungated_best_candidate")


@pytest.mark.parametrize(
    ("rewrite", "returned"),
    [
        ("Answer the user's request!", False),  # no change in meaning words
        (PROMPT + " Be thorough." * 40, False),  # over the length cap
        (PLAIN, True),
    ],
)
def test_a_rewrite_that_failed_a_free_gate_is_never_returned_ungated(tmp_path, rewrite, returned):
    outcome = run(tmp_path, World(rewrites=(rewrite,)), K1M2, ungated=True).outcome
    assert outcome.prompt == (rewrite if returned else PROMPT)


@pytest.mark.parametrize(("rewrite", "won"), [(PLAIN, False), (BETTER, True)])
def test_an_ungated_checked_run_makes_no_held_out_call_unless_a_rewrite_won(tmp_path, rewrite, won):
    """A winner goes to stage E as without the flag; the ungated pick skips it."""
    result = run(tmp_path, World(rewrites=(rewrite,)), CHECKED, ungated=True)
    outcome = result.outcome
    on_target = [call for call in result.calls("task") if call.model == MODELS.target]
    assert (outcome.prompt, outcome.verified, outcome.mode) == (rewrite, won, "checked")
    assert outcome.reason_code == ("improved" if won else "ungated_best_candidate")
    assert bool(on_target) is won
    if not won:
        assert outcome.reason == f"tier checked: {LABEL}"
        assert (outcome.score_before, outcome.score_after) == (None, None)


def test_a_rewrite_that_wins_is_returned_as_without_the_flag(tmp_path):
    (tmp_path / "gated").mkdir()
    (tmp_path / "ungated").mkdir()
    gated = run(tmp_path / "gated", World(), K1M2).outcome
    ungated = run(tmp_path / "ungated", World(), K1M2, ungated=True).outcome
    assert gated.reason_code == "improved" and gated.prompt == BETTER
    assert dataclasses.replace(ungated, run_dir="") == dataclasses.replace(gated, run_dir="")


def test_the_ungated_pick_ranks_the_second_generation_too(tmp_path):
    """PLAIN loses both scenarios; the reflection SECOND ties both, so it ranks first."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    world = World(rewrites=(PLAIN,), reflections=(SECOND,), task=original_wins)
    assert run(tmp_path / "one", world, K1M2, ungated=True).outcome.prompt == PLAIN
    outcome = run(tmp_path / "two", world, TWO_K1, ungated=True).outcome
    assert (outcome.prompt, outcome.reason_code) == (SECOND, "ungated_best_candidate")


@pytest.mark.parametrize("cut", [False, True])
def test_a_second_generation_that_was_cut_is_not_ranked(tmp_path, cut):
    """SECOND's first task run passes the deadline, so its second is refused: the second
    generation was cut, and the ungated pick is the first generation's PLAIN."""
    clock = FakeClock()

    def hook(call: Call) -> None:
        if cut and call.role == "task" and prompt_of(call) == SECOND:
            clock.advance(1000)

    world = World(rewrites=(PLAIN,), reflections=(SECOND,), task=original_wins, hook=hook)
    outcome = run(tmp_path, world, TWO_K1, clock=clock, deadline=1000, ungated=True).outcome
    assert (outcome.prompt, outcome.stop) == ((PLAIN, "clock") if cut else (SECOND, None))
    assert outcome.reason_code == "ungated_best_candidate"
