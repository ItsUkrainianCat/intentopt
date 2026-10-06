"""Robust fast runs (SPEC R17, R25; ADR-008, ADR-011), from the live bench of 2026-10-06.

A rewrite reply whose delimiter lines are the bare `<<<` and `>>>` (or carry a space before or
after INSTRUCTION) still holds its prompt: the long-2 run lost its only rewrite to that shape. Every
rejection of `parse_rewrite` stays.

A stage B that overruns no longer ends the run out of time: its calls end by the deadline less the
cheapest stage C (ADR-011 decision 2: a run that cannot finish by its stage's deadline is
abandoned), and the scenarios every run answered in time go on to stage C, provided the time left
covers one pairwise call; the hard deadline is never passed. The long-1 run lost stage C so.
Estimates on the mechanics model of `test_fast_world.py` (2.4 s per call): J1 = 3.471 s, P1 =
2.971 s.
"""

import io
import math
from collections.abc import Callable

import pytest
from fakes import FakeClock, ScriptedBackend
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2,
    PLAN,
    PROMPT,
    Verbatim,
    World,
    is_pairwise,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
)

from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.fast import improve_fast
from autoimprover.fast_prompts import parse_rewrite
from autoimprover.runstore import RunStore
from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END, Call, CallError, Reply

# --- the delimiter lines of a rewrite reply -------------------------------------------------------

# The long-2 reply of the live bench (its cache entry, shortened): bare delimiters, no INSTRUCTION.
LONG2_PROMPT = (
    "We are writing the design document for a new feature in our habit-tracking app: shared "
    "habits, where two or more friends track the same habit and see each other's progress.\n\n"
    "Write the section of the design document that covers the data model and the privacy rules. "
    "Keep it to what a backend engineer needs to start building."
)
LONG2_REPLY = f"<<<\n{LONG2_PROMPT}\n>>>"


def test_the_live_reply_with_bare_delimiters_holds_its_prompt():
    assert parse_rewrite(LONG2_REPLY) == LONG2_PROMPT


@pytest.mark.parametrize(
    ("begin", "end"),
    [
        ("<<<", ">>>"),
        ("<<< INSTRUCTION", "INSTRUCTION >>>"),
        ("  <<<  ", "\t>>> "),
        (INSTRUCTION_BEGIN, ">>>"),
        ("<<<", INSTRUCTION_END),
        ("<<< INSTRUCTION", INSTRUCTION_END),
    ],
)
def test_bare_or_spaced_delimiter_lines_are_delimiter_lines(begin, end):
    reply = f"Here it is:\n{begin}\n\nBe brief.\nUse `code`.\n\n{end}\nDone."
    assert parse_rewrite(reply) == "Be brief.\nUse `code`."


@pytest.mark.parametrize(
    "reply", ["<<<\n>>>", "<<<\n  \n\t\n>>>", "<<< INSTRUCTION\n\nINSTRUCTION >>>"]
)
def test_bare_delimiters_around_nothing_are_an_empty_prompt(reply):
    with pytest.raises(ValueError, match="empty"):
        parse_rewrite(reply)


@pytest.mark.parametrize(
    "reply",
    [
        ">>>\nBe brief.\n<<<",
        "<<<\nBe brief.",
        "Be brief.\n>>>",
        "<<< Be brief. >>>",
        "<<<< \nBe brief.\n>>>>",
        "<<< instruction\nBe brief.\ninstruction >>>",
    ],
)
def test_lines_that_are_not_delimiter_lines_are_still_refused(reply):
    """Guards: refused before WP17 and after (end before begin, one side only, inline, other
    marks)."""
    with pytest.raises(ValueError, match="delimiter"):
        parse_rewrite(reply)


@pytest.mark.parametrize("bad", ["<curr_param>", "<side_info>", "\0"])
def test_a_bare_delimited_prompt_with_a_template_token_or_a_nul_is_refused(bad):
    with pytest.raises(ValueError, match="token|NUL"):
        parse_rewrite(f"<<<\nBe {bad} brief.\n>>>")


def test_a_rewrite_with_bare_delimiters_is_a_candidate_of_the_run(tmp_path):
    """The run of long-2 kept the original because its one rewrite was dropped."""
    result = run(tmp_path, World(rewrites=(Verbatim(f"<<<\n{BETTER}\n>>>"),)), K1M2)
    assert (result.outcome.prompt, result.outcome.reason_code) == (BETTER, "improved")
    assert "dropped" not in result.log


# --- a stage B that overruns ----------------------------------------------------------------------


class Timed:
    """The raw model timed as `claude_cli` times it (SPEC R17): a call takes `seconds(call)` on the
    fake clock, and one that would end after the current deadline is stopped there, `stop_s` later
    (the child must end), and fails; its retry then finds the deadline passed."""

    def __init__(self, world: World, clock: FakeClock, seconds: Callable[[Call], float], stop_s):
        self.raw, self.clock, self.seconds, self.stop_s = (
            ScriptedBackend(world),
            clock,
            seconds,
            stop_s,
        )
        self.deadline: Callable[[], float] = lambda: math.inf

    def complete(self, call: Call) -> Reply:
        end, deadline = self.clock.t + self.seconds(call), self.deadline()
        if end > deadline:
            self.clock.t = deadline + self.stop_s
            raise CallError("claude did not answer before the deadline and was stopped")
        self.clock.t = end
        return self.raw.complete(call)


def timed_run(root, fplan, seconds, deadline: float, stop_s: float = 0.0):
    """improve_fast on one worker over the real stack, the raw layer `Timed` by `seconds`, with
    `deadline` on a fake clock; the outcome, the raw calls and the clock at the end."""
    store = RunStore.open_or_create(root, PLAN, PROMPT)
    try:
        fake = FakeClock()
        raw = Timed(World(), fake, seconds, stop_s)
        clock = Clock(now=fake.now)
        budgeted = BudgetedBackend(raw, PLAN.budget, 0, clock, deadline)
        raw.deadline = lambda: budgeted.deadline
        outcome = improve_fast(
            PROMPT,
            PLAN,
            fplan,
            backend=CachedBackend(ResilientBackend(budgeted), store),
            budgeted=budgeted,
            clock=clock,
            store=store,
            scenarios=None,
            kind=None,
            log=io.StringIO(),
            workers=1,
        )
        return outcome, raw.raw.calls, fake.t
    finally:
        store.close()


def slow_second_run(call: Call) -> float:
    """1 s a call; the rewrite's run on the second scenario writes a whole document: 100 s."""
    slow = call.role == "task" and prompt_of(call) == BETTER and scenario_of(call) == "situation 2"
    return 100.0 if slow else 1.0


def pairwise_scenarios(calls: list[Call]) -> set[tuple[str, ...]]:
    return {tuple(judged_scenarios(c)) for c in calls if is_pairwise(c)}


# Stage B's calls end by the deadline less the cheapest stage C, here 5 J1 = 17.357 s on one
# worker (and stage E in the checked tier, 49.714 s), so its calls end at 42.643 s of 60 (32.929 s
# of 100 in the checked tier). The slow run is stopped there, 0.2 s later.
@pytest.mark.parametrize(
    ("fplan", "deadline", "verified"), [(K1M2, 60, False), (CHECKED, 100, True)]
)
def test_a_stage_b_cut_at_its_deadline_still_judges_what_every_run_answered(
    tmp_path, fplan, deadline, verified
):
    """The original's two runs and the rewrite's answered the first scenario in time: stage C runs
    on it, inside the deadline (SPEC R17, R25; ADR-011 decision 2)."""
    outcome, calls, end = timed_run(tmp_path, fplan, slow_second_run, deadline, stop_s=0.2)
    assert (outcome.prompt, outcome.reason_code, outcome.stop) == (BETTER, "improved", "clock")
    assert outcome.verified is verified
    assert pairwise_scenarios(calls) == {("s1",)}
    assert end <= deadline


def test_a_stage_b_that_ends_in_time_judges_every_scenario(tmp_path):
    """The twin: no run is slow, so stage B's own deadline never cuts it."""
    outcome, calls, end = timed_run(tmp_path, K1M2, lambda _call: 1.0, 60)
    assert (outcome.prompt, outcome.reason_code, outcome.stop) == (BETTER, "improved", None)
    assert pairwise_scenarios(calls) == {("s1", "s2")}
    assert end <= 60


def test_a_stage_b_cut_with_no_time_for_one_pairwise_call_keeps_the_original(tmp_path):
    """Guard: the slow run takes 15 s to stop, so 2.357 s are left, under one pairwise call (P1 =
    2.971 s): no judge call is made and the original is kept, out of time."""
    outcome, calls, end = timed_run(tmp_path, K1M2, slow_second_run, 60, stop_s=15.0)
    assert (outcome.prompt, outcome.reason_code, outcome.stop) == (
        PROMPT,
        "unconfirmed_out_of_budget",
        "clock",
    )
    assert [c for c in calls if c.role == "judge"] == []
    assert end <= 60
