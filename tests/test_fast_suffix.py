"""The concise scoring runs of the fast tiers (SPEC R25 "Scoring runs are concise"; ADR-011
amendment 2026-10-06, third live run): every task run of the original and of every rewrite, in
both generations and in the checked tier's held-out check, ends its user text with the same fixed
suffix asking for at most 120 words; the judge calls do not carry it and no returned prompt holds
it. The task-call builder itself is pinned in `test_fast_prompts.py`.
"""

import dataclasses

import pytest
from fakes import intake_reply
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    MODELS,
    PROMPT,
    TWO_K1,
    World,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
)

from autoimprover.fast_prompts import FAST_TASK_SUFFIX

PLAIN = "Answer the user's request plainly."  # a rewrite that runs no better
SHORT_BETTER = f"Answer {BETTER.split(' ', 4)[-1]}"  # the marker, in fewer words


@pytest.mark.parametrize("kind", ["task", "template"])
def test_every_scoring_run_of_both_generations_asks_for_120_words_and_no_judge_call_does(
    tmp_path, kind
):
    world = World(rewrites=(PLAIN,), reflections=(BETTER, SHORT_BETTER), intake=intake_reply(kind))
    result = run(tmp_path, world, TWO_K1)
    tasks = result.calls("task")
    assert len(tasks) == (1 + 2) * 2 + 2 * 2  # stage B: the original twice and 1 rewrite; B2: 2
    assert all(call.user.endswith(FAST_TASK_SUFFIX) for call in tasks)
    assert {prompt_of(call) for call in tasks} == {PROMPT, PLAIN, BETTER, SHORT_BETTER}
    assert not any("120 words" in call.user for call in result.calls("judge"))
    assert result.outcome.prompt == SHORT_BETTER  # never the suffix


def test_the_checked_tiers_held_out_runs_on_the_target_ask_for_120_words(tmp_path):
    two = dataclasses.replace(CHECKED, generations=2, rewrites2=1)
    result = run(tmp_path, World(rewrites=(PLAIN,), reflections=(BETTER,)), two)
    on_target = [call for call in result.calls("task") if call.model == MODELS.target]
    assert {prompt_of(call) for call in on_target} == {PROMPT, BETTER}
    assert all(call.user.endswith(FAST_TASK_SUFFIX) for call in on_target)
    assert (result.outcome.prompt, result.outcome.verified) == (BETTER, True)
