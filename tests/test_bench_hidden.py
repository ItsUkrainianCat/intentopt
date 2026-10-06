"""The hidden examples of a bench item (SPEC R26, with R10, R11, R14; WP21): the original and the
returned prompt (and with the baseline the naive rewrite) run on each hidden input on the target
model under a fresh sample (7000 and up), and one batched judge call per prompt (per 6 examples)
checks agreement with the references; an example passes when every one of its checks passes. Wins
are the hidden examples the returned prompt passes and the original fails, losses the reverse; an
unchanged result is a tie."""

import json

import pytest
from fakes import MARKER, ScriptedBackend, judge_reply
from test_bench_world import BETTER, NAIVE, ORIGINAL

from autoimprover.bench_hidden import (
    Hidden,
    Passed,
    hidden_calls,
    hidden_seconds,
    score_hidden,
    versus,
)
from autoimprover.bench_judge import BENCH_SAMPLE, Comparison
from autoimprover.fastplan import reference_seconds
from autoimprover.types import (
    INSTRUCTION_BEGIN,
    INSTRUCTION_END,
    Call,
    CallFailed,
    Models,
    Scenario,
)

MODELS = Models(
    task="claude-haiku-4-5-20251001",
    judge="claude-opus-5-5",
    reflect="claude-sonnet-5-5",
    target="claude-fable-5-1",
)
HIDDEN = tuple(Scenario(f"e{n}", f"ticket {n}", expected=f"P{n % 4 + 1}") for n in range(9, 17))


def world(good, naive: str = NAIVE):
    """GOOD when `good(prompt, example input)`; the judge passes a check of a GOOD output only."""

    def answer(call: Call) -> str | Exception:
        if call.role == "reflect":
            return f"{INSTRUCTION_BEGIN}\n{naive}\n{INSTRUCTION_END}"
        if call.role == "task":
            situation, prompt = call.user.split("\n\n", 1)
            return f"{'GOOD' if good(prompt, situation) else 'BAD'} {situation}"
        return judge_reply(call, lambda _s, _c, output: output.startswith("GOOD"))

    return ScriptedBackend(answer)


def marked(prompt: str, _situation: str) -> bool:
    return MARKER in prompt


def score(raw, candidate, naive=False, hidden=HIDDEN, seed=0):
    return score_hidden(
        raw,
        ORIGINAL,
        candidate,
        naive=naive,
        kind="task",
        models=MODELS,
        workers=1,
        seed=seed,
        hidden=hidden,
    )


def test_the_returned_prompt_wins_the_examples_it_passes_and_the_original_fails():
    raw = world(marked)
    judged, hidden = score(raw, BETTER)
    assert judged.tool == Comparison("win", 8, 0, 0) and judged.naive is None
    assert hidden == Hidden(original=Passed(0, 8), returned=Passed(8, 8), naive=None)
    tasks = [call for call in raw.calls if call.role == "task"]
    assert {(call.model, call.sample) for call in tasks} == {(MODELS.target, BENCH_SAMPLE)}
    assert sorted(call.user.split("\n\n")[0] for call in tasks) == sorted(
        [s.input for s in HIDDEN] * 2
    )
    judges = [call for call in raw.calls if call.role == "judge"]
    assert len(judges) == 2 * 2  # two prompts, 8 examples in batches of at most 6
    assert {call.model for call in judges} == {MODELS.judge}
    for call in judges:
        for item in json.loads(call.user)["scenarios"]:
            assert [check["id"] for check in item["checks"]] == ["s:expected"]


@pytest.mark.parametrize(
    ("original_passes", "returned_passes", "verdict"),
    [
        ({"ticket 9"}, {"ticket 10"}, ("tie", 1, 6, 1)),
        ({"ticket 9", "ticket 10"}, {"ticket 9"}, ("loss", 0, 7, 1)),
        (set(), {"ticket 9"}, ("win", 1, 7, 0)),
    ],
)
def test_wins_and_losses_by_example(original_passes, returned_passes, verdict):
    def good(prompt: str, situation: str) -> bool:
        return situation in (returned_passes if MARKER in prompt else original_passes)

    judged, hidden = score(world(good), BETTER)
    assert judged.tool == Comparison(*verdict)
    assert hidden.original == Passed(len(original_passes), 8)
    assert hidden.returned == Passed(len(returned_passes), 8)


def test_an_unchanged_result_is_a_tie_with_the_originals_counts():
    raw = world(marked)
    judged, hidden = score(raw, None)
    assert judged.tool == Comparison("tie", 0, 8, 0)
    assert hidden == Hidden(original=Passed(0, 8), returned=Passed(0, 8), naive=None)
    assert {call.user.split("\n\n", 1)[1] for call in raw.calls if call.role == "task"} == {
        ORIGINAL
    }


def test_the_naive_baseline_is_scored_on_the_same_examples():
    naive_better = f"{NAIVE} {MARKER}"
    judged, hidden = score(world(marked, naive=naive_better), BETTER, naive=True)
    assert judged.naive == Comparison("win", 8, 0, 0)
    assert hidden.naive == Passed(8, 8)


def test_a_failed_task_run_leaves_its_example_out_of_both_counts_of_that_prompt():
    def answer(call: Call) -> str | Exception:
        if call.role == "task" and call.user.startswith("ticket 9\n\n") and MARKER in call.user:
            return CallFailed("down")  # as the Resilient layer gives it
        return world(marked).complete(call).text

    judged, hidden = score(ScriptedBackend(answer), BETTER)
    assert hidden.returned == Passed(7, 7) and hidden.original == Passed(0, 8)
    assert judged.tool == Comparison("win", 7, 0, 0)


def test_versus_compares_only_the_examples_both_were_scored_on():
    assert versus({"a": False, "b": True}, {"a": True, "c": True}) == Comparison("win", 1, 0, 0)
    assert versus({"a": True}, {}).verdict == "error"


def test_a_fresh_sample_per_seed():
    raw = world(marked)
    score(raw, BETTER, seed=12)
    assert {call.sample for call in raw.calls} == {BENCH_SAMPLE + 12}


def test_the_calls_and_seconds_of_an_item():
    """10 hidden examples: per prompt 10 task runs and 2 judge calls (6 + 4)."""
    assert hidden_calls(True, False, 10) == 2 * (10 + 2)
    assert hidden_calls(True, True, 10) == 1 + 3 * (10 + 2)
    assert hidden_calls(False, False, 10) == 10 + 2
    seconds = hidden_seconds(True, False, 6, 20, 10, 1)
    plain = 3.4 + 500 / 70  # a plain answer on the target model (bench_judge.PLAIN_OUT_TOKENS)
    assert seconds == pytest.approx(4 * plain + reference_seconds(6, 1))
