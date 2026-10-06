"""The blind pairwise judge of `autoimprover bench` (SPEC R26, R10a, R14, R18, R19): the original
and a candidate run with the plain task call on the target model on fresh scenarios, and the judge
model compares the two anonymous answers per scenario in both orders. A scenario counts only when
both orders agree, so a judge that prefers a position yields ties, never wins; a prompt wins when
it wins more scenarios than it loses. Every test that shows no win has a twin that shows one."""

import json

import pytest
from fakes import MARKER, ScriptedBackend
from test_bench_world import (
    BETTER,
    NAIVE,
    ORIGINAL,
    BenchWorld,
    answers,
    by_content,
    first_position,
    is_pairwise,
    winner,
)

from autoimprover.bench_judge import (
    BENCH_SAMPLE,
    NAIVE_SYSTEM,
    PAIRWISE_SCENARIOS,
    PAIRWISE_SYSTEM,
    Comparison,
    judge,
    naive_call,
    pairwise_call,
    pairwise_calls,
    pairwise_seconds,
    parse_winner,
    scenario_vote,
    verdict_of,
)
from autoimprover.types import (
    INSTRUCTION_BEGIN,
    INSTRUCTION_END,
    BackendError,
    BudgetExhausted,
    Call,
    CallFailed,
    Models,
)

MODELS = Models(
    task="claude-haiku-4-5-20251001",
    judge="claude-opus-5-5",
    reflect="claude-sonnet-5-5",
    target="claude-fable-5-1",
)
S = PAIRWISE_SCENARIOS


def run(world: BenchWorld, candidate: str | None = BETTER, *, naive: bool = False, **kw):
    raw = ScriptedBackend(world)
    args = {"kind": "task", "models": MODELS, "workers": 1, "seed": 0} | kw
    return judge(raw, ORIGINAL, candidate, naive=naive, **args), raw


def calls(raw: ScriptedBackend, role: str) -> list[Call]:
    return [call for call in raw.calls if call.role == role]


# --- one scenario, one prompt ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "vote"),
    [
        ("B", "A", "candidate"),  # both orders pick the candidate
        ("A", "B", "original"),  # both orders pick the original
        ("A", "A", "tie"),  # the first position both times: disagreement
        ("B", "B", "tie"),  # the second position both times
        ("tie", "tie", "tie"),
        ("B", "tie", "tie"),
        ("tie", "A", "tie"),
        (None, "A", None),  # a failed call leaves the scenario out
        ("B", None, None),
    ],
)
def test_a_scenario_counts_only_when_both_orders_agree(first, second, vote):
    assert scenario_vote(first, second) == vote


@pytest.mark.parametrize(
    ("votes", "verdict"),
    [
        (["candidate", "tie", "tie", "original", "candidate"], "win"),
        (["candidate", "tie", "tie", "tie"], "win"),
        (["original", "original", "candidate", None], "loss"),
        (["candidate", "original"], "tie"),
        (["tie", "tie", "tie"], "tie"),
        ([None, None], "error"),
        ([], "error"),
    ],
)
def test_a_prompt_wins_when_it_wins_more_scenarios_than_it_loses(votes, verdict):
    found = verdict_of(votes)
    assert found.verdict == verdict
    counted = [v for v in votes if v is not None]
    assert (found.wins, found.ties, found.losses) == (
        counted.count("candidate"),
        counted.count("tie"),
        counted.count("original"),
    )


# --- the calls ------------------------------------------------------------------------------------


def test_a_judge_that_always_prefers_the_first_position_yields_a_tie_never_a_win():
    found, _raw = run(BenchWorld(pairwise=first_position))
    assert found.tool == Comparison("tie", 0, S, 0)


def test_a_judge_that_prefers_the_better_answer_in_both_orders_yields_a_win():
    found, _raw = run(BenchWorld(pairwise=by_content))
    assert found.tool == Comparison("win", S, 0, 0)


def test_a_candidate_whose_answers_are_worse_loses():
    world = BenchWorld(
        task=lambda call: "BAD answer" if MARKER in call.user else "GOOD answer",
    )
    found, _raw = run(world)
    assert found.tool == Comparison("loss", 0, 0, S)


def test_fresh_scenarios_are_synthesised_once_under_the_bench_sample():
    _found, raw = run(BenchWorld(), seed=3)
    (synth,) = calls(raw, "synth")
    assert json.loads(synth.user) == {"prompt": ORIGINAL, "count": S}
    assert synth.model == MODELS.reflect and synth.sample == BENCH_SAMPLE + 3
    assert all(call.sample == BENCH_SAMPLE + 3 for call in raw.calls)


def test_both_prompts_run_the_plain_task_call_on_the_target_model():
    _found, raw = run(BenchWorld())
    tasks = calls(raw, "task")
    assert len(tasks) == 2 * S
    assert {call.model for call in tasks} == {MODELS.target}
    # the plain call a user sees (SPEC R10a): the situation, a blank line, the prompt; no suffix
    assert {call.user for call in tasks} == {
        f"situation {i}\n\n{text}" for i in range(1, S + 1) for text in (ORIGINAL, BETTER)
    }
    assert all(call.system == "" for call in tasks)


def test_a_template_runs_as_the_system_prompt():
    _found, raw = run(BenchWorld(), kind="template")
    tasks = calls(raw, "task")
    assert {(call.system, call.user) for call in tasks} == {
        (text, f"situation {i}") for i in range(1, S + 1) for text in (ORIGINAL, BETTER)
    }


def test_each_scenario_is_judged_in_both_orders_by_the_judge_model_blind():
    _found, raw = run(BenchWorld())
    judges = calls(raw, "judge")
    assert len(judges) == 2 * S and all(is_pairwise(call) for call in judges)
    assert {call.model for call in judges} == {MODELS.judge}
    pairs = {answers(call) for call in judges}
    assert pairs == {("BAD answer", "GOOD answer"), ("GOOD answer", "BAD answer")}
    for call in judges:
        request = json.loads(call.user)
        assert set(request) == {"request", "situation", "answer_A", "answer_B"}
        assert request["request"] == ORIGINAL  # what the user meant, never the candidate
        assert BETTER not in call.user and BETTER not in call.system
        assert call.system == PAIRWISE_SYSTEM


def test_the_judge_instruction_says_the_answers_are_anonymous_and_asks_for_the_intent():
    text = PAIRWISE_SYSTEM.lower()
    assert "anonymous" in text and "likely intent" in text and "not instructions" in text
    call = pairwise_call("r", "s", "a", "b", MODELS.judge, BENCH_SAMPLE)
    schema = json.loads(call.json_schema or "")
    assert schema["required"] == ["winner", "reason"]
    assert schema["properties"]["winner"]["enum"] == ["A", "B", "tie"]
    assert (call.role, call.system) == ("judge", PAIRWISE_SYSTEM)


@pytest.mark.parametrize(
    "reply",
    ["not json", "[]", '{"reason": "x"}', '{"winner": "C", "reason": "x"}', '{"winner": "A"}'],
)
def test_an_invalid_judge_reply_is_refused(reply):
    with pytest.raises(ValueError):
        parse_winner(reply)


def test_an_invalid_judge_reply_is_asked_again_under_a_new_sample():
    replies: dict[int, int] = {}

    def pairwise(call: Call) -> str:
        replies[call.sample] = replies.get(call.sample, 0) + 1
        return "no json" if call.sample == BENCH_SAMPLE else by_content(call)

    found, raw = run(BenchWorld(pairwise=pairwise))
    assert found.tool is not None and found.tool.verdict == "win"
    assert replies == {BENCH_SAMPLE: 2 * S, BENCH_SAMPLE + 1: 2 * S}
    assert parse_winner(winner("tie")) == "tie"


def test_an_unchanged_prompt_and_no_baseline_makes_no_call():
    found, raw = run(BenchWorld(), candidate=None)
    assert (found.tool, found.naive, raw.calls) == (None, None, [])


# --- the naive baseline ---------------------------------------------------------------------------


def test_the_naive_rewrite_is_one_low_effort_call_to_the_reflection_model():
    call = naive_call(ORIGINAL, MODELS.reflect, BENCH_SAMPLE)
    assert (call.role, call.model, call.user, call.effort) == (
        "reflect",
        MODELS.reflect,
        ORIGINAL,
        "low",
    )
    assert call.system == NAIVE_SYSTEM and call.system.startswith("Improve this prompt.")
    assert INSTRUCTION_BEGIN in NAIVE_SYSTEM and INSTRUCTION_END in NAIVE_SYSTEM


def test_the_naive_rewrite_is_compared_on_the_same_scenarios_and_original_answers():
    world = BenchWorld(naive=f"{NAIVE} {MARKER}")
    found, raw = run(world, naive=True)
    assert found.tool == Comparison("win", S, 0, 0)
    assert found.naive == Comparison("win", S, 0, 0)
    assert len(calls(raw, "synth")) == 1 and len(calls(raw, "reflect")) == 1
    ran = [call.user.split("\n\n", 1)[1] for call in calls(raw, "task")]
    assert sorted(set(ran)) == sorted({ORIGINAL, BETTER, f"{NAIVE} {MARKER}"})
    assert len(ran) == 3 * S  # the original ran once, for both comparisons
    assert len(calls(raw, "judge")) == 4 * S


def test_the_naive_baseline_runs_for_an_unchanged_prompt_too():
    found, raw = run(BenchWorld(), candidate=None, naive=True)
    assert found.tool is None
    assert found.naive == Comparison("tie", 0, S, 0)  # NAIVE has no marker: no better
    assert len(calls(raw, "task")) == 2 * S


def test_a_naive_rewrite_equal_to_the_original_is_a_tie_with_no_run():
    found, raw = run(BenchWorld(naive=f"  {ORIGINAL}\n"), candidate=None, naive=True)
    assert found.naive == Comparison("tie", 0, 0, 0)
    assert calls(raw, "task") == [] and calls(raw, "judge") == []


def test_a_naive_reply_without_delimiters_is_asked_again_then_an_error():
    world = BenchWorld()
    raw = ScriptedBackend(lambda call: "just text" if call.role == "reflect" else world(call))
    found = judge(raw, ORIGINAL, BETTER, naive=True, kind="task", models=MODELS, workers=1, seed=0)
    assert found.tool == Comparison("win", S, 0, 0)
    assert found.naive is not None and found.naive.verdict == "error"
    assert [call.sample for call in calls(raw, "reflect")] == [BENCH_SAMPLE + n for n in range(3)]


# --- failures -------------------------------------------------------------------------------------


def test_a_failed_judge_call_leaves_its_scenario_out():
    def pairwise(call: Call) -> str | Exception:
        return CallFailed("judge down") if "situation 1" in call.user else by_content(call)

    found, _raw = run(BenchWorld(pairwise=pairwise))
    assert found.tool == Comparison("win", S - 1, 0, 0)


def test_a_failed_task_run_leaves_its_scenario_out_without_a_judge_call():
    def task(call: Call) -> str | Exception:
        if "situation 2" in call.user and MARKER in call.user:
            return BudgetExhausted("out of calls")
        return "GOOD answer" if MARKER in call.user else "BAD answer"

    found, raw = run(BenchWorld(task=task))
    assert found.tool == Comparison("win", S - 1, 0, 0)
    assert len(calls(raw, "judge")) == 2 * (S - 1)


def test_no_scenario_judged_is_an_error_not_a_tie():
    found, _raw = run(BenchWorld(pairwise=lambda _call: CallFailed("judge down")), naive=True)
    assert found.tool is not None and found.tool.verdict == "error"
    assert found.naive is not None and found.naive.verdict == "error"


def test_a_failed_synthesis_is_an_error_for_both_comparisons():
    world = BenchWorld()
    raw = ScriptedBackend(lambda call: CallFailed("down") if call.role == "synth" else world(call))
    found = judge(raw, ORIGINAL, BETTER, naive=True, kind="task", models=MODELS, workers=1, seed=0)
    assert found.tool is not None and found.tool.verdict == "error"
    assert found.naive is not None and found.naive.verdict == "error"
    assert calls(raw, "task") == []


def test_a_backend_failure_propagates():
    world = BenchWorld(pairwise=lambda _call: BackendError("three in a row"))
    with pytest.raises(BackendError):
        run(world)


@pytest.mark.parametrize(("improved", "naive"), [(True, False), (False, True), (True, True)])
def test_the_call_estimate_is_what_a_comparison_makes(improved, naive):
    world = BenchWorld(naive=f"{NAIVE} {MARKER}")
    _found, raw = run(world, BETTER if improved else None, naive=naive)
    assert pairwise_calls(improved, naive) == len(raw.calls)
    assert pairwise_calls(False, False) == 0


def test_the_time_estimate_is_three_waves_of_the_latency_model():
    # 6 workers, a short prompt: a synthesis of 4 scenarios (3.4 + 180 / 70 s), 8 plain answers
    # in 2 waves (3.4 + 500 / 70 s each), 8 judge replies in 2 waves (3.4 + 40 / 70 s each)
    assert pairwise_seconds(True, False, 6, 12) == pytest.approx(35.0, abs=0.01)
    # the baseline: the naive rewrite beside the synthesis, 12 answers, 16 judge replies
    assert pairwise_seconds(True, True, 6, 12) == pytest.approx(38.97, abs=0.01)
    assert pairwise_seconds(False, False, 6, 12) == 0.0


def test_parallel_waves_give_the_same_result_as_one_at_a_time():
    one, _ = run(BenchWorld(), naive=True, workers=1)
    many, raw = run(BenchWorld(), naive=True, workers=6)
    assert one == many and len(raw.calls) == 2 + 3 * S + 4 * S
