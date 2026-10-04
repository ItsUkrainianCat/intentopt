"""The evaluator without the judge's reply rules: one task call per scenario shaped by the prompt's
kind, programmatic checks on the output, the score and its ASI, failed calls, and purity (SPEC
R10, R10a, R16, R24). The judge request and the parsing of its reply are tested in
`test_evaluator_judge.py`."""

import dataclasses
import json
import random
from collections.abc import Callable

import pytest
from fakes import ScriptedBackend, judge_reply

from autoimprover.evaluator import Evaluator
from autoimprover.types import (
    CALL_RETRIES,
    JUDGE_BATCH_MAX,
    PROGRAMMATIC_RULES,
    BackendError,
    BudgetExhausted,
    Call,
    CallError,
    CallFailed,
    Check,
    Contract,
    Scenario,
    SessionNotLockedDown,
)

TASK = "claude-haiku-4-5-20251001"
JUDGE = "claude-opus-5-5"
CANDIDATE = "Answer the question in one short paragraph and name the source."
S1 = Scenario(id="s1", input="Why is the sky blue?")
S2 = Scenario(id="s2", input="Why is the sea salty?")
S3 = Scenario(id="s3", input="Why do leaves fall?")
BARE_TEMPLATE = Contract(goal="answer questions", kind="template")
BARE_TASK = Contract(goal="answer questions", kind="task")


def script_for(
    output: str | None = None, judge: Callable[[Call], str] = judge_reply
) -> Callable[[Call], str]:
    """Task calls answer `output` (default: an answer naming the scenario's input); judge calls
    answer `judge(call)` (default: every check passes, quoting the output's first characters)."""

    def script(call: Call) -> str:
        if call.role == "judge":
            return judge(call)
        return output if output is not None else f"Answer to: {call.user[:40]}"

    return script


def answering(
    output: str | None = None, judge: Callable[[Call], str] = judge_reply
) -> ScriptedBackend:
    return ScriptedBackend(script_for(output, judge))


def tasks(backend: ScriptedBackend) -> list[Call]:
    return [c for c in backend.calls if c.role == "task"]


# --- task calls (SPEC R10a; ADR-005, ADR-008 task row) -------------------------------------------


def test_a_template_runs_as_the_system_prompt_with_the_input_as_the_user_message():
    backend = answering()
    Evaluator(backend, BARE_TEMPLATE, TASK, JUDGE)(CANDIDATE, [S1])
    (call,) = tasks(backend)
    assert call == Call(role="task", model=TASK, user=S1.input, system=CANDIDATE)
    assert call.json_schema is None


def test_a_task_prompt_follows_its_situation_after_a_blank_line_with_no_system_prompt():
    backend = answering()
    Evaluator(backend, BARE_TASK, TASK, JUDGE)(CANDIDATE, [S1])
    (call,) = tasks(backend)
    assert call == Call(role="task", model=TASK, user=S1.input + "\n\n" + CANDIDATE, system="")
    assert call.json_schema is None


def test_one_task_call_per_scenario_in_order_with_the_task_model_and_the_sample():
    backend = answering()
    Evaluator(backend, BARE_TEMPLATE, "some-task-model", JUDGE, sample=3)(CANDIDATE, [S1, S2, S3])
    assert [c.user for c in tasks(backend)] == [S1.input, S2.input, S3.input]
    assert {(c.model, c.sample, c.system, c.json_schema) for c in tasks(backend)} == {
        ("some-task-model", 3, CANDIDATE, None)
    }


def test_the_sample_defaults_to_zero():
    backend = answering()
    Evaluator(backend, BARE_TASK, TASK, JUDGE)(CANDIDATE, [S1, S2])
    assert [c.sample for c in tasks(backend)] == [0, 0]


def test_an_empty_batch_returns_nothing_and_calls_nothing():
    backend = answering()
    assert Evaluator(backend, BARE_TASK, TASK, JUDGE)(CANDIDATE, []) == []
    assert backend.calls == []


def test_one_entry_per_scenario_each_a_score_and_a_dict():
    results = Evaluator(answering(), BARE_TASK, TASK, JUDGE)(CANDIDATE, [S1, S2, S3])
    assert len(results) == 3
    for score, side_info in results:
        assert isinstance(score, float) and isinstance(side_info, dict)
    assert json.dumps([info for _, info in results])  # plain data GEPA can store


# --- programmatic checks (SPEC R10 (a)) ----------------------------------------------------------

HAS_SOURCE = Check("has-source", "format", "names a source", "contains", "Source:")
NO_SORRY = Check("no-sorry", "constraints", "never apologises", "not_contains", "sorry")
SHORT = Check("short", "constraints", "at most 40 characters", "max_chars", "40")
LONG_ENOUGH = Check("long-enough", "content", "at least 20 characters", "min_chars", "20")
PROGRAMMATIC = Contract(
    goal="answer questions",
    kind="template",
    checks=(HAS_SOURCE, NO_SORRY, SHORT, LONG_ENOUGH),
)


def scored(contract: Contract, output: str, candidate: str = CANDIDATE) -> tuple[float, dict]:
    """The one entry for S1 when the task model answers `output`."""
    (entry,) = Evaluator(answering(output), contract, TASK, JUDGE)(candidate, [S1])
    return entry


def passes(check: Check, output: str) -> bool:
    score, _ = scored(Contract(goal="g", kind="template", checks=(check,)), output)
    return score == 1.0


CASES = {
    "contains: present": (HAS_SOURCE, "Rayleigh. Source: NASA", True),
    "contains: absent": (HAS_SOURCE, "Rayleigh scattering", False),
    "contains: other case": (HAS_SOURCE, "Rayleigh. source: NASA", False),
    "not_contains: absent": (NO_SORRY, "Rayleigh scattering", True),
    "not_contains: present": (NO_SORRY, "I am sorry", False),
    "not_contains: other case": (NO_SORRY, "Sorry, Rayleigh", True),
    "max_chars: at the limit": (SHORT, "x" * 40, True),
    "max_chars: one over": (SHORT, "x" * 41, False),
    "min_chars: at the limit": (LONG_ENOUGH, "x" * 20, True),
    "min_chars: one under": (LONG_ENOUGH, "x" * 19, False),
    "max_chars 0: empty output": (Check("e", "format", "empty", "max_chars", "0"), "", True),
    "max_chars 0: any output": (Check("e", "format", "empty", "max_chars", "0"), "x", False),
    "max_chars with leading zeros": (Check("z", "format", "z", "max_chars", "0003"), "abc", True),
    "min_chars with leading zeros": (Check("z", "format", "z", "min_chars", "0003"), "abc", True),
    "max_chars past int()": (Check("h", "format", "huge", "max_chars", "9" * 5000), "x", True),
    "min_chars past int()": (Check("h", "format", "huge", "min_chars", "9" * 5000), "x", False),
    "min_chars 0": (Check("m", "format", "anything", "min_chars", "0"), "", True),
}


@pytest.mark.parametrize(("check", "output", "expected"), CASES.values(), ids=CASES.keys())
def test_a_programmatic_check_runs_on_the_output(check, output, expected):
    assert passes(check, output) is expected


@pytest.mark.parametrize("rule", PROGRAMMATIC_RULES)
def test_every_programmatic_rule_is_checked_without_a_model(rule):
    backend = answering("12345")
    contract = Contract(goal="g", kind="template", checks=(Check("c", "format", "t", rule, "3"),))
    Evaluator(backend, contract, TASK, JUDGE)(CANDIDATE, [S1])
    assert [c.role for c in backend.calls] == ["task"]


def test_programmatic_checks_read_the_output_not_the_candidate():
    candidate = "Always end with 'Source:' and never say sorry. " + "pad " * 20
    output = "sorry"  # no source, apologises, 5 characters
    score, side_info = scored(PROGRAMMATIC, output, candidate)
    assert [f["id"] for f in side_info["failed"]] == ["has-source", "no-sorry", "long-enough"]
    assert score == 0.25  # only `short` passes, and the candidate itself is long


def test_the_score_is_the_share_of_checks_passed_with_a_share_per_group():
    score, side_info = scored(PROGRAMMATIC, "I am sorry. Source: none")
    assert score == 0.75  # the apology fails `no-sorry`
    assert side_info["scores"] == {"format": 1.0, "constraints": 0.5, "content": 1.0}
    assert side_info["failed"] == [{"id": "no-sorry", "text": "never apologises"}]
    assert "reason" not in side_info


def test_all_checks_passed_scores_one():
    score, side_info = scored(PROGRAMMATIC, "Rayleigh. Source: NASA")
    assert score == 1.0
    assert side_info["failed"] == []


def test_scores_list_only_the_groups_that_have_checks():
    contract = Contract(goal="g", kind="template", checks=(HAS_SOURCE, NO_SORRY))
    _, side_info = scored(contract, "Source: NASA, sorry")
    assert side_info["scores"] == {"format": 1.0, "constraints": 0.0}


def test_the_asi_names_the_scenario_and_quotes_the_first_300_characters_of_the_output():
    output = "".join(f"{i:04d}" for i in range(250))  # 1,000 characters
    _, side_info = scored(PROGRAMMATIC, output)
    assert side_info["scenario"] == "s1"
    assert side_info["output_excerpt"] == output[:300]


def test_a_scenario_with_no_check_scores_zero_with_the_reason_no_checks():
    score, side_info = scored(BARE_TEMPLATE, "anything")
    assert score == 0.0
    assert side_info["reason"] == "no_checks"
    assert side_info["scores"] == {} and side_info["failed"] == []


# --- failed calls (SPEC R24) ---------------------------------------------------------------------

CITES = Check("cites", "content", "cites a source")
JUDGED = Contract(goal="answer questions", kind="template", checks=(CITES, SHORT))


def batch(n: int) -> list[Scenario]:
    return [Scenario(id=f"s{i}", input=f"question {i}") for i in range(1, n + 1)]


def failing_on(role: str, when: Callable[[Call], bool], error: Exception) -> ScriptedBackend:
    """Answers like `answering()`, except that calls of `role` matching `when` raise `error`."""
    normal = script_for()
    return ScriptedBackend(lambda call: error if call.role == role and when(call) else normal(call))


def first_chunk(call: Call) -> bool:
    return '"s1"' in call.user


def sent_to_judge(backend: ScriptedBackend) -> list[list[str]]:
    return [
        [s["scenario"] for s in json.loads(c.user)["scenarios"]]
        for c in backend.calls
        if c.role == "judge"
    ]


def test_a_failed_task_call_makes_only_its_scenario_incomplete():
    backend = failing_on("task", lambda c: c.user == "question 2", CallFailed("task down"))
    results = Evaluator(backend, JUDGED, TASK, JUDGE)(CANDIDATE, batch(3))
    assert results[1] == (0.0, {"incomplete": True, "error": "task down", "scenario": "s2"})
    assert [score for score, _ in (results[0], results[2])] == [1.0, 1.0]
    assert sent_to_judge(backend) == [["s1", "s3"]]


def test_a_failed_judge_call_makes_its_chunk_incomplete_and_spares_the_other_chunks():
    backend = failing_on("judge", first_chunk, CallFailed("judge down"))
    results = Evaluator(backend, JUDGED, TASK, JUDGE)(CANDIDATE, batch(7))
    for (score, info), scenario in zip(results[:6], batch(6), strict=True):
        assert score == 0.0
        assert info == {"incomplete": True, "error": "judge down", "scenario": scenario.id}
    assert results[6][0] == 1.0 and "incomplete" not in results[6][1]
    assert sent_to_judge(backend) == [["s1", "s2", "s3", "s4", "s5", "s6"], ["s7"]]


def test_a_failed_chunk_is_left_out_of_the_30_percent_rule():
    backend = failing_on("judge", first_chunk, CallFailed("judge down"))
    score, info = Evaluator(backend, JUDGED, TASK, JUDGE)(CANDIDATE, batch(7))[6]
    assert score == 1.0  # 6 of 7 judged checks would be unknown if the failed chunk counted
    assert info["failed"] == [] and "reason" not in info


def test_three_invalid_judge_replies_make_the_chunk_incomplete_without_echoing_them():
    backend = answering("out", judge=lambda _call: "SECRET REPLY")
    ((score, info),) = Evaluator(backend, JUDGED, TASK, JUDGE)(CANDIDATE, [S1])
    assert score == 0.0 and info["incomplete"] is True and info["scenario"] == "s1"
    assert "judge" in info["error"] and "3 attempts" in info["error"]
    assert "SECRET REPLY" not in info["error"]
    assert [(c.role, c.sample) for c in backend.calls] == [
        ("task", 0),
        ("judge", 0),
        ("judge", 1),
        ("judge", 2),
    ]


def test_a_scenario_the_failed_judge_call_did_not_hold_keeps_its_score():
    programmatic = Contract(goal="g", kind="template", checks=(SHORT,))
    with_criteria = Scenario(id="c", input="question c", criteria=("is kind",))
    backend = failing_on("judge", lambda c: True, CallFailed("judge down"))
    plain, judged = Evaluator(backend, programmatic, TASK, JUDGE)(CANDIDATE, [S1, with_criteria])
    assert plain[0] == 1.0 and "incomplete" not in plain[1]
    assert judged == (0.0, {"incomplete": True, "error": "judge down", "scenario": "c"})


PROPAGATED = [
    CallError("down"),
    BackendError("down"),
    BudgetExhausted("down"),
    SessionNotLockedDown("down"),
    ValueError("down"),
    RuntimeError("down"),
]


@pytest.mark.parametrize("error", PROPAGATED, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("role", ["task", "judge"])
def test_every_other_exception_propagates_and_stops_the_batch(role, error):
    backend = failing_on(role, lambda c: True, error)
    with pytest.raises(type(error), match="^down$"):
        Evaluator(backend, JUDGED, TASK, JUDGE)(CANDIDATE, batch(3))
    assert backend.calls[-1].role == role
    assert len(backend.calls) == (1 if role == "task" else 4)


@pytest.mark.parametrize("n", range(1, 14))
def test_the_calls_never_exceed_one_task_call_per_scenario_plus_the_judge_attempts(n):
    happy, broken = answering(), answering("out", judge=lambda _call: "not json")
    Evaluator(happy, JUDGED, TASK, JUDGE)(CANDIDATE, batch(n))
    Evaluator(broken, JUDGED, TASK, JUDGE)(CANDIDATE, batch(n))
    chunks = -(-n // JUDGE_BATCH_MAX)
    assert len(happy.calls) == n + chunks
    assert len(broken.calls) == n + chunks * (1 + CALL_RETRIES)


# --- purity, repeated scenarios, bounds -----------------------------------------------------------


def test_the_inputs_are_left_as_they_were():
    scenarios = [S1, Scenario(id="e", input="q", expected="a", criteria=("c",)), S3]
    snapshot, contract = list(scenarios), dataclasses.replace(JUDGED)
    Evaluator(answering(), JUDGED, TASK, JUDGE)(CANDIDATE, scenarios)
    assert scenarios == snapshot and JUDGED == contract


def test_the_same_inputs_on_a_deterministic_model_give_equal_results():
    first, second = answering(), answering()
    evaluator = Evaluator(first, JUDGED, TASK, JUDGE)
    once = evaluator(CANDIDATE, batch(8))
    assert evaluator(CANDIDATE, batch(8)) == once
    assert Evaluator(second, JUDGED, TASK, JUDGE)(CANDIDATE, batch(8)) == once
    assert first.calls == second.calls + second.calls


def test_a_repeated_scenario_is_run_once_and_gets_an_entry_of_its_own_in_place():
    backend = answering()
    results = Evaluator(backend, JUDGED, TASK, JUDGE)(CANDIDATE, [S1, S2, S1])
    assert [c.user for c in tasks(backend)] == [S1.input, S2.input]
    assert sent_to_judge(backend) == [["s1", "s2"]]
    assert [info["scenario"] for _, info in results] == ["s1", "s2", "s1"]
    assert results[0] == results[2] == (1.0, results[0][1])
    results[0][1]["failed"].append("x")
    assert results[2][1]["failed"] == []


def test_two_different_scenarios_with_one_id_are_refused_before_any_call():
    backend = answering()
    with pytest.raises(ValueError, match="share an id"):
        Evaluator(backend, JUDGED, TASK, JUDGE)(CANDIDATE, [S1, Scenario(id="s1", input="other")])
    assert backend.calls == []


@pytest.mark.parametrize(
    ("taken", "scenario"),
    [
        ("expected", Scenario(id="e", input="q", expected="a")),
        ("crit-1", Scenario(id="e", input="q", criteria=("c",))),
    ],
    ids=["expected", "crit-1"],
)
def test_a_judged_check_id_given_twice_is_refused_before_any_call(taken, scenario):
    contract = Contract(goal="g", kind="template", checks=(Check(taken, "content", "t"),))
    backend = answering()
    with pytest.raises(ValueError, match="share an id"):
        Evaluator(backend, contract, TASK, JUDGE)(CANDIDATE, [scenario])
    assert backend.calls == []


def test_a_programmatic_check_may_share_an_id_with_a_judged_one():
    contract = Contract(
        goal="g", kind="template", checks=(Check("expected", "format", "t", "min_chars", "1"),)
    )
    ((score, _),) = Evaluator(answering(), contract, TASK, JUDGE)(
        CANDIDATE, [Scenario(id="e", input="q", expected="Answer")]
    )
    assert score == 1.0


def test_every_score_and_every_share_lies_between_zero_and_one():
    rng = random.Random(4)
    contract = Contract(
        goal="g",
        kind="template",
        checks=(
            CITES,
            SHORT,
            NO_SORRY,
            Check("j2", "format", "t"),
            Check("j3", "constraints", "t"),
        ),
    )

    def judge(call: Call) -> str:
        return judge_reply(call, lambda *_: rng.random() < 0.5)

    for _ in range(50):
        output = rng.choice(["", "sorry", "x" * 50, "Answer: fine"])
        results = Evaluator(answering(output, judge), contract, TASK, JUDGE)(CANDIDATE, batch(7))
        for score, info in results:
            assert 0.0 <= score <= 1.0
            assert all(0.0 <= share <= 1.0 for share in info["scores"].values())
