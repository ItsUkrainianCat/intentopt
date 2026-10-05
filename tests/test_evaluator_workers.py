"""The evaluator with `workers` (SPEC R25; ADR-011, and R10, R10b, R24 as before): the task calls
of distinct scenarios and the judge chunks run side by side, and the entries, scores, side info and
calls are those of a serial run whatever order the calls end in. A failed call still makes only
its own scenarios incomplete; the first propagating error by scenario order still ends the batch.

The scripted model answers by the content of each call, never by the order of the calls. No test
sleeps: threads wait on events whose timeouts fire only when the code under test is wrong.
"""

import json
import threading
from collections import Counter

import pytest
from fakes import ScriptedBackend, judge_reply

from autoimprover.evaluator import Evaluator
from autoimprover.types import (
    BackendError,
    BudgetExhausted,
    Call,
    CallFailed,
    Check,
    Contract,
    Scenario,
    SessionNotLockedDown,
)

WAIT = 10.0  # seconds; only a broken implementation ever waits this long
TASK = "claude-haiku-4-5-20251001"
JUDGE = "claude-opus-5-5"
CANDIDATE = "Answer in one short paragraph and name the topic."
CONTRACT = Contract(
    goal="answer questions",
    kind="template",
    checks=(
        Check(id="short", group="format", text="at most 40 characters", rule="max_chars", arg="40"),
        Check(id="says", group="format", text="starts with Answer", rule="contains", arg="Answer"),
        Check(id="topic", group="content", text="names the topic"),
        Check(id="kind", group="constraints", text="stays polite"),
    ),
)


def scenario(n: int) -> Scenario:
    criteria = ("mentions light",) if n % 3 == 0 else ()
    expected = "blue" if n % 4 == 0 else None
    tail = " [fail]" if n == 5 else ""
    return Scenario(id=f"s{n}", input=f"question {n}{tail}", expected=expected, criteria=criteria)


SCENARIOS = [scenario(n) for n in range(1, 15)]


def number(text: str) -> int:
    """The scenario number in a task call's user text (`question N`)."""
    return int(text.split()[1])


def answer(call: Call) -> str | Exception:
    """By content only: scenario 5's task call fails, the judge chunk holding s7 replies invalid
    JSON to its first sample and the chunk holding s14 (the third, alone) always fails; checks
    pass by their ids."""
    if call.role == "task":
        if "[fail]" in call.user:
            return CallFailed(f"task call failed for {call.user}")
        return f"Answer on {call.user}" + " and more" * (number(call.user) % 4)
    ids = [item["scenario"] for item in json.loads(call.user)["scenarios"]]
    if "s14" in ids:
        return CallFailed("judge chunk with s14 failed")
    if "s7" in ids and call.sample < 1000:
        return "not json"
    return judge_reply(call, lambda sid, cid, _output: (len(sid) + len(cid)) % 2 == 0)


def run(script, workers: int, scenarios=SCENARIOS) -> tuple[list, ScriptedBackend]:
    backend = ScriptedBackend(script)
    evaluator = Evaluator(backend, CONTRACT, TASK, JUDGE, sample=3, workers=workers)
    return evaluator(CANDIDATE, [*scenarios, scenarios[0]]), backend


def waited(event: threading.Event) -> None:
    assert event.wait(WAIT), "a call waited in vain: the calls did not run side by side"


@pytest.mark.parametrize("workers", [2, 4, 16])
def test_workers_give_the_entries_and_the_calls_of_a_serial_run(workers):
    serial, serial_backend = run(answer, 1)
    threaded, threaded_backend = run(answer, workers)
    assert threaded == serial
    assert Counter(threaded_backend.calls) == Counter(serial_backend.calls)
    incomplete = [side["scenario"] for _, side in serial if side.get("incomplete")]
    assert incomplete == ["s5", "s14"]  # the fixture reaches every path
    assert {c.sample for c in serial_backend.calls if c.role == "judge"} == {3, 1003}


def test_entries_follow_the_scenarios_not_the_order_the_calls_end():
    """Eight task calls end last to first, then the second judge chunk ends before the first."""
    eight = [scenario(n) for n in (1, 2, 3, 4, 6, 7, 8, 9)]
    task_done = {n: threading.Event() for n in (1, 2, 3, 4, 6, 7, 8, 9)}
    later = {1: 2, 2: 3, 3: 4, 4: 6, 6: 7, 7: 8, 8: 9}
    second_chunk_done = threading.Event()
    ended: list[str] = []

    def backwards(call: Call) -> str | Exception:
        if call.role == "task":
            n = number(call.user)
            if n in later:
                waited(task_done[later[n]])
            ended.append(f"task {n}")
            task_done[n].set()
            return answer(call)
        ids = [item["scenario"] for item in json.loads(call.user)["scenarios"]]
        if "s1" in ids and call.sample < 1000:
            waited(second_chunk_done)
        reply = answer(call)
        ended.append(f"judge {ids[0]} sample {call.sample}")
        if "s9" in ids:
            second_chunk_done.set()
        return reply

    threaded, _ = run(backwards, 8, eight)
    serial, _ = run(answer, 1, eight)
    assert threaded == serial
    assert ended[:8] == [f"task {n}" for n in (9, 8, 7, 6, 4, 3, 2, 1)]
    assert ended[8] == "judge s8 sample 3"  # the chunks are s1-s7 (without s5) and s8-s9


def test_by_default_every_call_runs_on_the_calling_thread():
    threads: set[int] = set()

    def record(call: Call) -> str | Exception:
        threads.add(threading.get_ident())
        return answer(call)

    backend = ScriptedBackend(record)
    Evaluator(backend, CONTRACT, TASK, JUDGE)(CANDIDATE, SCENARIOS)
    assert threads == {threading.get_ident()}


@pytest.mark.parametrize("workers", [1, 4])
def test_the_first_task_error_by_scenario_order_ends_the_batch(workers):
    """With threads, scenario 4's error comes first in time; scenario 2's still wins."""
    four = [scenario(n) for n in (1, 2, 3, 4)]
    fourth_failed = threading.Event()

    def script(call: Call) -> str | Exception:
        if call.role == "task" and number(call.user) == 2:
            if workers > 1:
                waited(fourth_failed)
            return BackendError("scenario 2 ends the run")
        if call.role == "task" and number(call.user) == 4:
            fourth_failed.set()
            return BudgetExhausted("scenario 4 found the budget spent")
        return answer(call)

    backend = ScriptedBackend(script)
    with pytest.raises(BackendError, match="scenario 2"):
        Evaluator(backend, CONTRACT, TASK, JUDGE, workers=workers)(CANDIDATE, four)
    assert backend.count("judge") == 0


@pytest.mark.parametrize("workers", [1, 4])
def test_a_judge_chunk_that_is_not_locked_down_ends_the_batch(workers):
    def script(call: Call) -> str | Exception:
        if call.role == "judge" and '"s8"' in call.user:
            return SessionNotLockedDown("the judge session reported a tool")
        return answer(call)

    with pytest.raises(SessionNotLockedDown):
        run(script, workers, [scenario(n) for n in (1, 2, 3, 4, 6, 7, 8, 9)])
