"""The shared test doubles behave the way the other tests assume."""

import pytest
from fakes import by_role, failing

from autoimprover.types import Call, CallError


def call(role: str = "task") -> Call:
    return Call(role=role, model="m", user="u")  # type: ignore[arg-type]


def test_scripted_backend_records_calls_and_counts_by_role():
    backend = by_role({"task": "out", "judge": "ok"})
    backend.complete(call("task"))
    backend.complete(call("task"))
    backend.complete(call("judge"))
    assert (backend.count(), backend.count("task"), backend.count("judge")) == (3, 2, 1)


def test_by_role_serves_a_list_in_order_and_repeats_the_last():
    backend = by_role({"task": ["a", "b"]})
    assert [backend.complete(call()).text for _ in range(3)] == ["a", "b", "b"]


def test_by_role_raises_an_exception_entry():
    backend = by_role({"task": CallError("boom")})
    with pytest.raises(CallError, match="boom"):
        backend.complete(call())


def test_failing_backend_fails_every_attempt_and_still_records_it():
    backend = failing()
    for _ in range(2):
        with pytest.raises(CallError):
            backend.complete(call())
    assert backend.count() == 2


def test_builders_produce_replies_with_exactly_the_schema_fields():
    import json

    from fakes import intake_reply, judge_reply, reflection_reply, synth_reply

    from autoimprover.types import INTAKE_SCHEMA, JUDGE_SCHEMA, SYNTH_SCHEMA

    assert set(json.loads(intake_reply())) == set(INTAKE_SCHEMA["required"])
    assert set(json.loads(synth_reply(3))) == set(SYNTH_SCHEMA["required"])
    request = {
        "scenarios": [
            {
                "scenario": "s1",
                "input": "i",
                "output": "an output",
                "checks": [{"id": "c1", "text": "t"}],
            }
        ]
    }
    reply = json.loads(judge_reply(Call(role="judge", model="m", user=json.dumps(request))))
    assert set(reply) == set(JUDGE_SCHEMA["required"])
    check = reply["results"][0]["checks"][0]
    assert set(check) == {"id", "pass", "quote"}
    assert check["quote"] in "an output"
    assert reflection_reply("new text").startswith("```\nnew text\n```\n- ")


def test_happy_backend_is_a_positive_control_and_its_twin_is_a_negative_one():
    import json

    from fakes import MARKER, happy_backend

    win = happy_backend(f"better prompt {MARKER}")
    lose = happy_backend("same old prompt")
    task = Call(role="task", model="m", user="situation 1", system="")
    with_marker = Call(role="task", model="m", user=f"situation 1 {MARKER}")
    assert win.complete(task).text == "BAD answer"
    assert win.complete(with_marker).text == "GOOD answer"
    request = {
        "scenarios": [
            {
                "scenario": "s1",
                "input": "i",
                "output": "GOOD answer",
                "checks": [{"id": "c1", "text": "t"}],
            }
        ]
    }
    judged = json.loads(win.complete(Call(role="judge", model="m", user=json.dumps(request))).text)
    assert judged["results"][0]["checks"][0]["pass"] is True
    assert MARKER in win.complete(Call(role="reflect", model="m", user="x")).text
    assert MARKER not in lose.complete(Call(role="reflect", model="m", user="x")).text


def test_happy_backend_passes_the_r6_contract_check_for_its_winner():
    # The contract check sends the candidate as the judge's "output" (ADR-008); without a pass for
    # scenario `contract` the positive control could never return an improvement.
    import json

    from fakes import MARKER, happy_backend

    win = happy_backend(f"better {MARKER}")
    request = {
        "scenarios": [
            {
                "scenario": "contract",
                "input": "the original prompt",
                "output": f"better {MARKER}",
                "checks": [{"id": "k1", "text": "keeps every literal"}],
            }
        ]
    }
    judged = json.loads(win.complete(Call(role="judge", model="m", user=json.dumps(request))).text)
    assert judged["results"][0]["scenario"] == "contract"
    assert judged["results"][0]["checks"][0]["pass"] is True


def test_scripted_backend_reports_a_duration_and_advances_a_fake_clock():
    from fakes import FakeClock, ScriptedBackend

    clock = FakeClock(start=100.0)
    backend = ScriptedBackend(lambda _call: "ok", duration_s=4.5, clock=clock)
    assert backend.complete(call()).duration_s == 4.5
    backend.complete(call())
    assert clock.now() == 109.0


def test_the_default_reflection_reply_has_three_why_lines():
    from fakes import reflection_reply

    assert reflection_reply("x").count("\n- ") == 3
