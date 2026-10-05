"""Synthesis of a given number of scenarios (SPEC R11, R25; ADR-008): the fast tiers ask for the few
scenarios their plan affords, so `synthesize` takes `count`. The request, the schema's item limits
and the "exactly count items" rule follow it; the default stays SYNTH_COUNT with the very call
(and so the very cache key) the deep tier made before. The other synthesis tests are in
`test_scenarios.py` and `test_scenarios_text.py`.
"""

import json

import pytest
from fakes import by_role, synth_reply

from autoimprover.scenarios import synthesize
from autoimprover.types import SYNTH_COUNT, SYNTH_SCHEMA, CallFailed, Contract, Scenario

MODEL = "claude-sonnet-5-5"
PROMPT = "Summarise the meeting notes for the team in five bullet points."
CONTRACT = Contract(goal="summarise meeting notes", kind="task")


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_synthesize_asks_for_count_scenarios_and_returns_them(count):
    backend = by_role({"synth": synth_reply(count)})
    found = synthesize(backend, MODEL, PROMPT, CONTRACT, count=count)
    assert found == [Scenario(id=f"s{i}", input=f"situation {i}") for i in range(1, count + 1)]
    (call,) = backend.calls
    assert json.loads(call.user)["count"] == count
    schema = json.loads(call.json_schema or "")
    items = schema["properties"]["scenarios"]
    assert (items["minItems"], items["maxItems"]) == (count, count)


@pytest.mark.parametrize("given", [3, 5, SYNTH_COUNT])
def test_a_reply_with_another_number_of_scenarios_is_asked_again_then_fails(given):
    backend = by_role({"synth": synth_reply(given)})
    with pytest.raises(CallFailed, match="exactly 4 scenarios"):
        synthesize(backend, MODEL, PROMPT, CONTRACT, count=4)
    assert [call.sample for call in backend.calls] == [0, 1, 2]


def test_a_wrong_first_reply_is_followed_by_a_right_one_under_a_new_sample():
    backend = by_role({"synth": [synth_reply(SYNTH_COUNT), synth_reply(2)]})
    assert len(synthesize(backend, MODEL, PROMPT, CONTRACT, count=2)) == 2
    assert [call.sample for call in backend.calls] == [0, 1]


def test_the_default_count_makes_the_call_the_deep_tier_always_made():
    backend = by_role({"synth": synth_reply(SYNTH_COUNT)})
    assert len(synthesize(backend, MODEL, PROMPT, CONTRACT)) == SYNTH_COUNT
    (call,) = backend.calls
    assert call.json_schema == json.dumps(SYNTH_SCHEMA)
    assert json.loads(call.user)["count"] == SYNTH_COUNT


def test_a_count_does_not_change_the_shared_schema_constant():
    before = json.dumps(SYNTH_SCHEMA)
    synthesize(by_role({"synth": synth_reply(3)}), MODEL, PROMPT, CONTRACT, count=3)
    assert json.dumps(SYNTH_SCHEMA) == before


@pytest.mark.parametrize("count", [0, -1])
def test_a_count_below_one_is_refused_before_any_call(count):
    backend = by_role({"synth": synth_reply(1)})
    with pytest.raises(ValueError, match="at least 1"):
        synthesize(backend, MODEL, PROMPT, CONTRACT, count=count)
    assert backend.calls == []
