"""`Evaluator.score` (SPEC R10, R10b, R11, R24, R25): the judging half of the evaluator for outputs
a stage already has, so the fast tiers' reference scoring (WP21) and the bench's hidden examples
use the very checks, calls and quote rule of a full evaluation."""

import json

from fakes import ScriptedBackend, judge_reply

from autoimprover.evaluator import Evaluator
from autoimprover.types import CallFailed, Contract, Scenario

SCENARIOS = [
    Scenario("e1", "ticket one", expected="P1"),
    Scenario("e2", "ticket two", expected="P3", criteria=("names the priority",)),
]
BARE = Contract(goal="g", kind="task")


def by_output(call):
    if call.role == "task":
        return f"GOOD answer to {call.user}"
    return judge_reply(call, lambda _s, _c, output: output.startswith("GOOD"))


def test_score_judges_given_outputs_as_a_full_evaluation_would_and_runs_no_task():
    full_raw, raw = ScriptedBackend(by_output), ScriptedBackend(by_output)
    full = Evaluator(full_raw, BARE, "task-m", "judge-m")("the prompt", SCENARIOS)
    outputs = {s.id: f"GOOD answer to {s.input}\n\nthe prompt" for s in SCENARIOS}
    found = Evaluator(raw, BARE, "", "judge-m").score(SCENARIOS, outputs)
    assert found == full
    assert [call.role for call in raw.calls] == ["judge"]
    sent = json.loads(raw.calls[0].user)["scenarios"]
    assert [[check["id"] for check in item["checks"]] for item in sent] == [
        ["s:expected"],
        ["s:crit-1", "s:expected"],
    ]


def test_a_failed_output_is_an_incomplete_entry_and_is_not_judged():
    raw = ScriptedBackend(by_output)
    outputs = {"e1": CallFailed("task call failed"), "e2": "GOOD x"}
    found = Evaluator(raw, BARE, "", "judge-m", sample=7).score(SCENARIOS, outputs)
    assert found[0] == (0.0, {"incomplete": True, "error": "task call failed", "scenario": "e1"})
    assert found[1][0] == 1.0
    assert [item["scenario"] for item in json.loads(raw.calls[0].user)["scenarios"]] == ["e2"]
    assert raw.calls[0].sample == 7
