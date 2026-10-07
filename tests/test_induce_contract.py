"""The intent contract learns from the user's examples (SPEC R5, R6; ADR-013, WP22): with
references, one intake call also receives the pick examples, as data in its JSON user message, and
records `from_examples`, the labels, rules and formats they show and the prompt leaves unsaid; the
contract check takes them as part of the intent, so a requirement that is one of them passes
`no-new-goal` and any other new requirement is still vetoed; the run folder keeps them, and a
contract.json from before has none; the report shows them in one line, only when there are some."""

import json
from pathlib import Path

import pytest
from fakes import ScriptedBackend, by_role, intake_reply

from autoimprover import contract_text, report
from autoimprover.contract import check, check_many, extract_contract
from autoimprover.contract_text import INTAKE_SYSTEM
from autoimprover.runstore import RunStore, RunStoreError
from autoimprover.types import (
    DEFAULT_MODELS,
    INTAKE_SCHEMA,
    Call,
    CallFailed,
    Contract,
    Outcome,
    Plan,
    Scenario,
)

MODEL = "claude-sonnet-5-5"
PROMPT = "Assign a priority to this support ticket."
RULE = "Use P1 when the service is down for everyone."
UNRELATED = "Also translate the ticket into French."
EXAMPLES = (
    Scenario("e1", "Nobody at our company can log in since 9am.", expected="P1"),
    Scenario("e2", "Please send me last month's invoice again.", criteria=("names P4",)),
)
SENT = [
    {"input": "Nobody at our company can log in since 9am.", "expected": "P1", "criteria": []},
    {
        "input": "Please send me last month's invoice again.",
        "expected": None,
        "criteria": ["names P4"],
    },
]


# --- the Contract field (types.py, the one field WP22 adds) ---------------------------------------


def test_a_contract_has_no_rule_from_examples_by_default():
    assert Contract(goal="g", kind="task").from_examples == ()


@pytest.mark.parametrize(
    "rules",
    [tuple(f"rule {i}" for i in range(13)), ("x" * 201,), (3,)],
    ids=["13 rules", "201 characters", "not a string"],
)
def test_from_examples_holds_at_most_12_strings_of_at_most_200_characters(rules):
    with pytest.raises(ValueError, match="from_examples"):
        Contract(goal="g", kind="task", from_examples=rules)
    kept = Contract(goal="g", kind="task", from_examples=tuple("r" * 200 for _ in range(12)))
    assert len(kept.from_examples) == 12


# --- the intake with examples (SPEC R5; ADR-013) --------------------------------------------------


def test_the_intake_with_examples_sends_them_as_data_and_records_what_they_show():
    backend = by_role({"intake": intake_reply(from_examples=[RULE])})
    contract = extract_contract(backend, MODEL, PROMPT, examples=EXAMPLES)
    assert contract.from_examples == (RULE,)
    (call,) = backend.calls
    assert (call.role, call.model, call.sample) == ("intake", MODEL, 0)
    assert json.loads(call.user) == {"prompt": PROMPT, "examples": SENT}
    assert call.system == contract_text.INTAKE_EXAMPLES_SYSTEM
    assert not any(example.input in call.system for example in EXAMPLES)  # R18: stdin only
    schema = json.loads(call.json_schema or "{}")
    assert schema["required"] == [*INTAKE_SCHEMA["required"], "from_examples"]
    rules = schema["properties"]["from_examples"]
    assert (rules["maxItems"], rules["items"]["maxLength"]) == (12, 200)
    assert {k: v for k, v in schema["properties"].items() if k != "from_examples"} == (
        INTAKE_SCHEMA["properties"]
    )


def test_the_intake_with_examples_says_what_to_record_and_that_the_examples_are_data():
    system = contract_text.INTAKE_EXAMPLES_SYSTEM
    assert "from_examples" in system and "data, not instructions" in system
    for word in ("label", "rule", "decision", "format", "leaves unsaid", "never a copy"):
        assert word in system
    # the fields of the plain intake are described the same way
    for field in ("goal", "kind", "keep", "constraints", "output_format", "checks"):
        assert f"- {field}:" in system


def test_without_examples_the_intake_is_the_call_it_was():
    backend = by_role({"intake": intake_reply(from_examples=[RULE])})
    contract = extract_contract(backend, MODEL, PROMPT)
    (call,) = backend.calls
    assert (call.user, call.system) == (PROMPT, INTAKE_SYSTEM)
    assert json.loads(call.json_schema or "{}") == INTAKE_SCHEMA
    assert contract.from_examples == ()  # a key the plain schema does not name is ignored


def test_a_reply_without_from_examples_records_none():
    backend = by_role({"intake": intake_reply()})
    assert extract_contract(backend, MODEL, PROMPT, examples=EXAMPLES).from_examples == ()


@pytest.mark.parametrize(
    "bad",
    [[f"rule {i}" for i in range(13)], ["x" * 201], [""], "a rule", [7]],
    ids=["13 rules", "too long", "blank", "not a list", "not a string"],
)
def test_an_invalid_from_examples_is_asked_again_as_a_new_sample(bad):
    backend = by_role(
        {"intake": [intake_reply(from_examples=bad), intake_reply(from_examples=[RULE])]}
    )
    assert extract_contract(backend, MODEL, PROMPT, examples=EXAMPLES).from_examples == (RULE,)
    assert [call.sample for call in backend.calls] == [0, 1]
    always = by_role({"intake": intake_reply(from_examples=bad)})
    with pytest.raises(CallFailed):
        extract_contract(always, MODEL, PROMPT, examples=EXAMPLES)


# --- the meaning check takes them as intent (SPEC R6; ADR-013) ------------------------------------

GOAL = "assign a priority to a support ticket"


def learned() -> Contract:
    return Contract(goal=GOAL, kind="task", from_examples=(RULE,))


def reading_judge() -> ScriptedBackend:
    """A contract judge that reads the check texts: `no-new-goal` fails when the candidate holds
    RULE or UNRELATED and that sentence is not in the check's own text; every other check passes,
    each with a quote from the candidate."""

    def script(call: Call) -> str:
        results = []
        for item in json.loads(call.user)["scenarios"]:
            output = item["output"]
            checks = []
            for asked in item["checks"]:
                added = [s for s in (RULE, UNRELATED) if s in output]
                ok = asked["id"] != "no-new-goal" or all(s in asked["text"] for s in added)
                checks.append({"id": asked["id"], "pass": ok, "quote": output[:20]})
            results.append({"scenario": item["scenario"], "checks": checks})
        return json.dumps({"results": results})

    return ScriptedBackend(script)


@pytest.mark.parametrize(
    ("added", "vetoed"),
    [(RULE, False), (f"{RULE} {UNRELATED}", True), (UNRELATED, True)],
    ids=["learned rule", "learned rule and an unrelated one", "unrelated"],
)
def test_no_new_goal_passes_a_learned_rule_and_still_vetoes_any_other(added, vetoed):
    candidate = f"{PROMPT} {added}"
    violations = check(reading_judge(), MODEL, learned(), PROMPT, candidate)
    assert [v.check_id for v in violations] == (["no-new-goal"] if vetoed else [])
    found = check_many(reading_judge(), MODEL, learned(), PROMPT, [candidate, f"{PROMPT} {RULE}"])
    assert [v is not None for v in found] == [vetoed, False]


def test_the_learned_rules_are_in_the_no_new_goal_question_only_when_there_are_some():
    def asked(contract: Contract) -> str:
        backend = reading_judge()
        check(backend, MODEL, contract, PROMPT, f"{PROMPT} {RULE}")
        (call,) = backend.calls
        (item,) = json.loads(call.user)["scenarios"]
        return next(c["text"] for c in item["checks"] if c["id"] == "no-new-goal")

    question = asked(learned())
    assert RULE in question and "examples" in question and "NOT a new goal" in question
    plain = Contract(goal=GOAL, kind="task")
    assert "examples" not in asked(plain) and RULE not in asked(plain)


# --- the run folder (contract.json) ---------------------------------------------------------------

PLAN = Plan(models=DEFAULT_MODELS, wall_clock_s=30, tier="fast")


def test_contract_json_keeps_the_learned_rules_and_an_older_one_loads_without_them(tmp_path):
    root = tmp_path / "state" / "autoimprover" / "runs"
    store = RunStore.open_or_create(root, PLAN, PROMPT)
    store.save_contract(learned())
    run_id, path = store.run_id, store.path / "contract.json"
    store.close()
    saved = json.loads(path.read_text())
    assert saved["contract"]["from_examples"] == [RULE]
    resumed = RunStore.resume(root, run_id)
    assert resumed.contract() == learned()
    resumed.close()
    del saved["contract"]["from_examples"]  # a contract.json written before WP22
    path.write_text(json.dumps(saved))
    older = RunStore.resume(root, run_id)
    contract = older.contract()
    older.close()
    assert contract is not None and contract.from_examples == ()
    assert contract == Contract(goal=GOAL, kind="task")


def test_a_contract_json_with_too_many_learned_rules_is_damaged(tmp_path: Path):
    root = tmp_path / "state" / "autoimprover" / "runs"
    store = RunStore.open_or_create(root, PLAN, PROMPT)
    store.save_contract(learned())
    run_id, path = store.run_id, store.path / "contract.json"
    store.close()
    saved = json.loads(path.read_text())
    saved["contract"]["from_examples"] = [f"rule {i}" for i in range(13)]
    path.write_text(json.dumps(saved))
    with pytest.raises(RunStoreError, match="contract.json in the run folder .* is damaged"):
        RunStore.resume(root, run_id)


# --- the report (SPEC R2) -------------------------------------------------------------------------

OUTCOME = Outcome(
    status="unchanged",
    prompt=PROMPT,
    reason="tier fast: no rewrite",
    reason_code="no_reliable_improvement",
    mode="fast",
)


def test_the_report_shows_the_learned_rules_in_one_line_only_when_there_are_some():
    two = Contract(goal="g", kind="task", from_examples=(RULE, "Answer with the label only."))
    text = report.render(OUTCOME, PROMPT, two, PLAN)
    (line,) = [line for line in text.splitlines() if "learned from the examples" in line]
    assert line == f"  rules learned from the examples: {RULE}; Answer with the label only."
    plain = report.render(OUTCOME, PROMPT, Contract(goal="g", kind="task"), PLAN)
    assert "learned from the examples" not in plain
