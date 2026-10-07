"""End to end, a representative pick set (SPEC R11, R22, R25; WP25). The prompt "Should this support
ticket be escalated?" hides a yes/no policy that only the references show, and the file lists the
'no' tickets first, as the bench's ex-escalate did. In this world the task model answers a ticket's
label only when the prompt it runs states RULE, and "no" otherwise; the reference judge passes an
output that is the label; the one rewrite states RULE.

Picking on the first examples of the file, the original passes every pick example, so no failure
shows and the original is kept (the twin, with the order of the examples taken away); ordered by
label, the pick holds both labels, the original fails the 'yes' ones and the rewrite is returned,
verified on the held-out examples in the checked tier. No held-out example reaches a generating
call or the contract check; a resumed run splits as the cut one did; a pick shorter than the
labels says so in the log."""

import dataclasses
import json

import pytest
from fakes import FakeClock, ScriptedBackend, intake_reply
from test_cli_fast import Cut, comparable, folders
from test_cli_fast import run as run_cli
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    CHECKED,
    MODELS,
    PLAN,
    World,
    is_contract_check,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
)

from autoimprover import cli, fast
from autoimprover.strata import ordered
from autoimprover.types import Call, Scenario

PROMPT = "Should this support ticket be escalated? Answer yes or no."
RULE = "Escalate (yes) a ticket that reports lost data or someone else in an account."
LEARNED = f"{PROMPT} {RULE}"
TICKETS = {
    "n1": ("The invoice PDF shows the wrong company logo.", "no"),
    "n2": ("How do I change the language of the dashboard?", "no"),
    "n3": ("Please add a dark mode to the mobile app.", "no"),
    "n4": ("The export button label has a typo in it.", "no"),
    "n5": ("Can I get a copy of last month's receipt?", "no"),
    "n6": ("The help page loads slowly in the evening.", "no"),
    "y1": ("Every file in our shared folder vanished overnight.", "yes"),
    "y2": ("Someone in another country logged into my account.", "yes"),
    "y3": ("Our customer records were deleted by the last update.", "yes"),
    "y4": ("I received a password reset mail for another customer.", "yes"),
}
LABEL_OF = dict(TICKETS.values())
GIVEN = [Scenario(sid, text, expected=label) for sid, (text, label) in TICKETS.items()]
# `--examples` names the examples of a file e1, e2, ... in file order.
LABEL_BY_ID = {
    **{sid: label for sid, (_text, label) in TICKETS.items()},
    **{f"e{n}": label for n, (_text, label) in enumerate(TICKETS.values(), start=1)},
}
BALANCED = dataclasses.replace(PLAN, strictness="balanced", budget=300)


def task(call: Call) -> str:
    return LABEL_OF[scenario_of(call)] if RULE in prompt_of(call) else "no"


def is_label(scenario: str, _check: str, output: str) -> bool:
    return output == LABEL_BY_ID[scenario]


def escalation() -> World:
    intake = intake_reply("task", from_examples=["yes for lost data"])
    return World(rewrites=(LEARNED,), task=task, passes=is_label, intake=intake)


def plan(tier: str, scenarios: int = 6, holdout: int = 0):
    """K=1, one generation, on `scenarios` pick examples and `holdout` held out: the runner reads
    only the shape."""
    shape = {"tier": tier, "scenarios": scenarios, "holdout": holdout, "time_s": 300}
    return dataclasses.replace(CHECKED, **shape)


def file_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The twin: the split takes the examples in file order, as before WP25."""
    monkeypatch.setattr(fast, "ordered", list)


def picked(result) -> set[str]:
    """The scenarios the reference judge scored on the task model (stage C)."""
    task_runs = {scenario_of(c) for c in result.calls("task") if c.model == MODELS.task}
    return {sid for sid, (text, _label) in TICKETS.items() if text in task_runs}


# --- the order exposes the failures -----------------------------------------------------------


def test_ordered_by_label_the_fast_pick_shows_the_failures_and_the_rewrite_is_returned(tmp_path):
    result = run(tmp_path, escalation(), plan("fast"), prompt=PROMPT, plan=BALANCED, examples=GIVEN)
    outcome = result.outcome
    assert picked(result) == {"n1", "y1", "n2", "y2", "n3", "y3"}
    assert (outcome.status, outcome.prompt, outcome.verified) == ("improved", LEARNED, False)
    assert (outcome.score_before, outcome.score_after) == (0.5, 1.0)


def test_its_twin_in_file_order_sees_no_failure_and_keeps_the_original(tmp_path, monkeypatch):
    file_order(monkeypatch)
    result = run(tmp_path, escalation(), plan("fast"), prompt=PROMPT, plan=BALANCED, examples=GIVEN)
    assert picked(result) == {"n1", "n2", "n3", "n4", "n5", "n6"}
    assert (result.outcome.status, result.outcome.prompt) == ("unchanged", PROMPT)


@pytest.mark.parametrize("ordering", [True, False], ids=["by label", "file order"])
def test_a_checked_run_verifies_the_rewrite_only_when_its_pick_held_both_labels(
    tmp_path, monkeypatch, ordering
):
    if not ordering:
        file_order(monkeypatch)
    fplan = plan("checked", holdout=3)
    result = run(tmp_path, escalation(), fplan, prompt=PROMPT, plan=BALANCED, examples=GIVEN)
    outcome = result.outcome
    assert (outcome.status == "improved", outcome.verified) == (ordering, ordering)
    assert outcome.prompt == (LEARNED if ordering else PROMPT)


# --- hygiene: only the pick examples are shown -----------------------------------------------


def test_no_held_out_example_reaches_a_generating_call_or_the_contract_check(tmp_path):
    """The checked run picks on the first 6 of the order (n1 y1 n2 y2 n3 y3) and holds out the
    next 3 (n4 y4 n5); n6 is not used. The intake, the rewrite and the contract check are shown
    the pick examples, in the order; a held-out example reaches only the task runs on the target
    model and their judge calls, and n6 no call at all."""
    order = ordered(GIVEN)
    pick, held, unused = order[:6], order[6:9], order[9:]
    assert [e.id for e in pick] == ["n1", "y1", "n2", "y2", "n3", "y3"]
    fplan = plan("checked", holdout=3)
    result = run(tmp_path, escalation(), fplan, prompt=PROMPT, plan=BALANCED, examples=GIVEN)
    assert result.outcome.verified
    for call in result.raw.calls:
        text = call.user + call.system
        assert not any(e.input in text for e in unused), (call.role, call.sample)
        if not any(e.input in text for e in held):
            continue
        stage_e = call.role == "task" and call.model == MODELS.target
        judged = call.role == "judge" and set(judged_scenarios(call)) <= {e.id for e in held}
        assert stage_e or judged, (call.role, call.sample)
    generating = [c for c in result.raw.calls if c.role in ("intake", "reflect")]
    checks = [c for c in result.raw.calls if is_contract_check(c)]
    assert len(generating) == 2 and len(checks) == 1
    for call in [*generating, *checks]:
        shown = [e["input"] for e in json.loads(call.user)["examples"]]
        assert shown == [e.input for e in pick], call.role


# --- resume -----------------------------------------------------------------------------------


def examples_file(tmp_path) -> str:
    path = tmp_path / "tickets.jsonl"
    rows = [{"input": e.input, "expected": e.expected} for e in GIVEN]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def cutting(at: int):
    world, seen = escalation(), []

    def script(call: Call) -> str | Exception:
        seen.append(call)
        if len(seen) == at:
            raise Cut
        return world(call)

    return script


@pytest.mark.parametrize("at", [2, 6, 12])
def test_a_resumed_run_splits_its_saved_examples_as_the_cut_run_did(tmp_path, capsys, at):
    """The run folder keeps the examples in file order and the resume orders them again: cut in
    stage A or stage B, it asks only calls the uncut run asked, and returns its result."""
    argv = ["--json", "--workers", "1", "--time", "5m", "--strictness", "balanced"]
    argv += ["--examples", examples_file(tmp_path), PROMPT]
    whole = run_cli(capsys, *argv, world=escalation())
    assert (whole.obj()["status"], whole.obj().get("prompt")) == ("improved", LEARNED), whole.out
    cli.main(["clean"])
    with pytest.raises(Cut):
        cli.main(argv, backend=ScriptedBackend(cutting(at)), now=FakeClock().now)
    capsys.readouterr()
    [folder] = folders()
    saved = json.loads((folder / "scenarios.json").read_text())
    assert [s["input"] for s in saved["scenarios"]] == [e.input for e in GIVEN]
    resumed = run_cli(capsys, "--json", "--resume", folder.name, world=escalation())
    assert comparable(resumed.obj()) == comparable(whole.obj())
    asked = sorted((c.role, c.sample, c.model, c.user) for c in whole.raw.calls)
    again = sorted((c.role, c.sample, c.model, c.user) for c in resumed.raw.calls)
    assert set(again) <= set(asked) and len(again) == len(asked) - (at - 1)


# --- a pick shorter than the labels ---------------------------------------------------------------

THREE = [
    Scenario("a1", "A blender for $70 arrived cracked.", expected="approve"),
    Scenario("a2", "A mug for $20 arrived chipped.", expected="approve"),
    Scenario("d1", "A jacket bought 45 days ago has a broken zipper.", expected="deny"),
    Scenario("e1", "A laptop for $1,200 has a flickering screen.", expected="escalate"),
]


@pytest.mark.parametrize(("scenarios", "said"), [(2, True), (3, False)])
def test_a_pick_that_misses_a_label_says_how_many_it_covers(tmp_path, scenarios, said):
    fplan = plan("fast", scenarios=scenarios)
    world = World(rewrites=(LEARNED,), task=lambda _call: "approve")
    result = run(tmp_path, world, fplan, prompt=PROMPT, plan=BALANCED, examples=THREE)
    assert ("fast: pick examples cover 2 of 3 labels" in result.log) is said
