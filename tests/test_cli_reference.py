"""`autoimprover --examples` with a reference in every example (SPEC R4, R11, R22, R25; WP21): the
plan of `--dry` decides by the references (it prices the reference judge, picks on up to 6 of the
examples and holds out at least 4 in the checked tier when they fit, and says reference-scored),
the run is reference-scored, and a resumed run rebuilds the same plan from its run folder."""

import json

import pytest
from fakes import FakeClock, ScriptedBackend
from test_cli_fast import Cut, comparable, cutting, folders, run
from test_fast_world import BETTER, PROMPT, World, is_pairwise, no_disk_flush  # noqa: F401

from autoimprover import cli, reference_text
from autoimprover.cli_fast import plan_for, saved_fast_plan
from autoimprover.fastplan import Reference, fast_plan
from autoimprover.runner import count_tokens
from autoimprover.runstore import RunStore
from autoimprover.types import DEFAULT_MODELS, Plan, Scenario


def examples_file(tmp_path, n: int, reference: bool = True) -> str:
    path = tmp_path / "examples.jsonl"
    rows = [
        {"input": f"example {i}", **({"expected": f"reference {i}"} if reference else {})}
        for i in range(1, n + 1)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def test_dry_with_18_referenced_examples_at_2_minutes_picks_on_6_and_holds_out_4(tmp_path, capsys):
    path = examples_file(tmp_path, 18)
    done = run(capsys, "--dry", "--json", "--time", "2m", "--examples", path, PROMPT)
    plan = done.obj()
    assert (done.code, plan["tier"]) == (0, "checked")
    assert plan["scenarios"] >= 6 + 4 and plan["holdout"] >= 4
    assert "C: reference judge and contract checks" in [s["name"] for s in plan["stages"]]
    text = run(capsys, "--dry", "--time", "2m", "--examples", path, PROMPT).out
    assert f"evidence: {reference_text.EVIDENCE['checked']}" in text
    plain = run(
        capsys,
        "--dry",
        "--json",
        "--time",
        "2m",
        "--examples",
        examples_file(tmp_path, 18, False),
        PROMPT,
    )
    names = [stage["name"] for stage in plain.obj()["stages"]]
    assert "C: pairwise judge and contract checks" in names and plain.obj()["scenarios"] <= 8


def test_a_run_with_referenced_examples_is_reference_scored(tmp_path, capsys):
    done = run(capsys, "--json", "--examples", examples_file(tmp_path, 3), PROMPT)
    obj = done.obj()
    assert (obj["status"], obj["prompt"], obj["verified"]) == ("improved", BETTER, False)
    assert "reference-scored" in obj["reason"]
    assert obj["meaning"] == reference_text.MEANING_UNVERIFIED
    assert not [call for call in done.raw.calls if is_pairwise(call)]


def test_plan_for_reads_the_references_of_the_examples():
    with_refs = [Scenario(f"e{i}", "x", expected="y") for i in range(8)]
    tokens = count_tokens(PROMPT)
    assert plan_for(120, 6, PROMPT, with_refs) == fast_plan(120, 6, tokens, True, Reference(8, 1))
    mixed = [*with_refs, Scenario("e9", "x")]
    assert plan_for(120, 6, PROMPT, mixed) == fast_plan(120, 6, tokens, True)
    assert plan_for(120, 6, PROMPT, None) == fast_plan(120, 6, tokens, False)


def test_a_resumed_run_rebuilds_the_reference_plan_from_its_folder(tmp_path):
    plan = Plan(models=DEFAULT_MODELS, wall_clock_s=120, tier="checked")
    store = RunStore.open_or_create(tmp_path, plan, PROMPT)
    try:
        store.save_scenarios([Scenario(f"e{i}", "x", expected="y") for i in range(18)])
        rebuilt = saved_fast_plan(store, True)
    finally:
        store.close()
    assert rebuilt.reference == Reference(18, 1) and rebuilt.scenarios == 6


def test_a_cut_run_with_referenced_examples_resumes_to_the_same_calls(tmp_path, capsys):
    """The start path and `--resume` plan alike from the examples (SPEC R22): a run cut in stage
    A asks, after the resume, exactly the calls an uncut run asks. At 5 minutes on 1 worker the
    reference plan (K=1 on 6, 5 held out) is not the pairwise one (K=2 on 4, 3 held out)."""
    tokens = count_tokens(PROMPT)
    assert fast_plan(300, 1, tokens, True) != fast_plan(300, 1, tokens, True, Reference(18, 1))
    path = examples_file(tmp_path, 18)
    argv = ["--json", "--workers", "1", "--time", "5m", "--examples", path, PROMPT]
    whole = run(capsys, *argv)
    cli.main(["clean"])
    with pytest.raises(Cut):
        cli.main(argv, backend=ScriptedBackend(cutting(2)), now=FakeClock().now)
    [folder] = folders()
    resumed = run(capsys, "--json", "--resume", folder.name)
    assert comparable(resumed.obj()) == comparable(whole.obj())
    asked = sorted((c.role, c.sample, c.user) for c in whole.raw.calls)
    again = sorted((c.role, c.sample, c.user) for c in resumed.raw.calls)
    assert set(again) <= set(asked) and len(again) == len(asked) - 1  # the one before the cut


def test_the_world_of_these_tests_answers_the_reference_judge():
    """The default World passes a check of a GOOD output only: the original's answers fail the
    reference, the marked rewrite's pass it."""
    assert World().passes("e1", "s:expected", "GOOD answer")
    assert not World().passes("e1", "s:expected", "BAD answer")
