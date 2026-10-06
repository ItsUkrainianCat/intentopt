"""`--ungated` on the command line, in the run folder and in the bench (SPEC R2, R4, R22, R25, R26;
WP20): the flag is off unless given, refused in the deep and quick tiers before any call, shown by
`--dry`, saved with the run so a resume stays as it began, labelled in the report and the JSON
object, and taken by `autoimprover bench`, whose summary then says its results are ungated and
tallies the returned prompts against the originals. The pick itself is in `test_ungated.py`.
Every test that shows a rewrite is NOT returned has a twin showing one IS."""

import json
from collections.abc import Callable

import pytest
from fakes import FakeClock, ScriptedBackend
from test_bench_cli import OTHER, one_object, prompt_set
from test_bench_cli import main as bench
from test_bench_world import ORIGINAL, BenchWorld
from test_cli_fast import Cut, folders, run
from test_fast_world import PROMPT, World, no_disk_flush  # noqa: F401  (an autouse fixture)

from autoimprover import cli
from autoimprover.cli_options import parse
from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END, Call

PLAIN = "Answer the request well."  # no marker: its answers tie the original's
LABEL = "ungated: the best-ranked candidate; no win over the original was shown"
HEAD = "a returned prompt is the best-ranked candidate, not gate-verified"  # the bench's


def test_the_flag_is_off_unless_given():
    assert parse([PROMPT]).ungated is False
    assert parse(["--ungated", PROMPT]).ungated is True


@pytest.mark.parametrize("flags", [["--deep"], ["--time", "10m"], ["--time", "15s"]])
def test_ungated_is_refused_in_the_deep_and_quick_tiers_before_any_call(capsys, flags):
    done = run(capsys, "--ungated", *flags, PROMPT)
    assert (done.code, done.out, done.raw.calls, folders()) == (2, "", [], [])
    assert done.err.startswith("error: --ungated applies to the fast and checked tiers only")
    assert ("--deep" if flags == ["--deep"] else f"--time {flags[1]}") in done.err


@pytest.mark.parametrize("time", ["30s", "2m"])
def test_its_twin_in_the_fast_and_checked_tiers_plans_and_says_so(capsys, time):
    done = run(capsys, "--dry", "--ungated", "--time", time, PROMPT)
    assert (done.code, done.raw.calls) == (0, [])
    assert "evidence: ungated (--ungated)" in done.out
    assert run(capsys, "--dry", "--json", "--ungated", "--time", time, PROMPT).obj()["ungated"]
    plain = run(capsys, "--dry", "--json", "--time", time, PROMPT).obj()
    assert plain["ungated"] is False
    assert "--ungated" not in run(capsys, "--dry", "--time", time, PROMPT).out


# --- the result (SPEC R2) -------------------------------------------------------------------------


def test_an_ungated_result_says_so_in_its_json_and_its_notices(capsys):
    done = run(capsys, "--json", "--ungated", PROMPT, world=World(rewrites=(PLAIN,)))
    obj = done.obj()
    assert (done.code, obj["status"], obj["prompt"], obj["verified"]) == (
        0,
        "improved",
        PLAIN,
        False,
    )
    assert (obj["reason_code"], obj["reason"]) == ("ungated_best_candidate", f"tier fast: {LABEL}")
    assert "--ungated" in obj["meaning"] and "no win over the original was shown" in obj["meaning"]
    assert obj["verified_text"] == f"no. NOT VERIFIED (tier fast: {LABEL})"
    assert obj["margin_text"].endswith("; ungated, so no lead was required")
    assert f"notice: NOT VERIFIED (tier fast: {LABEL})" in done.err.splitlines()
    twin = run(capsys, "--json", PROMPT, world=World(rewrites=(PLAIN,))).obj()
    assert (twin["status"], twin["reason_code"]) == ("unchanged", "no_reliable_improvement")


def test_without_json_the_report_names_the_code_the_label_and_the_shares(capsys):
    done = run(capsys, "--ungated", PROMPT, world=World(rewrites=(PLAIN,)))
    assert (done.code, done.out) == (0, PLAIN + "\n")
    lines = done.err.splitlines()
    assert "result: improved (ungated_best_candidate)" in lines
    assert f"verified: no. NOT VERIFIED (tier fast: {LABEL})" in lines
    assert any(line.startswith("preference on the scenarios it was picked on") for line in lines)
    assert any(line.startswith("margin: lead 0.00 vs noise 0.00 on the") for line in lines)
    assert not any("had to lead by more than" in line for line in lines)


# --- the run folder (SPEC R22) --------------------------------------------------------------------


def cutting(at: int) -> Callable[[Call], str | Exception]:
    world, seen = World(rewrites=(PLAIN,)), []

    def script(call: Call) -> str | Exception:
        seen.append(call)
        if len(seen) == at:
            raise Cut
        return world(call)

    return script


@pytest.mark.parametrize("ungated", [False, True])
def test_a_resumed_run_stays_as_it_began_whatever_the_resume_line_says(capsys, ungated):
    argv = ["--json", *(["--ungated"] if ungated else []), PROMPT]
    with pytest.raises(Cut):
        cli.main(argv, backend=ScriptedBackend(cutting(4)), now=FakeClock().now)
    [folder] = folders()
    assert json.loads((folder / "manifest.json").read_text())["opts"]["ungated"] is ungated
    again = ["--json", "--resume", folder.name, "--ungated"]
    resumed = run(capsys, *again, world=World(rewrites=(PLAIN,)))
    code = "ungated_best_candidate" if ungated else "no_reliable_improvement"
    assert (resumed.code, resumed.obj()["reason_code"]) == (0, code)
    assert "ignoring --ungated" in resumed.err


def test_a_manifest_without_the_flag_resumes_gated_and_a_damaged_one_is_refused(capsys):
    with pytest.raises(Cut):
        argv = ["--json", "--ungated", PROMPT]
        cli.main(argv, backend=ScriptedBackend(cutting(1)), now=FakeClock().now)
    [folder] = folders()
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["opts"]["ungated"] = "yes"
    path.write_text(json.dumps(manifest))
    refused = run(capsys, "--resume", folder.name)
    assert (refused.code, refused.out, refused.raw.calls) == (2, "", [])
    assert "manifest.json" in refused.err and "damaged" in refused.err
    del manifest["opts"]["ungated"]  # as a run from before the flag saved it
    path.write_text(json.dumps(manifest))
    resumed = run(capsys, "--json", "--resume", folder.name, world=World(rewrites=(PLAIN,)))
    assert (resumed.obj()["status"], resumed.obj()["reason_code"]) == (
        "unchanged",
        "no_reliable_improvement",
    )


# --- the bench (SPEC R26) -------------------------------------------------------------------------


def test_the_bench_takes_the_flag_and_its_plan_says_so(capsys):
    code, out, err, raw = bench(capsys, ["--dry", "--limit", "1", "--ungated"])
    assert (code, err, raw.calls) == (0, "", [])
    # the plan's first line, after "dry run: ..." (its path may hold the word too)
    assert out.splitlines()[1].endswith(", ungated (--ungated): " + HEAD)
    code, out, _err, _raw = bench(capsys, ["--dry", "--json", "--limit", "1", "--ungated"])
    assert (code, one_object(out)["ungated"]) == (0, True)
    code, out, _err, _raw = bench(capsys, ["--dry", "--json", "--limit", "1"])
    assert (code, one_object(out)["ungated"]) == (0, False)
    assert "--ungated" not in bench(capsys, ["--dry", "--limit", "1"])[1]  # the path may say it


@pytest.mark.parametrize("time", ["20m", "15s"])
def test_the_bench_refuses_the_flag_in_the_deep_and_quick_tiers(capsys, time):
    code, out, err, raw = bench(capsys, ["--dry", "--limit", "1", "--ungated", "--time", time])
    assert (code, out, raw.calls) == (2, "", [])
    assert err.startswith("error: --ungated applies to the fast and checked tiers only")


TRIP = ORIGINAL.removesuffix(".") + ", please."
NOTE = OTHER.removesuffix(".") + ", please."


def mixed(call: Call) -> str | Exception:
    """Every run's answers tie, so no rewrite wins there; on the bench's fresh scenarios TRIP beats
    its original and NOTE loses to its own."""
    if call.role == "reflect" and not call.system.startswith("Improve this prompt."):
        text = TRIP if ORIGINAL in call.user else NOTE
        return f"{INSTRUCTION_BEGIN}\n{text}\n{INSTRUCTION_END}"
    if call.role == "task":
        if "120 words" in call.user:  # a scoring run inside a run
            return "BAD answer"
        return "GOOD answer" if call.user.split("\n\n", 1)[1] in (TRIP, OTHER) else "BAD answer"
    return BenchWorld()(call)


@pytest.mark.parametrize("ungated", [False, True])
def test_an_ungated_bench_compares_the_best_ranked_rewrites_and_tallies_them(
    capsys, tmp_path, ungated
):
    path = prompt_set(tmp_path, ("trip", ORIGINAL), ("note", OTHER))
    argv = ["--prompts", path, *(["--ungated"] if ungated else [])]
    code, out, _err, _raw = bench(capsys, ["--json", *argv], ScriptedBackend(mixed))
    found = one_object(out)
    rows = [(r["status"], r["reason_code"], r["verified"], r["verdict"]) for r in found["rows"]]
    if not ungated:
        assert rows == [("unchanged", "no_reliable_improvement", False, "tie")] * 2
        assert (found["ungated"], found["ungated_tally"]) == (False, None)
        return
    code_ = "ungated_best_candidate"
    assert rows == [("improved", code_, False, "win"), ("improved", code_, False, "loss")]
    assert (found["improved"], found["wins"], found["ties"], found["losses"]) == (2, 1, 0, 1)
    assert found["ungated"] is True
    assert found["ungated_tally"] == {
        "rows": 2,
        "wins": 1,
        "ties": 0,
        "losses": 1,
        "decided": 2,
        "win_share_of_decided": 0.5,
    }
    code, out, _err, _raw = bench(capsys, argv, ScriptedBackend(mixed))
    lines = out.splitlines()
    assert code == 0 and lines[0].endswith(", ungated (--ungated): " + HEAD)
    assert (
        "ungated: the returned prompts against the original over the 2 improved rows compared: "
        "1 win, 0 ties, 1 loss; wins among the decided: 50% (1 of 2)"
    ) in lines
