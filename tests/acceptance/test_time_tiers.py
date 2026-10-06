"""Acceptance tests for SPEC R25 (time is the knob) with R2 (labels, exit codes), R4 (`--dry`),
R17 (the hard clock), R21 (cancel) and R22 (resume), written from SPEC R25 and ADR-011: `--time`
picks the tier (quick 15-24 s, fast 25-59 s and the default 30 s, checked 1-9 min, deep from
10 min, `--deep` = 20 min), below 15 s the run is refused, `--dry` shows the tier, the stages,
the calls and the estimated seconds; a quick or fast result is labelled not verified and only
checked and deep results are verified; `mode` names the tier in the JSON.

The fast tiers' contract check judges every rewrite in one call, one scenario `contract-k` each
(ADR-011 decision 3), so the judge here passes those as `happy_backend`'s passes `contract`. A
synthesis call returns as many scenarios as the plan says (`--dry` tells how many). Every test
that shows a rewrite is NOT returned has a twin on the same model showing one IS."""

import json
from collections.abc import Callable

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend, happy_backend, judge_reply

from autoimprover import cli
from autoimprover.types import Call

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."
PLAIN = f"{ORIGINAL} Keep each bullet short."  # a rewrite that runs no better
FAST_LABEL = "not verified on held-out scenarios"


@pytest.fixture
def run_bare(capsys) -> Callable:
    """`cli.main` exactly as given (the shared `run_cli` makes a run without `--time` deep)."""

    def run(argv, backend=None, clock=None):
        backend = backend or ScriptedBackend(lambda call: AssertionError(f"{call.role} call"))
        capsys.readouterr()
        code = cli.main(list(argv), backend=backend, now=(clock or FakeClock()).now)
        out, err = capsys.readouterr()
        return code, out, err, backend

    return run


def dry_plan(run_bare, *argv: str) -> dict:
    code, out, err, backend = run_bare(["--dry", "--json", *argv, ORIGINAL])
    assert (code, err, backend.calls) == (0, "", [])
    return json.loads(out)


def model(override, run_bare, proposal: str, *argv: str) -> ScriptedBackend:
    """`happy_backend` proposing `proposal`, synthesising the plan's scenario count, its judge
    passing the contract check of every rewrite."""
    n = dry_plan(run_bare, *argv)["scenarios"] or 1

    def judge(call: Call) -> str:
        return judge_reply(
            call,
            lambda scenario, _c, output: (
                scenario.startswith("contract") or output.startswith("GOOD")
            ),
        )

    return override(happy_backend(proposal, n=n), judge=judge)


# --- the tiers (SPEC R25) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "tier", "seconds"),
    [
        ([], "fast", 30),
        (["--time", "15s"], "quick", 15),
        (["--time", "24s"], "quick", 24),
        (["--time", "25s"], "fast", 25),
        (["--time", "59s"], "fast", 59),
        (["--time", "1m"], "checked", 60),
        (["--time", "9m"], "checked", 540),
        (["--time", "10m"], "deep", 600),
        (["--deep"], "deep", 1200),
    ],
)
def test_the_time_picks_the_tier_and_is_the_runs_clock(run_bare, argv, tier, seconds):
    plan = dry_plan(run_bare, *argv)
    assert (plan["tier"], plan["plan"]["tier"], plan["plan"]["wall_clock_s"]) == (
        tier,
        tier,
        seconds,
    )


@pytest.mark.parametrize("dry", [[], ["--dry"]], ids=["run", "dry"])
def test_below_15_seconds_a_run_is_refused_naming_the_flag(run_bare, files_under, tmp_path, dry):
    before = files_under(tmp_path)
    code, out, err, backend = run_bare([*dry, "--time", "14s", ORIGINAL])
    assert (code, out, backend.calls) == (2, "", [])
    assert err.startswith("error: ") and "--time" in err and "15" in err
    assert files_under(tmp_path) == before


def test_a_deep_only_flag_with_a_shorter_tier_is_a_usage_error(run_bare):
    code, out, err, _ = run_bare(["--trust-search", ORIGINAL])
    assert (code, out) == (2, "") and "--trust-search" in err


# --- --dry (SPEC R4, R25) ---------------------------------------------------------------------


def test_dry_shows_the_tier_the_stages_the_calls_and_the_seconds(run_bare, files_under, tmp_path):
    before = files_under(tmp_path)
    plan = dry_plan(run_bare)
    assert files_under(tmp_path) == before
    assert (plan["tier"], plan["workers"], plan["refusal"]) == ("fast", 6, None)
    assert plan["efforts"] == {"task": "low", "judge": "low", "reflect": "low"}
    assert plan["est_calls"] == sum(stage["calls"] for stage in plan["stages"]) > 0
    assert plan["est_seconds"] == pytest.approx(sum(stage["seconds"] for stage in plan["stages"]))
    assert 0 < plan["est_seconds"] <= 30
    assert plan["rewrites"] >= 1 and plan["scenarios"] >= 1
    assert plan["plan"]["budget"] >= plan["est_calls"]
    models = plan["plan"]["models"]
    assert (models["task"], models["reflect"], models["judge"]) == (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-5-5",
        "claude-opus-5-5",
    )
    code, text, _, _ = run_bare(["--dry", ORIGINAL])
    assert code == 0 and "tier: fast" in text and f"{plan['est_calls']} calls" in text


def test_dry_says_a_real_run_would_refuse_a_plan_longer_than_its_time(run_bare):
    plan = dry_plan(run_bare, "--workers", "1")
    assert plan["est_seconds"] > 30 and "--time" in plan["refusal"]
    code, out, err, backend = run_bare(["--workers", "1", ORIGINAL])
    assert (code, out, backend.calls) == (2, "", []) and "--time" in err


# --- what each tier returns, and how it is labelled (SPEC R2, R25) -------------------------------


@pytest.mark.parametrize("proposal", [IMPROVED, PLAIN], ids=["improves", "twin-no-better"])
def test_a_fast_run_returns_a_rewrite_only_when_it_scores_better_and_labels_it(
    run_bare, override, proposal
):
    code, out, err, _ = run_bare(["--json", ORIGINAL], model(override, run_bare, proposal))
    obj = json.loads(out)
    assert code == 0 and (obj["mode"], obj["verified"]) == ("fast", False)
    if proposal == IMPROVED:
        assert (obj["status"], obj["prompt"]) == ("improved", IMPROVED)
        assert FAST_LABEL in obj["reason"] and "fast check" in obj["reason"]
        assert any("NOT VERIFIED" in line for line in err.splitlines())
    else:
        assert (obj["status"], obj["prompt"]) == ("unchanged", ORIGINAL)


def test_a_fast_result_never_reads_as_verified_in_the_human_report(run_bare, override):
    code, out, err, _ = run_bare([ORIGINAL], model(override, run_bare, IMPROVED))
    assert (code, out) == (0, IMPROVED + "\n")
    assert "verified: yes" not in err and "NOT VERIFIED" in err and "fast check" in err
    assert "mode: fast" in err and "holdout" not in err


def test_the_quick_tier_labels_its_result_as_a_quick_check(run_bare, override):
    argv = ["--json", "--time", "15s", ORIGINAL]
    obj = json.loads(run_bare(argv, model(override, run_bare, IMPROVED, "--time", "15s"))[1])
    assert (obj["status"], obj["mode"], obj["verified"]) == ("improved", "quick", False)
    assert "quick check" in obj["reason"] and "no noise measured" in obj["reason"]


@pytest.mark.parametrize("wins_on_target", [True, False], ids=["verified", "twin-loses"])
def test_a_checked_result_is_verified_on_held_out_scenarios_on_the_target_model(
    run_bare, override, wins_on_target
):
    inner = model(override, run_bare, IMPROVED, "--time", "2m")

    def task(call: Call) -> str:
        good = MARKER in call.system + call.user
        on_target = call.model == "claude-sonnet-5-5"
        return "GOOD answer" if good and (wins_on_target or not on_target) else "BAD answer"

    obj = json.loads(run_bare(["--json", "--time", "2m", ORIGINAL], override(inner, task=task))[1])
    assert obj["mode"] == "checked"
    if wins_on_target:
        assert (obj["status"], obj["prompt"], obj["verified"]) == ("improved", IMPROVED, True)
    else:
        assert (obj["status"], obj["prompt"]) == ("unchanged", ORIGINAL)


def test_a_deep_run_names_its_tier_too(run_cli):
    obj = run_cli(["--json", "--deep", ORIGINAL], happy_backend(IMPROVED)).json()
    assert (obj["status"], obj["mode"], obj["verified"]) == ("improved", "deep", True)


def test_the_stages_report_their_progress_on_stderr_while_stdout_holds_one_object(
    run_bare, override
):
    code, out, err, _ = run_bare(["--json", ORIGINAL], model(override, run_bare, IMPROVED))
    assert code == 0 and len(out.splitlines()) == 1
    stages = [line for line in err.splitlines() if "stage" in line and line.startswith("fast")]
    assert len(stages) >= 3


# --- the clock, cancel and resume (SPEC R17, R21, R22) -----------------------------------------


@pytest.mark.parametrize("seconds", [100.0, 1.0], ids=["out-of-time", "in-time"])
def test_no_call_starts_after_the_time_and_the_original_is_kept_when_it_runs_out(
    run_cli, override, run_bare, timed, seconds
):
    """One call at a time, so the fake clock moves as one: 9 minutes on 1 worker."""
    argv = ["--time", "9m", "--workers", "1"]
    clock = FakeClock()
    backend = timed(model(override, run_bare, IMPROVED, *argv), clock, seconds)
    obj = run_cli(["--json", *argv, ORIGINAL], backend, clock).json()
    assert max(backend.starts) < 540
    if seconds == 100.0:
        assert (obj["prompt"], obj["reason_code"], obj["stop"]) == (
            ORIGINAL,
            "unconfirmed_out_of_budget",
            "clock",
        )
    else:
        assert (obj["prompt"], obj["status"]) == (IMPROVED, "improved")


def test_an_interrupted_fast_run_is_130_and_resumes_to_the_same_result(
    run_bare, run_cli, override, cut_at
):
    reference = json.loads(run_bare(["--json", ORIGINAL], model(override, run_bare, IMPROVED))[1])
    code, out, err, _ = run_bare([ORIGINAL], cut_at(model(override, run_bare, IMPROVED), 4))
    assert (code, out) == (130, "") and "error: interrupted" in err
    run_id = err.split("autoimprover --resume ")[1].split()[0]
    resumed = run_cli(["--json", "--resume", run_id], model(override, run_bare, IMPROVED)).json()
    keep = ("status", "prompt", "mode", "verified", "reason_code")
    assert {k: resumed[k] for k in keep} == {k: reference[k] for k in keep}
