"""Acceptance tests for SPEC R4 (`--dry`, the low-budget floor) and R17 (one call budget, fixed
costs reserved up front, hard limit, cache hits free, the 45-minute clock and its 75 % share).

Expected numbers are worked out by hand from SPEC R15 and R17 (split sizes, fixed costs, iteration
cost with the minibatch of 3 from ADR-006, a judge call per at most 6 scenarios), not read from the
tool. The `--dry --json` key names are the tool's own: SPEC R4 fixes what the plan shows, not the
names, so they appear only in PLAN_KEYS below.
"""

import json

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend, happy_backend, judge_reply

from autoimprover.types import Call, CallError

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."
MODELS = {  # SPEC R14 defaults
    "task": "claude-haiku-4-5-20251001",
    "judge": "claude-opus-5-5",
    "reflect": "claude-opus-5-5",
    "target": "claude-sonnet-5-5",
}
WALL_S, SEARCH_S = 45 * 60, 0.75 * 45 * 60  # SPEC R17
PLAN_KEYS = {  # meaning -> key in the `--dry --json` object
    "before": "calls_before_search",
    "after": "calls_after_search",
    "iterations": "iterations",
    "scenarios": "scenarios",
    "holdout": "holdout",
    "valset": "valset",
    "dataset": "dataset",
    "final_clock_s": "final_clock_s",
}
# (extra argv, examples to write or None for synthesis) -> expected plan at budget 100.
# n=12 synthesised: holdout 4, valset 3, dataset 5; before 1+1+2*(4+1)+(3+1)=16, after
# 3*(4+1)+3=18, iteration 1+2*(3+1)+(3+1)=13, (100-34)//13=5.
PLANS = {
    "synth-12": ([], None, dict(before=16, after=18, iterations=5, h=4, v=3, d=5, n=12)),
    # n=8: 3/2/3; before 1+2*(3+1)+(2+1)=12, after 3*4+3=15, iteration 1+8+3=12, 73//12=6
    "examples-8": ([], 8, dict(before=12, after=15, iterations=6, h=3, v=2, d=3, n=8)),
    # n=10: 4/3/3; before 1+2*5+4=15, after 18, iteration 13, 67//13=5
    "examples-10": ([], 10, dict(before=15, after=18, iterations=5, h=4, v=3, d=3, n=10)),
    # n=40: 6/4/30 (holdout capped at 6); before 1+2*7+5=20, after 3*7+3=24, iteration 14, 4
    "examples-40": ([], 40, dict(before=20, after=24, iterations=4, h=6, v=4, d=30, n=40)),
    # n=7 with --trust-search: no holdout, dataset = valset = 7 (2 judge calls); before 1+(7+2)=10,
    # after 3 contract checks, iteration 1+2*(3+1)+9=18, 87//18=4
    "trust-7": (["--trust-search"], 7, dict(before=10, after=3, iterations=4, h=0, v=7, d=7, n=7)),
}


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_dry_makes_no_call_writes_nothing_and_exits_0(
    run_cli, tmp_path, state_home, files_under, as_json
):
    """R4: --dry prints the plan (the four models, the budget), makes zero model calls, writes
    nothing (no run folder, no file anywhere) and exits 0; stderr stays empty."""
    backend = happy_backend(IMPROVED)
    before = files_under(tmp_path)
    r = run_cli((["--json"] if as_json else []) + ["--dry", ORIGINAL], backend)
    assert r.code == 0, r.err
    assert backend.calls == []
    assert files_under(tmp_path) == before
    assert not (state_home / "autoimprover").exists()
    assert r.err == ""
    for model in MODELS.values():
        assert model in r.out
    assert "100" in r.out
    assert "a real run would refuse" not in r.out
    if as_json:
        plan = r.json()
        assert plan["plan"]["models"] == MODELS
        assert plan["plan"]["budget"] == 100


@pytest.mark.parametrize("case", list(PLANS))
def test_dry_plan_shows_fixed_costs_split_and_iterations(run_cli, examples, case):
    """R4, R15, R17: the plan shows the fixed costs before and after the search, the scenario count
    (and the split), the estimated iterations the budget affords, and the clock share kept for the
    final steps (25 % of 45 minutes)."""
    extra, n, want = PLANS[case]
    argv = extra + (["--examples", examples(n)] if n else [])
    r = run_cli(["--dry", "--json", *argv, ORIGINAL])
    assert r.code == 0, r.err
    plan = r.json()
    got = {meaning: plan[key] for meaning, key in PLAN_KEYS.items()}
    assert got["before"] == want["before"]
    assert got["after"] == want["after"]
    assert got["iterations"] == want["iterations"]
    assert got["scenarios"] == want["n"]
    assert (got["holdout"], got["valset"], got["dataset"]) == (want["h"], want["v"], want["d"])
    assert got["final_clock_s"] == pytest.approx(WALL_S - SEARCH_S)


def test_dry_target_opus_switches_to_the_fallback_judge(run_cli):
    """R14: when the target is the default judge (Opus 5.5), the judge falls back to Sonnet 5.5."""
    r = run_cli(["--dry", "--json", "--target-model", "opus", ORIGINAL])
    assert r.code == 0, r.err
    models = r.json()["plan"]["models"]
    assert models["target"] == "claude-opus-5-5"
    assert models["judge"] == "claude-sonnet-5-5"


def _git_state(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("XDG_STATE_HOME", str(repo / "state"))


def _read_only_state(tmp_path, monkeypatch):
    folder = tmp_path / "read-only"
    folder.mkdir()
    folder.chmod(0o500)
    monkeypatch.setenv("XDG_STATE_HOME", str(folder))


REFUSALS = {  # name -> (extra argv, environment change, words the reason must name)
    "fixed-costs": (["--budget", "30"], None, "--budget"),
    "fewer-than-4-iterations": (["--budget", "85"], None, "--force-low-budget"),
    "state-in-git": ([], _git_state, "XDG_STATE_HOME"),
    "state-not-writable": ([], _read_only_state, "XDG_STATE_HOME"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_dry_exits_0_and_says_a_real_run_would_refuse(
    run_cli, tmp_path, monkeypatch, files_under, case
):
    """R4, R23: --dry exits 0 once the arguments parse, even when a real run would refuse, and says
    "a real run would refuse: <reason>"; zero calls and nothing written."""
    extra, env, names = REFUSALS[case]
    if env:
        env(tmp_path, monkeypatch)
    backend = happy_backend(IMPROVED)
    before = files_under(tmp_path)
    try:
        r = run_cli(["--dry", *extra, ORIGINAL], backend)
    finally:
        if (tmp_path / "read-only").exists():
            (tmp_path / "read-only").chmod(0o700)
    assert r.code == 0, r.err
    assert backend.calls == []
    assert files_under(tmp_path) == before
    line = next(x for x in r.out.splitlines() if "a real run would refuse: " in x)
    assert names in line


@pytest.mark.parametrize("case", list(REFUSALS))
def test_real_run_refuses_with_exit_2_before_any_call(
    run_cli, tmp_path, monkeypatch, files_under, case
):
    """R2, R4, R17, R23: the same conditions refuse a real run with exit 2 before any paid call
    (intake included), name the flag or folder to change, and leave no file behind."""
    extra, env, names = REFUSALS[case]
    if env:
        env(tmp_path, monkeypatch)
    backend = happy_backend(IMPROVED)
    before = files_under(tmp_path)
    try:
        r = run_cli([*extra, ORIGINAL], backend)
    finally:
        if (tmp_path / "read-only").exists():
            (tmp_path / "read-only").chmod(0o700)
    assert r.code == 2, r.err
    assert backend.calls == []
    assert r.out == ""
    assert files_under(tmp_path) == before
    error = next(x for x in r.err.splitlines() if x.startswith("error: "))
    assert names in error


def test_dry_usage_error_is_still_exit_2(run_cli):
    """R4: a usage error is exit 2 even with --dry."""
    r = run_cli(["--dry", "--budget", "301", ORIGINAL])
    assert r.code == 2
    assert r.out == ""


def test_low_budget_runs_with_force_and_stays_inside_the_budget(run_cli):
    """R4 twin of the refusal: --force-low-budget lets 3 iterations run, within the 85 calls."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--json", "--budget", "85", "--force-low-budget", ORIGINAL], backend)
    assert r.code == 0, r.err
    assert 0 < len(backend.calls) <= 85
    assert r.json()["calls_used"] == len(backend.calls)


def test_four_iterations_run_without_force(run_cli):
    """R4 boundary: a budget of 86 affords exactly 4 worst-case iterations, so no refusal; it is
    also the twin of the zero-search run below: with calls left for the search, a candidate wins."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--budget", "86", ORIGINAL], backend)
    assert r.code == 0, r.err
    assert 0 < len(backend.calls) <= 86
    assert backend.count("reflect") > 0
    assert r.out == IMPROVED + "\n"


def test_fixed_costs_above_budget_refuse_even_with_force(run_cli):
    """R17: when the fixed costs alone (34 at 12 scenarios) exceed the budget the run refuses
    (exit 2) before the first paid call, --force-low-budget or not."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--budget", "33", "--force-low-budget", ORIGINAL], backend)
    assert r.code == 2
    assert backend.calls == []
    assert "--budget" in next(x for x in r.err.splitlines() if x.startswith("error: "))


def test_fixed_costs_equal_to_budget_run_with_force(run_cli):
    """R17 twin: fixed costs equal to the budget do not exceed it; the run starts and never spends
    more than the 34 calls. No call is left for one iteration, so the search stops before its first
    reflection and the original is returned (a budget stop, not a failure)."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--json", "--budget", "34", "--force-low-budget", ORIGINAL], backend)
    assert r.code == 0, r.err
    obj = r.json()
    assert 0 < len(backend.calls) <= 34
    assert obj["calls_used"] == len(backend.calls)
    assert backend.count("reflect") == 0
    assert obj["prompt"] == ORIGINAL
    assert obj["stop"] == "budget"


@pytest.mark.parametrize("budget", [86, 100, 173, 300])
def test_every_call_counts_and_the_budget_is_a_hard_limit(run_cli, budget):
    """R17: every call (intake, synthesis, task, judge, reflection) counts toward one budget and the
    run never makes more; identical calls are cached, so none is made twice and a hit is not
    counted (calls_used equals the live calls)."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--json", "--budget", str(budget), ORIGINAL], backend)
    assert r.code == 0, r.err
    obj = r.json()
    assert len(backend.calls) <= budget
    assert obj["calls_used"] == len(backend.calls)
    assert len(set(backend.calls)) == len(backend.calls)
    assert {c.role for c in backend.calls} == {"intake", "synth", "task", "judge", "reflect"}
    assert obj["status"] == "improved"


def test_retries_count_toward_the_budget_and_never_exceed_it(run_cli):
    """R17, R24: each retry is a paid call; with every call failing once before it answers, the run
    still spends at most the budget and counts every attempt."""
    inner = happy_backend(IMPROVED)
    failed_once: set[Call] = set()

    def flaky(call: Call) -> str | Exception:
        if call not in failed_once:
            failed_once.add(call)
            return CallError("first attempt fails")
        return inner.complete(call).text

    backend = ScriptedBackend(flaky)
    r = run_cli(["--json", ORIGINAL], backend)
    assert r.code == 0, r.err
    assert len(backend.calls) <= 100
    assert len(backend.calls) == 2 * len(failed_once)
    assert r.json()["calls_used"] == len(backend.calls)


def _search_call(call: Call) -> bool:
    return call.role == "reflect" or (call.role == "task" and call.model == MODELS["task"])


@pytest.mark.parametrize("seconds", [60.0, 1.0], ids=["slow-clock-stop", "fast-budget-stop"])
def test_clock_share_ends_the_search_and_the_report_says_cut_short(run_cli, timed, seconds):
    """R17, R2: the search may use 75 % of the 45-minute monotonic clock; when the clock ends it the
    result still comes from what it fully scored, `stop` is "clock" and stderr says it was cut
    short. Twin: fast calls end on the budget (the normal ending) with no such notice."""
    clock = FakeClock()
    backend = timed(happy_backend(IMPROVED), clock, seconds)
    r = run_cli(["--json", ORIGINAL], backend, clock)
    assert r.code == 0, r.err
    obj = r.json()
    assert obj["status"] == "improved"
    assert obj["prompt"] == IMPROVED
    search_starts = [
        t for c, t in zip(backend.calls, backend.starts, strict=True) if _search_call(c)
    ]
    assert max(search_starts) < SEARCH_S
    assert max(backend.starts) < WALL_S
    if seconds == 60.0:
        assert obj["stop"] == "clock"
        assert "cut short" in r.err
    else:
        assert obj["stop"] == "budget"
        assert "cut short" not in r.err


@pytest.mark.parametrize("seconds", [1000.0, 1.0], ids=["out-of-time", "in-time"])
def test_time_running_out_before_a_finalist_is_confirmed_returns_the_original(
    run_cli, timed, seconds
):
    """R17: if the clock runs out before a finalist is confirmed the original is returned (a stop
    never returns a candidate that skipped R3), no call starts after the deadline, and the
    reason_code tells it apart from "no reliable improvement". Twin: in time, the winner."""
    clock = FakeClock()

    def duration(call: Call) -> float:
        slow = call.model == MODELS["target"] and MARKER in call.user
        return seconds if slow else 1.0

    backend = timed(happy_backend(IMPROVED), clock, duration)
    r = run_cli(["--json", ORIGINAL], backend, clock)
    assert r.code == 0, r.err
    obj = r.json()
    assert max(backend.starts) < WALL_S
    if seconds == 1000.0:
        assert obj["prompt"] == ORIGINAL
        assert obj["status"] == "unchanged"
        assert obj["reason_code"] == "unconfirmed_out_of_budget"
        # R2, R12: both seed runs were measured, so the report still has the original's holdout
        # score and the noise
        assert obj["score_before"] == pytest.approx(0.0)
        assert obj["noise"] == pytest.approx(0.0)
    else:
        assert obj["prompt"] == IMPROVED
        assert obj["reason_code"] == "improved"


@pytest.mark.parametrize("passes", [False, True], ids=["contract-violation", "twin-passes"])
def test_a_clock_stop_never_returns_a_candidate_that_skipped_the_gates(
    run_cli, timed, override, passes
):
    """R17, R6: a search the clock cut short still sends its finalist through the gates; one that
    fails the contract check is not returned. Twin: one that passes is returned."""
    clock = FakeClock()

    def judge(call: Call) -> str:
        if json.loads(call.user)["scenarios"][0]["scenario"] == "contract":
            return judge_reply(call, lambda *_: passes)
        return judge_reply(call, lambda _s, _c, output: output.startswith("GOOD"))

    backend = timed(override(happy_backend(IMPROVED), judge=judge), clock, 60.0)
    r = run_cli(["--json", ORIGINAL], backend, clock)
    assert r.code == 0, r.err
    obj = r.json()
    assert obj["stop"] == "clock"
    assert obj["prompt"] == (IMPROVED if passes else ORIGINAL)
