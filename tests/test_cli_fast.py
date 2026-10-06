"""`autoimprover` in the quick, fast and checked tiers (SPEC R2, R4, R22, R25; ADR-011): without
`--time` a run is the 30-second fast pipeline, through `Effort(Cached(Resilient(Budgeted(raw))))`
with the deadline at `--time`; its outcome names its tier and says it is not verified; `--dry`
prints its stages and estimate with no call and nothing written; the stages' progress goes to
stderr; a resume keeps the saved plan. The raw model is the content-addressed `World` of the fast
pipeline tests. Every test that shows a rewrite is NOT returned has a twin showing one IS."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from fakes import FakeClock, ScriptedBackend, happy_backend
from test_fast_world import (  # noqa: F401  (no_disk_flush is an autouse fixture)
    BETTER,
    PROMPT,
    World,
    no_disk_flush,
)

from autoimprover import cli
from autoimprover.backend import BudgetedBackend
from autoimprover.cli_plan import DRY_KEYS
from autoimprover.fastplan import fast_plan
from autoimprover.runner import count_tokens
from autoimprover.runstore import runs_root
from autoimprover.types import Call, Reply

PLAIN = "Answer the request well."  # a rewrite without the marker: it scores no better
# The default plan, 30 s on 6 workers (test_fastplan.py): K=2, M=2, 14 calls (A: intake, synthesis
# and 2 rewrites = 4; B: 3 prompts x 2 scenarios = 6; C: 3 judge calls and 1 contract check = 4),
# about 20.6 s (I 8.83 + T 6.26 + J2 5.54, one wave each); the budget is 3 x 14 = 42 calls.
DEFAULT_PLAN = fast_plan(30, 6, count_tokens(PROMPT), False)


@dataclass
class Run:
    code: int
    out: str
    err: str
    raw: ScriptedBackend

    def obj(self) -> dict:
        lines = self.out.splitlines()
        assert len(lines) == 1, self.out
        return json.loads(lines[0])

    def calls(self, role: str) -> list[Call]:
        return [call for call in self.raw.calls if call.role == role]


def run(capsys, *argv: str, world: World | None = None, raw=None, now=None) -> Run:
    raw = raw if raw is not None else ScriptedBackend(world or World())
    code = cli.main(list(argv), backend=raw, now=now or FakeClock().now)
    out, err = capsys.readouterr()
    return Run(code, out, err, raw if isinstance(raw, ScriptedBackend) else raw.inner)


def folders() -> list:
    root = runs_root()
    return sorted(root.iterdir()) if root.is_dir() else []


def comparable(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if k not in ("calls_used", "run_dir")}


# --- the default run is the fast tier (SPEC R25) -------------------------------------------------


def test_without_time_the_run_is_the_fast_tier_and_says_it_is_not_verified(capsys):
    done = run(capsys, "--json", PROMPT)
    obj = done.obj()
    assert (done.code, obj["status"], obj["prompt"], obj["mode"]) == (0, "improved", BETTER, "fast")
    assert (obj["verified"], obj["reason_code"]) == (False, "improved")
    assert "fast check" in obj["reason"] and "not verified on held-out scenarios" in obj["reason"]
    assert any(line.startswith("notice: NOT VERIFIED") for line in done.err.splitlines())
    assert obj["calls_used"] == len(done.raw.calls) <= 3 * DEFAULT_PLAN.est_calls
    [folder] = folders()
    plan = json.loads((folder / "manifest.json").read_text())["plan"]
    assert (plan["tier"], plan["wall_clock_s"], plan["workers"]) == ("fast", 30, 6)
    assert plan["budget"] == 3 * DEFAULT_PLAN.est_calls
    assert plan["efforts"] == {"task": "low", "judge": "low", "reflect": "low"}


def test_its_twin_whose_rewrites_score_no_better_keeps_the_original(capsys):
    obj = run(capsys, "--json", PROMPT, world=World(rewrites=(PLAIN,))).obj()
    assert (obj["status"], obj["prompt"], obj["mode"]) == ("unchanged", PROMPT, "fast")
    assert (obj["reason_code"], obj["verified"]) == ("no_reliable_improvement", False)


def test_without_json_the_prompt_goes_to_stdout_and_the_report_names_the_tier(capsys):
    done = run(capsys, PROMPT)
    assert (done.code, done.out) == (0, BETTER + "\n")
    assert "mode: fast (0 s)" in done.err
    assert "verified: no. NOT VERIFIED (tier fast: fast check" in done.err


def test_the_stages_progress_goes_to_stderr_one_flushed_line_each_and_never_to_stdout(capsys):
    done = run(capsys, "--json", PROMPT)
    stages = [line for line in done.err.splitlines() if line.startswith("fast: stage ")]
    assert [line[:13] for line in stages] == [
        "fast: stage A",
        "fast: stage B",
        "fast: stage C",
    ]
    assert done.err.index("fast: stage A") < done.err.index("fast: stage C")
    assert "fast: stage" not in done.out
    [folder] = folders()
    assert "fast: stage A" in (folder / "gepa.log").read_text()  # the run's log keeps them too


def test_every_call_asks_the_tiers_effort_and_the_fast_models(capsys):
    done = run(capsys, "--json", PROMPT)
    assert {call.effort for call in done.raw.calls} == {"low"}
    for role in ("intake", "synth", "reflect"):
        assert {call.model for call in done.calls(role)} == {"claude-sonnet-5-5"}
    assert {call.model for call in done.calls("task")} == {"claude-haiku-4-5-20251001"}
    assert {call.model for call in done.calls("judge")} == {"claude-opus-5-5"}


def test_effort_and_model_flags_win_over_the_tiers_defaults(capsys):
    argv = ("--json", "--effort", "high", "--task-effort", "default", "--reflect-model", "opus")
    done = run(capsys, *argv, PROMPT)
    assert {call.effort for call in done.calls("task")} == {None}
    assert {call.effort for call in done.raw.calls if call.role != "task"} == {"high"}
    assert {call.model for call in done.calls("reflect")} == {"claude-opus-5-5"}
    assert done.obj()["status"] == "improved"


def test_the_users_examples_are_the_scenarios_and_nothing_is_synthesised(tmp_path, capsys):
    path = tmp_path / "ex.jsonl"
    path.write_text("".join(json.dumps({"input": f"example {i}"}) + "\n" for i in range(3)))
    done = run(capsys, "--json", "--examples", str(path), PROMPT)
    assert done.calls("synth") == [] and done.obj()["prompt"] == BETTER
    # With examples the default plan is K=2 on M=2 (I + T + J2 = 20.6 s; M=3 needs a second wave
    # of stage B: I + 2 T + J3 = 28.0 s > 25.5): stage B runs on the first 2 of the 3.
    assert {call.user.split("\n\n", 1)[0] for call in done.calls("task")} == {
        "example 0",
        "example 1",
    }


# --- the other fast tiers -------------------------------------------------------------------------


def test_the_quick_tier_returns_a_rewrite_that_kept_the_contract_unscored(capsys):
    done = run(capsys, "--json", "--time", "15s", PROMPT)
    obj = done.obj()
    assert (obj["status"], obj["mode"], obj["verified"]) == ("improved", "quick", False)
    assert "quick check" in obj["reason"] and done.calls("task") == []


def test_its_twin_whose_rewrite_breaks_the_contract_keeps_the_original(capsys):
    world = World(contract_ok=lambda _candidate: False)
    obj = run(capsys, "--json", "--time", "15s", PROMPT, world=world).obj()
    assert (obj["status"], obj["prompt"], obj["mode"]) == ("unchanged", PROMPT, "quick")


def test_the_checked_tier_verifies_on_held_out_scenarios(capsys):
    obj = run(capsys, "--json", "--time", "2m", PROMPT).obj()
    assert (obj["status"], obj["mode"], obj["verified"]) == ("improved", "checked", True)
    assert obj["score_before"] == 0.0 and obj["score_after"] == 1.0


def test_its_twin_that_loses_on_the_target_model_keeps_the_original(capsys):
    def task(call: Call) -> str:
        wins = "[[better]]" in call.system + call.user and call.model != "claude-sonnet-5-5"
        return "GOOD answer" if wins else "BAD answer"

    obj = run(capsys, "--json", "--time", "2m", PROMPT, world=World(task=task)).obj()
    assert (obj["status"], obj["prompt"], obj["mode"]) == ("unchanged", PROMPT, "checked")


def test_the_outcome_does_not_depend_on_the_workers(capsys):
    """At 9 minutes 1 and 4 workers plan the same shape; the calls run one at a time or four at
    a time, and the result and the calls are the same."""
    tokens = count_tokens(PROMPT)
    one, four = fast_plan(540, 1, tokens, False), fast_plan(540, 4, tokens, False)
    assert (one.rewrites, one.scenarios, one.holdout) == (four.rewrites, four.scenarios, 4)
    serial = run(capsys, "--json", "--time", "9m", "--workers", "1", PROMPT)
    parallel = run(capsys, "--json", "--time", "9m", "--workers", "4", PROMPT)
    assert comparable(serial.obj()) == comparable(parallel.obj())
    assert sorted(map(repr, serial.raw.calls)) == sorted(map(repr, parallel.raw.calls))


# --- the clock (SPEC R17, R25) --------------------------------------------------------------------


@dataclass
class Timed:
    """A raw model whose every call takes `seconds` on a fake clock; it notes when each starts."""

    inner: ScriptedBackend
    clock: FakeClock
    seconds: float
    starts: list[float] = field(default_factory=list)

    def complete(self, call: Call) -> Reply:
        self.starts.append(self.clock.now())
        self.clock.advance(self.seconds)
        return self.inner.complete(call)


@pytest.mark.parametrize("seconds", [100.0, 0.0], ids=["out-of-time", "in-time"])
def test_no_call_starts_after_the_time_and_the_original_is_kept_when_it_runs_out(capsys, seconds):
    clock = FakeClock()
    raw = Timed(ScriptedBackend(World()), clock, seconds)
    obj = run(capsys, "--json", "--time", "9m", "--workers", "1", PROMPT, raw=raw, now=clock.now)
    obj = obj.obj()
    assert max(raw.starts) < 540
    if seconds:
        assert (obj["status"], obj["reason_code"], obj["stop"]) == (
            "unchanged",
            "unconfirmed_out_of_budget",
            "clock",
        )
    else:
        assert (obj["status"], obj["prompt"]) == ("improved", BETTER)


# --- --dry (SPEC R4, R25) -------------------------------------------------------------------------


def test_dry_prints_the_fast_plan_with_no_call_and_nothing_written(capsys):
    dry = run(capsys, "--dry", "--json", PROMPT)
    plan = dry.obj()
    assert (dry.code, dry.err, dry.raw.calls, folders()) == (0, "", [], [])
    assert tuple(plan) == DRY_KEYS
    assert (plan["tier"], plan["workers"], plan["rewrites"]) == ("fast", 6, 1)
    assert (plan["scenarios"], plan["holdout"], plan["synthesised"]) == (2, 0, True)
    assert plan["efforts"] == {"task": "low", "judge": "low", "reflect": "low"}
    assert [stage["calls"] for stage in plan["stages"]] == [3, 6, 4, 0]
    assert plan["est_calls"] == DEFAULT_PLAN.est_calls == 13
    assert plan["est_seconds"] == pytest.approx(DEFAULT_PLAN.est_seconds) == 20.628571
    assert plan["plan"]["budget"] == 39 and plan["plan"]["wall_clock_s"] == 30
    assert plan["refusal"] is None and plan["iterations"] is None


def test_dry_text_shows_the_tier_the_stages_and_the_estimate(capsys):
    text = run(capsys, "--dry", PROMPT).out
    assert text.startswith("dry run: no model call made, nothing written\n")
    assert "tier: fast (--time 30 s), 6 calls at a time\n" in text
    assert "effort: task low, judge low, reflection low\n" in text
    assert "rewrites: 1; scenarios: 2, synthesised by one call (2 to pick on, 0 held out)" in text
    assert "  B: task runs: 6 calls, about 6.3 s" in text
    assert "estimate: 13 calls in about 21 s of 30 s; budget: 39 calls" in text
    assert "evidence: a fast check" in text and "would refuse" not in text


def test_a_plan_that_cannot_fit_its_time_is_refused_and_dry_says_so(capsys):
    dry = run(capsys, "--dry", "--workers", "1", PROMPT)
    [line] = [x for x in dry.out.splitlines() if x.startswith("a real run would refuse: ")]
    assert dry.code == 0 and "--time" in line and "--workers" in line
    refused = run(capsys, "--workers", "1", PROMPT)
    assert (refused.code, refused.out, refused.raw.calls, folders()) == (2, "", [], [])
    assert refused.err.startswith("error: ") and "--time" in refused.err


@pytest.mark.parametrize(("extra", "keeps"), [(0, True), (1, False)], ids=["none-left", "one-left"])
def test_dry_of_the_checked_tier_says_when_the_examples_leave_nothing_to_hold_out(
    tmp_path, capsys, extra, keeps
):
    picks = fast_plan(120, 6, count_tokens(PROMPT), True).scenarios
    path = tmp_path / "ex.jsonl"
    lines = (json.dumps({"input": f"example {i}"}) + "\n" for i in range(picks + extra))
    path.write_text("".join(lines))
    plan = run(capsys, "--dry", "--json", "--time", "2m", "--examples", str(path), PROMPT).obj()
    assert plan["tier"] == "checked"
    assert (plan["keeps_original"] is not None) == keeps
    if keeps:
        assert "examples" in plan["keeps_original"]


def test_dry_of_the_quick_tier_plans_one_rewrite_and_no_scenario(capsys):
    plan = run(capsys, "--dry", "--json", "--time", "15s", PROMPT).obj()
    assert (plan["tier"], plan["rewrites"], plan["scenarios"]) == ("quick", 1, 0)
    assert (plan["synthesised"], plan["holdout"]) == (False, 0)


# --- resume (SPEC R22) ----------------------------------------------------------------------------


class Cut(BaseException):
    """The process dies in the middle of a live call."""


def cutting(at: int) -> Callable[[Call], str | Exception]:
    world, seen = World(), []

    def script(call: Call) -> str | Exception:
        seen.append(call)
        if len(seen) == at:
            raise Cut
        return world(call)

    return script


@pytest.mark.parametrize("at", [1, 3, 8, 15])
def test_a_fast_run_cut_anywhere_resumes_to_the_same_outcome_with_its_saved_plan(capsys, at):
    reference = run(capsys, "--json", "--workers", "1", "--time", "9m", PROMPT).obj()
    cli.main(["clean"])
    capsys.readouterr()
    with pytest.raises(Cut):
        argv = ["--json", "--workers", "1", "--time", "9m", PROMPT]
        cli.main(argv, backend=ScriptedBackend(cutting(at)), now=FakeClock().now)
    [folder] = folders()
    resumed = run(capsys, "--json", "--resume", folder.name, "--time", "15s", "--workers", "3")
    assert comparable(resumed.obj()) == comparable(reference)
    assert resumed.obj()["calls_used"] == at + len(resumed.raw.calls)
    assert "ignoring --time, --workers" in resumed.err


def test_a_resumed_fast_run_with_examples_rebuilds_the_same_plan(tmp_path, capsys):
    """At 59 s on 4 workers (50.15 s planned; stage B runs the original twice, (K + 2) M task
    runs, stage C K + 3 calls) the plan with examples is K=3 on M=3: no synthesis call, so stage A
    is one wave, I + 4 T + 2 J3 = 8.83 + 25.03 + 13.23 = 47.1 s (M=4: I + 5 T + 2 J4 = 55.5 s).
    Without them it is K=3 on M=2, as stage A then needs a second wave: 2 I + 3 T + 2 J3 = 17.66 +
    18.77 + 13.23 = 49.7 s (M=3: 2 I + 4 T + 2 J3 = 55.9 s). The resume knows from the run folder
    that the run had them, and runs stage B on all 3."""
    tokens = count_tokens(PROMPT)
    shapes = [fast_plan(59, 4, tokens, given) for given in (True, False)]
    assert [(plan.rewrites, plan.scenarios) for plan in shapes] == [(3, 3), (3, 2)]
    path = tmp_path / "ex.jsonl"
    path.write_text("".join(json.dumps({"input": f"example {i}"}) + "\n" for i in range(3)))
    argv = ["--json", "--time", "59s", "--workers", "4", "--examples", str(path), PROMPT]
    reference = run(capsys, *argv).obj()
    cli.main(["clean"])
    with pytest.raises(Cut):  # in stage A, before any task run
        cli.main(argv, backend=ScriptedBackend(cutting(2)), now=FakeClock().now)
    [folder] = folders()
    opts = json.loads((folder / "manifest.json").read_text())["opts"]
    assert opts["examples"] is True
    resumed = run(capsys, "--json", "--resume", folder.name)
    assert comparable(resumed.obj()) == comparable(reference) and resumed.calls("synth") == []
    assert {call.user.split("\n\n", 1)[0] for call in resumed.calls("task")} == {
        "example 0",
        "example 1",
        "example 2",
    }


def test_a_saved_tier_its_clock_does_not_select_is_a_damaged_manifest(capsys):
    with pytest.raises(Cut):
        cli.main(["--json", PROMPT], backend=ScriptedBackend(cutting(1)), now=FakeClock().now)
    [folder] = folders()
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["plan"]["tier"] = "quick"  # with its 30 s, which is the fast tier
    (folder / "manifest.json").write_text(json.dumps(manifest))
    refused = run(capsys, "--resume", folder.name)
    assert (refused.code, refused.out, refused.raw.calls) == (2, "", [])
    assert "manifest.json" in refused.err and "damaged" in refused.err


@pytest.mark.parametrize(
    ("argv", "limit", "deadline"), [([], 39, 30.0), (["--time", "2m"], None, 120.0)]
)
def test_a_fast_run_has_the_whole_budget_and_clock(monkeypatch, capsys, argv, limit, deadline):
    built: list[tuple[int, float]] = []

    class Spy(BudgetedBackend):
        def __init__(self, raw, limit, used, clock, deadline, on_call=None):
            built.append((limit, deadline))
            super().__init__(raw, limit, used, clock, deadline, on_call)

    monkeypatch.setattr(cli, "BudgetedBackend", Spy)
    obj = run(capsys, "--json", *argv, PROMPT).obj()
    [folder] = folders()
    budget = json.loads((folder / "manifest.json").read_text())["plan"]["budget"]
    assert built == [(limit or budget, deadline)] and obj["status"] == "improved"


def test_the_deep_tiers_keep_the_original_outcome_names_its_tier(tmp_path, capsys):
    path = tmp_path / "ex.jsonl"
    path.write_text("".join(json.dumps({"input": f"example {i}"}) + "\n" for i in range(7)))
    obj = run(capsys, "--json", "--deep", "--examples", str(path), PROMPT).obj()
    assert (obj["reason_code"], obj["mode"], obj["run_dir"]) == ("no_holdout", "deep", "")
    assert folders() == []


# --- the deep tier's effort (SPEC R25; ADR-011: above the cache) ---------------------------------


@pytest.mark.parametrize(
    ("argv", "efforts"),
    [([], {None}), (["--effort", "low"], {"low"}), (["--judge-effort", "high"], {None, "high"})],
)
def test_a_deep_run_asks_the_effort_its_flags_choose(capsys, argv, efforts):
    raw = happy_backend(BETTER)
    done = run(capsys, "--json", "--deep", *argv, PROMPT, raw=raw)
    assert done.obj()["status"] == "improved"
    assert {call.effort for call in raw.calls} == efforts
    if argv[:1] == ["--judge-effort"]:
        assert {call.effort for call in raw.calls if call.role == "judge"} == {"high"}
