"""The run of a bench (SPEC R26, R17, R19, R23, R24): each prompt runs the ordinary pipeline in its
own run folder under the bench's folder, one prompt after the other; an improved prompt is then
judged pairwise through its own stack (`Effort(Cached(Resilient(Budgeted(raw))))`, its own call
limit and clock) into that run folder's cache; an unchanged one is a tie with no call; a failed
run is an error and the bench goes on; Ctrl-C ends the bench after the prompts already measured.
The pipeline here is a fake that opens a run folder the way the real one does."""

from collections.abc import Callable
from pathlib import Path

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend
from test_bench_world import BETTER, NAIVE, ORIGINAL, BenchWorld, first_position, is_pairwise

from autoimprover.backend import Clock
from autoimprover.bench import BenchPlan, BenchPrompt, Collected, run_bench
from autoimprover.bench_judge import PAIRWISE_SCENARIOS, Comparison
from autoimprover.runstore import RunStore
from autoimprover.types import (
    BackendError,
    Call,
    CallFailed,
    Contract,
    Efforts,
    Models,
    Outcome,
    Plan,
    SessionNotLockedDown,
)

MODELS = Models(
    task="claude-haiku-4-5-20251001",
    judge="claude-opus-5-5",
    reflect="claude-sonnet-5-5",
    target="claude-fable-5-1",
)
EFFORTS = Efforts("low", "medium", "high")
S = PAIRWISE_SCENARIOS


def bench_plan(baseline: bool = False, seed: int = 0) -> BenchPlan:
    return BenchPlan(models=MODELS, efforts=EFFORTS, workers=1, baseline=baseline, seed=seed)


class Pipeline:
    """A fake `run_one`: opens a run folder under the root it is given, saves a contract of
    `kind`, and returns `outcomes[id]` (an Outcome, a failure text, or an exception to raise)."""

    def __init__(self, outcomes: dict[str, Outcome | str | BaseException], kind: str = "task"):
        self.outcomes = outcomes
        self.kind = kind
        self.roots: list[Path] = []

    def __call__(self, item: BenchPrompt, root: Path) -> Collected:
        self.roots.append(root)
        found = self.outcomes[item.id]
        if isinstance(found, BaseException):
            raise found
        store = RunStore.open_or_create(root, Plan(models=MODELS, wall_clock_s=30), item.prompt)
        try:
            store.save_contract(Contract(goal="g", kind=self.kind))  # type: ignore[arg-type]
            if isinstance(found, str):
                return Collected(None, 3.0, 5, store.run_id, failure=found)
            return Collected(found, 21.5, 17, store.run_id)
        finally:
            store.close()


def improved(prompt: str = BETTER) -> Outcome:
    return Outcome(status="improved", prompt=prompt, reason="r", reason_code="improved", noise=0.1)


def unchanged() -> Outcome:
    return Outcome(
        status="unchanged", prompt=ORIGINAL, reason="r", reason_code="no_reliable_improvement"
    )


class Raw:
    """The raw layer the bench makes for its own calls: records each maker call (the deadline as
    read by the first call, as the real layer reads it per call) and terminate."""

    def __init__(self, world: Callable[[Call], str | Exception]):
        self.inner = ScriptedBackend(world)
        self.made: list[tuple[Clock, RunStore, float]] = []
        self.terminated = 0
        self._deadline: Callable[[], float] | None = None

    def make(self, clock: Clock, store: RunStore, deadline: Callable[[], float]) -> "Raw":
        self.made.append((clock, store, -1.0))
        self._deadline = deadline
        return self

    def complete(self, call: Call):
        clock, store, seen = self.made[-1]
        if seen < 0 and self._deadline is not None:
            self.made[-1] = (clock, store, self._deadline())
        return self.inner.complete(call)

    def terminate(self) -> None:
        self.terminated += 1

    def calls(self, role: str | None = None) -> list[Call]:
        return [c for c in self.inner.calls if role is None or c.role == role]


def bench(
    tmp_path: Path,
    pipeline: Pipeline,
    raw: Raw,
    ids: tuple[str, ...] = ("a",),
    plan: BenchPlan | None = None,
    notes: list[str] | None = None,
):
    prompts = [BenchPrompt(id=i, prompt=ORIGINAL) for i in ids]
    root = tmp_path / "state" / "bench" / "20261007-010203-0123abcd"
    said = notes if notes is not None else []
    measured = run_bench(
        prompts,
        plan or bench_plan(),
        root,
        run_one=pipeline,
        make_raw=raw.make,
        now=FakeClock().now,
        note=said.append,
    )
    return measured, root


# --- one prompt ----------------------------------------------------------------------------------


def test_an_improved_prompt_is_judged_into_its_own_run_folder(tmp_path):
    pipeline, raw = Pipeline({"a": improved()}), Raw(BenchWorld())
    measured, root = bench(tmp_path, pipeline, raw)
    (row,) = measured.rows
    assert pipeline.roots == [root / "a"]
    assert (row.id, row.status, row.reason_code, row.seconds, row.calls) == (
        "a",
        "improved",
        "improved",
        21.5,
        17,
    )
    assert row.noise == 0.1 and row.tool == Comparison("win", S, 0, 0) and row.naive is None
    assert row.bench_calls == len(raw.calls()) == 1 + 2 * S + 2 * S
    (run_dir,) = (root / "a").iterdir()
    ((_clock, store, deadline),) = raw.made
    assert store.path == run_dir and deadline > 0
    cached = list((run_dir / "cache").iterdir())
    assert len(cached) == len(raw.calls())  # every bench call is in the run folder's cache


def test_an_unchanged_prompt_is_a_tie_with_no_call(tmp_path):
    raw = Raw(BenchWorld())
    measured, _root = bench(tmp_path, Pipeline({"a": unchanged()}), raw)
    (row,) = measured.rows
    assert (row.status, row.tool, row.naive, row.bench_calls) == (
        "unchanged",
        Comparison("tie", 0, 0, 0),
        None,
        0,
    )
    assert raw.calls() == [] and raw.made == []


def tagged(call: Call) -> str:
    """GOOD or BAD as the world says, each prompt's answers told apart by its length."""
    return f"{'GOOD' if MARKER in call.user else 'BAD'} answer {len(call.user)}"


def test_with_the_baseline_an_unchanged_prompt_still_gets_its_naive_comparison(tmp_path):
    raw = Raw(BenchWorld(task=tagged))
    measured, _root = bench(tmp_path, Pipeline({"a": unchanged()}), raw, plan=bench_plan(True))
    (row,) = measured.rows
    assert row.tool == Comparison("tie", 0, 0, 0)
    assert row.naive == Comparison("tie", 0, S, 0)  # NAIVE answers no better
    assert row.bench_calls == 2 + 2 * S + 2 * S
    (naive,) = raw.calls("reflect")
    assert (naive.user, naive.effort) == (ORIGINAL, "low")


def test_two_identical_answers_are_one_judge_call_for_both_orders(tmp_path):
    raw = Raw(BenchWorld())  # the original and the naive rewrite both answer "BAD answer"
    measured, _root = bench(tmp_path, Pipeline({"a": unchanged()}), raw, plan=bench_plan(True))
    assert measured.rows[0].naive == Comparison("tie", 0, S, 0)
    assert len(raw.calls("judge")) == S and measured.rows[0].bench_calls == 2 + 2 * S + S


def test_the_bench_calls_carry_the_plans_efforts_and_seed(tmp_path):
    raw = Raw(BenchWorld())
    bench(tmp_path, Pipeline({"a": improved()}), raw, plan=bench_plan(seed=5))
    assert {c.effort for c in raw.calls("task")} == {"low"}
    assert {c.effort for c in raw.calls("judge")} == {"medium"}
    assert {c.effort for c in raw.calls("synth")} == {"high"}
    assert {c.sample for c in raw.calls()} == {7005}


def test_the_kind_comes_from_the_runs_contract(tmp_path):
    raw = Raw(BenchWorld())
    bench(tmp_path, Pipeline({"a": improved()}, kind="template"), raw)
    assert {c.system for c in raw.calls("task")} == {ORIGINAL, BETTER}


def test_the_bench_calls_have_their_own_limit_and_clock(tmp_path):
    clock = FakeClock(10_000.0)  # far past any run's deadline
    raw = Raw(BenchWorld())
    measured = run_bench(
        [BenchPrompt(id="a", prompt=ORIGINAL)],
        bench_plan(),
        tmp_path / "b",
        run_one=Pipeline({"a": improved()}),
        make_raw=raw.make,
        now=clock.now,
        note=lambda _line: None,
    )
    assert measured.rows[0].tool == Comparison("win", S, 0, 0)
    ((made_clock, _store, deadline),) = raw.made
    assert made_clock.elapsed() == 0.0 and deadline >= 300


# --- many prompts, failures, Ctrl-C --------------------------------------------------------------


def test_prompts_run_one_after_the_other_and_a_failed_run_is_an_error(tmp_path):
    pipeline = Pipeline({"a": "backend failure: down", "b": improved(), "c": unchanged()})
    notes: list[str] = []
    measured, root = bench(tmp_path, pipeline, Raw(BenchWorld()), ("a", "b", "c"), notes=notes)
    assert [row.id for row in measured.rows] == ["a", "b", "c"]
    assert pipeline.roots == [root / "a", root / "b", root / "c"]
    error = measured.rows[0]
    assert (error.status, error.tool, error.calls, error.seconds) == ("error", None, 5, 3.0)
    assert error.error == "backend failure: down"
    assert measured.rows[1].tool == Comparison("win", S, 0, 0)
    assert not measured.interrupted
    assert any("a" in line and "error" in line for line in notes)


def test_a_position_biased_judge_never_makes_a_win(tmp_path):
    measured, _root = bench(
        tmp_path, Pipeline({"a": improved()}), Raw(BenchWorld(pairwise=first_position))
    )
    assert measured.rows[0].tool == Comparison("tie", 0, S, 0)


def test_a_backend_failure_while_judging_is_an_error_and_the_bench_goes_on(tmp_path):
    world = BenchWorld()

    def script(call: Call) -> str | Exception:
        if is_pairwise(call):
            return BackendError("three judge calls failed")
        return world(call)

    pipeline = Pipeline({"a": improved(), "b": unchanged()})
    measured, _root = bench(tmp_path, pipeline, Raw(script), ("a", "b"))
    tool = measured.rows[0].tool
    assert tool is not None and tool.verdict == "error" and "three judge" in tool.why
    assert measured.rows[1].tool == Comparison("tie", 0, 0, 0)


def test_ctrl_c_during_a_run_ends_the_bench_with_the_prompts_measured(tmp_path):
    pipeline = Pipeline({"a": improved(), "b": KeyboardInterrupt(), "c": improved()})
    measured, root = bench(tmp_path, pipeline, Raw(BenchWorld()), ("a", "b", "c"))
    assert measured.interrupted and [row.id for row in measured.rows] == ["a"]
    assert pipeline.roots == [root / "a", root / "b"]  # c never started


def test_ctrl_c_while_judging_cancels_the_bench_calls(tmp_path):
    world = BenchWorld()

    def script(call: Call) -> str | Exception:
        if call.role == "judge":
            raise KeyboardInterrupt
        return world(call)

    raw = Raw(script)
    measured, _root = bench(
        tmp_path, Pipeline({"a": unchanged(), "b": improved()}), raw, ("a", "b")
    )
    assert measured.interrupted and [row.id for row in measured.rows] == ["a"]
    assert raw.terminated == 1


def test_a_session_that_is_not_locked_down_ends_the_bench(tmp_path):
    raw = Raw(lambda _call: SessionNotLockedDown("tools listed"))
    with pytest.raises(SessionNotLockedDown):
        bench(tmp_path, Pipeline({"a": improved()}), raw)


def test_a_failed_call_of_the_bench_leaves_only_its_scenario_out(tmp_path):
    world = BenchWorld()

    def script(call: Call) -> str | Exception:
        if call.role == "task" and "situation 1" in call.user:
            return CallFailed("task down")
        return world(call)

    measured, _root = bench(tmp_path, Pipeline({"a": improved()}), Raw(script))
    assert measured.rows[0].tool == Comparison("win", S - 1, 0, 0)


def test_the_naive_rewrite_against_an_improved_prompt(tmp_path):
    raw = Raw(BenchWorld(naive=NAIVE))
    measured, _root = bench(tmp_path, Pipeline({"a": improved()}), raw, plan=bench_plan(True))
    row = measured.rows[0]
    assert row.tool == Comparison("win", S, 0, 0) and row.naive == Comparison("tie", 0, S, 0)
