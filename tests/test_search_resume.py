"""The whole search on the pinned GEPA 0.1.4 with the real evaluator and backend layers (SPEC R15,
R15a, R16, R17, R22, R24; ADR-004): the wiring, the result, the feedback reaching reflection, the
output kept off stdout, failed calls, and the replay of a run cut at every live call."""

import io
import os
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend, happy_backend, judge_reply, reflection_reply

from autoimprover import search
from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.evaluator import Evaluator
from autoimprover.runstore import RunStore, cache_key
from autoimprover.scenarios import split
from autoimprover.search import Candidate, RunState, SearchResult, run_search
from autoimprover.types import (
    DEFAULT_MODELS,
    JUDGE_BATCH_MAX,
    MINIBATCH_SIZE,
    Backend,
    BackendError,
    Call,
    CallError,
    CallFailed,
    Check,
    Contract,
    Plan,
    Reply,
    Scenario,
)

TASK = "claude-haiku-4-5-20251001"
JUDGE = "claude-opus-5-5"
SEED = "Answer the user's request."
BETTER = f"Answer the user's request {MARKER} and cite a source."
WORSE = "Answer the user's request, and be quick about it."
CONTRACT = Contract(
    goal="answer the request",
    kind="task",
    checks=(Check(id="c1", group="content", text="answers the request"),),
)
TEMPLATE = "Improve:\n<curr_param>\nFeedback:\n<side_info>\nReply between the delimiter lines."
WHY = ("tightened the wording", "kept the output format", "kept every literal")


def scenarios(n: int) -> list[Scenario]:
    return [Scenario(id=f"s{i}", input=f"situation {i}") for i in range(1, n + 1)]


def iter_cost(train: int, val: int) -> int:
    """An accepted child: reflect, parent and child minibatches, a valset pass (ARCHITECTURE)."""
    m = min(MINIBATCH_SIZE, train)
    return 1 + 2 * (m + 1) + val + -(-val // JUDGE_BATCH_MAX)


def search_with(
    backend: Backend,
    n: int = 8,
    log: io.StringIO | None = None,
    state: RunState | None = None,
    **overrides: object,
) -> SearchResult:
    parts = split(scenarios(n), seed=0)
    args: dict = {
        "seed": SEED,
        "train": parts.train,
        "val": parts.val,
        "make_evaluator": lambda b: Evaluator(b, CONTRACT, TASK, JUDGE),
        "backend": backend,
        "reflect_model": JUDGE,
        "reflection_template": TEMPLATE,
        "rng_seed": 0,
        "calls_left_at_start": 40,
        "iter_cost": iter_cost(len(parts.train), len(parts.val)),
        "search_start": (5, 0.0),
        "wall_clock_s": 2700.0,
        "clock_share": 0.75,
        "log": log if log is not None else io.StringIO(),
        "state": state,
    }
    return run_search(**{**args, **overrides})


def answering(improved: str) -> Callable[[Call], str]:
    """`happy_backend`'s answers: good output exactly when the prompt holds MARKER."""
    model = happy_backend(improved)
    return lambda call: model.complete(call).text


# --- the search on GEPA (SPEC R15, R15a; ADR-004) -------------------------------------------------


def test_an_improving_proposal_is_completed_with_its_valset_score_and_its_notes():
    result = search_with(ScriptedBackend(answering(BETTER)))
    assert result.candidates == (Candidate(SEED, 0.0, ()), Candidate(BETTER, 1.0, WHY))
    assert result.seed_val_score == 0.0 and result.stop == "budget" and result.iterations > 1


def test_a_proposal_that_does_not_beat_its_minibatch_is_never_run_on_the_valset():
    raw = ScriptedBackend(answering(WORSE))
    result = search_with(raw)
    assert result.candidates == (Candidate(SEED, 0.0, ()),) and result.seed_val_score == 0.0
    samples = [call.sample for call in raw.calls if call.role == "reflect"]
    assert result.stop == "budget" and samples == list(range(result.iterations))
    val_inputs = {s.input for s in split(scenarios(8), seed=0).val}
    worse = [c for c in raw.calls if c.role == "task" and WORSE in c.user]
    assert worse and not any(c.user.split("\n\n")[0] in val_inputs for c in worse)


@pytest.mark.parametrize("n", [2, 3])
def test_below_8_scenarios_a_minibatch_over_the_whole_valset_completes_nothing(n: int):
    raw = ScriptedBackend(answering(WORSE))
    result = search_with(raw, n=n, iter_cost=iter_cost(n, n))
    assert result.candidates == (Candidate(SEED, 0.0, ()),) and result.iterations > 0
    worse = [c.user.split("\n\n")[0] for c in raw.calls if c.role == "task" and WORSE in c.user]
    assert sorted(set(worse)) == [s.input for s in scenarios(n)]  # it ran on every scenario


def test_the_meter_charges_each_distinct_issued_call_once_and_a_failed_one_at_0_s():
    failing = Call(role="task", model=TASK, user=f"situation 1\n\n{BETTER}")

    def script(call: Call) -> str | Exception:
        return CallFailed("task down") if call == failing else answering(BETTER)(call)

    raw = ScriptedBackend(script, duration_s=2.0)  # no cache below: GEPA's repeats are live
    state = RunState()
    search_with(raw, state=state)
    keys = {cache_key(call) for call in raw.calls}
    assert failing in raw.calls and len(raw.calls) > len(keys)
    assert (state.meter.calls, state.meter.seconds) == (len(keys), 2.0 * (len(keys) - 1))


def test_the_gepa_configuration_is_the_one_the_spec_asks_for(monkeypatch: pytest.MonkeyPatch):
    seen = []
    real = search.optimize_anything

    def spy(*args, **kwargs):
        seen.append(kwargs["config"])
        return real(*args, **kwargs)

    monkeypatch.setattr(search, "optimize_anything", spy)
    search_with(ScriptedBackend(answering(BETTER)), rng_seed=7)
    search_with(ScriptedBackend(answering(BETTER)), merge=True)
    search_with(ScriptedBackend(answering(BETTER)), n=2, iter_cost=iter_cost(2, 2))
    off, on, small = seen
    assert small.reflection.reflection_minibatch_size == 2  # min(MINIBATCH_SIZE, len(train))
    assert (off.engine.parallel, off.engine.run_dir, off.engine.seed) == (False, None, 7)
    assert (off.engine.frontier_type, off.engine.acceptance_criterion) == (
        "hybrid",
        "strict_improvement",
    )
    assert off.reflection.reflection_prompt_template == TEMPLATE
    assert off.reflection.reflection_minibatch_size == MINIBATCH_SIZE
    assert (off.merge, on.merge is not None) == (None, True)


def test_nothing_reaches_stdout_or_stderr_and_gepa_progress_goes_to_the_log(capfd):
    def noisy(call: Call) -> str:
        print("printed by a call")
        print("written to stderr by a call", file=sys.stderr)
        return answering(BETTER)(call)

    log = io.StringIO()
    search_with(ScriptedBackend(noisy), log=log)
    assert capfd.readouterr() == ("", "")
    assert "printed by a call" in log.getvalue() and "stderr by a call" in log.getvalue()
    assert "Iteration" in log.getvalue()


def test_an_instruction_with_code_blocks_becomes_the_candidate_unchanged():
    code = f"{MARKER} Reply like this:\n```python\nprint('keep')\n```\nand end with\n```\nok\n```"

    def script(call: Call) -> str:
        return reflection_reply(code) if call.role == "reflect" else answering(code)(call)

    result = search_with(ScriptedBackend(script))
    assert [c.text for c in result.candidates] == [SEED, code]


# --- a search that climbs: each accepted child passes one more check -----------------------------

LEVELS = Contract(
    goal="answer the request",
    kind="task",
    checks=tuple(
        Check(id=f"c{j}", group=group, text=f"reaches level {j}")
        for j, group in enumerate(("format", "constraints", "content"), start=1)
    ),
)


def climbing(call: Call) -> str:
    """Task output names the candidate's level (its count of "+"), the judge passes check cJ from
    level J on, and reflection proposes the parent with one more "+" (two notes for level 2,
    three otherwise, so a lineage can hold more than NOTES_MAX)."""
    if call.role == "task":
        return f"answer at level {call.user.count('+')}"
    if call.role == "judge":
        return judge_reply(call, lambda _s, check, out: int(out.split()[-1]) >= int(check[3:]))
    parent = call.user.split("Improve:\n", 1)[1].split("\nFeedback:", 1)[0]
    level = parent.count("+") + 1
    notes = (f"raised to level {level}", "kept format", "kept tone")
    return reflection_reply(parent + " +", notes[:2] if level == 2 else notes)


@pytest.fixture(autouse=True)
def _no_disk_flush(monkeypatch: pytest.MonkeyPatch):
    """The run folder's writes stay atomic but are not flushed to the disk: these tests write a
    few thousand cache entries, and durability is tested with the run store, not here."""
    monkeypatch.setattr(os, "fsync", lambda _fd: None)


class Cut(BaseException):
    """The process dies in the middle of a live call (a kill, a crash, Ctrl-C)."""


class Raw:
    """The raw model layer: answers by `script`, dies with Cut at live call `cut_at`, fails every
    attempt of a call for which `fails` is true; `keys` lists every live call, `ok` the successful
    ones and `failed` the failed attempts, by cache key."""

    def __init__(self, script=climbing, *, cut_at=0, fails=None, clock=None, duration_s=0.0):
        self.script, self.cut_at, self.fails = script, cut_at, fails or (lambda _call: False)
        self.clock, self.duration_s = clock or FakeClock(), duration_s
        self.calls: list[Call] = []
        self.keys: list[str] = []
        self.ok: list[str] = []
        self.failed: list[str] = []

    @property
    def live(self) -> int:
        return len(self.keys)

    def complete(self, call: Call) -> Reply:
        key = cache_key(call)
        self.calls.append(call)
        self.keys.append(key)
        if self.live == self.cut_at:
            raise Cut
        if self.fails(call):
            self.failed.append(key)
            raise CallError("exit 1")
        self.clock.advance(self.duration_s)
        self.ok.append(key)
        return Reply(text=self.script(call), duration_s=self.duration_s)


def new_run(root: Path) -> str:
    """A new run folder under `root`, named at a fixed time; its id."""
    now = datetime(2026, 10, 4, 12, tzinfo=UTC)
    store = RunStore.open_or_create(root, Plan(models=DEFAULT_MODELS), SEED, utcnow=lambda: now)
    store.close()
    return store.run_id


def attempt(
    root: Path, raw: Raw, run_id: str = "", *, n: int = 8, limit: int = 60, **overrides: object
) -> tuple[str, SearchResult | None]:
    """One process: open (or resume) the run folder, build the real stack over `raw` and search
    from the checkpoint; the result is None when the process was cut."""
    store = RunStore.resume(root, run_id or new_run(root))
    try:
        clock = Clock(now=raw.clock.now, elapsed=store.elapsed_s)
        budgeted = BudgetedBackend(
            raw,
            limit=limit,
            used=store.calls_used,
            clock=clock,
            deadline=float(overrides.pop("deadline", 2025.0)),
            on_call=store.save_progress,
        )
        cached = CachedBackend(ResilientBackend(budgeted), store)
        start = store.search_start_or_record(budgeted.used, clock.elapsed())
        settings: dict = {"calls_left_at_start": limit - start[0], "search_start": start}
        settings["make_evaluator"] = lambda b: Evaluator(b, LEVELS, TASK, JUDGE)
        result = search_with(cached, n=n, **{**settings, **overrides})
        assert cached.record_failures is False  # on for the search only
    except Cut:
        return store.run_id, None
    finally:
        store.close()
    return store.run_id, result


def replays_every_cut(
    tmp_path: Path, make_raw: Callable[..., Raw], **settings: object
) -> tuple[SearchResult, Raw, list[tuple[int, Raw]]]:
    """Cut a run at each of its live calls and resume it: the same result as the uninterrupted
    run, the same successful calls in the same order, none of them twice (SPEC R22). Returns the
    uninterrupted run's result and raw layer and, per cut, the resumed run's raw layer."""
    reference_raw = make_raw()
    _, reference = attempt(tmp_path / "reference", reference_raw, **settings)
    assert reference is not None
    resumes = []
    for k in range(1, reference_raw.live + 1):
        cut_raw, resumed_raw = make_raw(cut_at=k), make_raw()
        run_id, cut = attempt(tmp_path / f"cut-{k}", cut_raw, **settings)
        assert cut is None and cut_raw.live == k
        _, resumed = attempt(tmp_path / f"cut-{k}", resumed_raw, run_id, **settings)
        assert resumed == reference, f"cut at live call {k}"
        assert cut_raw.ok + resumed_raw.ok == reference_raw.ok, f"cut at live call {k}"
        assert max(Counter(cut_raw.ok + resumed_raw.ok).values()) == 1
        resumes.append((k, resumed_raw))
    return reference, reference_raw, resumes


def test_the_climbing_search_completes_each_level_with_its_lineage_notes(tmp_path: Path):
    raw = Raw()
    _, result = attempt(tmp_path, raw)
    assert result is not None
    first_reflection = next(call.user for call in raw.calls if call.role == "reflect")
    assert SEED in first_reflection and "answer at level 0" in first_reflection  # SPEC R16
    assert "reaches level 1" in first_reflection and "Scores (Higher is Better)" in first_reflection
    assert [(c.text, round(c.val_score, 3)) for c in result.candidates] == [
        (SEED + " +" * level, round(level / 3, 3)) for level in range(4)
    ]
    own = [(f"raised to level {level}", "kept format", "kept tone") for level in range(1, 4)]
    own[1] = own[1][:2]
    lineage = [(), own[0], own[1] + own[0], (own[2] + own[1] + own[0])[:6]]  # 8 cut to 6
    assert [c.notes for c in result.candidates] == lineage
    assert result.stop == "budget" and result.seed_val_score == 0.0 and result.iterations > 3


@pytest.mark.parametrize("n", [8, 12])
def test_a_run_cut_at_any_live_call_resumes_to_the_same_result(tmp_path: Path, n: int):
    replays_every_cut(tmp_path, lambda **kw: Raw(**kw), n=n)


FAILURES = {
    # a task call of the level-2 candidate: that candidate never completes, the next level does
    "task": (lambda c: c.role == "task" and c.user.count("+") == 2, (0, 1, 3)),
    # the first reflection call: GEPA asks again under the next sample, nothing is lost
    "reflect": (lambda c: c.role == "reflect", (0, 1, 2, 3)),
}


@pytest.mark.parametrize("kind", FAILURES)
def test_an_in_search_failure_is_replayed_as_failed_and_never_retried_after_the_cut(
    tmp_path: Path, kind: str
):
    match, levels = FAILURES[kind]
    failing = first_call(tmp_path, match)
    result, reference, resumes = replays_every_cut(
        tmp_path, lambda **kw: Raw(fails=lambda call: call == failing, **kw)
    )
    assert reference.failed == [cache_key(failing)] * 3
    assert [c.text for c in result.candidates] == [SEED + " +" * level for level in levels]
    last_attempt = max(i for i, c in enumerate(reference.calls, start=1) if c == failing)
    late = [resumed for k, resumed in resumes if k > last_attempt]
    assert late and all(resumed.failed == [] for resumed in late)


def test_a_clock_bound_search_stops_on_its_clock_share_and_resumes_to_the_same_result(
    tmp_path: Path,
):
    result, reference, _ = replays_every_cut(
        tmp_path, lambda **kw: Raw(duration_s=10.0, **kw), wall_clock_s=400.0, deadline=300.0
    )
    assert result.stop == "clock" and len(result.candidates) >= 2
    assert reference.clock.t < 300.0  # the look-ahead stopped it, not the hard deadline


@pytest.mark.parametrize(
    ("limits", "cause"), [({"limit": 20}, "budget"), ({"deadline": 160.0}, "clock")]
)
def test_a_hard_limit_mid_search_is_a_stop_that_keeps_the_completed_candidates(
    tmp_path: Path, limits: dict, cause: str
):
    _, full = attempt(tmp_path / "full", Raw())
    _, result = attempt(tmp_path / "hard", Raw(duration_s=10.0), calls_left_at_start=100, **limits)
    assert full is not None and result is not None and result.stop == cause
    texts = [c.text for c in result.candidates]
    assert len(texts) >= 2 and texts == [c.text for c in full.candidates][: len(texts)]


# --- failed calls inside the search (SPEC R24) ----------------------------------------------------


def first_call(tmp_path: Path, match: Callable[[Call], bool]) -> Call:
    """The first live call `match` picks in an uninterrupted run of the climbing search."""
    probe = Raw()
    attempt(tmp_path / "probe", probe)
    return next(call for call in probe.calls if match(call))


def test_one_failed_reflection_call_only_skips_its_proposal(tmp_path: Path):
    reflect = first_call(tmp_path, lambda c: c.role == "reflect")
    raw = Raw(fails=lambda call: call == reflect)
    _, result = attempt(tmp_path / "run", raw)
    assert raw.failed == [cache_key(reflect)] * 3
    assert result is not None and [c.text for c in result.candidates] == [
        SEED + " +" * level for level in range(4)
    ]


def test_three_consecutive_failed_reflection_calls_end_the_run(tmp_path: Path):
    raw = Raw(fails=lambda call: call.role == "reflect")
    with pytest.raises(BackendError):
        attempt(tmp_path, raw)
    assert len(raw.failed) == 9  # three calls, three attempts each


def test_a_failed_seed_call_ends_the_run_and_is_tried_again_on_resume(tmp_path: Path):
    seed_call = first_call(tmp_path, lambda c: c.role == "task")
    run_id = new_run(tmp_path / "run")
    with pytest.raises(BackendError, match="original prompt"):
        attempt(tmp_path / "run", Raw(fails=lambda call: call == seed_call), run_id)
    resumed = Raw()
    _, result = attempt(tmp_path / "run", resumed, run_id)
    assert seed_call in resumed.calls and result is not None and result.seed_val_score == 0.0
