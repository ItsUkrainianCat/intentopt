"""The backend layers shared by the threads of a fast-pipeline stage (SPEC R25; ADR-011, and the
R17 and R24 rules they keep): the budget counts every attempt once and never passes its limit,
the failure count per (role, model) stays exact, identical calls in flight make one raw call
(single-flight), and no layer holds a lock around the raw call.

No test sleeps. Threads start together at a barrier, and the switch interval is cut to a
microsecond while they run so that a missing lock shows as a wrong count; waits time out only
when the code under test is wrong.
"""

import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import FakeClock, ScriptedBackend

from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.runstore import RunStore
from autoimprover.types import (
    DEFAULT_MODELS,
    BackendError,
    BudgetExhausted,
    Call,
    CallError,
    CallFailed,
    Plan,
)

WAIT = 10.0  # seconds; only a broken implementation ever waits this long
THREADS, EACH = 8, 50
TASK = Call(role="task", model="claude-haiku-4-5-20251001", user="say hi")
JUDGE = Call(role="judge", model="claude-opus-5-5", user="{}")


@pytest.fixture(autouse=True)
def fast_switching():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


@pytest.fixture
def store(tmp_path: Path):
    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    s = RunStore.open_or_create(
        tmp_path / "runs", Plan(models=DEFAULT_MODELS), "p", utcnow=lambda: now
    )
    yield s
    s.close()


def in_threads(work: Callable[[int], object], n: int = THREADS) -> list[object]:
    """`work(0..n-1)`, one thread each, started together: each result or the exception raised."""
    start = threading.Barrier(n, timeout=WAIT)
    results: list[object] = [None] * n

    def run(i: int) -> None:
        try:
            start.wait()
            results[i] = work(i)
        except Exception as error:
            results[i] = error

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads), "a thread hung"
    return results


def repeat(backend, call: Call, times: int = EACH) -> Callable[[int], list[object]]:
    """Work for `in_threads`: `times` calls, each outcome a reply text or the exception."""

    def work(_i: int) -> list[object]:
        out: list[object] = []
        for _ in range(times):
            try:
                out.append(backend.complete(call).text)
            except Exception as error:
                out.append(error)
        return out

    return work


def flat(results: list[object]) -> list[object]:
    return [outcome for per_thread in results for outcome in per_thread]  # type: ignore[union-attr]


# --- BudgetedBackend (SPEC R17) -------------------------------------------------------------------


def test_eight_threads_count_every_attempt_once_and_report_it_in_order():
    raw, log, each = ScriptedBackend(lambda _c: "ok"), [], 1000
    clock = Clock(now=FakeClock().now)
    budgeted = BudgetedBackend(raw, 10_000, 3, clock, 1e9, lambda used, _s: log.append(used))
    outcomes = flat(in_threads(repeat(budgeted, TASK, each)))
    assert outcomes == ["ok"] * THREADS * each
    assert budgeted.used == 3 + THREADS * each == 3 + raw.count()
    assert len(log) == THREADS * each and log == sorted(log) and log[-1] == budgeted.used


def test_the_limit_holds_exactly_under_threads():
    raw = ScriptedBackend(lambda _c: "ok")
    budgeted = BudgetedBackend(raw, 137, 0, Clock(now=FakeClock().now), 1e9)
    outcomes = flat(in_threads(repeat(budgeted, TASK)))
    refused = [o for o in outcomes if isinstance(o, BudgetExhausted)]
    assert outcomes.count("ok") == raw.count() == budgeted.used == 137
    assert len(refused) == THREADS * EACH - 137 and {o.cause for o in refused} == {"budget"}


def test_raise_limit_under_threads_opens_exactly_the_new_share():
    raw = ScriptedBackend(lambda _c: "ok")
    budgeted = BudgetedBackend(raw, 100, 0, Clock(now=FakeClock().now), 1e9)
    first = flat(in_threads(repeat(budgeted, TASK, 20)))
    budgeted.raise_limit(250, 1e9)
    second = flat(in_threads(repeat(budgeted, TASK, 20)))
    assert first.count("ok") == 100 and second.count("ok") == 150
    assert raw.count() == budgeted.used == budgeted.limit == 250


# --- ResilientBackend (SPEC R24) ------------------------------------------------------------------


def test_the_failure_count_per_role_and_model_stays_exact_under_threads():
    """Each attempt waits for all eight threads, so their failed calls reach the count together;
    per model exactly two are CallFailed and the other six end the run."""
    models = [f"claude-model-{n}" for n in range(5)]
    together = threading.Barrier(THREADS, timeout=WAIT)

    def script(call: Call) -> str | Exception:
        if call.role == "task":
            return "ok"
        together.wait()
        return CallError(f"{call.model} down")

    raw = ScriptedBackend(script)
    resilient = ResilientBackend(raw)

    def work(_i: int) -> list[object]:
        out: list[object] = []
        for model in models:
            try:
                resilient.complete(Call(role="judge", model=model, user="{}"))
            except (CallFailed, BackendError) as error:
                out.append((model, type(error)))
        return out

    outcomes = flat(in_threads(work))
    for model in models:
        assert outcomes.count((model, CallFailed)) == 2
        assert outcomes.count((model, BackendError)) == THREADS - 2
    assert raw.count() == 3 * THREADS * len(models)
    assert resilient.complete(TASK).text == "ok"  # another role is not affected


# --- CachedBackend: single-flight (SPEC R17, R22) -------------------------------------------------


class Gate:
    """Counts the threads whose first cache lookup has happened; `wait_all` holds a raw call until
    all of them are past it, so every identical call is in flight at once."""

    def __init__(self, store: RunStore, n: int) -> None:
        self.n, self.seen, self.lock = n, set(), threading.Lock()
        self.all_in = threading.Event()
        lookup = store.cache_get

        def cache_get(call: Call):
            with self.lock:
                self.seen.add(threading.get_ident())
                if len(self.seen) >= self.n:
                    self.all_in.set()
            return lookup(call)

        store.cache_get = cache_get  # type: ignore[method-assign]

    def wait_all(self) -> None:
        assert self.all_in.wait(WAIT), "not every thread reached the cache"


def full_stack(store: RunStore, script) -> tuple[CachedBackend, BudgetedBackend, ScriptedBackend]:
    raw = ScriptedBackend(script)
    budgeted = BudgetedBackend(
        raw, 1000, 0, Clock(now=FakeClock().now), 1e9, on_call=store.save_progress
    )
    return CachedBackend(ResilientBackend(budgeted), store), budgeted, raw


def test_identical_calls_in_flight_make_one_raw_call_and_share_its_reply(store: RunStore):
    gate = Gate(store, THREADS)

    def script(call: Call) -> str:
        gate.wait_all()
        return f"reply to {call.user}"

    cached, budgeted, raw = full_stack(store, script)
    outcomes = in_threads(lambda _i: cached.complete(TASK))
    assert raw.count() == 1 == budgeted.used
    assert {o.text for o in outcomes} == {"reply to say hi"}  # type: ignore[union-attr]
    assert sorted(o.cached for o in outcomes) == [False] + [True] * (THREADS - 1)  # type: ignore[union-attr]


def test_a_failing_leader_does_not_block_the_others_who_each_try_again(store: RunStore):
    gate, attempts = Gate(store, THREADS), []

    def script(_call: Call) -> str | Exception:
        gate.wait_all()
        attempts.append(1)
        return CallError("down") if len(attempts) <= 3 else "ok"  # the first call fails

    cached, budgeted, raw = full_stack(store, script)
    outcomes = in_threads(lambda _i: cached.complete(TASK))
    assert [type(o) for o in outcomes].count(CallFailed) == 1
    assert sum(getattr(o, "text", None) == "ok" for o in outcomes) == THREADS - 1
    assert raw.count() == 4 == budgeted.used


def test_with_failures_recorded_the_waiters_replay_the_leaders_tombstone(store: RunStore):
    gate = Gate(store, THREADS)

    def script(_call: Call) -> Exception:
        gate.wait_all()
        return CallError("down")

    cached, budgeted, raw = full_stack(store, script)
    cached.record_failures = True
    outcomes = in_threads(lambda _i: cached.complete(TASK))
    assert [type(o) for o in outcomes] == [CallFailed] * THREADS
    assert raw.count() == 3 == budgeted.used


def test_raw_calls_overlap_through_the_whole_stack(store: RunStore):
    """Two different calls meet inside the raw layer: no layer holds a lock around it."""
    meet = threading.Barrier(2, timeout=WAIT)

    def script(call: Call) -> str:
        meet.wait()
        return f"reply to {call.user}"

    cached, budgeted, raw = full_stack(store, script)
    calls = [Call(role="task", model=TASK.model, user=f"question {i}") for i in range(2)]
    outcomes = in_threads(lambda i: cached.complete(calls[i]).text, 2)
    assert outcomes == ["reply to question 0", "reply to question 1"]
    assert raw.count() == 2 == budgeted.used == store.calls_used
