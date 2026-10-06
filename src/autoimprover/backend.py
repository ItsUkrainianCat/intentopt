"""The backend layers every model call passes, outermost first `Cached(Resilient(Budgeted(raw)))`
(ADR-004): the disk cache, the retries with the per-(role, model) failure count, and the call
limit with its deadline on the shared monotonic clock (SPEC R17, R24). The raw layer that starts
`claude -p` lives in `claude_cli.py`; tests put `tests/fakes.py` there.

Prompts and replies are data: they are hashed and stored, never evaluated (SPEC R19).

The threads of a fast-pipeline stage share one stack (SPEC R25; ADR-011): every layer keeps its
counters under a lock, and none holds a lock while the raw call runs.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator

from autoimprover.runstore import RunStore
from autoimprover.types import (
    CALL_RETRIES,
    MAX_CONSECUTIVE_FAILURES,
    Backend,
    BackendError,
    BudgetExhausted,
    Call,
    CallError,
    CallFailed,
    Reply,
)


class Clock:
    """Monotonic seconds used by a run, carried across `--resume` (SPEC R17, ADR-007). `now` is
    the time source (tests pass `FakeClock.now`); `elapsed` is the time a resumed run had already
    used. Deadlines are in the same elapsed seconds."""

    def __init__(self, now: Callable[[], float] = time.monotonic, elapsed: float = 0.0) -> None:
        self._now = now
        self._carried = elapsed
        self._start = now()

    def elapsed(self) -> float:
        return self._carried + self._now() - self._start

    def remaining(self, deadline: float) -> float:
        """Seconds left until `deadline`; negative once it has passed."""
        return deadline - self.elapsed()


class BudgetedBackend:
    """The single enforcement point of the call limit and the deadline (SPEC R17, ADR-004).

    Every attempt that reaches it is counted before the raw call starts, so a failed or
    interrupted attempt still counts; `used` starts from the run folder's saved total, so a resume
    continues the same budget. A call is refused, uncounted and without a raw call, once the clock
    has reached `deadline` (cause "clock", checked first) or `used` has reached `limit` (cause
    "budget"). `on_call(used, elapsed)` runs after every attempt that reached the raw layer, so
    the runner can save the progress (ADR-007).

    Threads (SPEC R25): the checks and the count are one step under a lock, so `used` never
    passes `limit`; the raw call runs outside it. `on_call` runs outside it too, one at a time,
    with the total and the time read at that moment, so the saved progress never goes back.

    `cancel()` (SPEC R2 exit 130, R21 cancel) refuses every later call, before the other checks
    and uncounted, as a "clock" stop; a raw call already running is not undone here.
    """

    def __init__(
        self,
        raw: Backend,
        limit: int,
        used: int,
        clock: Clock,
        deadline: float,
        on_call: Callable[[int, float], None] | None = None,
    ) -> None:
        self._raw = raw
        self._limit = limit
        self._used = used
        self._clock = clock
        self._deadline = deadline
        self._on_call = on_call
        self._cancelled = False
        self._lock = threading.Lock()
        self._report_lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def limit(self) -> int:
        with self._lock:
            return self._limit

    @property
    def deadline(self) -> float:
        with self._lock:
            return self._deadline

    def raise_limit(self, limit: int, deadline: float) -> None:
        """Open the final steps' share of calls and clock once the search has ended (SPEC R17)."""
        with self._lock:
            self._limit = limit
            self._deadline = deadline

    def cancel(self) -> None:
        """Refuse every call from now on: the run is being stopped."""
        with self._lock:
            self._cancelled = True

    def complete(self, call: Call) -> Reply:
        with self._lock:
            if self._cancelled:
                raise BudgetExhausted("the run was cancelled", cause="clock")
            elapsed = self._clock.elapsed()
            if elapsed >= self._deadline:
                raise BudgetExhausted(
                    f"the clock reached its deadline ({elapsed:.0f} of {self._deadline:.0f} s)",
                    cause="clock",
                )
            if self._used >= self._limit:
                raise BudgetExhausted(
                    f"the call limit is used up ({self._used} of {self._limit} calls)",
                    cause="budget",
                )
            self._used += 1
        try:
            return self._raw.complete(call)
        finally:
            if self._on_call is not None:
                with self._report_lock:
                    self._on_call(self.used, self._clock.elapsed())


class ResilientBackend:
    """Retries a failed attempt (`CallError`) `CALL_RETRIES` times, each attempt going through the
    budget below (SPEC R24). A call that fails every attempt raises `CallFailed`; the
    `MAX_CONSECUTIVE_FAILURES`-th consecutive one for the same (role, model) raises `BackendError`
    instead, and the count stays there until that role and model succeed again. Another role's or
    model's success leaves the count alone, so a dead judge cannot hide behind a live task model.
    Every other exception (`BudgetExhausted`, `SessionNotLockedDown`, `BackendError`, a bug)
    propagates at once, unretried and uncounted. The counts are kept under a lock, so calls on
    several threads count in the order they end (SPEC R25).
    """

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        self._failures: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def complete(self, call: Call) -> Reply:
        key = (call.role, call.model)
        last: CallError | None = None
        for _attempt in range(1 + CALL_RETRIES):
            try:
                reply = self._inner.complete(call)
            except CallError as error:
                last = error
                continue
            with self._lock:
                self._failures.pop(key, None)
            return reply
        with self._lock:
            count = min(self._failures.get(key, 0) + 1, MAX_CONSECUTIVE_FAILURES)
            self._failures[key] = count
        if count >= MAX_CONSECUTIVE_FAILURES:
            raise BackendError(
                f"{count} consecutive {call.role} calls to {call.model} failed; "
                f"the last error: {last}"
            ) from last
        raise CallFailed(
            f"{call.role} call to {call.model} failed {1 + CALL_RETRIES} attempts: {last}"
        ) from last


class CachedBackend:
    """The disk cache, outermost (ADR-004): a call whose every field matches a stored one is
    answered from the run folder without a live call, so a hit costs nothing and never reaches
    the budget, and a resume replays the run (SPEC R17, R22). Only successful replies are stored,
    with the duration of the live call; a hit carries that duration so the search meter advances
    as it did. A tombstone replays as `CallFailed`. Exceptions from below are not stored, except
    while `record_failures` is on (the search sets it): then a `CallFailed`, a call that failed
    all its attempts, is stored as a tombstone before it propagates, so a resume decides as the
    original run did (SPEC R22; ADR-004). The tombstone's duration is 0: the attempts' time is not
    known here. Every answered call is logged in `calls.jsonl`.

    Single-flight (SPEC R25): identical calls on several threads are answered one after the other,
    so N of them in flight make one live call and the others read its cached reply; when it
    fails, the next one tries for itself as it would have alone, or replays its tombstone.
    Different calls never wait for each other.
    """

    def __init__(self, inner: Backend, store: RunStore) -> None:
        self._inner = inner
        self._store = store
        self.record_failures = False
        self._lock = threading.Lock()
        self._flights: dict[Call, threading.Event] = {}

    def complete(self, call: Call) -> Reply:
        reply = _stored(self._store, call)
        if reply is None:
            with self._flight(call):
                reply = _stored(self._store, call)  # an identical call may have stored it meanwhile
                if reply is None:
                    reply = self._live(call)
        self._store.log_call(
            call.role, call.model, reply.tokens_in, reply.tokens_out, reply.cached, reply.duration_s
        )
        return reply

    def _live(self, call: Call) -> Reply:
        try:
            reply = self._inner.complete(call)
        except CallFailed as error:
            if self.record_failures:
                self._store.record_failure(call, str(error), 0.0)
            raise
        self._store.cache_put(call, reply)
        return reply

    @contextlib.contextmanager
    def _flight(self, call: Call) -> Iterator[None]:
        """Hold the one flight of `call`; an identical call waits here until it has ended."""
        while True:
            with self._lock:
                ahead = self._flights.get(call)
                if ahead is None:
                    mine = self._flights[call] = threading.Event()
                    break
            ahead.wait()
        try:
            yield
        finally:
            with self._lock:
                del self._flights[call]
            mine.set()


def _stored(store: RunStore, call: Call) -> Reply | None:
    """The cached reply of `call`, None for a miss; a tombstone raises its `CallFailed`."""
    entry = store.cache_get(call)
    if entry is None:
        return None
    if entry.reply is None:
        raise CallFailed(entry.error)
    return entry.reply
