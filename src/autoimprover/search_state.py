"""What a search has decided and produced (ADR-004 item 3), apart from the GEPA seam in
`search.py`, which keeps the only `gepa` import: `RunState` holds what was completed, why the
search stopped and what aborted it; its stopper reads only the `SearchMeter`, which charges each
distinct call once, live or cached, so a resumed run replays to the same stop and the same
candidates (SPEC R17, R22). `Candidate` and `SearchResult` are what `run_search` returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from autoimprover.types import (
    BackendError,
    BudgetExhausted,
    SessionNotLockedDown,
    StopCause,
)


class Stopper(Protocol):
    """GEPA's stop callback (its `StopperProtocol`), declared here so that `search.py` stays the
    only module that imports `gepa`; GEPA's state is never read."""

    def __call__(self, gepa_state: object) -> bool: ...


# Why a search was abandoned, and the exit it becomes once the search returns: backend 3,
# lockdown 4, bug 1 (any other exception) (SPEC R24; ADR-004 item 4).
AbortCause = Literal["backend", "lockdown", "bug"]


@dataclass(frozen=True)
class Abort:
    """The first failure that ended the search, kept to be re-raised as itself."""

    cause: AbortCause
    error: Exception


@dataclass(frozen=True)
class Candidate:
    """A fully scored candidate: its mean valset score on the search model and the "what changed
    and why" lines of its lineage, newest first, at most NOTES_MAX (ADR-006)."""

    text: str
    val_score: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    """Every candidate scored on the whole valset without a failed call, in completion order, the
    original included (`seed_val_score`); why the search ended; the GEPA iterations started."""

    candidates: tuple[Candidate, ...]
    seed_val_score: float | None
    stop: StopCause | None
    iterations: int


class SearchMeter:
    """The calls and seconds the search has issued: each distinct call (by cache key) once, the
    first time, with that first occurrence's duration; a repeat costs nothing (ADR-004)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.seconds = 0.0

    @property
    def calls(self) -> int:
        return len(self._seen)

    @property
    def seconds_per_call(self) -> float:
        return self.seconds / self.calls if self.calls else 0.0

    def issue(self, key: str, duration_s: float) -> None:
        if key not in self._seen:
            self._seen.add(key)
            self.seconds += duration_s


class SkipProposal(Exception):
    """A reflection that yields no candidate: a reply without an instruction between the delimiter
    lines, an empty one, one holding a GEPA placeholder, or a failed call. GEPA asks once more,
    then skips the iteration (ADR-004 item 3, SPEC R24)."""


class RunState:
    """What the search has decided (ADR-004 item 3): the meter, `stop` (why it ended normally),
    `abort` (the failure that ends the run; first of each kept), GEPA `iterations` started, and
    `completed`: each candidate scored on the whole valset without a failed call, with its mean
    valset score, in completion order. `notes` holds the "what changed and why" lines of each
    proposed instruction, from its latest proposal until it is first completed (ADR-006)."""

    def __init__(self) -> None:
        self.meter = SearchMeter()
        self.stop: StopCause | None = None
        self.abort: Abort | None = None
        self.iterations = 0
        self.completed: dict[str, float] = {}
        self.notes: dict[str, tuple[str, ...]] = {}
        self._settled: set[str] = set()
        self._reflections = 0

    def next_reflect_index(self) -> int:
        """The next reflection call's `sample`: 0, 1, 2... over the search, retries included, so
        a stalled search pays for each proposal and a replay asks the same calls (ADR-004)."""
        index = self._reflections
        self._reflections += 1
        return index

    def complete(self, candidate: str, val_score: float) -> None:
        self.completed[candidate] = val_score
        self._settled.add(candidate)

    def note(self, instruction: str, notes: tuple[str, ...]) -> None:
        if instruction not in self._settled:
            self.notes[instruction] = notes

    @property
    def stopping(self) -> bool:
        return self.abort is not None or self.stop is not None

    def record_stop(self, cause: StopCause) -> None:
        if self.stop is None:
            self.stop = cause

    def record(self, error: Exception) -> None:
        """A failure caught inside the search. `BudgetExhausted` (the hard limit or the deadline)
        is a stop with its own cause, never an abort (SPEC R17); anything else aborts: a
        `BackendError` as backend, a `SessionNotLockedDown` as lockdown, the rest as a bug."""
        if isinstance(error, BudgetExhausted):
            self.record_stop("clock" if error.cause == "clock" else "budget")
        elif self.abort is None:
            cause: AbortCause = "bug"
            if isinstance(error, BackendError):
                cause = "backend"
            elif isinstance(error, SessionNotLockedDown):
                cause = "lockdown"
            self.abort = Abort(cause, error)

    def raise_if_aborted(self) -> None:
        """Re-raise the abort's original exception (exit 3, 4 or 1); a stop raises nothing."""
        if self.abort is not None:
            raise self.abort.error

    def stopper(
        self,
        *,
        calls_left_at_start: int,
        iter_cost: int,
        search_start: tuple[int, float],
        wall_clock_s: float,
        clock_share: float,
    ) -> Stopper:
        """GEPA's stop callback, asked before every iteration (ADR-004). It reads only the meter,
        this state and these arguments, never the saved call total or the clock, so a replay
        stops where the original did (SPEC R17, R22); `calls_left_at_start` is
        `(budget - final) - search_start[0]`. The calls condition comes first ("budget", the
        normal ending), then the clock share's one-iteration look-ahead ("clock"); each go-ahead
        counts an iteration."""
        meter = self.meter
        share = clock_share * wall_clock_s

        def stop(gepa_state: object) -> bool:  # GEPA's state is not read: replays must agree
            if self.stopping:
                return True
            if calls_left_at_start - meter.calls < iter_cost:
                self.record_stop("budget")
            elif search_start[1] + meter.seconds + meter.seconds_per_call * iter_cost >= share:
                self.record_stop("clock")
            else:
                self.iterations += 1
            return self.stopping

        return stop
