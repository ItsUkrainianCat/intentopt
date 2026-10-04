"""The GEPA seam (ADR-004) and the only module that imports `gepa`: `run_search` runs
`optimize_anything` with `parallel=False` (SPEC R15, R15a) and keeps backend exceptions away from
GEPA, which swallows reflection errors and loses the result of a search that raises (SPEC R17,
R24). `EvaluatorAdapter` scores candidates with the run's evaluator, whose side info is the
feedback GEPA renders for reflection (SPEC R16); `ReflectionWrapper` takes the new instruction
from between delimiter lines (ADR-006, ADR-008); `RunState` keeps what was completed, why the
search stopped and what aborted it. Its stopper reads only the `SearchMeter`, which charges each
distinct call once, live or cached, so a resumed run replays to the same stop and the same
candidates (SPEC R22). GEPA's progress, and anything printed meanwhile, goes to the run's log.

A call that failed all its attempts is charged 0 s, and its tombstone stores 0 s: above
`Resilient` the attempts' time is unknown, and a replay must charge what the original did.
Known limit: a judge call holds outputs, not the candidate, so candidates with identical outputs
on the same scenarios share it; a child's tombstone on it then fails the seed's scoring with no
live call, which ends the run (exit 3), and a resume ends it the same way.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TextIO

from gepa.core.result import GEPAResult
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    MergeConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)
from gepa.utils import StopperProtocol

from autoimprover.backend import CachedBackend
from autoimprover.runstore import cache_key
from autoimprover.types import (
    INSTRUCTION_BEGIN,
    INSTRUCTION_END,
    MINIBATCH_SIZE,
    Backend,
    BackendError,
    BatchEvaluator,
    BudgetExhausted,
    Call,
    CallFailed,
    Reply,
    Scenario,
    SessionNotLockedDown,
    StopCause,
)

# GEPA splices feedback into these tokens by plain replacement: never in a candidate (ADR-006).
_PLACEHOLDERS = ("<curr_param>", "<side_info>")
# "What changed and why" lines kept per reflection reply, and per lineage (ADR-006).
_NOTES_PER_REPLY = 3
NOTES_MAX = 6

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
    ) -> StopperProtocol:
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


Entry = tuple[float, dict[str, Any]]


class Example(NamedTuple):
    """One item of GEPA's dataset or valset: the scenario, and whether it is a valset item, so a
    valset pass is told apart from a minibatch even when both hold the same scenarios (below 8
    scenarios every scenario is in both, SPEC R15)."""

    scenario: Scenario
    val: bool


class EvaluatorAdapter:
    """GEPA's `batch_evaluator` (ADR-004 item 3): the pairs of one GEPA call, grouped by
    candidate, each candidate's scenarios scored in order; results aligned with the pairs.

    One valset pass that scored every valset scenario completes a candidate, a minibatch never
    does (SPEC R15a). A failed call keeps a candidate from completing for good (SPEC R24); on the
    original prompt (`seed`) it aborts the run as a backend failure, since a 0 would lower the bar
    for every candidate, and `cache` (the run's disk cache, if any) records no tombstone for it,
    so a resume tries it again (SPEC R22). Once the search is stopping, pairs are neutral and make
    no call. Other exceptions are recorded in the run state (`BudgetExhausted` as a stop, the rest
    as an abort); only `Exception` is caught, so an interrupt passes through."""

    def __init__(
        self,
        evaluator: BatchEvaluator,
        state: RunState,
        *,
        seed: str,
        val: Sequence[Scenario],
        cache: CachedBackend | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._state = state
        self._seed = seed
        self._val = tuple(val)
        self._cache = cache
        self._failed: set[str] = set()

    def __call__(self, pairs: Sequence[tuple[str, Example]]) -> list[Entry]:
        groups: dict[str, list[int]] = {}
        for index, (candidate, _example) in enumerate(pairs):
            groups.setdefault(candidate, []).append(index)
        # A pair left unscored (the search is stopping) is neutral: no call, never completing.
        results: list[Entry] = [(0.0, {"incomplete": True}) for _ in pairs]
        for candidate, indices in groups.items():
            if self._state.stopping:
                break
            examples = [pairs[index][1] for index in indices]
            entries = self._score(candidate, [example.scenario for example in examples])
            if entries is None:
                continue
            for index, entry in zip(indices, entries, strict=True):
                results[index] = entry
            self._settle(candidate, examples, entries)
        return results

    def _score(self, candidate: str, scenarios: list[Scenario]) -> list[Entry] | None:
        """The evaluator's entries, or None when it raised something other than CallFailed."""
        try:
            with _recording_failures(self._cache, candidate != self._seed):
                return list(self._evaluator(candidate, scenarios))
        except CallFailed as error:
            return [(0.0, {"incomplete": True, "error": str(error)}) for _ in scenarios]
        except Exception as error:  # recorded; `raise_if_aborted` re-raises it after the search
            self._state.record(error)
            return None

    def _settle(self, candidate: str, examples: list[Example], entries: list[Entry]) -> None:
        failed = [info for _score, info in entries if info.get("incomplete")]
        if failed:
            if candidate == self._seed:
                self._state.record(
                    BackendError(
                        "a call failed while scoring the original prompt: "
                        f"{failed[0].get('error', 'no reason given')}"
                    )
                )
            self._failed.add(candidate)
            self._state.completed.pop(candidate, None)
            return
        pairs = zip(examples, entries, strict=True)
        scores = {ex.scenario: score for ex, (score, _info) in pairs if ex.val}
        if candidate not in self._failed and self._val and all(s in scores for s in self._val):
            self._state.complete(candidate, sum(scores[s] for s in self._val) / len(self._val))


class ReflectionWrapper:
    """GEPA's `reflection_lm` (ADR-004 item 3; ADR-008 reflect row): asks the reflection model with
    GEPA's rendered prompt (the feedback of SPEC R16) and the next running sample. The instruction
    between the first `INSTRUCTION_BEGIN` line and the last `INSTRUCTION_END` line goes back in one
    outer fence (GEPA keeps all from the first fence to the last, so inner code blocks stay whole);
    the `- ` lines after it are its notes (ADR-006).

    GEPA swallows what this raises, so a failed call or an unusable reply is `SkipProposal` (GEPA
    retries, then skips the iteration, SPEC R24), and anything else is first recorded in the run
    state (`BudgetExhausted` as a stop, the rest as an abort). Once the search is stopping it
    raises without a call. Only `Exception` is caught, so an interrupt passes through."""

    def __init__(self, backend: Backend, state: RunState, *, model: str) -> None:
        self._backend = backend
        self._state = state
        self._model = model

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if self._state.stopping:
            raise SkipProposal("the search is stopping")
        try:
            return self._reflect(prompt)
        except SkipProposal:
            raise
        except Exception as error:  # recorded first: GEPA swallows it, `raise_if_aborted` does not
            self._state.record(error)
            raise

    def _reflect(self, prompt: str | list[dict[str, Any]]) -> str:
        if not isinstance(prompt, str):  # GEPA sends messages only for images, never used here
            raise TypeError("the reflection prompt is not text")
        sample = self._state.next_reflect_index()
        call = Call(role="reflect", model=self._model, user=prompt, sample=sample)
        try:
            reply = self._backend.complete(call)
        except CallFailed as error:
            raise SkipProposal(str(error)) from error
        instruction, notes = _parse_reflection(reply.text)
        self._state.note(instruction, notes)
        return f"```\n{instruction}\n```"


def _parse_reflection(text: str) -> tuple[str, tuple[str, ...]]:
    """The instruction between the delimiter lines, without the blank space around it, and up to
    three notes from the `- ` lines after it; SkipProposal when there is no usable instruction."""
    lines = text.splitlines()
    marks = [line.strip() for line in lines]
    begin = marks.index(INSTRUCTION_BEGIN) if INSTRUCTION_BEGIN in marks else None
    end = len(marks) - 1 - marks[::-1].index(INSTRUCTION_END) if INSTRUCTION_END in marks else None
    if begin is None or end is None or end < begin:
        raise SkipProposal("the reply has no instruction between the delimiter lines")
    instruction = "\n".join(lines[begin + 1 : end]).strip()
    if not instruction:
        raise SkipProposal("the reply's instruction is empty")
    if any(token in instruction for token in _PLACEHOLDERS):
        raise SkipProposal("the reply's instruction holds a GEPA placeholder")
    notes = [mark[2:].strip() for mark in marks[end + 1 :] if mark.startswith("- ")]
    return instruction, tuple(note for note in notes if note)[:_NOTES_PER_REPLY]


class RunLogger:
    """GEPA's logger: its progress lines go to `out`, the run folder's `gepa.log` (ADR-004)."""

    def __init__(self, out: TextIO) -> None:
        self._out = out

    def log(self, message: str) -> None:
        self._out.write(f"{message}\n")
        self._out.flush()


@contextlib.contextmanager
def _recording_failures(cache: CachedBackend | None, on: bool) -> Iterator[None]:
    """`cache.record_failures` set to `on` for the block, then put back, also on a raise."""
    before = on if cache is None else cache.record_failures
    if cache is not None:
        cache.record_failures = on
    try:
        yield
    finally:
        if cache is not None:
            cache.record_failures = before


class _Metered:
    """The backend the search sees: every call it issues goes to the meter under its cache key,
    with the reply's duration (live, or stored with a cache hit), and 0 s for a failed call."""

    def __init__(self, inner: Backend, meter: SearchMeter) -> None:
        self._inner = inner
        self._meter = meter

    def complete(self, call: Call) -> Reply:
        key = cache_key(call)
        try:
            reply = self._inner.complete(call)
        except CallFailed:
            self._meter.issue(key, 0.0)
            raise
        self._meter.issue(key, reply.duration_s)
        return reply


def run_search(
    *,
    seed: str,
    train: Sequence[Scenario],
    val: Sequence[Scenario],
    make_evaluator: Callable[[Backend], BatchEvaluator],
    backend: Backend,
    reflect_model: str,
    reflection_template: str,
    rng_seed: int,
    calls_left_at_start: int,
    iter_cost: int,
    search_start: tuple[int, float],
    wall_clock_s: float,
    clock_share: float,
    log: TextIO,
    merge: bool = False,
    state: RunState | None = None,
    cache: CachedBackend | None = None,
) -> SearchResult:
    """GEPA's search from `seed`, minibatches from `train`, acceptance on `val`, until the stopper
    ends it (SPEC R15, R15a, R17); the evaluator and the reflection calls use `backend` through
    the meter. GEPA runs with `parallel=False`, no run folder, the hybrid frontier, strict
    improvement, merge only with `merge`, and its output (and anything printed meanwhile) in `log`.
    `cache`, the run's disk cache, has `record_failures` on for the search except while the seed
    is scored, then put back, also on a raise: an in-search failure replays as failed on resume,
    the one that ended the run is tried again (SPEC R22). A stop returns every candidate completed
    before it; an abort raises its original exception (`RunState.raise_if_aborted`)."""
    if state is None:
        state = RunState()
    metered = _Metered(backend, state.meter)
    stopper = state.stopper(
        calls_left_at_start=calls_left_at_start,
        iter_cost=iter_cost,
        search_start=search_start,
        wall_clock_s=wall_clock_s,
        clock_share=clock_share,
    )
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=None,
            seed=rng_seed,
            parallel=False,
            frontier_type="hybrid",
            acceptance_criterion="strict_improvement",
        ),
        reflection=ReflectionConfig(
            reflection_lm=ReflectionWrapper(metered, state, model=reflect_model),
            reflection_prompt_template=reflection_template,
            reflection_minibatch_size=min(MINIBATCH_SIZE, len(train)),
        ),
        tracking=TrackingConfig(logger=RunLogger(log)),
        merge=MergeConfig() if merge else None,
        stop_callbacks=[stopper],
    )
    adapter = EvaluatorAdapter(make_evaluator(metered), state, seed=seed, val=val, cache=cache)
    with (
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
        _recording_failures(cache, True),
    ):
        try:
            result = optimize_anything(
                seed,
                batch_evaluator=adapter,
                dataset=[Example(scenario, val=False) for scenario in train],
                valset=[Example(scenario, val=True) for scenario in val],
                config=config,
            )
        except Exception:
            state.raise_if_aborted()  # the recorded cause decides the exit, not its aftermath
            raise
    state.raise_if_aborted()
    return SearchResult(
        candidates=tuple(
            Candidate(text, score, _lineage_notes(result, state, text))
            for text, score in state.completed.items()
        ),
        seed_val_score=state.completed.get(seed),
        stop=state.stop,
        iterations=state.iterations,
    )


def _lineage_notes(result: GEPAResult, state: RunState, text: str) -> tuple[str, ...]:
    """The notes of `text` and of each ancestor in GEPA's lineage, newest first, at most
    NOTES_MAX: from the first candidate GEPA holds with that text, by first parents (a parent
    always comes before its child, so the walk ends)."""
    texts = [next(iter(candidate.values())) for candidate in result.candidates]
    index = texts.index(text) if text in texts else None
    notes: list[str] = []
    while index is not None and len(notes) < NOTES_MAX:
        notes += state.notes.get(texts[index], ())
        parents = result.parents[index]
        index = parents[0] if parents else None
    return tuple(notes[:NOTES_MAX])
