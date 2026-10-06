"""`autoimprover bench` (SPEC R26): the prompt set and the run of a bench.

The set is JSON Lines, one prompt per non-blank line: `id`, `prompt`, optional `kind` (`template`
or `task`, handed to the run as `--kind`) and optional `examples` (a list of objects as in an
examples file, SPEC R11); other keys are ignored. A prompt passes the checks of SPEC R1 (CRLF and
CR become LF); an id names the prompt's folder, so it is a short run of letters, digits, `-` and
`_` (SPEC R19); a bench takes at most MAX_PROMPTS. Every refusal names its line and never quotes
the prompt.

A bench runs the prompts one after the other. Each runs the ordinary pipeline (`run_one`, the
command line's own run of a prompt) in a run folder under `<bench folder>/<prompt id>/`; a run
that ends in a backend failure is an error and the bench goes on (SPEC R24). An improved prompt,
and with the naive baseline every prompt, is then judged pairwise (`bench_judge`) through a stack
of its own, `Effort(Cached(Resilient(Budgeted(raw))))` (SPEC R17, R24; ADR-004, ADR-011): the
plan's efforts, the run folder's cache, its own call limit and its own clock, so neither the run's
limit nor its deadline cuts the measurement. The plan's models and efforts are the runs' own (the
run's flags a bench takes, `cli_bench`), so the comparisons follow the run's choice: the target
model answers the fresh scenarios, both prompts alike, at the task role's effort (the task model
works only inside the runs); the judge model, never the task or target model (SPEC R14), judges
the answers at the judge role's effort; the reflection model writes the scenarios at the
reflection role's effort and the naive rewrite at low effort (SPEC R26). An unchanged prompt
without the baseline is a tie and costs no call. Ctrl-C ends the bench after the prompts already
measured; the prompt it cut is not counted, and the bench's own calls in flight are cancelled
(SPEC R2, R21).

DEBT, a private name used here until its owner adds a public seam: `scenarios._example`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.bench_judge import Comparison, Judged, judge, pairwise_calls
from autoimprover.efforts import EffortBackend
from autoimprover.runstore import RunStore
from autoimprover.scenarios import _example
from autoimprover.types import (
    CALL_RETRIES,
    PROMPT_MAX_CHARS,
    Backend,
    BackendError,
    Efforts,
    Kind,
    Models,
    Outcome,
    Scenario,
)

# The most prompts one bench measures: each costs about 40 calls at 30 s (SPEC R26).
MAX_PROMPTS = 25
# The set this repository ships, found from the package in a checkout (SPEC R26).
SHIPPED_PROMPTS = Path(__file__).resolve().parents[2] / "bench" / "prompts.jsonl"
# A prompt id names a folder (bench/<bench id>/<prompt id>/), so it is never a path (SPEC R19).
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_ID_RULE = "1 to 64 letters, digits, '-' or '_', starting with a letter or a digit"
# GEPA renders its templates by plain replacement of these, so a prompt may not hold one (R1).
_GEPA_TOKENS = ("<curr_param>", "<side_info>")
_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class BenchPrompt:
    """One prompt of the set: its id, its text (line endings LF), the `--kind` it runs with, the
    user's examples it runs on (None: synthesised), and its line in the file."""

    id: str
    prompt: str
    kind: Kind | None = None
    examples: tuple[Scenario, ...] | None = None
    line: int = 0


def load_prompts(path: Path, limit: int | None = None) -> list[BenchPrompt]:
    """The prompts of the set at `path`, in order, the first `limit` of them when given. Every
    line is checked; ValueError("line N: ...") for the first bad one, for a duplicate id (naming
    the line of the first), and for a set of more than MAX_PROMPTS without `limit`."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error.strerror or error}") from None
    found: list[BenchPrompt] = []
    lines: dict[str, int] = {}
    for number, raw in enumerate(data.removeprefix(_BOM).split(b"\n"), start=1):
        try:
            text = raw.removesuffix(b"\r").decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"line {number}: not valid UTF-8") from None
        if not text.strip():
            continue
        item = _prompt(text, number)
        if item.id in lines:
            raise ValueError(f"line {number}: the id {item.id} is already on line {lines[item.id]}")
        if limit is None and len(found) == MAX_PROMPTS:
            raise ValueError(
                f"line {number}: more than {MAX_PROMPTS} prompts; a bench measures at most "
                f"{MAX_PROMPTS}, choose the first ones with --limit"
            )
        lines[item.id] = number
        found.append(item)
    if not found:
        raise ValueError(f"no prompts in {path}")
    return found if limit is None else found[:limit]


def _prompt(text: str, number: int) -> BenchPrompt:
    where = f"line {number}"
    try:
        entry = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"{where}: not valid JSON ({type(error).__name__})") from None
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: not a JSON object")
    prompt_id = entry.get("id")
    if not isinstance(prompt_id, str) or not _ID.fullmatch(prompt_id):
        raise ValueError(f"{where}: `id` must be {_ID_RULE}")
    prompt = entry.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{where}: `prompt` must be a string")
    prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")
    if problem := _problem(prompt):
        raise ValueError(f"{where}: `prompt` {problem}")
    kind = entry.get("kind", ...)
    if kind is not ... and kind not in get_args(Kind):
        raise ValueError(f'{where}: `kind` must be "template" or "task"')
    examples = entry.get("examples")
    if examples is not None and (not isinstance(examples, list) or not examples):
        raise ValueError(f"{where}: `examples` must be a non-empty list of objects with `input`")
    scenarios = None
    if examples is not None:
        scenarios = tuple(
            _example(json.dumps(example), f"{where}: examples[{n}]", f"e{n + 1}")
            for n, example in enumerate(examples)
        )
    return BenchPrompt(
        id=prompt_id,
        prompt=prompt,
        kind=None if kind is ... else kind,
        examples=scenarios,
        line=number,
    )


def _problem(prompt: str) -> str:
    """Why `prompt` cannot be run, as SPEC R1 refuses a prompt, or ""."""
    if not prompt.strip():
        return "is empty"
    if "\0" in prompt:
        return "contains a NUL character"
    if len(prompt) > PROMPT_MAX_CHARS:
        return f"is longer than {PROMPT_MAX_CHARS:,} characters"
    if token := next((token for token in _GEPA_TOKENS if token in prompt), None):
        return f"contains {token}, which GEPA cannot escape"
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError:
        return "contains a lone surrogate (an unpaired \\ud800-\\udfff escape)"
    return ""


# --- the run of a bench ---------------------------------------------------------------------------

# The bench's own calls per prompt get this clock (each call's timeout is capped by what is left)
# and this many times their estimate as their call limit, room for retries (SPEC R17, R24).
PAIRWISE_CLOCK_S = 600
PAIRWISE_LIMIT_FACTOR = 1 + CALL_RETRIES

# The raw model layer of the bench's calls for one prompt, made with their clock, the prompt's run
# folder (the empty working folder of `claude -p`) and their current deadline.
RawMaker = Callable[[Clock, RunStore, Callable[[], float]], Backend]


@dataclass(frozen=True)
class Collected:
    """What one prompt's run left (SPEC R26): its Outcome with its tier, or the failure that ended
    it; the seconds on the run's clock, the calls it used, and the id of its run folder (None when
    it kept the original before making one)."""

    outcome: Outcome | None
    seconds: float | None
    calls: int
    run_id: str | None
    failure: str | None = None


# One prompt through the ordinary pipeline in a new run folder under the given folder.
RunOne = Callable[[BenchPrompt, Path], Collected]


@dataclass(frozen=True)
class BenchPlan:
    """What every prompt of a bench shares: the models and efforts of its runs, which its
    comparisons use too (the target model answers, the plan's judge judges, SPEC R14, R26), the
    calls at a time, the naive baseline, and `--seed`."""

    models: Models
    efforts: Efforts
    workers: int
    baseline: bool
    seed: int = 0


@dataclass(frozen=True)
class Row:
    """One measured prompt: its id, its run's status and reason code, whether the result is
    verified, the seconds on the run's clock, the run's calls and the bench's own, the noise the
    run measured, the tool's comparison with the original (an unchanged prompt ties; None when
    the run failed), the naive rewrite's (None without the baseline), and what failed."""

    id: str
    status: Literal["improved", "unchanged", "error"]
    reason_code: str | None
    verified: bool | None
    seconds: float | None
    calls: int
    bench_calls: int
    noise: float | None
    tool: Comparison | None
    naive: Comparison | None
    error: str = ""


@dataclass(frozen=True)
class Measured:
    """The rows of the prompts measured, in order, and whether Ctrl-C ended the bench early."""

    rows: tuple[Row, ...]
    interrupted: bool


def new_bench_folder(
    state: Path, data: bytes, utcnow: Callable[[], datetime] | None = None
) -> Path:
    """`<state>/bench/<id>`, the id like a run id: the UTC second and 8 hex of the SHA-256 of the
    set (`-2`, `-3`... when that folder exists). Reads only; the runs create the folders."""
    now = (utcnow or (lambda: datetime.now(UTC)))()
    base = f"{now:%Y%m%d-%H%M%S}-{hashlib.sha256(data).hexdigest()[:8]}"
    root = state / "bench"
    n = 1
    while os.path.lexists(root / (base if n == 1 else f"{base}-{n}")):
        n += 1
    return root / (base if n == 1 else f"{base}-{n}")


def run_bench(
    prompts: Sequence[BenchPrompt],
    plan: BenchPlan,
    root: Path,
    *,
    run_one: RunOne,
    make_raw: RawMaker,
    now: Callable[[], float],
    note: Callable[[str], None],
) -> Measured:
    """Measure `prompts` in order under the bench folder `root` (SPEC R26); `note` gets a line per
    prompt. SessionNotLockedDown, a usage error and anything unexpected end the bench."""
    rows: list[Row] = []
    for number, item in enumerate(prompts, start=1):
        note(f"bench: prompt {number} of {len(prompts)}: {item.id}")
        try:
            row = _measure(item, plan, root / item.id, run_one, make_raw, now)
        except KeyboardInterrupt:
            note(f"bench: interrupted at {item.id}; the summary covers {len(rows)} prompts")
            return Measured(tuple(rows), interrupted=True)
        rows.append(row)
        note(f"bench: {item.id}: {_said(row)}")
    return Measured(tuple(rows), interrupted=False)


def _measure(
    item: BenchPrompt,
    plan: BenchPlan,
    folder: Path,
    run_one: RunOne,
    make_raw: RawMaker,
    now: Callable[[], float],
) -> Row:
    collected = run_one(item, folder)
    outcome = collected.outcome
    if outcome is None:
        return Row(
            id=item.id,
            status="error",
            reason_code=None,
            verified=None,
            seconds=collected.seconds,
            calls=collected.calls,
            bench_calls=0,
            noise=None,
            tool=None,
            naive=None,
            error=collected.failure or "the run failed",
        )
    improved = outcome.status == "improved"
    judged, used = Judged(None, None), 0
    if (improved or plan.baseline) and collected.run_id is not None:
        candidate = outcome.prompt if improved else None
        judged, used = _pairwise(item, plan, folder, collected.run_id, candidate, make_raw, now)
    return Row(
        id=item.id,
        status="improved" if improved else "unchanged",
        reason_code=outcome.reason_code,
        verified=outcome.verified,
        seconds=collected.seconds,
        calls=collected.calls,
        bench_calls=used,
        noise=outcome.noise,
        tool=judged.tool if improved else Comparison("tie", 0, 0, 0),  # unchanged: a tie
        naive=judged.naive,
    )


def _pairwise(
    item: BenchPrompt,
    plan: BenchPlan,
    folder: Path,
    run_id: str,
    candidate: str | None,
    make_raw: RawMaker,
    now: Callable[[], float],
) -> tuple[Judged, int]:
    """The comparisons of one prompt through the bench's own stack over its run folder, and the
    calls they used. A backend failure makes them errors; Ctrl-C cancels the calls in flight."""
    store = RunStore.resume(folder, run_id)
    try:
        clock = Clock(now)

        def deadline() -> float:
            return budgeted.deadline

        raw = make_raw(clock, store, deadline)
        estimate = pairwise_calls(candidate is not None, plan.baseline)
        budgeted = BudgetedBackend(
            raw, PAIRWISE_LIMIT_FACTOR * estimate, used=0, clock=clock, deadline=PAIRWISE_CLOCK_S
        )
        stack = EffortBackend(CachedBackend(ResilientBackend(budgeted), store), plan.efforts)
        contract = store.contract()
        kind: Kind = contract.kind if contract is not None else item.kind or "task"
        try:
            judged = judge(
                stack,
                store.prompt,
                candidate,
                naive=plan.baseline,
                kind=kind,
                models=plan.models,
                workers=plan.workers,
                seed=plan.seed,
            )
        except BackendError as error:
            failed = Comparison("error", 0, 0, 0, str(error))
            judged = Judged(
                None if candidate is None else failed, failed if plan.baseline else None
            )
        except KeyboardInterrupt:
            budgeted.cancel()
            terminate = getattr(raw, "terminate", None)
            if callable(terminate):
                terminate()
            raise
        return judged, budgeted.used
    finally:
        store.close()


def _said(row: Row) -> str:
    """A prompt's line on stderr: ids, codes and numbers only, never a prompt (SPEC R26)."""
    if row.status == "error":
        return f"error: {row.error}"
    verdict = "not compared" if row.tool is None else row.tool.verdict
    naive = "" if row.naive is None else f", naive {row.naive.verdict}"
    return f"{row.status} ({row.reason_code}), {verdict}{naive}"
