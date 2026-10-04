"""Command-line entry point: `autoimprover [flags] <prompt>` (or `--file`, or stdin), `--resume
<id>` and `clean [<id>]`. `backend` replaces the raw model layer and `now` the monotonic clock, so
tests inject a fake model and a fake clock (SPEC R17, R20). Before the first paid call (ARCHITECTURE
section 1) `main` checks the prompt at the boundary (SPEC R1), the flags, the models (SPEC R14),
the examples, the fixed costs and the state folder (SPEC R4, R17, R23); `--dry` stops there, with
no call and nothing written, and fewer than 8 scenarios without `--trust-search` keep the original
(SPEC R11). A run gets its folder, `Cached(Resilient(Budgeted(raw)))` on one clock and
`runner.improve`; `--resume <id>` continues it with its saved budget and clock (SPEC R22). Every
ending maps to an exit code of SPEC R2; `report.Emitter` is the only writer to stdout."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, NoReturn, cast, get_args

from autoimprover import runner
from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.report import Emitter, PlanView
from autoimprover.runstore import RunStore, RunStoreError, runs_root
from autoimprover.scenarios import MIN_SCENARIOS_FOR_HOLDOUT, read_examples
from autoimprover.types import (
    BUDGET_CEILING,
    BUDGET_DEFAULT,
    EXIT_BACKEND,
    EXIT_INTERNAL,
    EXIT_INTERRUPTED,
    EXIT_NOT_LOCKED_DOWN,
    EXIT_OK,
    EXIT_USAGE,
    LENGTH_CAP,
    PROMPT_MAX_CHARS,
    SEARCH_CLOCK_SHARE,
    SYNTH_COUNT,
    Backend,
    BackendError,
    Kind,
    Models,
    Outcome,
    Plan,
    Scenario,
    SessionNotLockedDown,
    Strictness,
    canonical_model,
    default_models,
)

# GEPA renders its reflection template by plain replacement of these, so a prompt holding one
# could not be told apart from the feedback spliced in (SPEC R1).
_GEPA_TOKENS = ("<curr_param>", "<side_info>")
_BOM = b"\xef\xbb\xbf"
# The most bytes a prompt of PROMPT_MAX_CHARS characters can take: 4 per character, and a BOM.
_READ_LIMIT = 4 * PROMPT_MAX_CHARS + len(_BOM)
_CLAUDE_CLI = "autoimprover.claude_cli"
_MODEL_FLAGS = {role: f"{role}_model" for role in ("task", "judge", "reflect", "target")}


class UsageError(Exception):
    """Bad input or usage, or a refusal before any paid call (exit 2); the message names the flag
    or folder to change (SPEC R2)."""


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: Backend | None = None,
    now: Callable[[], float] | None = None,
) -> int:
    """Run the tool and return the exit code of SPEC R2."""
    args = list(sys.argv[1:] if argv is None else argv)
    emit = Emitter(sys.stdout, sys.stderr, json_mode=_json_requested(args))
    session = _Session(emit, backend, now)
    fail, folder = emit.error, session.run_dir
    try:
        return session.run(args)
    except (UsageError, RunStoreError) as error:
        return fail(EXIT_USAGE, str(error), folder())
    except SessionNotLockedDown as error:
        return fail(EXIT_NOT_LOCKED_DOWN, f"the session is not locked down: {error}", folder())
    except BackendError as error:
        return fail(EXIT_BACKEND, f"backend failure: {error}", folder(), session.resume_line())
    except KeyboardInterrupt:
        return fail(EXIT_INTERRUPTED, "interrupted", folder(), session.resume_line())
    except Exception as error:  # the top-level boundary: a bug is reported as exit 1 (SPEC R2)
        message = f"internal error: {type(error).__name__}: {error}"
        return fail(EXIT_INTERNAL, message, folder(), session.resume_line())
    finally:
        session.close()


def read_prompt(argument: str | None, file: str | None, stdin: IO[bytes] | None) -> str:
    """The prompt from the argument, else the file, else stdin, with CRLF and CR made LF; a
    UsageError naming its source unless it is UTF-8, not blank, at most PROMPT_MAX_CHARS
    characters, without NUL and without a GEPA template token (SPEC R1). A file or stdin is read
    only up to the bytes such a prompt can take."""
    if argument is not None and file is not None:
        raise UsageError("--file: give the prompt either as an argument or with --file, not both")
    if argument is not None:
        source, text = "the prompt argument", argument
        try:
            argument.encode("utf-8")
        except UnicodeEncodeError:
            raise UsageError(f"{source} is not valid UTF-8") from None
    else:
        source = "stdin" if file is None else f"--file {file}"
        data = _read(file, stdin)
        if len(data) > _READ_LIMIT:
            raise UsageError(f"{source}: the prompt is longer than {PROMPT_MAX_CHARS:,} characters")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise UsageError(f"{source}: the prompt is not valid UTF-8") from None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise UsageError(f"{source}: the prompt is empty")
    if "\0" in text:
        raise UsageError(f"{source}: the prompt contains a NUL character")
    if len(text) > PROMPT_MAX_CHARS:
        raise UsageError(f"{source}: the prompt is longer than {PROMPT_MAX_CHARS:,} characters")
    if token := next((token for token in _GEPA_TOKENS if token in text), None):
        raise UsageError(f"{source}: the prompt contains {token}, which GEPA cannot escape")
    return text


def _read(file: str | None, stdin: IO[bytes] | None) -> bytes:
    """At most one byte more than a prompt can take, from the file or else from stdin."""
    if file is not None:
        try:
            with open(file, "rb") as stream:
                return stream.read(_READ_LIMIT + 1)
        except OSError as error:
            raise UsageError(f"--file: cannot read {file}: {error.strerror or error}") from None
    if stdin is None:
        raise UsageError("no prompt: give it as one quoted argument, with --file, or on stdin")
    try:
        return stdin.read(_READ_LIMIT + 1)
    except OSError as error:
        raise UsageError(f"stdin: cannot read the prompt: {error}") from None


# --- the command line ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Options:
    """The parsed command line; `given` holds the dest of every flag written on it."""

    given: frozenset[str] = frozenset()
    words: tuple[str, ...] = ()
    help: bool = False
    file: str | None = None
    examples: str | None = None
    kind: Kind | None = None
    budget: int = BUDGET_DEFAULT
    strictness: Strictness = "conservative"
    allow_growth: bool = False
    task_model: str | None = None
    judge_model: str | None = None
    reflect_model: str | None = None
    target_model: str | None = None
    merge: bool = False
    trust_search: bool = False
    force_low_budget: bool = False
    dry: bool = False
    json: bool = False
    resume: str | None = None


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def _parser() -> _Parser:
    parser = _Parser(
        prog="autoimprover",
        description="Improve a prompt by GEPA search, scored by running it on test scenarios.",
        epilog="autoimprover clean [<id>] removes one run folder, or all of them. Exit codes: 0 "
        "done (also when the original is kept), 1 internal error, 2 bad input or a refusal "
        "before any paid call, 3 backend failure, 4 claude session not locked down, 130 "
        "interrupted.",
        argument_default=argparse.SUPPRESS,
        allow_abbrev=False,
        add_help=False,
    )
    flag = parser.add_argument
    flag("words", nargs="*", metavar="prompt", help="the prompt, as one quoted argument")
    flag("-h", "--help", action="store_true", help="show this help")
    flag("--file", metavar="PATH", help="read the prompt from this UTF-8 file")
    flag("--examples", metavar="PATH", help="test scenarios, JSON Lines: input, expected, criteria")
    flag("--kind", choices=get_args(Kind), help="template or task (default: guessed)")
    flag("--budget", type=_budget, help=f"model calls (default {BUDGET_DEFAULT})")
    flag("--strictness", choices=tuple(LENGTH_CAP), help="how far a rewrite may go")
    flag("--allow-growth", action="store_true", help="no length cap")
    for role, dest in _MODEL_FLAGS.items():
        flag(f"--{dest.replace('_', '-')}", metavar="MODEL", help=f"the {role} model")
    flag("--merge", action="store_true", help="let GEPA merge candidates")
    flag("--trust-search", action="store_true", help="below 8 scenarios, accept an unverified win")
    flag("--force-low-budget", action="store_true", help="run on fewer than 4 iterations")
    flag("--dry", action="store_true", help="print the plan; no model call, nothing written")
    flag("--json", action="store_true", help="print one JSON object on stdout")
    flag("--resume", metavar="ID", help="continue the run with this id")
    return parser


def _budget(text: str) -> int:
    if not (text.isascii() and text.isdigit() and 1 <= int(text) <= BUDGET_CEILING):
        raise argparse.ArgumentTypeError(f"must be a whole number from 1 to {BUDGET_CEILING}")
    return int(text)


def parse(argv: Sequence[str]) -> Options:
    found: dict[str, Any] = vars(_parser().parse_intermixed_args(list(argv)))
    words = tuple(found.pop("words", ()))
    return Options(given=frozenset(found), words=words, **found)


def _flag(dest: str) -> str:
    return f"--{dest.replace('_', '-')}"


def _json_requested(args: list[str]) -> bool:
    """Whether `--json` is among the flags, known before parsing so a usage error is an object."""
    return "--json" in (args[: args.index("--")] if "--" in args else args)


# --- the flows -----------------------------------------------------------------------------------

# The raw model layer of a run, made once its clock and folder exist: an injected test double,
# or the real `ClaudeCliBackend` with the run's clock, empty working folder and current deadline.
RawMaker = Callable[[Clock, RunStore, Callable[[], float]], Backend]


class _Session:
    """One invocation: its writer, the injected raw layer and clock, and the run folder it holds
    once there is one (named in error output, released at exit)."""

    def __init__(
        self, emit: Emitter, backend: Backend | None, now: Callable[[], float] | None
    ) -> None:
        self.emit, self.backend, self.now = emit, backend, now
        self.store: RunStore | None = None

    def run_dir(self) -> str:
        return "" if self.store is None else str(self.store.path)

    def resume_line(self) -> str:
        """`autoimprover --resume <id>`, after `XDG_STATE_HOME=<dir> ` (shell-quoted) when the
        run's state folder is not the default one (ADR-007)."""
        if self.store is None:
            return ""
        state = self.store.path.parent.parent.parent
        home = os.environ.get("HOME", "")
        default = Path(home, ".local", "state") if os.path.isabs(home) else None
        prefix = "" if state == default else f"XDG_STATE_HOME={shlex.quote(str(state))} "
        return f"{prefix}autoimprover --resume {self.store.run_id}"

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def run(self, args: list[str]) -> int:
        opts = parse(args)
        if opts.help:
            self.emit.help(_parser().format_help())
            return EXIT_OK
        if opts.words[:1] == ("clean",):
            return _clean(opts, self.emit)
        return self.resume(opts) if opts.resume is not None else self.start(opts)

    def start(self, opts: Options) -> int:
        """Everything checked before the first paid call, in the order of ARCHITECTURE section
        1; then a new run folder and the run."""
        prompt = _prompt(opts)
        models, merge, growth = _models(opts), opts.merge, opts.allow_growth
        plan = Plan(models, opts.strictness, opts.budget, allow_growth=growth, merge=merge)
        examples = _examples(opts.examples)
        n = SYNTH_COUNT if examples is None else len(examples)
        costs = runner.fixed_costs(plan, n, synthesising=examples is None)
        keeps = None
        if n < MIN_SCENARIOS_FOR_HOLDOUT and not opts.trust_search:
            keeps = (
                f"{n} scenarios are fewer than {MIN_SCENARIOS_FOR_HOLDOUT}: no holdout to verify "
                "a result on, so the original is kept; --trust-search accepts an unverified one"
            )
        refusal = _refusal(costs, opts.force_low_budget)
        view = PlanView(plan, n, examples is None, costs, refusal, keeps)
        if opts.dry:
            self.emit.plan(view, dry=True)
            return EXIT_OK
        if refusal is not None:
            raise UsageError(refusal)
        if keeps is not None:
            kept = Outcome(
                status="unchanged", prompt=prompt, reason=keeps, reason_code="no_holdout"
            )
            self.emit.outcome(kept, prompt, None, plan)
            return EXIT_OK
        make_raw = self.raw_maker()
        saved = {"kind": opts.kind, "trust_search": opts.trust_search}
        store = self.store = RunStore.open_or_create(runs_root(), plan, prompt, saved)
        if examples is not None:  # saved now, so a resume never needs the file again (SPEC R22)
            store.save_scenarios(examples)
        self.emit.plan(view, dry=False)
        self.emit.notice(f"run folder: {store.path}")
        return self.improve(make_raw, costs, examples, opts.kind, opts.trust_search)

    def resume(self, opts: Options) -> int:
        """The run `--resume` names, with the prompt, plan, flags, scenarios, call count and
        clock saved in its folder (SPEC R22)."""
        ignored, make_raw = _ignored(opts), self.raw_maker()
        try:
            store = self.store = RunStore.resume(runs_root(), cast(str, opts.resume))
        except RunStoreError as error:
            raise UsageError(f"--resume: {error}") from error
        if ignored:
            self.emit.notice(
                f"notice: --resume continues the saved run; ignoring {', '.join(ignored)}"
            )
        kind, trust_search = _saved(store)
        saved = store.scenarios()
        costs = runner.fixed_costs(store.plan, len(saved or ()) or SYNTH_COUNT, not saved)
        self.emit.notice(
            f"resuming run {store.run_id}: {store.calls_used} of {store.plan.budget} calls used, "
            f"{store.elapsed_s:.0f} s of {store.plan.wall_clock_s} s of the clock"
        )
        self.emit.notice(f"run folder: {store.path}")
        return self.improve(make_raw, costs, None, kind, trust_search)

    def improve(
        self,
        make_raw: RawMaker,
        costs: runner.FixedCosts,
        scenarios: Sequence[Scenario] | None,
        kind: Kind | None,
        trust_search: bool,
    ) -> int:
        """The run in the held folder: one clock carried across resumes, the call limit at what
        the final steps leave, the deadline at the search's share of the clock (SPEC R17), all
        printed output meanwhile in the run's log, the Outcome reported (SPEC R2)."""
        store = cast(RunStore, self.store)
        plan = store.plan
        clock = Clock(self.now or time.monotonic, elapsed=store.elapsed_s)

        def deadline() -> float:  # the current one: the search's, until the final steps open
            return budgeted.deadline

        budgeted = BudgetedBackend(
            make_raw(clock, store, deadline),
            limit=plan.budget - costs.final,
            used=store.calls_used,
            clock=clock,
            deadline=SEARCH_CLOCK_SHARE * plan.wall_clock_s,
            on_call=store.save_progress,
        )
        with store.open_log() as log, contextlib.redirect_stdout(log):
            outcome = runner.improve(
                store.prompt,
                plan,
                backend=CachedBackend(ResilientBackend(budgeted), store),
                budgeted=budgeted,
                clock=clock,
                store=store,
                scenarios=scenarios,
                kind=kind,
                trust_search=trust_search,
                log=log,
            )
        self.emit.outcome(outcome, store.prompt, store.contract(), plan)
        return EXIT_OK

    def raw_maker(self) -> RawMaker:
        """The injected raw layer, or the real one (WP1b), imported only when none is injected,
        so tests never load it; a build without it is a backend failure, before any folder."""
        backend = self.backend
        if backend is not None:
            return lambda _clock, _store, _deadline: backend
        try:
            module = importlib.import_module(_CLAUDE_CLI)
        except ModuleNotFoundError as error:
            if error.name != _CLAUDE_CLI:
                raise
            raise BackendError(
                f"the claude backend ({_CLAUDE_CLI}) is not part of this build, so no model can "
                "be called; install a complete release"
            ) from error
        real = module.ClaudeCliBackend
        return lambda clock, store, deadline: real(clock, cwd=store.cwd(), deadline=deadline)


def _prompt(opts: Options) -> str:
    if len(opts.words) > 1:
        raise UsageError(
            f"give the prompt as one quoted argument (got {len(opts.words)} arguments), or use "
            "--file"
        )
    argument = opts.words[0] if opts.words else None
    piped = _stdin() if argument is None and opts.file is None else None
    return read_prompt(argument, opts.file, piped)


def _ignored(opts: Options) -> list[str]:
    """What a resumed run ignores, because its prompt, plan and flags come from its folder
    (SPEC R22; ADR-007): every flag but --resume and --json, and a prompt argument. `--dry` is a
    usage error: there is no plan left to choose."""
    if opts.dry:
        raise UsageError("--dry cannot be combined with --resume: a resumed run keeps its plan")
    ignored = sorted(_flag(dest) for dest in opts.given - {"resume", "json"})
    return ignored + ["the prompt argument"] * bool(opts.words)


def _saved(store: RunStore) -> tuple[Kind | None, bool]:
    """The flags a run saved in its manifest (`kind`, `trust_search`)."""
    opts = store.opts
    kind, trust_search = opts.get("kind"), opts.get("trust_search", False)
    if kind not in (None, *get_args(Kind)) or type(trust_search) is not bool:
        raise RunStoreError(
            f"manifest.json in the run folder {store.path} is damaged (its saved flags); start a "
            f"new run, or remove this one with `autoimprover clean {store.run_id}`"
        )
    return cast(Kind | None, kind), trust_search


def _clean(opts: Options, emit: Emitter) -> int:
    extra = sorted(_flag(dest) for dest in opts.given - {"json"})
    if extra:
        raise UsageError(f"clean takes a run id and --json only; drop {', '.join(extra)}")
    ids = opts.words[1:]
    if len(ids) > 1:
        raise UsageError(f"clean takes at most one run id, got {len(ids)}")
    root = runs_root()
    try:
        removed, skipped = RunStore.clean(root, ids[0] if ids else None)
    except RunStoreError as error:
        raise UsageError(f"clean: {error}") from error
    emit.cleaned(removed, skipped, str(root))
    return EXIT_OK


def _models(opts: Options) -> Models:
    """The four models as full ids: the defaults for the target (SPEC R14's fallback judge when
    the target is the default judge), overridden by the flags given; a judge equal to the task or
    target model is refused naming the flags that chose it."""
    chosen: dict[str, str] = {}
    for role, dest in _MODEL_FLAGS.items():
        name = getattr(opts, dest)
        if name is not None:
            try:
                chosen[role] = canonical_model(name)
            except ValueError as error:
                raise UsageError(f"{_flag(dest)}: {error}") from None
    defaults = default_models(chosen.get("target"))
    models: dict[str, str] = {r: chosen.get(r) or getattr(defaults, r) for r in _MODEL_FLAGS}
    for role in ("task", "target"):
        if models["judge"] == models[role]:
            flags = [_flag(_MODEL_FLAGS[r]) for r in ("judge", role) if r in chosen]
            raise UsageError(
                f"{' and '.join(flags) or '--judge-model'}: the judge model {models['judge']} "
                f"is also the {role} model; the judge must differ from the task and target "
                "models, so choose another --judge-model"
            )
    return Models(**models)


def _examples(path: str | None) -> list[Scenario] | None:
    if path is None:
        return None
    try:
        return read_examples(Path(path))
    except ValueError as error:
        raise UsageError(f"--examples: {error}") from None


def _refusal(costs: runner.FixedCosts, force_low_budget: bool) -> str | None:
    """Why a real run would not start, the state folder first (SPEC R4, R17, R23); reads only."""
    try:
        RunStore.check_root(runs_root())
    except RunStoreError as error:
        return str(error)
    return runner.refusal(costs, force_low_budget=force_low_budget)


def _stdin() -> IO[bytes] | None:
    """Piped stdin; None for a terminal, so the tool never waits for typing."""
    stream = sys.stdin
    if stream is None or stream.isatty():
        return None
    return stream.buffer
