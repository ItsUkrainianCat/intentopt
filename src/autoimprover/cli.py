"""Command-line entry point: `autoimprover [flags] <prompt>` (or `--file`, or stdin), `--resume
<id>` and `clean [<id>]`. `backend` replaces the raw model layer and `now` the monotonic clock, so
tests inject a fake model and a fake clock (SPEC R17, R20). Before the first paid call (ARCHITECTURE
section 1) `main` checks the prompt at the boundary (SPEC R1), the flags (`cli_options`), the
models (SPEC R14), the examples, the plan and the state folder (SPEC R4, R17, R23); `--dry` stops
there, with no call and nothing written (`cli_plan`).

`--time` picks the tier (SPEC R25): the deep tier gets the fixed costs of the GEPA search, keeps
the original without a call below 8 scenarios unless `--trust-search` (SPEC R11), and runs
`runner.improve` with the search's share of the clock; the quick, fast and checked tiers get a fast
plan and run `fast.improve_fast` with the whole clock, its stages reported on stderr as they go
(`cli_fast`). Every run gets its folder and `Cached(Resilient(Budgeted(raw)))` on one clock, under
`EffortBackend` in the fast tiers; `--resume <id>` continues it with its saved plan, budget and
clock (SPEC R22). The outcome gets its tier in one place. Ctrl-C and SIGTERM end a run with exit
130: no call starts any more and the raw layer's children are killed (SPEC R2, R21). Every ending
maps to an exit code of SPEC R2; `report.Emitter` is the only writer to stdout."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import cast, get_args

from autoimprover import runner
from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.cli_fast import Progress, checked_keeps, fast_refusal, saved_fast_plan
from autoimprover.cli_input import examples_of, prompt_of
from autoimprover.cli_input import read_prompt as read_prompt  # cli.read_prompt (SPEC R1)
from autoimprover.cli_options import (
    Options,
    Settings,
    UsageError,
    _parser,
    budget,
    flag_name,
    json_requested,
    parse,
    settings,
)
from autoimprover.cli_plan import FastView, PlanView
from autoimprover.efforts import EffortBackend
from autoimprover.fast import improve_fast
from autoimprover.fastplan import FastPlan, fast_plan
from autoimprover.report import Emitter
from autoimprover.runner import count_tokens
from autoimprover.runstore import RunStore, RunStoreError, runs_root
from autoimprover.scenarios import MIN_SCENARIOS_FOR_HOLDOUT
from autoimprover.types import (
    EXIT_BACKEND,
    EXIT_INTERNAL,
    EXIT_INTERRUPTED,
    EXIT_NOT_LOCKED_DOWN,
    EXIT_OK,
    EXIT_USAGE,
    SEARCH_CLOCK_SHARE,
    SYNTH_COUNT,
    Backend,
    BackendError,
    Contract,
    Kind,
    Outcome,
    Plan,
    Scenario,
    SessionNotLockedDown,
)

_CLAUDE_CLI = "autoimprover.claude_cli"


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: Backend | None = None,
    now: Callable[[], float] | None = None,
) -> int:
    """Run the tool and return the exit code of SPEC R2."""
    args = list(sys.argv[1:] if argv is None else argv)
    emit = Emitter(sys.stdout, sys.stderr, json_mode=json_requested(args))
    session = _Session(emit, backend, now)
    fail, folder = emit.error, session.run_dir
    with _sigterm_interrupts():
        try:
            return session.run(args)
        except (UsageError, RunStoreError) as error:
            return fail(EXIT_USAGE, str(error), folder())
        except SessionNotLockedDown as error:
            return fail(EXIT_NOT_LOCKED_DOWN, f"the session is not locked down: {error}", folder())
        except BackendError as error:
            return fail(EXIT_BACKEND, f"backend failure: {error}", folder(), session.resume_line())
        except KeyboardInterrupt:
            session.cancel()
            return fail(EXIT_INTERRUPTED, "interrupted", folder(), session.resume_line())
        except Exception as error:  # the top-level boundary: a bug is reported as exit 1 (SPEC R2)
            message = f"internal error: {type(error).__name__}: {error}"
            return fail(EXIT_INTERNAL, message, folder(), session.resume_line())
        finally:
            session.close()


@contextlib.contextmanager
def _sigterm_interrupts() -> Iterator[None]:
    """SIGTERM (`/improve cancel`, SPEC R21) ends the run as Ctrl-C does: the handler raises
    KeyboardInterrupt in the main thread, once; the previous handler is back when the run ends.
    Off the main thread, where no handler can be set, nothing changes."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    raised = False

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal raised
        if not raised:
            raised = True
            raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


# --- the flows -----------------------------------------------------------------------------------

# The raw model layer of a run, made once its clock and folder exist: an injected test double,
# or the real `ClaudeCliBackend` with the run's clock, empty working folder and current deadline.
RawMaker = Callable[[Clock, RunStore, Callable[[], float]], Backend]
# What a run of a tier is planned with: the fixed costs of the deep search, or a fast plan.
TierPlan = runner.FixedCosts | FastPlan


class _Session:
    """One invocation: its writer, the injected raw layer and clock, the run folder it holds once
    there is one (named in error output, released at exit), and the run's call limit and raw
    layer once they exist (stopped by `cancel`)."""

    def __init__(
        self, emit: Emitter, backend: Backend | None, now: Callable[[], float] | None
    ) -> None:
        self.emit, self.backend, self.now = emit, backend, now
        self.store: RunStore | None = None
        self.budgeted: BudgetedBackend | None = None
        self.raw: Backend | None = None

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

    def cancel(self) -> None:
        """A run being interrupted: no call starts any more, and the raw layer, when it can,
        kills the children it started (SPEC R2, R21, R25)."""
        if self.budgeted is not None:
            self.budgeted.cancel()
        terminate = getattr(self.raw, "terminate", None)
        if callable(terminate):
            terminate()

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
        1; then a new run folder and the run of the tier `--time` picks (SPEC R25)."""
        prompt = prompt_of(opts)
        chosen = settings(opts)
        examples = examples_of(opts.examples)
        if chosen.tier == "deep":
            return self.start_deep(opts, prompt, chosen, examples)
        given = examples is not None
        fplan = fast_plan(chosen.time_s, chosen.workers, count_tokens(prompt), given)
        refusal = _root_refusal() or fast_refusal(fplan)
        plan = _plan(opts, chosen, budget(opts, chosen.tier, chosen.time_s, fplan.est_calls))
        view = FastView(plan, fplan, examples is None, refusal, checked_keeps(fplan, examples))
        return self.open(opts, plan, prompt, view, examples, fplan)

    def start_deep(
        self, opts: Options, prompt: str, chosen: Settings, examples: list[Scenario] | None
    ) -> int:
        """The deep tier: the fixed costs of the search, and the original kept without a call
        below 8 scenarios unless `--trust-search` (SPEC R4, R11, R17)."""
        plan = _plan(opts, chosen, budget(opts, "deep", chosen.time_s, 0))
        n = SYNTH_COUNT if examples is None else len(examples)
        costs = runner.fixed_costs(plan, n, synthesising=examples is None)
        keeps = None
        if n < MIN_SCENARIOS_FOR_HOLDOUT and not opts.trust_search:
            keeps = (
                f"{n} scenarios are fewer than {MIN_SCENARIOS_FOR_HOLDOUT}: no holdout to verify "
                "a result on, so the original is kept; --trust-search accepts an unverified one"
            )
        refusal = _root_refusal() or runner.refusal(costs, force_low_budget=opts.force_low_budget)
        view = PlanView(plan, n, examples is None, costs, refusal, keeps)
        if keeps is not None and not opts.dry and refusal is None:
            kept = Outcome(
                status="unchanged", prompt=prompt, reason=keeps, reason_code="no_holdout"
            )
            self.report(kept, prompt, None, plan, 0.0)
            return EXIT_OK
        return self.open(opts, plan, prompt, view, examples, costs)

    def open(
        self,
        opts: Options,
        plan: Plan,
        prompt: str,
        view: PlanView | FastView,
        examples: list[Scenario] | None,
        tier_plan: TierPlan,
    ) -> int:
        """`--dry` prints the plan and stops; a refusal is exit 2; else a new run folder, the
        user's examples saved in it (a resume never needs the file again, SPEC R22), and the
        run."""
        if opts.dry:
            self.emit.plan(view, dry=True)
            return EXIT_OK
        if view.refusal is not None:
            raise UsageError(view.refusal)
        make_raw = self.raw_maker()
        given = examples is not None
        saved = {"kind": opts.kind, "trust_search": opts.trust_search, "examples": given}
        store = self.store = RunStore.open_or_create(runs_root(), plan, prompt, saved)
        if examples is not None:
            store.save_scenarios(examples)
        self.emit.plan(view, dry=False)
        self.emit.notice(f"run folder: {store.path}")
        return self.improve(make_raw, tier_plan, examples, opts.kind, opts.trust_search)

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
        kind, trust_search, had_examples = _saved(store)
        plan, saved = store.plan, store.scenarios()
        tier_plan: TierPlan = (
            runner.fixed_costs(plan, len(saved or ()) or SYNTH_COUNT, not saved)
            if plan.tier == "deep"
            else saved_fast_plan(store, had_examples)
        )
        self.emit.notice(
            f"resuming run {store.run_id}: {store.calls_used} of {plan.budget} calls used, "
            f"{store.elapsed_s:.0f} s of {plan.wall_clock_s} s of the clock"
        )
        self.emit.notice(f"run folder: {store.path}")
        return self.improve(make_raw, tier_plan, None, kind, trust_search)

    def improve(
        self,
        make_raw: RawMaker,
        tier_plan: TierPlan,
        scenarios: Sequence[Scenario] | None,
        kind: Kind | None,
        trust_search: bool,
    ) -> int:
        """The run in the held folder: one clock carried across resumes; for the deep tier the
        call limit at what the final steps leave and the deadline at the search's share of the
        clock (SPEC R17), for the fast tiers the whole budget and clock (SPEC R25); all printed
        output meanwhile in the run's log, the Outcome reported (SPEC R2)."""
        store = cast(RunStore, self.store)
        plan = store.plan
        clock = Clock(self.now or time.monotonic, elapsed=store.elapsed_s)
        limit, share = plan.budget, 1.0
        if isinstance(tier_plan, runner.FixedCosts):
            limit, share = plan.budget - tier_plan.final, SEARCH_CLOCK_SHARE

        def deadline() -> float:  # the current one: the search's, until the final steps open
            return budgeted.deadline

        self.raw = make_raw(clock, store, deadline)
        budgeted = self.budgeted = BudgetedBackend(
            self.raw,
            limit=limit,
            used=store.calls_used,
            clock=clock,
            deadline=share * plan.wall_clock_s,
            on_call=store.save_progress,
        )
        cached = CachedBackend(ResilientBackend(budgeted), store)
        stack = EffortBackend(cached, plan.efforts)  # above the cache: part of its key (ADR-011)
        with store.open_log() as log, contextlib.redirect_stdout(log):
            if isinstance(tier_plan, FastPlan):
                outcome = improve_fast(
                    store.prompt,
                    plan,
                    tier_plan,
                    backend=stack,
                    budgeted=budgeted,
                    clock=clock,
                    store=store,
                    scenarios=scenarios,
                    kind=kind,
                    log=Progress(self.emit, log),
                    workers=plan.workers,
                )
            else:
                outcome = runner.improve(
                    store.prompt,
                    plan,
                    backend=stack,
                    cache=cached,
                    budgeted=budgeted,
                    clock=clock,
                    store=store,
                    scenarios=scenarios,
                    kind=kind,
                    trust_search=trust_search,
                    log=log,
                )
        self.report(outcome, store.prompt, store.contract(), plan, clock.elapsed())
        return EXIT_OK

    def report(
        self,
        outcome: Outcome,
        original: str,
        contract: Contract | None,
        plan: Plan,
        elapsed_s: float,
    ) -> None:
        """The one place an Outcome gets its tier (SPEC R25) before it is shown (SPEC R2)."""
        mode = dataclasses.replace(outcome, mode=plan.tier)
        self.emit.outcome(mode, original, contract, plan, elapsed_s)

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


def _plan(opts: Options, chosen: Settings, calls: int) -> Plan:
    return Plan(
        chosen.models,
        opts.strictness,
        calls,
        wall_clock_s=chosen.time_s,
        allow_growth=opts.allow_growth,
        merge=opts.merge,
        tier=chosen.tier,
        workers=chosen.workers,
        efforts=chosen.efforts,
    )


def _ignored(opts: Options) -> list[str]:
    """What a resumed run ignores, because its prompt, plan and flags come from its folder
    (SPEC R22; ADR-007): every flag but --resume and --json, and a prompt argument. `--dry` is a
    usage error: there is no plan left to choose."""
    if opts.dry:
        raise UsageError("--dry cannot be combined with --resume: a resumed run keeps its plan")
    ignored = sorted(flag_name(dest) for dest in opts.given - {"resume", "json"})
    return ignored + ["the prompt argument"] * bool(opts.words)


def _saved(store: RunStore) -> tuple[Kind | None, bool, bool]:
    """The flags a run saved in its manifest (`kind`, `trust_search`, and whether it had the
    user's examples, false for a run from before the time tiers)."""
    opts = store.opts
    kind, trust_search = opts.get("kind"), opts.get("trust_search", False)
    examples = opts.get("examples", False)
    if (
        kind not in (None, *get_args(Kind))
        or type(trust_search) is not bool
        or type(examples) is not bool
    ):
        raise RunStoreError(
            f"manifest.json in the run folder {store.path} is damaged (its saved flags); start a "
            f"new run, or remove this one with `autoimprover clean {store.run_id}`"
        )
    return cast(Kind | None, kind), trust_search, examples


def _clean(opts: Options, emit: Emitter) -> int:
    extra = sorted(flag_name(dest) for dest in opts.given - {"json"})
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


def _root_refusal() -> str | None:
    """Why no run can start in the state folder (SPEC R23); reads only, so `--dry` can say it."""
    try:
        RunStore.check_root(runs_root())
    except RunStoreError as error:
        return str(error)
    return None
