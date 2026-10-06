"""`autoimprover bench [--prompts FILE] [--time T] [--limit N] [--baseline naive|none]
[--judge-model M] [--seed S] [--json] [--dry]` on the command line (SPEC R26, with R2, R4, R14,
R23, R25).

Before the first paid call: the flags, the set (`bench.load_prompts`, the repository's own set by
default), the tier, models and efforts every run shares (`cli_options.settings`, so a judge equal
to the target is refused, SPEC R14), each prompt's plan and whether a real run of it would refuse
(SPEC R4, R25), and the state folder (SPEC R23). `--dry` prints the plan (`bench_report.DryView`)
with no call and nothing written. A real bench measures the prompts (`bench.run_bench`) in a new
folder `<state>/bench/<id>/`; each prompt runs through `run_prompt`, the command line's own run
of a prompt, injected by `cli`, as is the raw model layer. The summary is the one result on
stdout. Exit 0 once the flags parse, whatever the tool's results (it measures, it does not gate);
2 for bad usage or a refusal; 3 when every prompt's run ended in a backend failure; a session that
is not locked down ends the bench with exit 4 (`cli.main`). Ctrl-C is exit 130 as SPEC R2 has it
(stdout empty, or the error object with `--json`): the `error:` line, then the partial summary of
the prompts measured on stderr, and its object saved as `<state>/bench/<id>/summary.json`.

DEBT, a private name used here until its owner adds a public seam: `cli_options._Parser`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from autoimprover import runner
from autoimprover.bench import (
    MAX_PROMPTS,
    SHIPPED_PROMPTS,
    BenchPlan,
    BenchPrompt,
    Collected,
    RawMaker,
    load_prompts,
    new_bench_folder,
    run_bench,
)
from autoimprover.bench_judge import PAIRWISE_SCENARIOS, pairwise_calls, pairwise_seconds
from autoimprover.bench_report import DryRow, DryView, Summary
from autoimprover.cli_fast import fast_refusal
from autoimprover.cli_options import Options, Settings, UsageError, _Parser, budget, settings
from autoimprover.fastplan import fast_plan
from autoimprover.report import Emitter
from autoimprover.runner import count_tokens
from autoimprover.runstore import RunStore, RunStoreError, make_dirs, runs_root, write_json
from autoimprover.types import (
    EXIT_INTERRUPTED,
    EXIT_OK,
    SYNTH_COUNT,
    BackendError,
    Plan,
    Scenario,
)

SEED_MAX = 999
# Where an interrupted bench keeps its partial summary, in its folder.
SUMMARY_FILE = "summary.json"
_BASELINES = ("naive", "none")


class RunPrompt(Protocol):
    """The command line's run of one prompt: `opts` as for `autoimprover <prompt>`, the user's
    examples (None: synthesised), and the folder its run folder goes in."""

    def __call__(self, opts: Options, examples: list[Scenario] | None, root: Path) -> Collected: ...


@dataclass(frozen=True)
class BenchOptions:
    """The parsed flags of `autoimprover bench`."""

    help: bool = False
    prompts: str | None = None
    time: str | None = None
    limit: int | None = None
    baseline: str = "none"
    judge_model: str | None = None
    seed: int = 0
    json: bool = False
    dry: bool = False


def bench_parser() -> _Parser:
    parser = _Parser(
        prog="autoimprover bench",
        description="Measure the tool: run every prompt of a set through the ordinary pipeline, "
        "then compare each returned rewrite with its original by a blind pairwise judge on fresh "
        "scenarios, in both orders. Spends subscription calls; --dry shows how many.",
        argument_default=argparse.SUPPRESS,
        allow_abbrev=False,
        add_help=False,
    )
    flag = parser.add_argument
    flag("-h", "--help", action="store_true", help="show this help")
    flag(
        "--prompts",
        metavar="FILE",
        help="the set, JSON Lines: id, prompt, kind, examples (default: bench/prompts.jsonl of "
        "this repository)",
    )
    flag("--time", metavar="DURATION", help="each run's clock and tier, as for a run (default 30s)")
    flag("--limit", type=_limit, help=f"the first N prompts, 1 to {MAX_PROMPTS}")
    flag("--baseline", choices=_BASELINES, help="also compare a naive one-call rewrite (naive)")
    flag("--judge-model", metavar="MODEL", help="the judge of the runs and of the comparisons")
    flag("--seed", type=_seed, help=f"a number from 0 to {SEED_MAX} for the bench's own calls")
    flag("--json", action="store_true", help="print one JSON object on stdout")
    flag("--dry", action="store_true", help="print the plan; no model call, nothing written")
    return parser


def bench_command(
    args: Sequence[str],
    emit: Emitter,
    run_prompt: RunPrompt,
    raw_maker: Callable[[], RawMaker],
    now: Callable[[], float] | None,
) -> int:
    """Run `autoimprover bench` with `args` (what follows `bench`); the exit code of SPEC R2."""
    opts = BenchOptions(**vars(bench_parser().parse_args(list(args))))
    if opts.help:
        emit.help(bench_parser().format_help())
        return EXIT_OK
    path = SHIPPED_PROMPTS if opts.prompts is None else Path(opts.prompts)
    try:
        prompts = load_prompts(path, opts.limit)
    except ValueError as error:
        raise UsageError(f"--prompts: {error}") from None
    shared = Options(time=opts.time, judge_model=opts.judge_model)
    chosen = settings(shared)
    data = "\n".join(f"{item.id}\t{item.prompt}" for item in prompts).encode()
    folder = new_bench_folder(runs_root().parent, data)
    baseline = opts.baseline == "naive"
    rows, refusal = [], _root_refusal(folder.parent)
    for item in prompts:
        tokens = count_tokens(item.prompt)
        calls, seconds, why = _run_plan(item, shared, chosen, tokens)
        refusal = refusal or (None if why is None else f"prompt {item.id}: {why}")
        bench_s = pairwise_seconds(True, baseline, chosen.workers, tokens)
        rows.append(DryRow(item.id, calls, seconds, pairwise_calls(True, baseline), bench_s))
    view = DryView(
        str(path),
        chosen.tier,
        chosen.time_s,
        chosen.workers,
        chosen.models,
        baseline,
        PAIRWISE_SCENARIOS,
        tuple(rows),
        refusal,
    )
    if opts.dry:
        emit.plan(view, dry=True)
        return EXIT_OK
    if refusal is not None:
        raise UsageError(refusal)
    make_raw = raw_maker()

    def run_one(item: BenchPrompt, root: Path) -> Collected:
        """The prompt as `autoimprover --time T --judge-model M --kind K <prompt>` would run it."""
        examples = None if item.examples is None else list(item.examples)
        given = Options(
            words=(item.prompt,), time=opts.time, judge_model=opts.judge_model, kind=item.kind
        )
        return run_prompt(given, examples, root)

    plan = BenchPlan(chosen.models, chosen.efforts, chosen.workers, baseline, opts.seed)
    emit.notice(f"bench folder: {folder}")
    measured = run_bench(
        prompts,
        plan,
        folder,
        run_one=run_one,
        make_raw=make_raw,
        now=now or time.monotonic,
        note=emit.notice,
    )
    done = measured.rows
    if done and not measured.interrupted and all(row.status == "error" for row in done):
        raise BackendError(
            f"every prompt's run ended in a backend failure; the last: {done[-1].error}"
        )
    summary = Summary(measured, len(prompts), chosen.tier, chosen.time_s, baseline, str(folder))
    if measured.interrupted:
        return _interrupted(emit, summary, folder)
    found: Any = summary.object()
    emit.result(json.dumps(found, allow_nan=False) + "\n" if emit.json_mode else summary.text())
    return EXIT_OK


def _interrupted(emit: Emitter, summary: Summary, folder: Path) -> int:
    """Ctrl-C ended the bench (SPEC R2): exit 130 with its `error:` line (with `--json` the error
    object is stdout's one object), then the partial summary as text on stderr; the partial
    summary object is saved as the bench folder's SUMMARY_FILE. A summary that cannot be saved is
    said in the error line."""
    path = folder / SUMMARY_FILE
    try:
        make_dirs(folder)
        write_json(path, summary.object())
        saved = f"the partial summary is in {path}"
    except OSError as error:
        saved = f"the partial summary could not be saved to {path}: {error.strerror or error}"
    measured = len(summary.measured.rows)
    code = emit.error(
        EXIT_INTERRUPTED,
        f"interrupted after {measured} of {summary.prompts} prompts; {saved}",
        str(folder),
    )
    emit.notice(summary.text())
    return code


def _run_plan(
    item: BenchPrompt, shared: Options, chosen: Settings, tokens: int
) -> tuple[int, float, str | None]:
    """A prompt's run: its estimated calls and seconds, and why a real run would refuse. The deep
    tier's calls are its budget and its seconds its clock (SPEC R4, R17, R25)."""
    if chosen.tier != "deep":
        fplan = fast_plan(chosen.time_s, chosen.workers, tokens, item.examples is not None)
        return fplan.est_calls, fplan.est_seconds, fast_refusal(fplan)
    calls = budget(shared, "deep", chosen.time_s, 0)
    plan = Plan(chosen.models, chosen.strictness, calls, wall_clock_s=chosen.time_s, tier="deep")
    n = SYNTH_COUNT if item.examples is None else len(item.examples)
    costs = runner.fixed_costs(plan, n, synthesising=item.examples is None)
    return calls, float(chosen.time_s), runner.refusal(costs, force_low_budget=False)


def _root_refusal(root: Path) -> str | None:
    """Why no bench can be kept under `root` (SPEC R23); reads only, so `--dry` can say it."""
    try:
        RunStore.check_root(root)
    except RunStoreError as error:
        return str(error)
    return None


def _limit(text: str) -> int:
    if not (text.isascii() and text.isdigit() and 1 <= int(text) <= MAX_PROMPTS):
        raise argparse.ArgumentTypeError(f"a whole number from 1 to {MAX_PROMPTS}")
    return int(text)


def _seed(text: str) -> int:
    if not (text.isascii() and text.isdigit() and int(text) <= SEED_MAX):
        raise argparse.ArgumentTypeError(f"a whole number from 0 to {SEED_MAX}")
    return int(text)
