"""The command line of `autoimprover` (SPEC R2, R4, R14, R17, R22, R25): the flags, `parse`,
and `settings`, which turns the parsed flags of a new run into its tier, clock, workers, efforts
and models. `--time` picks the tier (quick, fast, checked, deep) and is the run's wall clock,
`--deep` is `--time 20m`; a flag given always wins over the tier's default (SPEC R25; ADR-011
"Customisation"). `--budget`, `--kind` and `--strictness` are checked while parsing; the other
values after it, as the model names are, and every error names its flag (exit 2)."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, get_args

from autoimprover.fastplan import tier_for
from autoimprover.types import (
    BUDGET_CEILING,
    BUDGET_DEFAULT,
    EFFORT_LEVELS,
    LENGTH_CAP,
    Efforts,
    Kind,
    Models,
    Strictness,
    Tier,
    canonical_model,
    default_efforts,
    default_models,
)

# The clock of a run (SPEC R25): the default, what `--deep` stands for, and the longest allowed.
TIME_DEFAULT = "30s"
DEEP_TIME_S = 20 * 60
TIME_MAX_S = 3 * 60 * 60
WORKERS_DEFAULT = 6
WORKERS_MAX = 16
# The deep tier's default budget: one call per 12 s of the clock, from 20 to 100 calls (SPEC R25).
DEEP_SECONDS_PER_CALL = 12
DEEP_BUDGET_MIN = 20
# The quick, fast and checked tiers' strictness when `--strictness` is not given (SPEC R25).
FAST_STRICTNESS: Strictness = "balanced"
# The fast tiers' default budget: three times the plan's estimate, room for retries.
FAST_BUDGET_FACTOR = 3
_UNITS = {"s": 1, "m": 60, "h": 3600}
_MODEL_FLAGS = {role: f"{role}_model" for role in ("task", "judge", "reflect", "target")}
_DEEP_ONLY = ("merge", "trust_search", "force_low_budget")


class UsageError(Exception):
    """Bad input or usage, or a refusal before any paid call (exit 2); the message names the flag
    or folder to change (SPEC R2)."""


@dataclass(frozen=True)
class Options:
    """The parsed command line; `given` holds the dest of every flag written on it."""

    given: frozenset[str] = frozenset()
    words: tuple[str, ...] = ()
    help: bool = False
    file: str | None = None
    examples: str | None = None
    kind: Kind | None = None
    budget: int | None = None
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
    time: str | None = None
    deep: bool = False
    workers: str | None = None
    effort: str | None = None
    task_effort: str | None = None
    judge_effort: str | None = None
    reflect_effort: str | None = None


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def _parser() -> _Parser:
    parser = _Parser(
        prog="autoimprover",
        description="Improve a prompt, scored by running it on test scenarios: by default a "
        "30-second pipeline of parallel stages, with --deep (or --time 10m and up) a GEPA search.",
        epilog="autoimprover clean [<id>] removes one run folder, or all of them. Exit codes: 0 "
        "done (also when the original is kept), 1 internal error, 2 bad input or a refusal "
        "before any paid call, 3 backend failure, 4 claude session not locked down, 130 "
        "interrupted.",
        argument_default=argparse.SUPPRESS,
        allow_abbrev=False,
        add_help=False,
    )
    flag = parser.add_argument
    levels = ", ".join((*EFFORT_LEVELS, "default"))
    flag("words", nargs="*", metavar="prompt", help="the prompt, as one quoted argument")
    flag("-h", "--help", action="store_true", help="show this help")
    flag("--file", metavar="PATH", help="read the prompt from this UTF-8 file")
    flag("--examples", metavar="PATH", help="test scenarios, JSON Lines: input, expected, criteria")
    flag("--kind", choices=get_args(Kind), help="template or task (default: guessed)")
    flag(
        "--time",
        metavar="DURATION",
        help="the run's clock, like 30s, 5m or 1h (default 30s): quick from 15s, fast from 25s, "
        "checked from 1m, deep from 10m; at most 3h",
    )
    flag("--deep", action="store_true", help="the GEPA search: --time 20m")
    flag(
        "--workers",
        metavar="N",
        help=f"calls at a time, 1 to {WORKERS_MAX} (default {WORKERS_DEFAULT})",
    )
    flag("--budget", type=_budget, help="model calls (default: from the tier)")
    flag("--strictness", choices=tuple(LENGTH_CAP), help="how far a rewrite may go")
    flag("--allow-growth", action="store_true", help="no length cap")
    for role, dest in _MODEL_FLAGS.items():
        flag(f"--{dest.replace('_', '-')}", metavar="MODEL", help=f"the {role} model")
    flag("--effort", metavar="LEVEL", help=f"claude effort of every role: {levels}")
    for role in ("task", "judge", "reflect"):
        flag(f"--{role}-effort", metavar="LEVEL", help=f"the {role} role's effort, over --effort")
    flag("--merge", action="store_true", help="let GEPA merge candidates (deep)")
    flag(
        "--trust-search",
        action="store_true",
        help="deep, below 8 scenarios: accept an unverified win",
    )
    flag("--force-low-budget", action="store_true", help="deep: run on fewer than 4 iterations")
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


def flag_name(dest: str) -> str:
    return f"--{dest.replace('_', '-')}"


def json_requested(args: list[str]) -> bool:
    """Whether `--json` is among the flags, known before parsing so a usage error is an object."""
    return "--json" in (args[: args.index("--")] if "--" in args else args)


# --- what the flags choose (SPEC R14, R25) -------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """What the flags choose for a new run: its tier, its clock in seconds, the calls at a time,
    the effort of each role, the models and the strictness (SPEC R8, R14, R25)."""

    tier: Tier
    time_s: int
    workers: int
    efforts: Efforts
    models: Models
    strictness: Strictness = "conservative"


def settings(opts: Options) -> Settings:
    """The tier, clock, workers, efforts and models `opts` choose; UsageError naming the flag
    for a bad value, for `--deep` beside `--time`, and for a flag of the deep tier only
    (`--merge`, `--trust-search`, `--force-low-budget`) with another tier."""
    if opts.deep and opts.time is not None:
        raise UsageError("--deep is --time 20m; give --deep or --time, not both")
    time_s = DEEP_TIME_S if opts.deep else duration(opts.time or TIME_DEFAULT)
    if time_s > TIME_MAX_S:
        raise UsageError(f"--time: at most 3h ({TIME_MAX_S} s), not {opts.time}")
    try:
        tier = tier_for(time_s)
    except ValueError as error:
        raise UsageError(str(error)) from None
    deep_only = [flag_name(dest) for dest in _DEEP_ONLY if getattr(opts, dest)]
    if tier != "deep" and deep_only:
        raise UsageError(
            f"{deep_only[0]} applies to the deep tier only (--deep, or --time 10m or more); "
            f"--time {opts.time or TIME_DEFAULT} is the {tier} tier"
        )
    # The fast tiers default to balanced, so a rewrite may change meaning-bearing structure
    # (SPEC R25 "Quality of the rewrites"); the deep tier keeps conservative; a flag given wins.
    given = "strictness" in opts.given or tier == "deep"
    strictness = opts.strictness if given else FAST_STRICTNESS
    workers = _workers(opts.workers)
    return Settings(tier, time_s, workers, _efforts(opts, tier), models(opts, tier), strictness)


def duration(text: str) -> int:
    """`<N>s`, `<N>m` or `<N>h` in seconds, N a whole number (SPEC R25)."""
    match = re.fullmatch(r"([0-9]+)([smh])", text)
    if match is None:
        raise UsageError(
            f"--time: give a whole number with s, m or h, like 30s, 5m or 1h; not {text!r}"
        )
    return int(match.group(1)) * _UNITS[match.group(2)]


def budget(opts: Options, tier: Tier, time_s: int, est_calls: int) -> int:
    """`--budget` when given, else the tier's default: for deep the clock at 12 s a call, from 20
    to 100 calls; for the fast tiers three times the plan's estimate `est_calls`, at most the
    ceiling (SPEC R17, R25)."""
    if opts.budget is not None:
        return opts.budget
    if tier == "deep":
        return min(BUDGET_DEFAULT, max(DEEP_BUDGET_MIN, time_s // DEEP_SECONDS_PER_CALL))
    return min(BUDGET_CEILING, FAST_BUDGET_FACTOR * est_calls)


def _workers(text: str | None) -> int:
    if text is None:
        return WORKERS_DEFAULT
    if not (text.isascii() and text.isdigit() and 1 <= int(text) <= WORKERS_MAX):
        raise UsageError(f"--workers: a whole number from 1 to {WORKERS_MAX}, not {text!r}")
    return int(text)


def _efforts(opts: Options, tier: Tier) -> Efforts:
    """Each role's effort: its own flag, else `--effort`, else the tier's default."""
    every = None if opts.effort is None else _level("effort", opts.effort)
    tier_default = default_efforts(tier)
    levels: dict[str, str | None] = {}
    for role in ("task", "judge", "reflect"):
        dest = f"{role}_effort"
        own = getattr(opts, dest)
        if own is not None:
            levels[role] = _level(dest, own)
        else:
            levels[role] = every if opts.effort is not None else getattr(tier_default, role)
    return Efforts(**levels)


def _level(dest: str, text: str) -> str | None:
    """An effort level; `default` is None, the model's own (SPEC R25)."""
    if text == "default":
        return None
    if text not in EFFORT_LEVELS:
        levels = ", ".join((*EFFORT_LEVELS, "default"))
        raise UsageError(f"{flag_name(dest)}: one of {levels}; not {text!r}")
    return text


def models(opts: Options, tier: Tier) -> Models:
    """The four models as full ids: the tier's defaults for the target (SPEC R14's fallback judge
    when the target is the default judge), overridden by the flags given; a judge equal to the
    task or target model is refused naming the flags that chose it."""
    chosen: dict[str, str] = {}
    for role, dest in _MODEL_FLAGS.items():
        name = getattr(opts, dest)
        if name is not None:
            try:
                chosen[role] = canonical_model(name)
            except ValueError as error:
                raise UsageError(f"{flag_name(dest)}: {error}") from None
    defaults = default_models(chosen.get("target"), tier)
    picked: dict[str, str] = {r: chosen.get(r) or getattr(defaults, r) for r in _MODEL_FLAGS}
    for role in ("task", "target"):
        if picked["judge"] == picked[role]:
            flags = [flag_name(_MODEL_FLAGS[r]) for r in ("judge", role) if r in chosen]
            raise UsageError(
                f"{' and '.join(flags) or '--judge-model'}: the judge model {picked['judge']} "
                f"is also the {role} model; the judge must differ from the task and target "
                "models, so choose another --judge-model"
            )
    return Models(**picked)
