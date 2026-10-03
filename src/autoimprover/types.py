"""Shared types and the Backend protocol. Every module depends on this one, and it on none."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

Role = Literal["intake", "synth", "task", "judge", "reflect"]
Kind = Literal["template", "task"]
Strictness = Literal["conservative", "balanced", "bold"]
CheckGroup = Literal["format", "constraints", "content"]
StopCause = Literal["finished", "budget", "clock"]

# Length cap per strictness level, as a multiple of the original's tokens (SPEC R7, R8).
LENGTH_CAP: dict[Strictness, float] = {"conservative": 1.25, "balanced": 1.5, "bold": 2.5}
# A very short prompt may always gain this many tokens (SPEC R7).
LENGTH_FLOOR_TOKENS = 40

BUDGET_DEFAULT = 100
BUDGET_CEILING = 300
WALL_CLOCK_DEFAULT_S = 45 * 60
# The search may use this share of the clock; the rest is kept for the final steps (SPEC R17).
SEARCH_CLOCK_SHARE = 0.75
PROMPT_MAX_CHARS = 20_000

# One call: its timeout, its retries and the failures that end a run (SPEC R17, R24).
CALL_TIMEOUT_S = 300
CALL_RETRIES = 2
MAX_CONSECUTIVE_FAILURES = 3
# A system prompt travels as one argv entry; Linux caps a single entry at 131,072 bytes (SPEC R18).
SYSTEM_PROMPT_MAX_BYTES = 100_000

# Exit codes (SPEC R2).
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BACKEND = 3
EXIT_NOT_LOCKED_DOWN = 4
EXIT_INTERRUPTED = 130

# Programmatic check rules. `regex` is not here on purpose: a pattern written by a model would run
# in this process (SPEC R19).
PROGRAMMATIC_RULES = ("contains", "not_contains", "max_chars", "min_chars")


class BackendError(Exception):
    """A model call failed after its retries (SPEC R24)."""


class BudgetExhausted(Exception):
    """The call limit is used up (SPEC R17). Raised by the backend seam only; the evaluator and
    reflection wrappers catch it and stop GEPA through a stopper, so it never escapes the search
    (ADR-004)."""


class SessionNotLockedDown(Exception):
    """The claude session reported plugins, MCP servers or tools (SPEC R18, exit code 4)."""


@dataclass(frozen=True)
class Call:
    """One model call. `json_schema` is set on intake, synthesis and judge calls (SPEC R18).

    `sample` separates repeated runs of an otherwise identical call: it is part of the cache key,
    so the second seed run of SPEC R12 (`sample=1`) is a new call, not a cache hit.
    """

    role: Role
    model: str
    user: str
    system: str = ""
    json_schema: str | None = None
    sample: int = 0


@dataclass(frozen=True)
class Reply:
    text: str
    cached: bool = False
    tokens_in: int = 0
    tokens_out: int = 0


class Backend(Protocol):
    """The only way any module talks to a model (ADR-004)."""

    def complete(self, call: Call) -> Reply: ...


@dataclass(frozen=True)
class Check:
    """One pass/fail check. `rule` is None for a judged check, else one of PROGRAMMATIC_RULES.

    `arg` is the needle (contains, not_contains) or the number (max_chars, min_chars).
    """

    id: str
    group: CheckGroup
    text: str
    rule: str | None = None
    arg: str | None = None

    def __post_init__(self) -> None:
        if self.rule is None:
            return
        if self.rule not in PROGRAMMATIC_RULES:
            raise ValueError(f"unknown check rule {self.rule!r}; allowed: {PROGRAMMATIC_RULES}")
        if not self.arg:
            raise ValueError(f"check rule {self.rule!r} needs an argument")


@dataclass(frozen=True)
class Contract:
    """What the user's prompt means; frozen for the whole run (SPEC R5)."""

    goal: str
    kind: Kind
    keep: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    output_format: str = ""
    language: str = ""
    tone: str = ""
    checks: tuple[Check, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """A test input (kind=template) or situation (kind=task); see ADR-005."""

    id: str
    input: str
    expected: str | None = None
    criteria: tuple[str, ...] = ()


class BatchEvaluator(Protocol):
    """Scores one candidate on a batch of scenarios with one judge call (SPEC R10).

    Returns one `(score, side_info)` per scenario, in order. `side_info["scores"]` maps each check
    group to its share passed; failed checks and short output excerpts (the ASI of SPEC R16) go in
    other keys, because GEPA's `oa.log` is not available on the batch path (ADR-002).
    """

    def __call__(
        self, candidate: str, scenarios: Sequence[Scenario]
    ) -> list[tuple[float, dict[str, Any]]]: ...


@dataclass(frozen=True)
class Models:
    """Model names per role. The judge is never the task model and never the target (SPEC R14)."""

    task: str
    judge: str
    reflect: str
    target: str

    def __post_init__(self) -> None:
        if self.judge == self.task:
            raise ValueError("judge model must differ from the task model (SPEC R14)")
        if self.judge == self.target:
            raise ValueError(
                "judge model must differ from the target model: the run that decides the result "
                "must not be graded by the model that produced it (SPEC R14)"
            )


DEFAULT_MODELS = Models(
    task="claude-haiku-4-5-20251001",
    judge="claude-opus-5-5",
    reflect="claude-opus-5-5",
    target="claude-sonnet-5-5",
)


@dataclass(frozen=True)
class Plan:
    """Everything decided before the first paid call; `--dry` prints it (SPEC R4)."""

    models: Models
    strictness: Strictness = "conservative"
    budget: int = BUDGET_DEFAULT
    wall_clock_s: int = WALL_CLOCK_DEFAULT_S
    allow_growth: bool = False
    merge: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.budget <= BUDGET_CEILING:
            raise ValueError(f"budget must be between 1 and {BUDGET_CEILING}")
        if self.wall_clock_s < 1:
            raise ValueError("wall_clock_s must be positive")


@dataclass(frozen=True)
class Outcome:
    """The result of a run. `prompt` is the original when status is "unchanged" (SPEC R3).

    `verified` is True only when the holdout comparison on the target model decided the result;
    a `--trust-search` win (SPEC R11) stays False. `stop` says why the search ended.
    """

    status: Literal["improved", "unchanged"]
    prompt: str
    reason: str
    verified: bool = False
    stop: StopCause = "finished"
    score_before: float | None = None
    score_after: float | None = None
    noise: float | None = None
    margin: float | None = None
    length_ratio: float | None = None
    calls_used: int = 0
    run_dir: str = ""
