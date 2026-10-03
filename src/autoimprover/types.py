"""Shared types and the Backend protocol. Every module depends on this one, and it on none."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["intake", "synth", "task", "judge", "reflect"]
Kind = Literal["template", "task"]
Strictness = Literal["conservative", "balanced", "bold"]
CheckGroup = Literal["format", "constraints", "content"]

# Length cap per strictness level, as a multiple of the original's tokens (SPEC R7, R8).
LENGTH_CAP: dict[Strictness, float] = {"conservative": 1.25, "balanced": 1.5, "bold": 2.5}
# A very short prompt may always gain this many tokens (SPEC R7).
LENGTH_FLOOR_TOKENS = 40

BUDGET_DEFAULT = 100
BUDGET_CEILING = 300
WALL_CLOCK_DEFAULT_S = 45 * 60
PROMPT_MAX_CHARS = 20_000


class BackendError(Exception):
    """A model call failed after its retries (SPEC R24)."""


class BudgetExhausted(Exception):
    """The call or wall-clock budget is used up (SPEC R17); the runner returns the best so far."""


class SessionNotLockedDown(Exception):
    """The claude session reported plugins, MCP servers or tools (SPEC R18, exit code 4)."""


@dataclass(frozen=True)
class Call:
    """One model call. `json_schema` is set on intake, synthesis and judge calls (SPEC R18)."""

    role: Role
    model: str
    user: str
    system: str = ""
    json_schema: str | None = None


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
    """One pass/fail check. `rule` is None for a judged check, else a programmatic rule.

    Rules: contains, not_contains, regex, max_chars, min_chars. `arg` is the needle, pattern or
    number.
    """

    id: str
    group: CheckGroup
    text: str
    rule: str | None = None
    arg: str | None = None


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


@dataclass(frozen=True)
class Models:
    """Model names per role. The judge is never the task model (SPEC R14)."""

    task: str
    judge: str
    reflect: str
    target: str

    def __post_init__(self) -> None:
        if self.judge == self.task:
            raise ValueError("judge model must differ from the task model (SPEC R14)")


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
    """The result of a run. `prompt` is the original when status is "unchanged" (SPEC R3)."""

    status: Literal["improved", "unchanged"]
    prompt: str
    reason: str
    score_before: float | None = None
    score_after: float | None = None
    calls_used: int = 0
