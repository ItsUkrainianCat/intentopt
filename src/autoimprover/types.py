"""Shared types and the Backend protocol. Every module depends on this one, and it on none."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, get_args

Role = Literal["intake", "synth", "task", "judge", "reflect"]
Kind = Literal["template", "task"]
Strictness = Literal["conservative", "balanced", "bold"]
CheckGroup = Literal["format", "constraints", "content"]
# Why the search ended. GEPA 0.1.4 never ends a search by itself (probe), so only a stopper does:
# "budget" is the normal ending (the search used its share of the calls), "clock" means the clock
# share ended it first. Only "clock" is reported as cut short (SPEC R2).
StopCause = Literal["budget", "clock"]
# Why a result was returned or kept; the fixed codes of the `--json` object (SPEC R2, R3, R11, R13).
REASON_CODES = (
    "improved",
    "no_reliable_improvement",
    "already_strong",
    "no_holdout",
    "no_candidate_beat_seed",
    "unconfirmed_out_of_budget",
)
# Delimiter lines the reflection reply puts around the new instruction (ADR-008). The instruction
# may itself contain fenced code blocks (SPEC R9), so a fence cannot be the delimiter.
INSTRUCTION_BEGIN = "<<<INSTRUCTION"
INSTRUCTION_END = "INSTRUCTION>>>"

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
# Scenarios one synthesis call must return (SPEC R11); the fixed costs of R17 assume this number.
SYNTH_COUNT = 12

# Sizes that keep the budget arithmetic bounded (SPEC R15, R17): the holdout never exceeds
# HOLDOUT_MAX scenarios however many the user supplies, and one judge call covers at most
# JUDGE_BATCH_MAX scenarios, so every scoring pass is one judge call.
HOLDOUT_MAX = 6
JUDGE_BATCH_MAX = 6
MINIBATCH_SIZE = 3

# One call: its timeout, its retries and the failures that end a run (SPEC R17, R24).
CALL_TIMEOUT_S = 300
CALL_RETRIES = 2
MAX_CONSECUTIVE_FAILURES = 3
# A system prompt travels as one argv entry; Linux caps a single entry at 131,072 bytes (SPEC R18).
SYSTEM_PROMPT_MAX_BYTES = 100_000

# Exit codes (SPEC R2).
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_BACKEND = 3
EXIT_NOT_LOCKED_DOWN = 4
EXIT_INTERRUPTED = 130

# A run id, the only form `--resume` and `clean` accept (ADR-007): never a path.
RUN_ID_PATTERN = r"\d{8}-\d{6}-[0-9a-f]{8}(-\d+)?"

# JSON schemas of the replies the model must give (SPEC R18 `--json-schema`; wire formats in
# ADR-008). Tests build matching replies with `tests/fakes.py`.
_CHECK_SCHEMA = {
    "type": "object",
    "required": ["id", "group", "text", "rule", "arg"],
    "properties": {
        "id": {"type": "string"},
        "group": {"enum": ["format", "constraints", "content"]},
        "text": {"type": "string"},
        "rule": {"enum": [None, "contains", "not_contains", "max_chars", "min_chars"]},
        "arg": {"type": ["string", "null"]},
    },
}
INTAKE_SCHEMA = {
    "type": "object",
    "required": [
        "goal",
        "kind",
        "keep",
        "constraints",
        "output_format",
        "language",
        "tone",
        "checks",
    ],
    "properties": {
        "goal": {"type": "string"},
        "kind": {"enum": ["template", "task"]},
        "keep": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "output_format": {"type": "string"},
        "language": {"type": "string"},
        "tone": {"type": "string"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA},
    },
}
SYNTH_SCHEMA = {
    "type": "object",
    "required": ["scenarios"],
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "input"],
                "properties": {"id": {"type": "string"}, "input": {"type": "string"}},
            },
            "minItems": SYNTH_COUNT,
            "maxItems": SYNTH_COUNT,
        }
    },
}
JUDGE_SCHEMA = {
    "type": "object",
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["scenario", "checks"],
                "properties": {
                    "scenario": {"type": "string"},
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "pass", "quote"],
                            "properties": {
                                "id": {"type": "string"},
                                "pass": {"type": "boolean"},
                                "quote": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        }
    },
}

# Programmatic check rules. `regex` is not here on purpose: a pattern written by a model would run
# in this process (SPEC R19).
PROGRAMMATIC_RULES = ("contains", "not_contains", "max_chars", "min_chars")
_NUMERIC_RULES = ("max_chars", "min_chars")

# Short names the claude CLI accepts, pinned to the ids this version was written for (SPEC R14).
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
DEFAULT_TASK_MODEL = MODEL_ALIASES["haiku"]
DEFAULT_JUDGE_MODEL = MODEL_ALIASES["opus"]
# The judge when the target is the default judge itself (for example `/improve` in an Opus session).
FALLBACK_JUDGE_MODEL = MODEL_ALIASES["sonnet"]
DEFAULT_REFLECT_MODEL = MODEL_ALIASES["opus"]
DEFAULT_TARGET_MODEL = MODEL_ALIASES["sonnet"]


def canonical_model(name: str) -> str:
    """The full id behind a model name: trims, lower-cases, drops a `[1m]` style suffix and maps
    the aliases above. Unknown names pass through, so R14 compares like with like."""
    key = re.sub(r"\[[^\]]*\]$", "", name.strip().lower())
    if not key:
        raise ValueError("model name is empty")
    return MODEL_ALIASES.get(key, key)


class CallError(Exception):
    """One attempt of a model call failed: non-zero exit, timeout, unreadable reply. Raw backends
    raise it; `Resilient` retries (SPEC R24)."""


class CallFailed(Exception):
    """A call failed all its attempts (SPEC R24). Inside the search the adapter gives that scenario
    a neutral score and keeps the candidate out of the finalists, and a failed reflection call
    skips the iteration; anywhere else (intake, synthesis, seed runs, finalist runs, contract
    checks, scoring the seed candidate) it ends the run as BackendError."""


class BackendError(Exception):
    """The run cannot go on (SPEC R24, exit code 3): three consecutive failed calls to the same
    model, or a call that failed outside the search or while scoring the seed candidate."""


class BudgetExhausted(Exception):
    """The call limit is used up or the clock deadline passed (SPEC R17). Raised by the backend
    seam only; the evaluator and reflection wrappers catch it and stop GEPA through a stopper, so
    it never escapes the search (ADR-004). `cause` is "budget" for the call limit and "clock" for
    the deadline; the runner turns it into the stop cause of the report."""

    def __init__(self, message: str, cause: Literal["budget", "clock"] = "budget") -> None:
        super().__init__(message)
        self.cause = cause


class SessionNotLockedDown(Exception):
    """The claude session reported tools, MCP servers, skills, slash commands, extra agents or a
    non-default output style (SPEC R18, ADR-009, exit code 4)."""


@dataclass(frozen=True)
class Call:
    """One model call. `json_schema` is set on intake, synthesis and judge calls (SPEC R18).

    `sample` separates repeated runs of an otherwise identical call: it is part of the cache key,
    so the second seed run of SPEC R12 (`sample=1`) is a new call, not a cache hit. Reflection
    calls carry their running index for the same reason (ADR-004).
    """

    role: Role
    model: str
    user: str
    system: str = ""
    json_schema: str | None = None
    sample: int = 0

    def __post_init__(self) -> None:
        if "\0" in self.system or len(self.system.encode()) > SYSTEM_PROMPT_MAX_BYTES:
            raise ValueError(
                f"system prompt has a NUL byte or exceeds {SYSTEM_PROMPT_MAX_BYTES} bytes "
                "(SPEC R18)"
            )


@dataclass(frozen=True)
class Reply:
    """`duration_s` is how long the call took when it ran live; a cache hit carries the stored
    value, so the search meter advances the same on a replay (ADR-004)."""

    text: str
    cached: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    duration_s: float = 0.0


class Backend(Protocol):
    """The only way any module talks to a model (ADR-004)."""

    def complete(self, call: Call) -> Reply: ...


@dataclass(frozen=True)
class Check:
    """One pass/fail check. `rule` is None for a judged check, else one of PROGRAMMATIC_RULES.

    `arg` is the needle (contains, not_contains) or a whole number (max_chars, min_chars).
    """

    id: str
    group: CheckGroup
    text: str
    rule: str | None = None
    arg: str | None = None

    def __post_init__(self) -> None:
        if self.group not in get_args(CheckGroup):
            raise ValueError(f"unknown check group {self.group!r}")
        if self.rule is None:
            if self.arg is not None:
                raise ValueError("a judged check takes no argument")
            return
        if self.rule not in PROGRAMMATIC_RULES:
            raise ValueError(f"unknown check rule {self.rule!r}; allowed: {PROGRAMMATIC_RULES}")
        if not self.arg:
            raise ValueError(f"check rule {self.rule!r} needs an argument")
        if self.rule in _NUMERIC_RULES and not (self.arg.isascii() and self.arg.isdigit()):
            raise ValueError(f"check rule {self.rule!r} needs a whole number, got {self.arg!r}")


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

    def __post_init__(self) -> None:
        if self.kind not in get_args(Kind):
            raise ValueError(f"unknown prompt kind {self.kind!r}")


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
    """Model names per role, stored as full ids. The judge is never the task model and never the
    target model, compared after aliases are resolved (SPEC R14)."""

    task: str
    judge: str
    reflect: str
    target: str

    def __post_init__(self) -> None:
        for name in ("task", "judge", "reflect", "target"):
            object.__setattr__(self, name, canonical_model(getattr(self, name)))
        if self.judge == self.task:
            raise ValueError("judge model must differ from the task model (SPEC R14)")
        if self.judge == self.target:
            raise ValueError(
                "judge model must differ from the target model: the run that decides the result "
                "must not be graded by the model that produced it (SPEC R14)"
            )


def default_models(target: str | None = None) -> Models:
    """The default roles for a target model. When the target is the default judge, the judge falls
    back to Sonnet 5.5, so `/improve` works in an Opus session (SPEC R14, R21)."""
    target_id = canonical_model(target) if target else DEFAULT_TARGET_MODEL
    judge = FALLBACK_JUDGE_MODEL if target_id == DEFAULT_JUDGE_MODEL else DEFAULT_JUDGE_MODEL
    return Models(
        task=DEFAULT_TASK_MODEL, judge=judge, reflect=DEFAULT_REFLECT_MODEL, target=target_id
    )


DEFAULT_MODELS = default_models()


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
        if self.strictness not in LENGTH_CAP:
            raise ValueError(f"unknown strictness {self.strictness!r}")
        if type(self.budget) is not int or not 1 <= self.budget <= BUDGET_CEILING:
            raise ValueError(f"budget must be a whole number between 1 and {BUDGET_CEILING}")
        if type(self.wall_clock_s) is not int or self.wall_clock_s < 1:
            raise ValueError("wall_clock_s must be a positive whole number of seconds")


@dataclass(frozen=True)
class Outcome:
    """The result of a run. `prompt` is the original when status is "unchanged" (SPEC R3).

    `verified` is True only when the holdout comparison on the target model decided the result;
    a `--trust-search` win (SPEC R11) stays False. `stop` says why the search ended (None when no
    search ran: already strong, no holdout). `reason` is the human text, `reason_code` one of
    REASON_CODES. `changes` are
    up to 6 lines of "what changed and why", newest first (SPEC R2). `score_*` are holdout scores on
    the target model; `search_score_*` are the seed's and the winner's valset scores on the search
    (task) model, taken during the search at no extra call, reported when it differs (R14a).
    """

    status: Literal["improved", "unchanged"]
    prompt: str
    reason: str
    reason_code: str
    verified: bool = False
    stop: StopCause | None = None
    changes: tuple[str, ...] = ()
    score_before: float | None = None
    score_after: float | None = None
    search_score_before: float | None = None
    search_score_after: float | None = None
    noise: float | None = None
    margin: float | None = None
    length_ratio: float | None = None
    calls_used: int = 0
    run_dir: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("improved", "unchanged"):
            raise ValueError(f"unknown outcome status {self.status!r}")
        if self.reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason code {self.reason_code!r}; allowed: {REASON_CODES}")
        if (self.status == "improved") != (self.reason_code == "improved"):
            raise ValueError("reason code `improved` goes with status improved, and only with it")
        if (
            self.status == "improved"
            and self.score_before is not None
            and self.score_after is not None
            and self.score_after <= self.score_before
        ):
            raise ValueError("an improved outcome must score higher than the original")
