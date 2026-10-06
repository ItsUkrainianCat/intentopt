"""The blind pairwise judge of `autoimprover bench` (SPEC R26, with R10a, R14, R18, R19).

For one prompt: one synthesis call writes PAIRWISE_SCENARIOS fresh scenarios (the fast tiers'
synthesis call under BENCH_SAMPLE + the seed, so never the scenarios a run picked on); the
original and the candidate run on each with the plain task call a user's model gets (the
evaluator's call on the target model, without the fast tiers' 120-word suffix; SPEC R10a); then
the judge model, never the target (SPEC R14), compares the two answers per scenario in both
orders. It sees the original prompt as the request, the situation and two anonymous answers,
never the candidate and never which prompt wrote which answer. A scenario counts for a side only
when both orders pick it; a disagreement is a tie, so a judge that prefers a position never makes
a win. A prompt wins when it wins more scenarios than it loses, loses mirror-wise, else ties.

The naive baseline (`--baseline naive`) is one call, "Improve this prompt.", to the reflection
model at low effort; its rewrite is compared with the original in the same way, on the same
scenarios and the same answers of the original. Each wave runs its calls side by side
(`parallel_map`): the synthesis and the naive rewrite, then the task runs, then the judge calls.
A call that fails all its attempts, or that the call limit or the clock refuses, leaves its
scenario out; no scenario judged is an error, never a tie. Answers and prompts are data: they
travel in the user JSON only (SPEC R19).

DEBT, private names used here until their owners add public seams: `contract._ask`,
`evaluator.Evaluator._task_call`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, cast

from autoimprover.contract import _ask
from autoimprover.evaluator import Evaluator
from autoimprover.fast_prompts import parse_rewrite, parse_synth, synth_call
from autoimprover.fastplan import (
    SYNTH_TOKENS_PER_SCENARIO,
    call_seconds,
    rewrite_tokens,
    wave_seconds,
)
from autoimprover.pairwise_text import (
    JUDGE_ANONYMOUS,
    JUDGE_CRITERIA,
    PAIRWISE_BATCH_SCHEMA,
    PAIRWISE_BATCH_SYSTEM,
)
from autoimprover.parallel import parallel_map
from autoimprover.types import (
    INSTRUCTION_BEGIN,
    INSTRUCTION_END,
    Backend,
    BudgetExhausted,
    Call,
    CallFailed,
    Contract,
    Kind,
    Models,
    Scenario,
)

# Fresh scenarios per prompt, and the sample every bench call carries (plus `--seed`), far from the
# samples of a run's own calls (0, 1, 100 and up, and their retries; SPEC R26).
PAIRWISE_SCENARIOS = 4
BENCH_SAMPLE = 7000

Winner = Literal["A", "B", "tie"]
Vote = Literal["candidate", "original", "tie"]
Verdict = Literal["win", "tie", "loss", "error"]

PAIRWISE_SCHEMA = {
    "type": "object",
    "required": ["winner", "reason"],
    "properties": {
        "winner": {"enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
}
PAIRWISE_SYSTEM = (
    "You compare two answers, for a tool that measures prompts. The user message is JSON: "
    "`request` is what a user asked an assistant for (for a reusable prompt, its instructions), "
    "`situation` is the context it was used in or the message it was applied to, and `answer_A` "
    "and `answer_B` are two anonymous answers, each written by an assistant given its own wording "
    f"of the request. {JUDGE_ANONYMOUS} Judge which answer better serves the likely intent of the "
    f"request in this situation: {JUDGE_CRITERIA} Reply only with JSON valid for the given schema: "
    '`winner` is "A", "B" or "tie" (when neither is clearly better), and `reason` says why in at '
    "most 20 words."
)
# The naive baseline's one call (SPEC R26): the user message is the prompt itself.
NAIVE_SYSTEM = (
    "Improve this prompt. Reply with only the improved prompt between "
    f"{INSTRUCTION_BEGIN} and {INSTRUCTION_END} lines."
)
NAIVE_EFFORT = "low"

# Output tokens the estimate of `--dry` assumes (ADR-011's latency model): a plain answer on the
# target model, which has no 120-word suffix, and a pairwise judge reply.
PLAIN_OUT_TOKENS = 500
PAIRWISE_REPLY_TOKENS = 40

# What a winner in each order means: the first order shows the original as A, the second as B.
_FIRST: dict[str, Vote] = {"A": "original", "B": "candidate", "tie": "tie"}
_SECOND: dict[str, Vote] = {"A": "candidate", "B": "original", "tie": "tie"}


@dataclass(frozen=True)
class Comparison:
    """A prompt against the original: its verdict and the scenarios it won, tied and lost (both
    orders agreeing); `why` says what made an error."""

    verdict: Verdict
    wins: int
    ties: int
    losses: int
    why: str = ""


@dataclass(frozen=True)
class Judged:
    """The tool's rewrite against the original (None: unchanged, nothing to compare) and the naive
    rewrite against it (None: no baseline)."""

    tool: Comparison | None
    naive: Comparison | None


class _Failed(NamedTuple):
    """A call that failed all its attempts, or that the call limit or the clock refused."""

    why: str


def scenario_vote(first: str | None, second: str | None) -> Vote | None:
    """One scenario's vote from the winner of each order (None: that call failed): a side only
    when both orders pick it, else a tie; None when an order has no winner."""
    if first is None or second is None:
        return None
    one, two = _FIRST[first], _SECOND[second]
    return one if one == two else "tie"


def verdict_of(votes: Sequence[Vote | None]) -> Comparison:
    """A prompt's verdict from its scenarios' votes: a win when it won more scenarios than it lost,
    a loss mirror-wise, else a tie; an error when no scenario was judged."""
    counted = [vote for vote in votes if vote is not None]
    wins, ties = counted.count("candidate"), counted.count("tie")
    losses = counted.count("original")
    if not counted:
        return Comparison("error", 0, 0, 0, "no scenario was judged in both orders")
    if wins > losses:
        return Comparison("win", wins, ties, losses)
    return Comparison("loss" if losses > wins else "tie", wins, ties, losses)


def pairwise_call(
    request: str, situation: str, answer_a: str, answer_b: str, model: str, sample: int
) -> Call:
    """The judge call for one scenario in one order (ADR-008's judge role, its own schema)."""
    user = {"request": request, "situation": situation, "answer_A": answer_a, "answer_B": answer_b}
    return Call(
        role="judge",
        model=model,
        user=json.dumps(user),
        system=PAIRWISE_SYSTEM,
        json_schema=json.dumps(PAIRWISE_SCHEMA),
        sample=sample,
    )


def pairwise_batch_call(
    request: str, items: Sequence[tuple[str, str, str, str]], model: str, sample: int
) -> Call:
    """The fast tiers' judge call for one pair in one order: per item (scenario name, situation,
    answer A, answer B), all in the user JSON beside the request (SPEC R19, R25; ADR-012)."""
    scenarios = [
        {"scenario": name, "situation": situation, "answer_A": a, "answer_B": b}
        for name, situation, a, b in items
    ]
    return Call(
        role="judge",
        model=model,
        user=json.dumps({"request": request, "scenarios": scenarios}),
        system=PAIRWISE_BATCH_SYSTEM,
        json_schema=json.dumps(PAIRWISE_BATCH_SCHEMA),
        sample=sample,
    )


def parse_pairwise_batch(text: str, names: Sequence[str]) -> dict[str, tuple[Winner, str]]:
    """Scenario name -> (winner, reason) of a batched pairwise reply; ValueError when it is not
    valid for PAIRWISE_BATCH_SCHEMA, has no result, or answers a scenario that was not asked, or
    one twice. A scenario left out is simply absent. The reply text is never echoed."""
    try:
        reply = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"not valid JSON ({type(error).__name__})") from None
    results = reply.get("results") if isinstance(reply, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError("not an object with a non-empty list of results")
    found: dict[str, tuple[Winner, str]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("a result is not an object")
        name, winner, reason = result.get("scenario"), result.get("winner"), result.get("reason")
        if not isinstance(name, str) or name not in names:
            raise ValueError("a result is not for a scenario that was asked")
        if name in found:
            raise ValueError("the reply answers a scenario twice")
        if winner not in ("A", "B", "tie") or not isinstance(reason, str):
            raise ValueError('a result lacks a winner "A", "B" or "tie" or a string reason')
        found[name] = (winner, reason)
    return found


def parse_winner(text: str) -> Winner:
    """The winner of a pairwise reply; ValueError when the reply is not valid for the schema. The
    reply text is never echoed."""
    try:
        reply = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"not valid JSON ({type(error).__name__})") from None
    if not isinstance(reply, dict) or not isinstance(reply.get("reason"), str):
        raise ValueError("not an object with a string `reason`")
    found = reply.get("winner")
    if found not in ("A", "B", "tie"):
        raise ValueError('`winner` is not "A", "B" or "tie"')
    return found


def naive_call(prompt: str, model: str, sample: int) -> Call:
    """The naive baseline's rewrite of `prompt` (SPEC R26)."""
    return Call(
        role="reflect",
        model=model,
        user=prompt,
        system=NAIVE_SYSTEM,
        sample=sample,
        effort=NAIVE_EFFORT,
    )


def pairwise_calls(improved: bool, naive: bool, count: int = PAIRWISE_SCENARIOS) -> int:
    """The calls `judge` makes for one prompt, retries not counted: the synthesis and the naive
    rewrite, a task run per prompt and scenario (the original's serve both comparisons), and two
    judge calls per comparison and scenario; none when there is nothing to compare."""
    sides = int(improved) + int(naive)
    if not sides:
        return 0
    return 1 + int(naive) + count * (1 + sides) + 2 * count * sides


def pairwise_seconds(
    improved: bool, naive: bool, workers: int, prompt_tokens: int, count: int = PAIRWISE_SCENARIOS
) -> float:
    """The estimated seconds of `judge` for a prompt of `prompt_tokens` tokens: three waves of
    `workers` calls (ADR-011's latency model, `fastplan`)."""
    sides = int(improved) + int(naive)
    if not sides:
        return 0.0
    first = call_seconds(SYNTH_TOKENS_PER_SCENARIO * count)
    if naive:
        first = max(first, call_seconds(rewrite_tokens(prompt_tokens)))
    runs = count * (1 + sides)
    return (
        wave_seconds(1 + int(naive), workers, first)
        + wave_seconds(runs, workers, call_seconds(PLAIN_OUT_TOKENS))
        + wave_seconds(2 * count * sides, workers, call_seconds(PAIRWISE_REPLY_TOKENS))
    )


def judge(
    backend: Backend,
    original: str,
    candidate: str | None,
    *,
    naive: bool,
    kind: Kind,
    models: Models,
    workers: int,
    seed: int,
    count: int = PAIRWISE_SCENARIOS,
) -> Judged:
    """`candidate` (None: the run returned the original) and, with `naive`, the naive rewrite,
    each against `original` on `count` fresh scenarios, a prompt of `kind` (SPEC R26).
    BackendError, SessionNotLockedDown and anything but a failed or refused call propagate."""
    if candidate is None and not naive:
        return Judged(None, None)
    sample = BENCH_SAMPLE + seed
    synthesis = dataclasses.replace(synth_call(original, count, models.reflect), sample=sample)
    rewrite = naive_call(original, models.reflect, sample)
    first: list[Callable[[], object]] = [
        lambda: _ask(backend, synthesis, lambda text: parse_synth(text, count))
    ]
    if naive:
        first.append(lambda: _ask(backend, rewrite, parse_rewrite))
    got = parallel_map(_tried, first, workers)
    scenarios = cast(list[Scenario] | _Failed, got[0])
    found: dict[str, Comparison] = {}
    if isinstance(scenarios, _Failed):
        error = Comparison("error", 0, 0, 0, f"no fresh scenarios: {scenarios.why}")
        return Judged(None if candidate is None else error, error if naive else None)
    sides = {} if candidate is None else {"tool": candidate}
    if naive:
        text = cast(str | _Failed, got[1])
        if isinstance(text, _Failed):
            found["naive"] = Comparison("error", 0, 0, 0, f"no naive rewrite: {text.why}")
        elif text.strip() == original.strip():
            found["naive"] = Comparison("tie", 0, 0, 0)  # nothing to compare
        else:
            sides["naive"] = text
    if sides:
        texts = list(dict.fromkeys([original, *sides.values()]))
        runner = Evaluator(backend, Contract(goal="", kind=kind), models.target, "", sample)
        outputs = _outputs(backend, runner, texts, scenarios, workers)
        found |= _compare(
            backend, original, sides, scenarios, outputs, models.judge, sample, workers
        )
    return Judged(found.get("tool"), found.get("naive"))


def _outputs(
    backend: Backend,
    runner: Evaluator,
    texts: Sequence[str],
    scenarios: Sequence[Scenario],
    workers: int,
) -> dict[tuple[str, str], str | None]:
    """(prompt, scenario id) -> the answer of the plain task call, None when it failed."""
    jobs = [(text, scenario) for text in texts for scenario in scenarios]
    calls = [runner._task_call(text, scenario) for text, scenario in jobs]
    results = parallel_map(
        _tried, [lambda call=call: backend.complete(call).text for call in calls], workers
    )
    return {
        (text, scenario.id): None if isinstance(result, _Failed) else cast(str, result)
        for (text, scenario), result in zip(jobs, results, strict=True)
    }


def _compare(
    backend: Backend,
    original: str,
    sides: Mapping[str, str],
    scenarios: Sequence[Scenario],
    outputs: Mapping[tuple[str, str], str | None],
    model: str,
    sample: int,
    workers: int,
) -> dict[str, Comparison]:
    """Each side's comparison with the original: two judge calls per scenario both answered."""
    asks: list[tuple[str, str, Call]] = []  # (side, scenario id, call), the two orders in turn
    for side, text in sides.items():
        for scenario in scenarios:
            mine, theirs = outputs[(original, scenario.id)], outputs[(text, scenario.id)]
            if mine is None or theirs is None:
                continue
            for a, b in ((mine, theirs), (theirs, mine)):
                asks.append(
                    (
                        side,
                        scenario.id,
                        pairwise_call(original, scenario.input, a, b, model, sample),
                    )
                )
    winners = parallel_map(
        _tried,
        [lambda call=call: _ask(backend, call, parse_winner) for _, _, call in asks],
        workers,
    )
    said: dict[tuple[str, str], list[str | None]] = {}
    for (side, scenario_id, _call), winner in zip(asks, winners, strict=True):
        named = None if isinstance(winner, _Failed) else cast(str, winner)
        said.setdefault((side, scenario_id), []).append(named)
    found: dict[str, Comparison] = {}
    for side in sides:
        votes = []
        for scenario in scenarios:
            first, second = said.get((side, scenario.id), [None, None])
            votes.append(scenario_vote(first, second))
        found[side] = verdict_of(votes)
    return found


def _tried(ask: Callable[[], object]) -> object:
    """`ask()`, or the _Failed of a call that failed all its attempts or was refused."""
    try:
        return ask()
    except (CallFailed, BudgetExhausted) as error:
        return _Failed(str(error))
