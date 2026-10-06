"""The hidden examples of a bench item (SPEC R26, with R10, R10b, R11, R14, R19; WP21).

An item of the set with `eval_from` gives the run its examples before that index and keeps the
rest hidden (`bench.load_prompts`). The bench then scores the original and the returned prompt
(and with `--baseline naive` the naive rewrite, `bench_judge.naive_call`) on each hidden example:
the plain task call a user's model gets, on the target model (the evaluator's call, without the
fast tiers' suffix; SPEC R10a), and one batched judge call per prompt per JUDGE_BATCH_MAX examples
of the judge model, never the target (SPEC R14), with the references' checks only (`expected` is
"the output agrees with the reference answer in substance", each `criteria` string one more, SPEC
R11; the evaluator's machinery and quote rule, SPEC R10b). Every call carries BENCH_SAMPLE + the
seed, so none is a run's cached call. An example passes for a prompt when every one of its checks
passed; an example whose task or judge call failed is left out of that prompt's count. Wins are the
examples the returned prompt passes and the original fails, losses the reverse, both counted on the
examples both were scored on; the verdict is win, loss or tie by wins against losses (`versus`).
An unchanged result is a tie, and the returned prompt's counts are the original's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from autoimprover.bench_judge import (
    BENCH_SAMPLE,
    PLAIN_OUT_TOKENS,
    Comparison,
    Judged,
    naive_call,
)
from autoimprover.contract import _ask
from autoimprover.evaluator import Evaluator
from autoimprover.fast_prompts import parse_rewrite
from autoimprover.fastplan import call_seconds, reference_seconds, rewrite_tokens, wave_seconds
from autoimprover.parallel import parallel_map
from autoimprover.reference_score import scored
from autoimprover.types import (
    JUDGE_BATCH_MAX,
    Backend,
    BudgetExhausted,
    CallFailed,
    Contract,
    Kind,
    Models,
    Scenario,
)


class Passed(NamedTuple):
    """How many hidden examples a prompt passed, of those it was scored on."""

    passed: int
    of: int


class Hidden(NamedTuple):
    """An item's hidden examples: the original's count, the returned prompt's (the original's
    when unchanged) and the naive rewrite's (None without the baseline or without a rewrite)."""

    original: Passed
    returned: Passed
    naive: Passed | None = None


def versus(original: Mapping[str, bool], mine: Mapping[str, bool]) -> Comparison:
    """`mine` against `original` on the examples both were scored on: a win when it passes more
    of those the original fails than the reverse, a loss mirror-wise, else a tie; an error when
    there is none."""
    common = [name for name in original if name in mine]
    if not common:
        return Comparison("error", 0, 0, 0, "no hidden example was scored for both prompts")
    wins = sum(mine[name] and not original[name] for name in common)
    losses = sum(original[name] and not mine[name] for name in common)
    verdict = "win" if wins > losses else "loss" if losses > wins else "tie"
    return Comparison(verdict, wins, len(common) - wins - losses, losses)


def hidden_calls(improved: bool, naive: bool, hidden: int) -> int:
    """The calls `score_hidden` makes for an item with `hidden` hidden examples, retries not
    counted: the naive rewrite, then per prompt a task run per example and a judge call per
    JUDGE_BATCH_MAX of them (the original's always: the pass rates need it)."""
    prompts = 1 + int(improved) + int(naive)
    return int(naive) + prompts * (hidden + math.ceil(hidden / JUDGE_BATCH_MAX))


def hidden_seconds(
    improved: bool, naive: bool, workers: int, prompt_tokens: int, hidden: int, checks: int
) -> float:
    """The estimated seconds of `score_hidden` (ADR-011's latency model): the naive rewrite, then
    a wave of the task runs, then a wave of the judge calls of `checks` checks per example."""
    prompts = 1 + int(improved) + int(naive)
    first = call_seconds(rewrite_tokens(prompt_tokens)) if naive else 0.0
    runs = wave_seconds(prompts * hidden, workers, call_seconds(PLAIN_OUT_TOKENS))
    batches = math.ceil(hidden / JUDGE_BATCH_MAX)
    judge = reference_seconds(min(hidden, JUDGE_BATCH_MAX), checks)
    return first + runs + wave_seconds(prompts * batches, workers, judge)


def score_hidden(
    backend: Backend,
    original: str,
    candidate: str | None,
    *,
    naive: bool,
    kind: Kind,
    models: Models,
    workers: int,
    seed: int,
    hidden: Sequence[Scenario],
) -> tuple[Judged, Hidden]:
    """`candidate` (None: the run returned the original) and, with `naive`, the naive rewrite,
    each against `original` on the `hidden` examples of a prompt of `kind` (SPEC R26).
    BackendError, SessionNotLockedDown and anything but a failed or refused call propagate."""
    sample = BENCH_SAMPLE + seed
    found: dict[str, Comparison] = {}
    texts = {"original": original} | ({} if candidate is None else {"tool": candidate})
    if naive:
        try:
            texts["naive"] = _ask(
                backend, naive_call(original, models.reflect, sample), parse_rewrite
            )
        except (CallFailed, BudgetExhausted) as error:
            found["naive"] = Comparison("error", 0, 0, 0, f"no naive rewrite: {error}")
    distinct = list(dict.fromkeys(texts.values()))
    inner = max(1, workers // len(distinct))
    contract = Contract(goal="", kind=kind)

    def passes(text: str) -> dict[str, bool]:
        evaluator = Evaluator(backend, contract, models.target, models.judge, sample, inner)
        try:
            entries = evaluator(text, hidden)
        except BudgetExhausted:
            return {}
        return {name: round(score, 9) >= 1 for name, score in scored(entries).items()}

    results = dict(zip(distinct, parallel_map(passes, distinct, workers), strict=True))
    base = results[original]

    def counted(text: str) -> Passed:
        return Passed(sum(results[text].values()), len(results[text]))

    tool = None if candidate is None else versus(base, results[candidate])
    if "naive" in texts:
        found["naive"] = versus(base, results[texts["naive"]])
    returned = counted(original if candidate is None else candidate)
    judged = Judged(tool or Comparison("tie", 0, len(base), 0), found.get("naive"))
    naive_count = counted(texts["naive"]) if "naive" in texts else None
    return judged, Hidden(counted(original), returned, naive_count)
