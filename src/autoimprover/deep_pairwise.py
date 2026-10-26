"""The deep tier's final decision when the examples carry no reference (SPEC R25; ADR-014): the
holdout answers of the original and of a finalist, both from the target model, go to the blind
pairwise judge of the fast tiers (`fast_pairwise`), anonymously and in both orders, the judge
seeing the original prompt as the request and never a candidate prompt. The original's two runs
compared with each other give the noise: the scenarios with an agreed winner. A call that fails
all its attempts is a CallFailed, which the flow turns into a BackendError (SPEC R24); unlike the
fast tiers there is no wave to fit in, so a call is never dropped as a tie."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from autoimprover.bench_judge import parse_pairwise_batch
from autoimprover.contract import _ask
from autoimprover.fast_pairwise import (
    Item,
    Preference,
    Said,
    noise_count,
    pair_calls,
    pair_items,
    preference,
)
from autoimprover.types import Backend, BackendError, CallFailed, Scenario

Outputs = Mapping[str, str | CallFailed]

# In the reason of every deep result decided this way, so the report words it as a preference.
MARK = "pairwise judge on the holdout"


def _orders(
    backend: Backend,
    model: str,
    request: str,
    holdout: Sequence[Scenario],
    original: Outputs,
    other: Outputs,
) -> tuple[list[Item], Said, Said]:
    """The scenarios of `holdout` both runs answered and the judge's replies to them in the two
    orders (`original` as A, then as B)."""
    items = pair_items(holdout, original, other)
    if not items:
        raise BackendError("the two runs left no holdout scenario both answered")
    names = [item[0] for item in items]
    first, second = (
        _ask(backend, call, lambda text: parse_pairwise_batch(text, names))
        for call in pair_calls(request, items, model)
    )
    return items, first, second


def noise(
    backend: Backend,
    model: str,
    request: str,
    holdout: Sequence[Scenario],
    first_run: Outputs,
    second_run: Outputs,
) -> int:
    """The scenarios of `holdout` where the original's two runs have an agreed winner."""
    items, first, second = _orders(backend, model, request, holdout, first_run, second_run)
    return noise_count(items, first, second)


def preferred(
    backend: Backend,
    model: str,
    request: str,
    holdout: Sequence[Scenario],
    original: Outputs,
    finalist: Outputs,
) -> Preference:
    """The scenarios of `holdout` the finalist won, tied and lost against the original."""
    items, first, second = _orders(backend, model, request, holdout, original, finalist)
    return preference(items, first, second)
