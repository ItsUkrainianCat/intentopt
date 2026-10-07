"""The order of the user's examples before a fast or checked run splits them into the pick examples
(the head) and the held-out ones (the rest) (SPEC R11, R25; WP25). When every example's `expected`
is a short label, at most LABEL_MAX_CHARS characters, and the examples have LABELS_MIN to LABELS_MAX
labels (case and whitespace runs ignored), a classification-like task, they are dealt round robin
over the labels, the labels in order of first appearance and each label's examples in file order,
so every head of the order covers the labels as evenly as their counts allow: a pick set made of
the first examples of the file could hold one label only and show no failure. Free-text references
(more labels, or a longer one), criteria only, a single label or an example without `expected`
keep the file order. Pure: the same examples give the same order, and ordering it again changes
nothing, so a resumed run splits as the run did (SPEC R22)."""

from __future__ import annotations

from collections.abc import Sequence

from autoimprover.types import Scenario

LABEL_MAX_CHARS = 40
LABELS_MIN = 2
LABELS_MAX = 12


def label(example: Scenario) -> str | None:
    """The example's `expected` with case and whitespace runs ignored, or None without one."""
    return None if example.expected is None else _flat(example.expected)


def _flat(text: str) -> str:
    return " ".join(text.split()).casefold()


def _labels(examples: Sequence[Scenario]) -> list[str] | None:
    """The labels in order of first appearance when the examples are classification-like (every
    `expected` short, LABELS_MIN to LABELS_MAX labels), else None."""
    found: dict[str, None] = {}
    for example in examples:
        if example.expected is None or len(example.expected) > LABEL_MAX_CHARS:
            return None
        found[_flat(example.expected)] = None
    return list(found) if LABELS_MIN <= len(found) <= LABELS_MAX else None


def ordered(examples: Sequence[Scenario]) -> list[Scenario]:
    """`examples` round robin over their labels when they are classification-like, else in file
    order."""
    labels = _labels(examples)
    if labels is None:
        return list(examples)
    queues = {name: [e for e in examples if label(e) == name] for name in labels}
    order: list[Scenario] = []
    for round_ in range(max(len(queue) for queue in queues.values())):
        order += [queue[round_] for queue in queues.values() if round_ < len(queue)]
    return order


def coverage(examples: Sequence[Scenario], pick: int) -> tuple[int, int] | None:
    """(k, m) when the first `pick` of `examples`, as a run split them, cover k of the m labels
    of all of them and k < m; None when they cover every label or the examples are not
    classification-like."""
    labels = _labels(examples)
    if labels is None:
        return None
    covered = {label(example) for example in examples[:pick]}
    return None if len(covered) == len(labels) else (len(covered), len(labels))
