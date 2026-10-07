"""What a generating call sees of the user's examples (SPEC R5, R6, R25; ADR-013): when every
example carries a reference, the intake and the rewrites of a fast or checked run receive the pick
examples, never a held-out one, as data in their user JSON (`example_data`), and the intake records
what they show in `from_examples` (`INTAKE_EXAMPLES_SCHEMA`: the plain intake schema with that key
required, possibly empty). The contract check of such a run receives the examples it picks on too,
at most CHECK_SHOWN_MAX, each field cut to FIELD_CHARS (`check_data`; the amendment of 2026-10-07).
Pure: no call."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autoimprover.fastsplit import REFERENCE_MAX_SCENARIOS
from autoimprover.refine import FIELD_CHARS
from autoimprover.types import (
    FROM_EXAMPLE_MAX_CHARS,
    FROM_EXAMPLES_MAX,
    INTAKE_SCHEMA,
    Scenario,
)

# A generating call sees at most this many examples: the most a run picks on (`fastplan`).
SHOWN_MAX = 6
# The contract check sees at most every example a run picks on (WP23), each field cut as the
# reflection's are (`refine.FIELD_CHARS`).
CHECK_SHOWN_MAX = REFERENCE_MAX_SCENARIOS

_RULE = {"type": "string", "maxLength": FROM_EXAMPLE_MAX_CHARS}
INTAKE_EXAMPLES_SCHEMA = {
    "type": "object",
    "required": [*INTAKE_SCHEMA["required"], "from_examples"],
    "properties": {
        **INTAKE_SCHEMA["properties"],
        "from_examples": {"type": "array", "items": _RULE, "maxItems": FROM_EXAMPLES_MAX},
    },
}


def shown(examples: Sequence[Scenario], pick: int) -> tuple[Scenario, ...]:
    """The examples a generating call may see: the first `pick`, those a run picks on, at most
    SHOWN_MAX; the held-out ones come after them and are never shown."""
    return tuple(examples[: min(pick, SHOWN_MAX)])


def example_data(examples: Sequence[Scenario]) -> list[dict[str, Any]]:
    """The examples as a generating call receives them, as data in its user JSON: each input with
    its reference answer (`expected`, None without) and its criteria."""
    return [
        {"input": example.input, "expected": example.expected, "criteria": list(example.criteria)}
        for example in examples
    ]


def check_data(examples: Sequence[Scenario]) -> list[dict[str, Any]]:
    """The examples as the contract check receives them, in its user JSON: the first
    CHECK_SHOWN_MAX, as `example_data` gives them, each field cut to FIELD_CHARS characters."""
    return [
        {
            "input": example.input[:FIELD_CHARS],
            "expected": None if example.expected is None else example.expected[:FIELD_CHARS],
            "criteria": [criterion[:FIELD_CHARS] for criterion in example.criteria],
        }
        for example in examples[:CHECK_SHOWN_MAX]
    ]
