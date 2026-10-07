"""What a generating call sees of the user's examples (SPEC R5, R25; ADR-013): when every example
carries a reference, the intake and the rewrites of a fast or checked run receive the pick examples,
never a held-out one, as data in their user JSON (`example_data`), and the intake records what they
show in `from_examples` (`INTAKE_EXAMPLES_SCHEMA`: the plain intake schema with that key required,
possibly empty). Pure: no call."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autoimprover.types import (
    FROM_EXAMPLE_MAX_CHARS,
    FROM_EXAMPLES_MAX,
    INTAKE_SCHEMA,
    Scenario,
)

# A generating call sees at most this many examples: the most a run picks on (`fastplan`).
SHOWN_MAX = 6

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
