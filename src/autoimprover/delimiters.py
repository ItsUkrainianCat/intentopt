"""The delimiter lines around a new prompt in a model's reply (ADR-008), read the same way by the
fast tiers' rewrites and reflections (`fast_prompts.parse_rewrite`) and the deep tier's
reflections (`search`): a begin line holds only INSTRUCTION_BEGIN or the bare `<<<`, an end line
only INSTRUCTION_END or the bare `>>>`, whitespace aside, because the model drops the word under
load (the live bench of 2026-10-06). What lies between them, and what each caller refuses in it,
is the caller's.
"""

from __future__ import annotations

from collections.abc import Sequence

from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END

# The marks of a delimiter line, its whitespace removed: the asked ones and the bare ones.
BEGIN_MARKS = frozenset({INSTRUCTION_BEGIN, "<<<"})
END_MARKS = frozenset({INSTRUCTION_END, ">>>"})


def span(lines: Sequence[str]) -> tuple[int, int] | None:
    """The indexes of the first begin line and of the last end line of `lines`, or None when
    either is missing or the end comes first."""
    marks = ["".join(line.split()) for line in lines]
    begins = [n for n, mark in enumerate(marks) if mark in BEGIN_MARKS]
    ends = [n for n, mark in enumerate(marks) if mark in END_MARKS]
    if not begins or not ends or ends[-1] < begins[0]:
        return None
    return begins[0], ends[-1]
