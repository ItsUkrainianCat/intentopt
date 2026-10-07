"""Text as the report prints it (SPEC R2, R19), split from `report.py`, which re-exports all three:
model-written text is data, never instructions to the terminal, so `clean_text` removes escape
sequences and control characters and `one_line` also folds whitespace; `word_diff` is the
word-level diff of a returned prompt. Pure: no I/O."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# A terminal escape sequence: CSI (ESC [ or the one-byte CSI, parameters, a final byte), OSC
# (ESC ] up to BEL or ESC \), or ESC and one more character. Every part is linear: the ranges
# of neighbouring quantified parts do not overlap.
_ESCAPE = re.compile(
    r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[@-Z\\-_]?"
)
# Control characters other than newline and tab, DEL, the C1 controls, and the bidirectional
# overrides and isolates that can reorder what a terminal shows.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def clean_text(text: str) -> str:
    """`text` without terminal escape sequences and control characters; newlines, tabs and
    ordinary Unicode stay (SPEC R19)."""
    return _CONTROL.sub("", _ESCAPE.sub("", text))


def one_line(text: str) -> str:
    """`clean_text` on one line: every run of whitespace becomes one space."""
    return " ".join(clean_text(text).split())


def word_diff(before: str, after: str) -> str:
    """A word-level diff: removed words in `[-...-]`, added words in `{+...+}` (SPEC R2)."""
    old, new = before.split(), after.split()
    parts: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            parts += old[i1:i2]
            continue
        if i2 > i1:
            parts.append(f"[-{' '.join(old[i1:i2])}-]")
        if j2 > j1:
            parts.append(f"{{+{' '.join(new[j1:j2])}+}}")
    return " ".join(parts)
