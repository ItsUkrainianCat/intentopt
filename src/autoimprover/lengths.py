"""The length of a candidate against its original (SPEC R7, R8; ADR-004): `count_tokens` is the
one approximate token count the whole tool uses, `length_ok` the free length gate."""

from __future__ import annotations

import math
import re

from autoimprover.types import (
    LENGTH_CAP,
    LENGTH_FLOOR_TOKENS,
    SYSTEM_PROMPT_MAX_BYTES,
    Strictness,
)

# A token, approximately: a run of letters, digits and underscores, or one other visible character.
_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """A model's token count, approximately and with no tokenizer to pin: each run of word
    characters (any script) is one token, each other visible character one more. It undercounts
    long words, but counts the original and the candidate alike, as the cap of SPEC R7 needs."""
    return sum(1 for _ in _TOKEN.finditer(text))


def _token_cap(original_tokens: int, strictness: Strictness) -> int:
    """The most tokens a result may have: LENGTH_CAP times the original's, at least the original
    plus LENGTH_FLOOR_TOKENS (SPEC R7, R8)."""
    capped = math.floor(LENGTH_CAP[strictness] * original_tokens)
    return max(capped, original_tokens + LENGTH_FLOOR_TOKENS)


def length_ok(
    original: str, candidate: str, strictness: Strictness, allow_growth: bool
) -> tuple[bool, float]:
    """Whether `candidate` is short enough to be the result, and its length in tokens relative to
    `original` (an original without tokens counts as one). The cap of `strictness` holds unless
    `allow_growth` (SPEC R7, R8); a candidate over SYSTEM_PROMPT_MAX_BYTES bytes is refused
    whatever the flags, because a template candidate travels as one argument (ADR-004)."""
    before, after = count_tokens(original), count_tokens(candidate)
    ratio = after / max(before, 1)
    if len(candidate.encode("utf-8", "surrogatepass")) > SYSTEM_PROMPT_MAX_BYTES:
        return False, ratio
    return allow_growth or after <= _token_cap(before, strictness), ratio
