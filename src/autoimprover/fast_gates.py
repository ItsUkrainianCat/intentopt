"""The free gates of a fast run's rewrites (SPEC R7, R9, R25; ADR-013), which cost no call. A
rewrite is dropped when its meaning words are the original's or an earlier rewrite's
(`meaning_words`), when it is over the length cap, when it loses a literal
(`contract.literals_preserved`, in `fast_stages`), and, when every example carries a reference,
when it copies a pick example's input (`copies_example`). With references the cap is the larger of
the strictness cap and the original plus GROWTH_TOKENS, so a short prompt can state the rules its
examples show, unless the strictness is conservative (`token_cap`, `length_fits`); the ratio
reported stays the candidate's tokens over the original's. Pure: no call."""

from __future__ import annotations

import re
from collections.abc import Sequence

from autoimprover.runner import _token_cap, count_tokens, length_ok
from autoimprover.types import SYSTEM_PROMPT_MAX_BYTES, Scenario, Strictness

# With references a rewrite may run to the original plus this many tokens (ADR-013).
GROWTH_TOKENS = 150
# An example input this long or longer, found in a rewrite, is a copy (ADR-013); a shorter one
# ("P1?", "yes") may well occur in a rule.
COPY_MIN_CHARS = 20

# A word that carries meaning: a run of letters or digits. Single letters and articles do not.
_WORD = re.compile(r"[^\W_]+")
_ARTICLES = frozenset({"an", "the"})


def meaning_words(text: str) -> tuple[str, ...]:
    """The words of `text` that carry its meaning, in order: lower-cased runs of letters or
    digits, without single letters and the articles a, an and the. Two prompts with the same
    meaning words differ only in case, punctuation, whitespace, single letters or articles."""
    words = _WORD.findall(text.lower())
    return tuple(w for w in words if not (len(w) == 1 and w.isalpha()) and w not in _ARTICLES)


def token_cap(original_tokens: int, strictness: Strictness, references: bool) -> int:
    """The most tokens a rewrite may have: the strictness cap (`runner`, SPEC R7), or with
    `references` the larger of that and the original plus GROWTH_TOKENS, unless conservative."""
    cap = _token_cap(original_tokens, strictness)
    if not references or strictness == "conservative":
        return cap
    return max(cap, original_tokens + GROWTH_TOKENS)


def length_fits(
    original: str, candidate: str, strictness: Strictness, allow_growth: bool, references: bool
) -> tuple[bool, float]:
    """`runner.length_ok`, with the cap of `token_cap` when `references`: whether `candidate` is
    short enough, and its tokens relative to `original`'s."""
    fits, ratio = length_ok(original, candidate, strictness, allow_growth)
    if fits or not references or allow_growth:
        return fits, ratio
    if len(candidate.encode("utf-8", "surrogatepass")) > SYSTEM_PROMPT_MAX_BYTES:
        return False, ratio
    return count_tokens(candidate) <= token_cap(count_tokens(original), strictness, True), ratio


def copies_example(text: str, examples: Sequence[Scenario], original: str) -> bool:
    """Whether `text` holds the input of one of `examples` verbatim: an input of COPY_MIN_CHARS
    characters or more, compared without case and with each run of whitespace as one space. An
    input the original itself holds is not a copy: the rewrite keeps it as a literal (SPEC R9)."""
    found, prompt = _flat(text), _flat(original)
    for example in examples:
        given = _flat(example.input)
        if len(given) >= COPY_MIN_CHARS and given in found and given not in prompt:
            return True
    return False


def _flat(text: str) -> str:
    return " ".join(text.split()).casefold()
