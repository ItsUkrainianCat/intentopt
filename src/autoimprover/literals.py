"""The literals of a prompt, the spans a rewrite must keep verbatim (SPEC R9): code blocks, inline
code, placeholders, URLs, file paths and quoted strings. Moved out of `contract.py`, which
re-exports `literals` and `literals_preserved`. Pure: no call.
"""

from __future__ import annotations

import re

# Every pattern below runs in linear time: no nested quantifiers, and no two neighbouring
# quantified parts that can match the same character, so a failed attempt gives back at most its
# own span once. Literals do not cross a line break except fenced code blocks, which are found
# line by line in `_fences`.
_INLINE_CODE = re.compile(r"`[^`\r\n]+`")
_PLACEHOLDER = re.compile(
    r"\{\{[ \t]*\w[\w.-]*[ \t]*\}\}"  # {{name}}, {{ name }}
    r"|\$?\{\w[\w.-]*\}"  # {name}, ${name}
    r"|%\(\w+\)[A-Za-z]"  # %(name)s
    r"|<[A-Za-z][\w-]*>"  # <topic>, <PATH>; no spaces, so `a < b and c > d` is not one
)
_DOUBLE_QUOTED = re.compile(r'"[^"\r\n]*"|“[^“”\r\n]*”')
# A single quote counts only around 2 or more characters, opening at a word start and closing
# before a non-word character, so the apostrophes of don't, it's, rock 'n' roll and the users' are
# not quotes. An opening that is an elision ('90s, 'em, 'til, ...) is not a quote either: it would
# pair with the next plural possessive ("the '90s kids' toys").
_SINGLE_QUOTED = re.compile(
    r"(?<!\w)'(?!(?i:\d\ds|em|til|cause|tis|twas|bout|round|n)(?!\w))"
    r"[^\s'][^'\r\n]*[^\s']'(?!\w)"
)
_URL = re.compile(r"(?i:https?)://\S+")
_URL_TRAILING = ".,;:!?)]}>'\"`*”’"
_PATH_RUN = re.compile(r"[\w.~/-]+")
_PATH_PREFIXES = ("~/", "./", "../", "/")
_FENCE_OPEN = re.compile(r"[ \t]*(`{3,}|~{3,})")


def literals(prompt: str) -> tuple[str, ...]:
    """The spans of `prompt` a rewrite must keep verbatim, deduplicated, in order of first
    appearance, SPEC R9: fenced code blocks (an unterminated fence runs to the end of the text),
    inline code, placeholders ({name}, {{name}}, ${name}, <name>, %(name)s), http and https URLs,
    file paths and quoted strings. A span inside another is kept too, after it (it starts later)."""
    spans = _fences(prompt)
    for pattern in (_INLINE_CODE, _PLACEHOLDER, _DOUBLE_QUOTED, _SINGLE_QUOTED):
        spans += [m.span() for m in pattern.finditer(prompt)]
    for m in _URL.finditer(prompt):
        url = m.group().rstrip(_URL_TRAILING)
        if not url.endswith("://"):
            spans.append((m.start(), m.start() + len(url)))
    for m in _PATH_RUN.finditer(prompt):
        if path := _path(m.group()):
            spans.append((m.start(), m.start() + len(path)))
    return tuple(dict.fromkeys(prompt[start:end] for start, end in sorted(spans)))


def literals_preserved(original: str, candidate: str) -> bool:
    """True exactly when every literal of `original` occurs in `candidate` as an exact substring,
    whitespace and line endings inside code blocks included (SPEC R9)."""
    return all(literal in candidate for literal in literals(original))


def _fences(text: str) -> list[tuple[int, int]]:
    """Spans of the fenced code blocks, from the first fence character of the opening line to the
    last of the closing line. A fence opens on a line that starts (after spaces or tabs) with 3 or
    more backticks or tildes; a backtick fence's info string has no backtick. It closes on a line
    holding only a run of the same character at least as long (CommonMark), so a longer fence can
    hold a shorter one. A "\\r" before a line break counts as trailing whitespace (CRLF text)."""
    spans: list[tuple[int, int]] = []
    start, fence, offset = -1, "", 0
    for line in text.split("\n"):
        if start < 0:
            m = _FENCE_OPEN.match(line)
            if m and not (m.group(1)[0] == "`" and "`" in line[m.end(1) :]):
                start, fence = offset + m.start(1), m.group(1)
        else:
            body = line.strip()
            if len(body) >= len(fence) and not body.strip(fence[0]):
                spans.append((start, offset + len(line) - len(line.lstrip()) + len(body)))
                start = -1
        offset += len(line) + 1
    if start >= 0:
        spans.append((start, len(text.rstrip())))
    return spans


def _path(run: str) -> str:
    """The file path a run of path characters is, or "": absolute (/a/b), ./a, ../a, ~/a, or
    relative with at least one slash and an extension on the last part (a/b.ext). Dots ending the
    run end a sentence, not the path. An empty part refuses the run, so the "//host/..." left of a
    URL after its scheme is not a path."""
    run = run.rstrip(".")
    prefix = next((p for p in _PATH_PREFIXES if run.startswith(p)), "")
    parts = run[len(prefix) :].split("/")
    if prefix:
        if parts[-1] == "":
            parts.pop()  # a trailing slash: a folder
        return run if parts and all(parts) else ""
    _, dot, extension = parts[-1].rpartition(".")
    return run if len(parts) > 1 and all(parts) and dot and extension[:1].isalpha() else ""
