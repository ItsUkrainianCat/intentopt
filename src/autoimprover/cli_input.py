"""The user's input at the boundary (SPEC R1, R11): the prompt from one quoted argument, a file
or piped stdin, checked before anything else sees it, and the examples file read into scenarios.
Every problem is a UsageError naming its source (exit 2)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO

from autoimprover.cli_options import Options, UsageError
from autoimprover.scenarios import read_examples
from autoimprover.types import PROMPT_MAX_CHARS, Scenario

# GEPA renders its reflection template by plain replacement of these, so a prompt holding one
# could not be told apart from the feedback spliced in (SPEC R1).
_GEPA_TOKENS = ("<curr_param>", "<side_info>")
_BOM = b"\xef\xbb\xbf"
# The most bytes a prompt of PROMPT_MAX_CHARS characters can take: 4 per character, and a BOM.
_READ_LIMIT = 4 * PROMPT_MAX_CHARS + len(_BOM)


def read_prompt(argument: str | None, file: str | None, stdin: IO[bytes] | None) -> str:
    """The prompt from the argument, else the file, else stdin, with CRLF and CR made LF; a
    UsageError naming its source unless it is UTF-8, not blank, at most PROMPT_MAX_CHARS
    characters, without NUL and without a GEPA template token (SPEC R1). A file or stdin is read
    only up to the bytes such a prompt can take."""
    if argument is not None and file is not None:
        raise UsageError("--file: give the prompt either as an argument or with --file, not both")
    if argument is not None:
        source, text = "the prompt argument", argument
        try:
            argument.encode("utf-8")
        except UnicodeEncodeError:
            raise UsageError(f"{source} is not valid UTF-8") from None
    else:
        source = "stdin" if file is None else f"--file {file}"
        data = _read(file, stdin)
        if len(data) > _READ_LIMIT:
            raise UsageError(f"{source}: the prompt is longer than {PROMPT_MAX_CHARS:,} characters")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise UsageError(f"{source}: the prompt is not valid UTF-8") from None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise UsageError(f"{source}: the prompt is empty")
    if "\0" in text:
        raise UsageError(f"{source}: the prompt contains a NUL character")
    if len(text) > PROMPT_MAX_CHARS:
        raise UsageError(f"{source}: the prompt is longer than {PROMPT_MAX_CHARS:,} characters")
    if token := next((token for token in _GEPA_TOKENS if token in text), None):
        raise UsageError(f"{source}: the prompt contains {token}, which GEPA cannot escape")
    return text


def _read(file: str | None, stdin: IO[bytes] | None) -> bytes:
    """At most one byte more than a prompt can take, from the file or else from stdin."""
    if file is not None:
        try:
            with open(file, "rb") as stream:
                return stream.read(_READ_LIMIT + 1)
        except OSError as error:
            raise UsageError(f"--file: cannot read {file}: {error.strerror or error}") from None
    if stdin is None:
        raise UsageError("no prompt: give it as one quoted argument, with --file, or on stdin")
    try:
        return stdin.read(_READ_LIMIT + 1)
    except OSError as error:
        raise UsageError(f"stdin: cannot read the prompt: {error}") from None


def prompt_of(opts: Options) -> str:
    """The prompt of a new run: one quoted argument, else `--file`, else piped stdin (SPEC R1)."""
    if len(opts.words) > 1:
        raise UsageError(
            f"give the prompt as one quoted argument (got {len(opts.words)} arguments), or use "
            "--file"
        )
    argument = opts.words[0] if opts.words else None
    piped = _stdin() if argument is None and opts.file is None else None
    return read_prompt(argument, opts.file, piped)


def examples_of(path: str | None) -> list[Scenario] | None:
    """The scenarios of `--examples`, or None without it; a bad line is exit 2 (SPEC R11)."""
    if path is None:
        return None
    try:
        return read_examples(Path(path))
    except ValueError as error:
        raise UsageError(f"--examples: {error}") from None


def _stdin() -> IO[bytes] | None:
    """Piped stdin; None for a terminal, so the tool never waits for typing."""
    stream = sys.stdin
    if stream is None or stream.isatty():
        return None
    return stream.buffer
