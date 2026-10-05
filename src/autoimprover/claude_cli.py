"""The raw model layer: one `claude -p` child process per call (ADR-004, ADR-009), and the only
code that builds that command (SPEC R18). The flags are those of R18; the system prompt is the
single argument `--system-prompt=<text>` and the reply schema follows `--json-schema`. The user
text goes to the child's stdin only, never into an argument, and no shell is involved (SPEC R19).
The child runs in the run's empty working folder with an allowlisted environment (no API key,
nothing `ANTHROPIC_*` or `CLAUDE_CODE_*`), so it uses the subscription login and reads no project
file.

Every call is checked for lockdown (ADR-009): each `init` line of the stream must report no tools
(none beyond `_ALLOWED_TOOLS`, which is empty), MCP servers, skills or slash commands, only the
built-in agents and the default output style, or the call raises `SessionNotLockedDown` (exit 4);
`plugins` is informational. The reply is the `result` of the last `{"type":"result"}` line; for a
call with a schema it is the parsed `structured_output` as JSON when that is an object (the
user's real schema call carried both, `result` holding the same answer as JSON text). Anything
else that goes wrong with one attempt is a `CallError`, which `ResilientBackend` retries (SPEC
R24); a missing `claude` binary can never succeed and is a `BackendError`. Each call has a timeout
of `min(CALL_TIMEOUT_S, clock left until the current deadline)` and no call starts once the
deadline has passed (SPEC R17); on expiry only the child this call started is killed, by its
exact pid.

Unverified against the real CLI (no test may run it; the user's `just smoke` does): whether the
`init` line of a schema call lists a tool for the schema; until it is known none is allowed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoimprover.backend import Clock
from autoimprover.types import (
    CALL_TIMEOUT_S,
    BackendError,
    Call,
    CallError,
    Reply,
    SessionNotLockedDown,
)

# The command of SPEC R18 around the model; the per-call parts follow it.
_HEAD = (
    "-p",
    "--safe-mode",
    "--settings",
    '{"outputStyle":"default"}',
    "--tools",
    "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--max-turns",
    "1",
)
_TAIL = ("--output-format", "stream-json", "--verbose")
# What the child may see of the environment (SPEC R18, R21; ARCHITECTURE section 9): the login's
# home and config folders, the locale, and the proxy and certificate settings a sandboxed shell
# needs. Nothing else gets through: no API key, no `ANTHROPIC_*`, no `CLAUDE_CODE_*` or
# `CLAUDECODE` nesting variable.
_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "TERM",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
        "all_proxy",
    }
)
_ENV_PREFIX = "LC_"
# The lockdown of ADR-009: these init fields must be empty lists, the tools and agents subsets of
# the sets below, the output style the default.
_EMPTY_FIELDS = ("mcp_servers", "skills", "slash_commands")
# Tools a session may report. Empty: `tools == []` holds strictly. Whether a call with
# `--json-schema` lists one for the schema (for example "StructuredOutput") is not known yet; if
# the user's real init shows one, its name goes here and nothing else changes.
_ALLOWED_TOOLS: frozenset[str] = frozenset()
_BUILTIN_AGENTS = frozenset({"claude", "Explore", "general-purpose", "Plan"})
_REPLY_MAX_BYTES = 1 << 20  # ARCHITECTURE section 9: a larger reply is a CallError
_TEXT_CHARS = 200  # the most model or stderr text an error message carries
_VALUE_CHARS = 80  # the most of an offending init value a lockdown message shows


class ClaudeCliBackend:
    """`Backend` over `claude -p`, built once per process by `cli.py`. `cwd` is the run's empty
    working folder; `deadline()` the current deadline in the clock's elapsed seconds (the
    search's until the final steps open, SPEC R17); `timeout_s` the longest a call may take."""

    def __init__(
        self,
        clock: Clock,
        *,
        cwd: Path,
        deadline: Callable[[], float],
        timeout_s: float = CALL_TIMEOUT_S,
    ) -> None:
        self._clock = clock
        self._cwd = cwd
        self._deadline = deadline
        self._timeout_s = timeout_s

    def complete(self, call: Call) -> Reply:
        env = _environment(os.environ)
        binary = shutil.which("claude", path=env.get("PATH", os.defpath))
        if binary is None:
            raise BackendError(
                "the `claude` command is not on PATH, so no model can be called; install Claude "
                "Code and log in (SPEC R18)"
            )
        start = self._clock.elapsed()
        timeout = min(self._timeout_s, self._deadline() - start)
        if timeout <= 0:
            raise CallError("no time left before the deadline; claude was not started (SPEC R17)")
        done = _run(_argv(binary, call), env, self._cwd, call.user.encode(), timeout)
        text, usage = _reply(done, call)
        return Reply(
            text=text,
            cached=False,
            tokens_in=_count(usage, "input_tokens"),
            tokens_out=_count(usage, "output_tokens"),
            duration_s=self._clock.elapsed() - start,
        )


def _environment(environ: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in environ.items() if k in _ENV_NAMES or k.startswith(_ENV_PREFIX)}


def _argv(binary: str, call: Call) -> list[str]:
    """The whole command; the user text is never part of it (SPEC R18, R19)."""
    argv = [binary, *_HEAD, "--model", call.model, *_TAIL]
    if call.system:
        argv.append(f"--system-prompt={call.system}")
    if call.json_schema is not None:
        argv += ["--json-schema", call.json_schema]
    return argv


@dataclass(frozen=True)
class _Done:
    code: int
    out: bytes
    err: bytes
    timeout: float
    timed_out: bool
    input_error: OSError | None


def _run(argv: list[str], env: dict[str, str], cwd: Path, data: bytes, timeout: float) -> _Done:
    """Start the child with `data` on its stdin and wait for it to end; after `timeout` seconds
    kill it, this child only. A thread feeds stdin, so a large input and a large reply cannot
    block each other, and a child that stops reading its input is noticed."""
    read_end, write_end = os.pipe()
    try:
        child = subprocess.Popen(
            argv, stdin=read_end, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env
        )
    except OSError as error:
        os.close(write_end)
        raise CallError(f"claude could not be started: {error}") from error
    finally:
        os.close(read_end)
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        child.kill()

    failed: list[OSError] = []
    feeder = threading.Thread(target=_feed, args=(write_end, data, failed), daemon=True)
    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    feeder.start()
    try:
        out, err = child.communicate()
    finally:
        timer.cancel()
        if child.poll() is None:  # communicate was interrupted (Ctrl-C): do not leave it running
            child.kill()
            child.wait()
        feeder.join()
    return _Done(child.returncode, out, err, timeout, expired.is_set(), next(iter(failed), None))


def _feed(fd: int, data: bytes, failed: list[OSError]) -> None:
    """Write `data` to the pipe and close it; an error (the child ended first) goes to `failed`."""
    view = memoryview(data)
    try:
        while view:
            view = view[os.write(fd, view) :]
    except OSError as error:
        failed.append(error)
    finally:
        os.close(fd)


def _reply(done: _Done, call: Call) -> tuple[str, object]:
    """The reply text and the `usage` of a finished call, checked in this order: lockdown of
    every init line seen (whatever else went wrong), then the process, then the stream."""
    if len(done.out) > _REPLY_MAX_BYTES:
        raise CallError(f"claude's reply is larger than {_REPLY_MAX_BYTES >> 20} MiB")
    objects, unreadable = _objects(done.out)
    inits = [o for o in objects if o.get("type") == "system" and o.get("subtype") == "init"]
    for init in inits:
        _check_lockdown(init)
    if done.timed_out:
        raise CallError(f"claude did not answer within {done.timeout:.1f} s and was stopped")
    if done.input_error is not None:
        raise CallError(f"claude did not read its whole input: {done.input_error}")
    final = next((o for o in reversed(objects) if o.get("type") == "result"), None)
    if done.code != 0:
        parts = [f"claude exited with code {done.code}"]
        stderr = _clip(done.err.decode(errors="replace"), tail=True)
        if stderr:
            parts.append(f"stderr: {stderr}")
        if final is not None:
            parts.append(f"result: {_clip(str(final.get('result', '')))}")
        raise CallError("; ".join(parts))
    if final is None:
        raise CallError("claude printed no result line")
    if not inits:
        raise SessionNotLockedDown(
            "the claude session printed no init line, so its lockdown cannot be checked "
            "(SPEC R18, ADR-009)"
        )
    if unreadable is not None:
        raise CallError(f"a line of claude's output is not a JSON object: {_clip(unreadable)}")
    if final.get("is_error") is not False:
        raise CallError(f"claude reported an error: {_clip(str(final.get('result', '')))}")
    if final.get("terminal_reason") != "completed":
        reason = _short(final.get("terminal_reason"))
        raise CallError(f"claude ended with terminal_reason {reason}, not 'completed'")
    shaped = final.get("structured_output")
    if call.json_schema is not None and isinstance(shaped, dict):
        return json.dumps(shaped), final.get("usage")
    text = final.get("result")
    if not isinstance(text, str):
        raise CallError(f"claude's result line has no result text: {_short(text)}")
    return text, final.get("usage")


def _objects(out: bytes) -> tuple[list[dict[str, Any]], str | None]:
    """The JSON objects of a stream-json reply, one per line, and the first line that is not one.
    Lines end at newline bytes only: a JSON string may hold U+2028, which `str.splitlines` would
    cut."""
    objects: list[dict[str, Any]] = []
    unreadable: str | None = None
    for line in out.split(b"\n"):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except ValueError:  # not UTF-8, not JSON
            value = None
        if isinstance(value, dict):
            objects.append(value)
        elif unreadable is None:
            unreadable = line.decode(errors="replace")
    return objects, unreadable


def _check_lockdown(init: Mapping[str, Any]) -> None:
    """Raise `SessionNotLockedDown` naming the first init field that fails ADR-009."""
    problem = _lockdown_problem(init)
    if problem is not None:
        raise SessionNotLockedDown(
            f"the claude session is not locked down: {problem}; it must report no tools, MCP "
            "servers, skills or slash commands, only the built-in agents and the default output "
            "style (SPEC R18, ADR-009)"
        )


def _lockdown_problem(init: Mapping[str, Any]) -> str | None:
    fields = ("tools", *_EMPTY_FIELDS, "agents", "output_style")
    missing = [name for name in fields if name not in init]
    if missing:
        return f"{missing[0]} is missing"
    for name in _EMPTY_FIELDS:
        if init[name] != []:
            return f"{name} = {_short(init[name])}"
    for name, allowed in (("tools", _ALLOWED_TOOLS), ("agents", _BUILTIN_AGENTS)):
        names = init[name]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            return f"{name} = {_short(names)} is not a list of names"
        if extra := sorted(set(names) - allowed):
            return f"{name} not allowed: {_short(extra)}"
    if init["output_style"] != "default":
        return f"output_style = {_short(init['output_style'])}, not 'default'"
    return None


def _count(usage: object, name: str) -> int:
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else 0


def _short(value: object) -> str:
    return _clip(repr(value), _VALUE_CHARS)


def _clip(text: str, limit: int = _TEXT_CHARS, *, tail: bool = False) -> str:
    """`text` for an error message: whitespace as spaces, control and format characters (escape
    sequences, bidi overrides) dropped, at most `limit` characters (the start, or the end)."""
    clean = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf")
    ).strip()
    if len(clean) <= limit:
        return clean
    return "..." + clean[-limit:] if tail else clean[:limit] + "..."
