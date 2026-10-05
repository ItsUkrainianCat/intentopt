"""The real model layer, `claude -p` as a child process (SPEC R17 timeout, R18, R19; ADR-009).

No test reaches the real `claude`: each puts a fake `claude` (a small Python script in tmp_path)
first on PATH. The fake records its argv, working folder, environment and stdin next to itself and
prints a configured stream, by default the real output in `tests/fixtures/claude_cli/`. Tests
that start it carry `real_process`; the others prove by the conftest guard that nothing starts.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
from fakes import FakeClock

from autoimprover.backend import Clock
from autoimprover.claude_cli import ClaudeCliBackend
from autoimprover.types import (
    JUDGE_SCHEMA,
    BackendError,
    Call,
    CallError,
    Reply,
    SessionNotLockedDown,
)

real = pytest.mark.real_process
MODEL = "claude-haiku-4-5-20251001"
STREAM = (Path(__file__).parent / "fixtures" / "claude_cli" / "stream.jsonl").read_text()
INIT, *MIDDLE, RESULT = (json.loads(line) for line in STREAM.split("\n") if line.strip())
BUILTIN_AGENTS = ["claude", "Explore", "general-purpose", "Plan"]
LS, RLO = "\N{LINE SEPARATOR}", "\N{RIGHT-TO-LEFT OVERRIDE}"
COMMAND = [
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
    "--model",
    MODEL,
    "--output-format",
    "stream-json",
    "--verbose",
]
# The variables the child may see (SPEC R18): the allowlist, LC_* included.
ALLOWED = {
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "TERM", "XDG_RUNTIME_DIR", "TMPDIR",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
}  # fmt: skip
SECRETS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_FOO", "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CONFIG_DIR", "SECRET_X", "PYTHONPATH",
)  # fmt: skip

FAKE = """#!PYTHON
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
def put(name, data):
    with open(os.path.join(here, name), "wb") as f:
        f.write(data)
conf = json.load(open(os.path.join(here, "conf.json")))
put("pid", str(os.getpid()).encode())
put("argv.json", json.dumps(sys.argv).encode())
put("cwd", os.getcwd().encode())
put("env.json", json.dumps(dict(os.environ)).encode())
if conf["read_stdin"]:
    put("stdin", sys.stdin.buffer.read())
sys.stdout.buffer.write(conf["stdout"].encode("utf-8", "surrogateescape"))
sys.stdout.flush()
time.sleep(conf["sleep"])
sys.stderr.buffer.write(conf["stderr"].encode())
sys.exit(conf["exit"])
"""


class FakeClaude:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def answer(
        self,
        *objects: object,
        raw: str | None = None,
        exit: int = 0,
        stderr: str = "",
        sleep: float = 0,
        read_stdin: bool = True,
    ) -> None:
        """Print `raw`, else one JSON line per object in raw UTF-8 as claude does (a str goes out
        as it is, and a lone surrogate U+DCxx in it as the byte xx)."""
        lines = (o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in objects)
        stdout = raw if raw is not None else "".join(f"{line}\n" for line in lines)
        conf = dict(stdout=stdout, exit=exit, stderr=stderr, sleep=sleep, read_stdin=read_stdin)
        (self.folder / "conf.json").write_text(json.dumps(conf))

    def read(self, name: str) -> bytes:
        return (self.folder / name).read_bytes()

    def argv(self) -> list[str]:
        return json.loads(self.read("argv.json"))

    def env(self) -> dict[str, str]:
        return json.loads(self.read("env.json"))

    def ran(self) -> bool:
        return (self.folder / "pid").exists()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    folder = tmp_path / "fakebin"
    folder.mkdir()
    script = folder / "claude"
    script.write_text(FAKE.replace("PYTHON", sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    claude = FakeClaude(folder)
    claude.answer(raw=STREAM)
    return claude


def backend(
    tmp_path: Path, deadline: float = 1e9, now=None, elapsed: float = 0.0, **kwargs
) -> ClaudeCliBackend:
    cwd = tmp_path / "run" / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    clock = Clock(now or FakeClock().now, elapsed=elapsed)
    return ClaudeCliBackend(clock, cwd=cwd, deadline=lambda: deadline, **kwargs)


def ask(tmp_path: Path, user: str = "Say OK.", **fields) -> Reply:
    return backend(tmp_path).complete(Call(role="task", model=MODEL, user=user, **fields))


def edited(base: dict, **changes) -> dict:
    """A copy of `base` with `changes`; a change to None removes the field."""
    out = {k: v for k, v in base.items() if k not in changes}
    out.update({k: v for k, v in changes.items() if v is not None})
    return out


# --- the command (SPEC R18, R19) -----------------------------------------------------------------

SCHEMA = json.dumps(JUDGE_SCHEMA)


@real
@pytest.mark.parametrize(
    ("fields", "tail"),
    [
        ({}, []),
        ({"system": "-x --tools Bash\nline 2"}, ["--system-prompt=-x --tools Bash\nline 2"]),
        ({"json_schema": SCHEMA}, ["--json-schema", SCHEMA]),
        (
            {"system": "You judge.", "json_schema": SCHEMA},
            ["--system-prompt=You judge.", "--json-schema", SCHEMA],
        ),
    ],
)
def test_the_command_is_exactly_the_r18_flags_plus_the_calls_own_parts(
    fake, tmp_path, fields, tail
):
    ask(tmp_path, **fields)
    argv = fake.argv()
    assert argv[0] == str(fake.folder / "claude")
    assert argv[1:] == COMMAND + tail


@real
def test_user_text_travels_on_stdin_only_however_hostile(fake, tmp_path):
    hostile = f"'\"$(touch /tmp/pwned)`id`\n--tools Bash; echo x && ünïcødé{LS}\r\n"
    user = (hostile * (100_000 // len(hostile) + 1))[:100_000]
    ask(tmp_path, user=user)
    assert fake.read("stdin") == user.encode()
    argv = fake.argv()
    assert not any("$(touch" in arg or "--tools Bash" in arg for arg in argv)
    assert argv[1:] == COMMAND


@real
def test_the_child_runs_in_the_given_empty_folder(fake, tmp_path):
    ask(tmp_path)
    assert fake.read("cwd").decode() == str(tmp_path / "run" / "cwd")


@real
def test_the_environment_is_an_allowlist(fake, tmp_path, monkeypatch):
    for name in SECRETS:
        monkeypatch.setenv(name, "must-not-leak")
    kept = {
        "HTTPS_PROXY": "http://127.0.0.1:3128",
        "https_proxy": "http://127.0.0.1:3129",
        "NO_PROXY": "localhost",
        "SSL_CERT_FILE": "/etc/ca.pem",
        "LC_ALL": "C.UTF-8",
    }
    for name, value in kept.items():
        monkeypatch.setenv(name, value)
    ask(tmp_path)
    env = fake.env()
    assert "must-not-leak" not in env.values()
    assert all(name in ALLOWED or name.startswith("LC_") for name in env), sorted(env)
    assert {name: env.get(name) for name in kept} == kept
    assert env["HOME"] == os.environ["HOME"]


def test_a_missing_claude_binary_ends_the_run_before_any_process(tmp_path, monkeypatch):
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(BackendError, match="claude"):
        ask(tmp_path)


# --- the timeout (SPEC R17) ----------------------------------------------------------------------


def pid_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@real
@pytest.mark.parametrize(
    ("deadline", "timeout_s"), [(1e9, 0.3), (0.3, 300)], ids=["call-timeout", "clock-left"]
)
def test_a_call_past_its_time_is_killed_and_fails(fake, tmp_path, deadline, timeout_s):
    fake.answer(INIT, sleep=30)
    started = time.monotonic()
    with pytest.raises(CallError, match=r"0\.3 s"):
        backend(tmp_path, deadline, timeout_s=timeout_s).complete(Call("task", MODEL, "hi"))
    assert time.monotonic() - started < 10
    assert pid_is_gone(int(fake.read("pid")))


@pytest.mark.parametrize("elapsed", [100.0, 100.5])
def test_no_call_starts_once_the_deadline_has_passed(fake, tmp_path, elapsed):
    with pytest.raises(CallError, match="no time left"):
        backend(tmp_path, 100.0, elapsed=elapsed).complete(Call("task", MODEL, "hi"))
    assert not fake.ran()


@real
def test_tokens_come_from_usage_and_the_duration_from_the_clock(fake, tmp_path):
    ticks = iter(range(0, 1000, 5))

    def now() -> float:
        return next(ticks) / 2

    reply = backend(tmp_path, now=now).complete(Call("task", MODEL, "hi"))
    assert reply == Reply(text="OK", cached=False, tokens_in=721, tokens_out=117, duration_s=2.5)


@real
def test_a_child_that_does_not_read_its_whole_input_fails(fake, tmp_path):
    fake.answer(raw=STREAM, read_stdin=False)
    with pytest.raises(CallError, match="input"):
        ask(tmp_path, user="x" * 4_000_000)


# --- lockdown on every call (SPEC R18, ADR-009) --------------------------------------------------


@real
@pytest.mark.parametrize(
    ("field", "value", "shown"),
    [
        ("tools", ["Bash", "Read"], "'Bash'"),
        ("mcp_servers", [{"name": "github", "status": "connected"}], "github"),
        ("skills", ["deploy"], "deploy"),
        ("slash_commands", ["compact"], "compact"),
        ("agents", [*BUILTIN_AGENTS, "rogue"], "rogue"),
        ("output_style", "Proactive", "Proactive"),
        ("tools", "none", "'none'"),
        ("agents", "claude", "'claude'"),
        ("mcp_servers", {}, "{}"),
    ],
)
def test_a_session_that_is_not_locked_down_is_refused(fake, tmp_path, field, value, shown):
    fake.answer(edited(INIT, **{field: value}), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown) as caught:
        ask(tmp_path)
    assert field in str(caught.value) and shown in str(caught.value)


@real
@pytest.mark.parametrize(
    "field", ["tools", "mcp_servers", "skills", "slash_commands", "agents", "output_style"]
)
def test_an_init_line_missing_a_field_is_refused(fake, tmp_path, field):
    fake.answer(edited(INIT, **{field: None}), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match=field):
        ask(tmp_path)


@real
def test_the_real_init_is_accepted_whatever_plugins_it_lists(fake, tmp_path):
    plugins = [{"name": f"plugin-{i}", "path": f"/p/{i}"} for i in range(70)]
    fake.answer(edited(INIT, plugins=plugins, agents=["Plan"]), *MIDDLE, RESULT)
    assert ask(tmp_path).text == "OK"


@real
def test_a_stream_without_an_init_line_is_refused(fake, tmp_path):
    fake.answer(*MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match="init"):
        ask(tmp_path)


@real
def test_every_init_line_is_checked(fake, tmp_path):
    fake.answer(INIT, edited(INIT, tools=["Bash"]), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match="Bash"):
        ask(tmp_path)


@real
def test_every_call_is_checked_not_only_the_first(fake, tmp_path):
    claude = backend(tmp_path)
    assert claude.complete(Call("task", MODEL, "one")).text == "OK"
    fake.answer(edited(INIT, mcp_servers=[{"name": "late"}]), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match="late"):
        claude.complete(Call("task", MODEL, "two"))


@real
def test_a_long_offending_value_is_shown_short(fake, tmp_path):
    fake.answer(edited(INIT, tools=[f"tool-{i}" for i in range(2000)]), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown) as caught:
        ask(tmp_path)
    assert "tool-0" in str(caught.value) and len(str(caught.value)) < 400


@real
def test_a_failed_lockdown_wins_over_a_failed_exit(fake, tmp_path):
    fake.answer(edited(INIT, tools=["Bash"]), exit=1, stderr="boom")
    with pytest.raises(SessionNotLockedDown, match="tools"):
        ask(tmp_path)


# --- the reply (ADR-009) -------------------------------------------------------------------------


@real
def test_the_reply_is_the_result_of_the_last_result_line(fake, tmp_path):
    first = edited(RESULT, result="an earlier result")
    fake.answer(INIT, first, *MIDDLE, edited(RESULT, result=f"last{LS}line"))
    assert ask(tmp_path).text == f"last{LS}line"


@real
def test_structured_output_is_the_reply_of_a_schema_call_only(fake, tmp_path):
    shaped = edited(RESULT, result="", structured_output={"results": [], "note": "é"})
    fake.answer(INIT, *MIDDLE, shaped)
    assert json.loads(ask(tmp_path, json_schema=SCHEMA).text) == {"results": [], "note": "é"}
    assert ask(tmp_path).text == ""


@real
def test_a_schema_call_without_structured_output_uses_the_result(fake, tmp_path):
    fake.answer(INIT, *MIDDLE, edited(RESULT, result='{"results": []}'))
    assert ask(tmp_path, json_schema=SCHEMA).text == '{"results": []}'


@real
@pytest.mark.parametrize(
    ("objects", "message"),
    [
        ((INIT, *MIDDLE, edited(RESULT, is_error=True, result="API Error: busy")), "busy"),
        ((INIT, *MIDDLE, edited(RESULT, is_error=None)), "error"),
        ((INIT, *MIDDLE, edited(RESULT, terminal_reason="max_turns")), "max_turns"),
        ((INIT, *MIDDLE, edited(RESULT, terminal_reason=None)), "terminal_reason"),
        ((INIT, *MIDDLE), "no result"),
        ((INIT, "not json {", *MIDDLE, RESULT), "not json {"),
        ((INIT, *MIDDLE, RESULT, "[1, 2]"), r"\[1, 2\]"),
        ((INIT, '{"type": "assistant", "x": "\udcff"}', *MIDDLE, RESULT), "JSON"),
        ((INIT, *MIDDLE, edited(RESULT, result=None)), "result text"),
        ((INIT, *MIDDLE, edited(RESULT, result=["OK"])), "result text"),
    ],
    ids=[
        "is-error",
        "no-is-error",
        "turns",
        "no-reason",
        "no-result",
        "garbage",
        "array",
        "not-utf8",
        "no-text",
        "list",
    ],
)
def test_a_failed_or_unreadable_reply_is_a_call_error(fake, tmp_path, objects, message):
    fake.answer(*objects)
    with pytest.raises(CallError, match=message):
        ask(tmp_path)


@real
def test_error_text_is_short_and_free_of_control_characters(fake, tmp_path):
    fake.answer(INIT, edited(RESULT, is_error=True, result=f"\x1b{RLO}\x07" + "x" * 1000))
    with pytest.raises(CallError) as caught:
        ask(tmp_path)
    message = str(caught.value)
    assert "x" * 200 in message and "x" * 201 not in message
    assert "\x1b" not in message and RLO not in message


@real
@pytest.mark.parametrize("with_result", [False, True])
def test_a_non_zero_exit_is_a_call_error_with_the_code_and_the_end_of_stderr(
    fake, tmp_path, with_result
):
    stderr = "\x1b[31mearly" + "e" * 500 + "\nboom at the end\n"
    fake.answer(INIT, *([RESULT] if with_result else []), exit=7, stderr=stderr)
    with pytest.raises(CallError) as caught:
        ask(tmp_path)
    message = str(caught.value)
    assert "code 7" in message and "boom at the end" in message
    assert "\x1b" not in message and "early" not in message and "e" * 201 not in message


@real
def test_a_reply_over_one_mebibyte_is_a_call_error(fake, tmp_path):
    fake.answer(INIT, *MIDDLE, edited(RESULT, result="y" * (1 << 20)))
    with pytest.raises(CallError, match="MiB"):
        ask(tmp_path)
