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
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "ALL_PROXY", "all_proxy",
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


def install_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    """A fake `claude` first on PATH, answering with the real stream (`test_claude_cli_stream.py`
    uses it too)."""
    folder = tmp_path / "fakebin"
    folder.mkdir()
    script = folder / "claude"
    script.write_text(FAKE.replace("PYTHON", sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    claude = FakeClaude(folder)
    claude.answer(raw=STREAM)
    return claude


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    return install_fake(tmp_path, monkeypatch)


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


@real
@pytest.mark.parametrize("name", ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "ALL_PROXY", "all_proxy"])
def test_config_folders_and_the_all_proxy_setting_reach_the_child(
    fake, tmp_path, monkeypatch, name
):
    monkeypatch.setenv(name, f"value of {name}")
    ask(tmp_path)
    assert fake.env()[name] == f"value of {name}"


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
