"""Tripwires that make SPEC R20 true for tests written in good faith: no network, no child
process, no sleep, no real home, state folder, working directory or temp folder, no real `claude`.
A test that reaches one of them fails with a message naming R20.

These guards patch the usual routes (they are not a sandbox): code that imports `time.sleep` before
the patch or calls `_socket` directly can slip past, and the agents that write tests are told never
to try (`.claude/agents/*.md`). What IS enforced: the `_ruv_guards` fixture below cannot be
redefined or shadowed (checked at collection), and the guard tests in `tests/test_guards.py` pin
each route. Owned by the lead.
"""

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

REAL_PROCESS = "real_process"
_STUB_CLAUDE = "#!/bin/sh\necho 'real claude reached by a test (SPEC R20)' >&2\nexit 97\n"
# Every way to start a program from Python without going through subprocess.Popen.
_PROCESS_FUNCTIONS = (
    "system",
    "popen",
    "fork",
    "forkpty",
    "posix_spawn",
    "posix_spawnp",
    *(f"exec{kind}" for kind in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
    *(f"spawn{kind}" for kind in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
)
_NETWORK_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
_NETWORK_FUNCTIONS = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "create_connection")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", f"{REAL_PROCESS}: the test may start a child process (backend tests only)"
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail the run if any test file defines its own `_ruv_guards` (a second definition shadows
    this one)."""
    shadowed = [
        item.nodeid
        for item in session.items
        if len(item._fixtureinfo.name2fixturedefs.get("_ruv_guards", ())) != 1  # type: ignore[attr-defined]
    ]
    if shadowed:
        raise pytest.UsageError(f"_ruv_guards is redefined or missing for: {shadowed[:3]}")


def _blocked(what: str):
    def refuse(*_args: object, **_kwargs: object):
        raise AssertionError(f"{what} in a test (SPEC R20)")

    return refuse


@pytest.fixture(autouse=True)
def _ruv_guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest):
    home = tmp_path / "home"
    for var, sub in (
        ("HOME", ""),
        ("XDG_STATE_HOME", ".local/state"),
        ("XDG_CACHE_HOME", ".cache"),
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_DATA_HOME", ".local/share"),
    ):
        folder = home / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(var, str(folder))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Working directory and temp folder live in the test folder, so a stray relative path or
    # `tempfile` call cannot touch the checkout or the system temp folder.
    cwd, tmp = tmp_path / "cwd", tmp_path / "tmp"
    cwd.mkdir()
    tmp.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "claude"
    stub.write_text(_STUB_CLAUDE)
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    for method in _NETWORK_METHODS:
        monkeypatch.setattr(socket.socket, method, _blocked("network access"))
    for function in _NETWORK_FUNCTIONS:
        monkeypatch.setattr(socket, function, _blocked("network access"))
    monkeypatch.setattr(time, "sleep", _blocked("sleep on the real clock"))
    if request.node.get_closest_marker(REAL_PROCESS) is None:
        monkeypatch.setattr(subprocess.Popen, "__init__", _blocked("child process"))
        for name in _PROCESS_FUNCTIONS:
            if hasattr(os, name):
                monkeypatch.setattr(os, name, _blocked("child process"))
