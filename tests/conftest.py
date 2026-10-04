"""Tripwires that make SPEC R20 true for tests written in good faith: no network, no child
process, no sleep, no real home, state folder, working directory or temp folder, no real `claude`.
A test that reaches one of them fails with a message naming R20.

The guards live on a private session-wide `MonkeyPatch`, so they also cover module- and
session-scoped fixtures and survive a test's own `monkeypatch.undo()`. They patch the usual routes
(this is not a sandbox): code that imported `time.sleep` before the patch, calls `_socket`
directly, starts a `multiprocessing` child with the spawn method, or runs in a thread a test left
behind can slip past, and the agents that write tests are told never to try
(`.claude/agents/*.md`). What IS enforced: the `_ruv_guards` fixture below cannot be redefined or
shadowed (checked at collection), and `tests/test_guards.py` pins each route. Owned by the lead.
"""

import os
import shutil
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
_ENV_FOLDERS = (
    ("HOME", ""),
    ("XDG_STATE_HOME", ".local/state"),
    ("XDG_CACHE_HOME", ".cache"),
    ("XDG_CONFIG_HOME", ".config"),
    ("XDG_DATA_HOME", ".local/share"),
)

_session_patch = pytest.MonkeyPatch()
# Child processes are blocked unless the running test carries the `real_process` marker.
_state: dict = {"process_allowed": False, "base": None}
_real_popen_init = subprocess.Popen.__init__


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", f"{REAL_PROCESS}: the test may start a child process (backend tests only)"
    )
    _install_session_guards()


def pytest_unconfigure(config: pytest.Config) -> None:
    _session_patch.undo()
    if _state["base"] is not None:
        shutil.rmtree(_state["base"], ignore_errors=True)


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


def _refuse(what: str):
    def refuse(*_args: object, **_kwargs: object):
        raise AssertionError(f"{what} in a test (SPEC R20)")

    return refuse


def _popen_init(*args: object, **kwargs: object):
    if not _state["process_allowed"]:
        raise AssertionError("child process in a test (SPEC R20)")
    return _real_popen_init(*args, **kwargs)  # type: ignore[arg-type]


def _install_session_guards() -> None:
    """Patch once, for the whole session, on a MonkeyPatch no test can reach."""
    base = Path(tempfile.mkdtemp(prefix="autoimprover-tests-"))
    _state["base"] = base
    for var, sub in _ENV_FOLDERS:
        folder = base / "home" / sub
        folder.mkdir(parents=True, exist_ok=True)
        _session_patch.setenv(var, str(folder))
    _session_patch.delenv("ANTHROPIC_API_KEY", raising=False)
    stub_dir = base / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "claude"
    stub.write_text(_STUB_CLAUDE)
    stub.chmod(0o755)
    _session_patch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    (base / "tmp").mkdir()
    _session_patch.setenv("TMPDIR", str(base / "tmp"))
    _session_patch.setattr(tempfile, "tempdir", str(base / "tmp"))

    for method in _NETWORK_METHODS:
        _session_patch.setattr(socket.socket, method, _refuse("network access"))
    for function in _NETWORK_FUNCTIONS:
        _session_patch.setattr(socket, function, _refuse("network access"))
    _session_patch.setattr(time, "sleep", _refuse("sleep on the real clock"))
    _session_patch.setattr(subprocess.Popen, "__init__", _popen_init)
    for name in _PROCESS_FUNCTIONS:
        if hasattr(os, name):
            _session_patch.setattr(os, name, _refuse("child process"))


@pytest.fixture(scope="session")
def session_home() -> Path:
    """The HOME the session guards installed before any fixture ran (for tests of the guards)."""
    return Path(_state["base"]) / "home"


@pytest.fixture(autouse=True)
def _ruv_guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest):
    """Per test: home, state, cache, config, cwd and temp folders inside tmp_path, and the
    `real_process` marker lifts the child-process guard for this one test."""
    home = tmp_path / "home"
    for var, sub in _ENV_FOLDERS:
        folder = home / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(var, str(folder))
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

    _state["process_allowed"] = request.node.get_closest_marker(REAL_PROCESS) is not None
    yield
    _state["process_allowed"] = False
