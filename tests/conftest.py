"""Guards that make SPEC R20 true for every test: no network, no child process, no sleep, no real
home or state folder, no real `claude`. A test that reaches one of them fails with a message naming
R20. Owned by the lead; the autouse fixture must not be redefined (see tests/test_guards.py)."""

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

REAL_PROCESS = "real_process"
_STUB_CLAUDE = "#!/bin/sh\necho 'real claude reached by a test (SPEC R20)' >&2\nexit 97\n"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", f"{REAL_PROCESS}: the test may start a child process (backend tests only)"
    )


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

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "claude"
    stub.write_text(_STUB_CLAUDE)
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    monkeypatch.setattr(socket.socket, "connect", _blocked("network connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked("network connect"))
    monkeypatch.setattr(time, "sleep", _blocked("sleep on the real clock"))
    if request.node.get_closest_marker(REAL_PROCESS) is None:
        monkeypatch.setattr(subprocess.Popen, "__init__", _blocked("child process"))
        monkeypatch.setattr(os, "system", _blocked("child process"))
