"""The R20 guards in tests/conftest.py actually fire: each reached resource fails the test."""

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest


def test_network_connect_is_blocked():
    with socket.socket() as s, pytest.raises(AssertionError, match="R20"):
        s.connect(("127.0.0.1", 9))


def test_child_process_is_blocked_by_default():
    with pytest.raises(AssertionError, match="R20"):
        subprocess.run(["true"], check=False)
    with pytest.raises(AssertionError, match="R20"):
        # Constant string, and the guard raises before any shell starts: this proves os.system is
        # blocked, it never runs a command.
        os.system("true")


def test_sleep_is_blocked():
    with pytest.raises(AssertionError, match="R20"):
        time.sleep(0)


def test_home_and_xdg_folders_are_inside_the_test_folder(tmp_path: Path):
    for var in ("HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        assert Path(os.environ[var]).is_relative_to(tmp_path), var
    assert Path.home().is_relative_to(tmp_path)


def test_api_key_is_removed(monkeypatch: pytest.MonkeyPatch):
    assert "ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.real_process
def test_claude_on_path_is_a_failing_stub():
    done = subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False)
    assert done.returncode == 97
    assert "R20" in done.stderr
