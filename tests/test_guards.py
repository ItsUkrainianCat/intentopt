"""The R20 guards in tests/conftest.py actually fire: each reached resource fails the test."""

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")


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


def test_other_ways_to_start_a_program_are_blocked():
    # None of these runs anything: each guard raises before the call is made.
    for name, args in (
        ("posix_spawn", ("/bin/true", ["true"], {})),
        ("fork", ()),
        ("execv", ("/bin/true", ["true"])),
        ("spawnv", (os.P_NOWAIT, "/bin/true", ["true"])),
    ):
        with pytest.raises(AssertionError, match="R20"):
            getattr(os, name)(*args)


def test_name_lookup_and_datagrams_are_blocked():
    with pytest.raises(AssertionError, match="R20"):
        socket.getaddrinfo("example.com", 80)
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s,
        pytest.raises(AssertionError, match="R20"),
    ):
        s.sendto(b"x", ("127.0.0.1", 9))


def test_working_directory_and_temp_folder_are_inside_the_test_folder(tmp_path: Path):
    assert Path.cwd().is_relative_to(tmp_path)
    assert Path(tempfile.gettempdir()).is_relative_to(tmp_path)
    assert Path(os.environ["TMPDIR"]).is_relative_to(tmp_path)


def test_a_test_file_cannot_shadow_the_guard_fixture(pytester: pytest.Pytester):
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(
        test_shadow="""
        import pytest

        @pytest.fixture(autouse=True)
        def _ruv_guards():
            yield

        def test_x():
            pass
        """
    )
    result = pytester.runpytest_inprocess()
    result.stderr.fnmatch_lines(["*_ruv_guards is redefined or missing*"])
    assert result.ret != 0


def test_the_guard_fixture_is_active_when_not_shadowed(pytester: pytest.Pytester):
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(
        test_ok="""
        import time

        def test_sleep_is_guarded():
            try:
                time.sleep(0)
            except AssertionError as error:
                assert "R20" in str(error)
            else:
                raise SystemExit("sleep was not blocked")
        """
    )
    pytester.runpytest_inprocess().assert_outcomes(passed=1)
