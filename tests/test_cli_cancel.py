"""Cancelling a run from outside (SPEC R2 exit 130, R21 `/improve cancel`, R22, R25): SIGTERM ends
a run as Ctrl-C does, with exit 130, the run folder kept and the resume line; no call starts after
it (pending stage calls, retries), the raw layer's children are killed, and the previous SIGTERM
handler is back when `main` returns. The signal goes to the main thread while the stage's calls
hang on worker threads; the tests wait on events and FIFOs, never on the clock."""

import signal
import threading

import pytest
from fakes import FakeClock
from test_cancel import WAIT, install_hanging, nothing_more, pid_is_gone

from autoimprover import cli
from autoimprover.types import Call, CallError, Reply

PROMPT = "Answer the user's request."
# 59 s on 2 workers: stage A holds 4 calls (intake, synthesis, 2 rewrites), 2 of them running.
ARGV = ["--time", "59s", "--workers", "2", PROMPT]


class Hanging:
    """A raw model whose calls hang until `terminate`, then fail as a killed child does."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.started = threading.Semaphore(0)
        self.released = threading.Event()
        self.terminated = False
        self._lock = threading.Lock()

    def complete(self, call: Call) -> Reply:
        with self._lock:
            self.calls.append(call)
        self.started.release()
        if not self.released.wait(WAIT):
            raise AssertionError("the call was never released")
        raise CallError("claude exited with code -9")

    def terminate(self) -> None:
        self.terminated = True
        self.released.set()


def sigterm_after(starts: int, wait_start) -> threading.Thread:
    """A thread that waits for `starts` calls to start, then sends SIGTERM to the main thread."""
    main = threading.main_thread().ident
    assert main is not None

    def send() -> None:
        for _ in range(starts):
            wait_start()
        signal.pthread_kill(main, signal.SIGTERM)

    thread = threading.Thread(target=send)
    thread.start()
    return thread


def run_until_sigterm(argv, backend, wait_start) -> tuple[int, list[int]]:
    """`cli.main(argv)` with SIGTERM sent once 2 calls run; returns the exit code and the
    signals the test's own handler saw (none when `main` installed its own), after every thread
    the run started has ended. The test's handler releases a `Hanging` raw so a run without a
    handler of its own still ends."""
    seen: list[int] = []

    def fallback(signum: int, _frame: object) -> None:
        seen.append(signum)
        if isinstance(backend, Hanging):
            backend.released.set()

    before = set(threading.enumerate())
    previous = signal.signal(signal.SIGTERM, fallback)
    try:
        sender = sigterm_after(2, wait_start)
        code = cli.main(argv, backend=backend, now=FakeClock().now)
        assert signal.getsignal(signal.SIGTERM) is fallback  # restored at exit
        sender.join(WAIT)
    finally:
        signal.signal(signal.SIGTERM, previous)
    for thread in set(threading.enumerate()) - before:
        thread.join(WAIT)
        assert not thread.is_alive(), thread.name
    return code, seen


def test_sigterm_ends_the_run_with_130_and_no_call_starts_after_it(capsys):
    raw = Hanging()
    code, seen = run_until_sigterm(ARGV, raw, lambda: raw.started.acquire(timeout=WAIT))
    out, err = capsys.readouterr()
    assert (code, out, seen) == (130, "", [])
    assert raw.terminated and len(raw.calls) == 2  # no pending call, no retry
    lines = err.splitlines()
    assert lines[-3] == "error: interrupted" and lines[-2].startswith("run folder: ")
    assert lines[-1].startswith("resume with: autoimprover --resume ")


def test_with_json_a_sigterm_is_one_error_object(capsys):
    raw = Hanging()
    argv = ["--json", *ARGV]
    code, _ = run_until_sigterm(argv, raw, lambda: raw.started.acquire(timeout=WAIT))
    out, _ = capsys.readouterr()
    assert code == 130 and out.count("\n") == 1 and '"code": 130' in out


@pytest.mark.real_process
def test_sigterm_kills_the_claude_children_by_their_own_process(tmp_path, monkeypatch, capsys):
    started = install_hanging(tmp_path, monkeypatch)
    pids: list[int] = []
    code, seen = run_until_sigterm(ARGV, None, lambda: pids.append(int(started.readline())))
    assert (code, seen) == (130, [])
    assert len(pids) == 2 and all(pid_is_gone(pid) for pid in pids)
    assert nothing_more(started)  # no child started after the signal
    assert "error: interrupted" in capsys.readouterr().err


def test_main_off_the_main_thread_leaves_the_handler_alone(capsys):
    before = signal.getsignal(signal.SIGTERM)
    codes: list[int] = []
    worker = threading.Thread(target=lambda: codes.append(cli.main(["--dry", PROMPT])))
    worker.start()
    worker.join(WAIT)
    assert codes == [0] and signal.getsignal(signal.SIGTERM) is before
