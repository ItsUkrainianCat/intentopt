"""Cancelling a run (SPEC R2 exit 130, R21 `/improve cancel`, R22, R25): `BudgetedBackend.cancel`
refuses every later call before it is counted, and `ClaudeCliBackend.terminate` kills the children
it started, by their own Popen objects, and makes later calls fail without a child. Their use by
the command line on SIGTERM and Ctrl-C is in `test_cli_cancel.py`."""

import os
import sys
import threading
from pathlib import Path
from typing import TextIO

import pytest
from fakes import FakeClock, ScriptedBackend

from autoimprover.backend import BudgetedBackend, Clock
from autoimprover.claude_cli import ClaudeCliBackend, _Children, _run
from autoimprover.types import BudgetExhausted, Call, CallError

MODEL = "claude-haiku-4-5-20251001"
WAIT = 30.0  # seconds; only a broken implementation waits this long (the fake lives 60 s)
# A fake `claude` that announces its pid on the `started` FIFO, then hangs until it is killed.
HANGING = """#!{python}
import os, time
with open({fifo!r}, "w") as started:
    started.write(f"{{os.getpid()}}\\n")
time.sleep(60)
"""


def install_hanging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TextIO:
    """Put the hanging fake first on PATH; return the reading end of its `started` FIFO, held
    open for writing too, so it never reads an end of file between two children."""
    folder = tmp_path / "fakebin"
    folder.mkdir()
    fifo = tmp_path / "started"
    os.mkfifo(fifo)
    script = folder / "claude"
    script.write_text(HANGING.format(python=sys.executable, fifo=str(fifo)))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    return os.fdopen(os.open(fifo, os.O_RDWR), "r")


def nothing_more(started: TextIO) -> bool:
    """True when no further child has announced itself on the FIFO."""
    os.set_blocking(started.fileno(), False)
    try:
        return os.read(started.fileno(), 64) == b""
    except BlockingIOError:
        return True


def pid_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def budgeted(raw: ScriptedBackend, used: int = 0) -> BudgetedBackend:
    return BudgetedBackend(raw, limit=10, used=used, clock=Clock(lambda: 0.0), deadline=100.0)


def test_after_cancel_no_call_reaches_the_raw_layer_or_the_count():
    raw = ScriptedBackend(lambda _call: "ok")
    layer = budgeted(raw, used=3)
    layer.complete(Call("task", "m", "before"))
    layer.cancel()
    for _ in range(3):
        with pytest.raises(BudgetExhausted, match="cancelled") as refused:
            layer.complete(Call("task", "m", "after"))
        assert refused.value.cause == "clock"
    assert [call.user for call in raw.calls] == ["before"]
    assert layer.used == 4


def test_cancel_is_checked_before_the_call_limit_and_the_deadline():
    raw = ScriptedBackend(lambda _call: "ok")
    full = BudgetedBackend(raw, limit=1, used=1, clock=Clock(lambda: 0.0), deadline=100.0)
    full.cancel()
    with pytest.raises(BudgetExhausted, match="cancelled"):
        full.complete(Call("task", "m", "x"))


def test_a_call_in_flight_ends_as_it_would_and_the_next_one_is_refused():
    """cancel() from another thread while a raw call runs: that call is not undone (the raw
    layer's terminate does that), the threads waiting behind it never start theirs."""
    entered, release = threading.Event(), threading.Event()

    def slow(call: Call) -> str:
        if call.user == "first":
            entered.set()
            assert release.wait(10)
        return "ok"

    raw = ScriptedBackend(slow)
    layer = budgeted(raw)
    results: list[object] = []

    def first() -> None:
        results.append(layer.complete(Call("task", "m", "first")))

    worker = threading.Thread(target=first)
    worker.start()
    assert entered.wait(10)
    layer.cancel()
    release.set()
    worker.join(10)
    with pytest.raises(BudgetExhausted):
        layer.complete(Call("task", "m", "second"))
    assert [call.user for call in raw.calls] == ["first"] and len(results) == 1


# --- the raw layer: terminate (exact children, no new ones) --------------------------------------


def cli_backend(tmp_path: Path) -> ClaudeCliBackend:
    cwd = tmp_path / "run" / "cwd"
    cwd.mkdir(parents=True)
    return ClaudeCliBackend(Clock(FakeClock().now), cwd=cwd, deadline=lambda: 1e9)


@pytest.mark.real_process
def test_terminate_kills_the_running_children_and_no_later_call_starts_one(tmp_path, monkeypatch):
    started = install_hanging(tmp_path, monkeypatch)
    backend = cli_backend(tmp_path)
    failed: list[CallError] = []

    def ask(n: int) -> None:
        try:
            backend.complete(Call("task", MODEL, f"call {n}"))
        except CallError as error:
            failed.append(error)

    threads = [threading.Thread(target=ask, args=(n,)) for n in range(3)]
    for thread in threads:
        thread.start()
    pids = [int(started.readline()) for _ in threads]
    backend.terminate()
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads)
    assert len(failed) == 3 and all(pid_is_gone(pid) for pid in pids)
    with pytest.raises(CallError, match="cancelled"):
        backend.complete(Call("task", MODEL, "after"))
    assert nothing_more(started)


def test_terminate_before_any_call_starts_no_process(tmp_path):
    backend = cli_backend(tmp_path)
    backend.terminate()
    with pytest.raises(CallError, match="cancelled"):  # the R20 guard refuses any child
        backend.complete(Call("task", MODEL, "hi"))


@pytest.mark.real_process
def test_a_child_started_after_terminate_is_killed_at_once(tmp_path, monkeypatch):
    """The race of a call that passed its check as `terminate` ran: its child, once started, is
    refused by the closed set and killed, so it never runs to its timeout."""
    started = install_hanging(tmp_path, monkeypatch)
    live = _Children()
    live.close()
    claude = str(tmp_path / "fakebin" / "claude")
    done = _run([claude], {"PATH": os.environ["PATH"]}, tmp_path, b"", WAIT, live)
    assert not done.timed_out and done.code != 0
    started.close()
