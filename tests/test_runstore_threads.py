"""The run folder written from the threads of a fast-pipeline stage (SPEC R25; ADR-011, and the
R17 and R22 files they share, ADR-007): `calls.jsonl` gets one whole line per call, the manifest
and the checkpoint stay valid and say one thing, and cache writes of one key leave one readable
entry. The file formats do not change.

No test sleeps: threads start together at a barrier, and the switch interval is cut to a
microsecond while they run.
"""

import json
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autoimprover.runstore import RunStore
from autoimprover.types import DEFAULT_MODELS, Call, Plan, Reply

WAIT = 10.0  # seconds; only a broken implementation ever waits this long
THREADS = 8
NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)
TASK = Call(role="task", model="claude-haiku-4-5-20251001", user="say hi")


@pytest.fixture(autouse=True)
def fast_switching():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def store(root: Path):
    s = RunStore.open_or_create(root, Plan(models=DEFAULT_MODELS), "p", utcnow=lambda: NOW)
    yield s
    s.close()


def in_threads(work: Callable[[int], object], n: int = THREADS) -> list[object]:
    """`work(0..n-1)`, one thread each, started together; a thread's exception fails the test."""
    start = threading.Barrier(n, timeout=WAIT)
    results: list[object] = [None] * n
    errors: list[BaseException] = []

    def run(i: int) -> None:
        try:
            start.wait()
            results[i] = work(i)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads), "a thread hung"
    assert errors == []
    return results


def reopened(root: Path, store: RunStore) -> RunStore:
    store.close()
    return RunStore.resume(root, store.run_id)


def test_log_lines_from_many_threads_never_interleave(store: RunStore):
    """Lines of 64 KiB take many pages each, so an unlocked writer could see another's half
    written line and start its own with a stray newline."""
    each, model = 25, "m" * 65_536

    def work(i: int) -> None:
        for n in range(each):
            store.log_call("task", f"{model}{i}", i, n, False, 0.5)

    in_threads(work)
    lines = (store.path / "calls.jsonl").read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == THREADS * each + 1
    entries = [json.loads(line) for line in lines[:-1]]
    assert sorted((e["tokens_in"], e["tokens_out"]) for e in entries) == [
        (i, n) for i in range(THREADS) for n in range(each)
    ]


def test_progress_saved_from_many_threads_leaves_a_valid_manifest(root: Path, store: RunStore):
    """The file and the totals the store holds say the same, whichever thread saved last."""
    in_threads(lambda i: [store.save_progress(i * 100 + n, float(n)) for n in range(20)])
    held = (store.calls_used, store.elapsed_s)
    resumed = reopened(root, store)
    try:
        assert (resumed.calls_used, resumed.elapsed_s) == held
        assert held[0] % 100 == 19 and held[1] == 19.0
    finally:
        resumed.close()


def test_the_search_start_is_recorded_once_whichever_thread_comes_first(
    root: Path, store: RunStore
):
    starts = in_threads(lambda i: store.search_start_or_record(i, float(i)))
    assert len(set(starts)) == 1
    resumed = reopened(root, store)
    try:
        assert resumed.search_start_or_record(99, 99.0) == starts[0]
    finally:
        resumed.close()


@pytest.mark.parametrize("folder_gone", [False, True])
def test_cache_writes_of_one_key_leave_one_readable_entry(store: RunStore, folder_gone: bool):
    if folder_gone:  # the first writers race to create the cache folder again
        (store.path / "cache").rmdir()
    texts = [f"reply {i} " + "x" * 50_000 for i in range(THREADS)]

    def work(i: int) -> None:
        store.cache_put(TASK, Reply(text=texts[i], tokens_in=i, duration_s=float(i)))
        store.record_failure(Call(role="judge", model="m", user=str(i)), f"down {i}", 0.0)

    in_threads(work)
    entry = store.cache_get(TASK)
    assert entry is not None and entry.reply is not None and entry.reply.text in texts
    assert entry.duration_s == float(texts.index(entry.reply.text))
    tombs = [store.cache_get(Call(role="judge", model="m", user=str(i))) for i in range(THREADS)]
    assert [t.error if t else None for t in tombs] == [f"down {i}" for i in range(THREADS)]
    assert [p.name for p in (store.path / "cache").iterdir() if p.name.startswith(".")] == []
