"""`parallel_map`, the stage runner of the fast pipeline (SPEC R25; ADR-011): results in input
order whatever order the items finish in, the error of the lowest input index, items not yet
started cancelled after an error or Ctrl-C, and a plain serial loop for `workers <= 1`.

No test sleeps: threads meet at barriers and events, and a timeout fires only when the code under
test is wrong. Two tests watch `Future.set_exception` and `Future.cancel` to order their threads.
"""

import concurrent.futures
import signal
import threading

import pytest

from autoimprover.parallel import parallel_map

WAIT = 10.0  # seconds; only a broken implementation ever waits this long


def waited(event: threading.Event) -> None:
    assert event.wait(WAIT), "an item waited in vain"


def new_threads(before: set[threading.Thread]) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t not in before]


@pytest.mark.parametrize("workers", [1, 0, -3])
def test_workers_up_to_one_run_inline_in_order_on_the_calling_thread(workers):
    seen: list[tuple[int, int]] = []

    def square(i: int) -> int:
        seen.append((i, threading.get_ident()))
        return i * i

    assert parallel_map(square, range(5), workers) == [0, 1, 4, 9, 16]
    assert seen == [(i, threading.get_ident()) for i in range(5)]


def test_inline_stops_at_the_first_error_and_runs_nothing_after_it():
    ran: list[int] = []

    def fn(i: int) -> int:
        ran.append(i)
        if i == 2:
            raise ValueError("item 2")
        return i

    with pytest.raises(ValueError, match="item 2"):
        parallel_map(fn, range(5), 1)
    assert ran == [0, 1, 2]


@pytest.mark.parametrize("workers", [1, 4])
def test_no_items_give_an_empty_list(workers):
    assert parallel_map(lambda i: i, [], workers) == []


def test_threaded_items_run_at_the_same_time_off_the_calling_thread():
    meet = threading.Barrier(4, timeout=WAIT)
    caller = threading.get_ident()

    def fn(i: int) -> tuple[int, bool]:
        meet.wait()
        return i, threading.get_ident() != caller

    assert parallel_map(fn, range(4), 4) == [(i, True) for i in range(4)]


def test_more_workers_than_items_and_no_thread_left_behind():
    before = set(threading.enumerate())
    assert parallel_map(lambda i: i + 1, range(3), 16) == [1, 2, 3]
    assert new_threads(before) == []


def test_results_keep_input_order_whatever_the_completion_order():
    done = [threading.Event() for _ in range(4)]
    finished: list[int] = []

    def fn(i: int) -> str:
        if i < 3:
            waited(done[i + 1])  # item 3 ends first, item 0 last
        finished.append(i)
        done[i].set()
        return f"result {i}"

    assert parallel_map(fn, range(4), 4) == [f"result {i}" for i in range(4)]
    assert finished == [3, 2, 1, 0]


def test_the_error_of_the_lowest_index_wins_even_when_a_later_one_failed_first(monkeypatch):
    recorded = threading.Event()
    original = concurrent.futures.Future.set_exception

    def set_exception(future, exception):
        original(future, exception)
        recorded.set()

    monkeypatch.setattr(concurrent.futures.Future, "set_exception", set_exception)

    def fn(i: int) -> int:
        if i == 3:
            raise ValueError("item 3")
        if i == 1:
            waited(recorded)  # item 3's error is on record before item 1 fails
            raise KeyError("item 1")
        return i

    with pytest.raises(KeyError, match="item 1"):
        parallel_map(fn, range(4), 4)


def test_an_error_cancels_the_items_not_started_and_waits_for_the_running_ones(monkeypatch):
    """Two workers, eight items: item 0 fails, items 1 and 2 hold their workers until every item
    from 3 on is cancelled; none of those may run, and the running ones finish first."""
    lock, cancelled, settled = threading.Lock(), [0], threading.Event()
    original = concurrent.futures.Future.cancel

    def cancel(future):
        done = original(future)
        if done:
            with lock:
                cancelled[0] += 1
                if cancelled[0] >= 5:  # items 3..7 (and item 2 if it had not started)
                    settled.set()
        return done

    monkeypatch.setattr(concurrent.futures.Future, "cancel", cancel)
    before = set(threading.enumerate())
    ran: list[int] = []
    finished: list[int] = []

    def fn(i: int) -> int:
        ran.append(i)
        if i == 0:
            raise RuntimeError("item 0")
        if i in (1, 2):
            waited(settled)
            finished.append(i)
        return i

    with pytest.raises(RuntimeError, match="item 0"):
        parallel_map(fn, range(8), 2)
    assert set(ran) <= {0, 1, 2} and 1 in finished
    assert new_threads(before) == []


def test_ctrl_c_while_waiting_cancels_what_has_not_started_and_propagates_at_once():
    """Item 0 sends SIGINT to the calling thread while items 0 and 1 run. KeyboardInterrupt leaves
    `parallel_map` while both are still held (they end on their own once released), and nothing
    from item 2 on ever runs."""
    caller = threading.main_thread().ident
    assert caller is not None and threading.get_ident() == caller
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    release, started = threading.Event(), threading.Barrier(2, timeout=WAIT)
    before = set(threading.enumerate())
    ran: list[int] = []
    finished: list[int] = []

    def fn(i: int) -> int:
        ran.append(i)
        if i in (0, 1):
            started.wait()
            if i == 0:
                signal.pthread_kill(caller, signal.SIGINT)
            release.wait(WAIT)
            finished.append(i)
        return i

    try:
        with pytest.raises(KeyboardInterrupt):
            parallel_map(fn, range(8), 2)
        assert finished == []  # it did not wait for the running items
        release.set()
        for thread in new_threads(before):
            thread.join(WAIT)
            assert not thread.is_alive()
    finally:
        release.set()
        signal.signal(signal.SIGINT, previous)
    assert sorted(ran) == [0, 1] and sorted(finished) == [0, 1]
