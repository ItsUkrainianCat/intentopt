"""Runs the calls of one stage side by side and gathers their results by input index, so a stage's
outcome never depends on the order in which its calls finish (SPEC R25; ADR-011). The calls go
through the backend stack as they are; its layers are safe to share between threads.

`workers <= 1` is a plain loop on the calling thread, in order, stopping at the first error (the
deep tier and the tests). Otherwise a thread pool of at most `workers` threads runs the items; the
first error by input order propagates once the items that had not started are cancelled and the
running ones have finished. Ctrl-C (or any other BaseException) while waiting cancels what has not
started and propagates at once; a running call ends on its own thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait


def parallel_map[T, R](fn: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    """`[fn(item) for item in items]`, with up to `workers` items running at a time."""
    todo = list(items)
    if workers <= 1 or len(todo) <= 1:
        return [fn(item) for item in todo]
    pool = ThreadPoolExecutor(max_workers=min(workers, len(todo)))
    try:
        futures = [pool.submit(fn, item) for item in todo]
        wait(futures, return_when=FIRST_EXCEPTION)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True, cancel_futures=True)
    for future in futures:
        error = None if future.cancelled() else future.exception()
        if error is not None:
            raise error
    return [future.result() for future in futures]
