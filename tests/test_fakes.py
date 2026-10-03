"""The shared test doubles behave the way the other tests assume."""

import pytest
from fakes import by_role, failing

from autoimprover.types import BackendError, Call


def call(role: str = "task") -> Call:
    return Call(role=role, model="m", user="u")  # type: ignore[arg-type]


def test_scripted_backend_records_calls_and_counts_by_role():
    backend = by_role({"task": "out", "judge": "ok"})
    backend.complete(call("task"))
    backend.complete(call("task"))
    backend.complete(call("judge"))
    assert (backend.count(), backend.count("task"), backend.count("judge")) == (3, 2, 1)


def test_by_role_serves_a_list_in_order_and_repeats_the_last():
    backend = by_role({"task": ["a", "b"]})
    assert [backend.complete(call()).text for _ in range(3)] == ["a", "b", "b"]


def test_by_role_raises_an_exception_entry():
    backend = by_role({"task": BackendError("boom")})
    with pytest.raises(BackendError, match="boom"):
        backend.complete(call())


def test_failing_backend_fails_every_call_and_still_records_it():
    backend = failing()
    for _ in range(2):
        with pytest.raises(BackendError):
            backend.complete(call())
    assert backend.count() == 2
