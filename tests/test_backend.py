"""The backend layers below the cache: the shared clock, the call limit and deadline, and the
retries with the per-(role, model) failure count (SPEC R17, R24; ADR-004)."""

import pytest
from fakes import FakeClock, ScriptedBackend

from autoimprover.backend import BudgetedBackend, Clock, ResilientBackend
from autoimprover.types import (
    CALL_RETRIES,
    MAX_CONSECUTIVE_FAILURES,
    BackendError,
    BudgetExhausted,
    Call,
    CallError,
    CallFailed,
    SessionNotLockedDown,
)

TASK = Call(role="task", model="claude-haiku-4-5-20251001", user="situation 1")


def ok(_call: Call) -> str:
    return "an answer"


# --- Clock (SPEC R17: monotonic, carried across --resume) ----------------------------------------


def test_clock_counts_from_its_creation_on_the_injected_now():
    fake = FakeClock(start=1000.0)
    clock = Clock(now=fake.now)
    assert clock.elapsed() == 0.0
    fake.advance(12.5)
    assert clock.elapsed() == 12.5


def test_clock_carries_the_elapsed_time_of_a_resumed_run():
    fake = FakeClock(start=50.0)
    clock = Clock(now=fake.now, elapsed=600.0)
    assert clock.elapsed() == 600.0
    fake.advance(30.0)
    assert clock.elapsed() == 630.0


def test_remaining_is_measured_to_a_deadline_in_elapsed_seconds_and_may_be_negative():
    fake = FakeClock()
    clock = Clock(now=fake.now, elapsed=100.0)
    assert clock.remaining(250.0) == 150.0
    fake.advance(200.0)
    assert clock.remaining(250.0) == -50.0


# --- BudgetedBackend (SPEC R17: one limit on live calls, a deadline on the shared clock) --------


def budgeted(raw: ScriptedBackend, fake: FakeClock, *, limit=10, used=0, deadline=1000.0, log=None):
    on_call = None if log is None else (lambda used, elapsed: log.append((used, elapsed)))
    return BudgetedBackend(
        raw, limit=limit, used=used, clock=Clock(now=fake.now), deadline=deadline, on_call=on_call
    )


def test_budgeted_starts_from_the_saved_total_and_refuses_at_the_limit():
    raw, fake = ScriptedBackend(ok), FakeClock()
    backend = budgeted(raw, fake, limit=8, used=5)
    for _ in range(3):
        assert backend.complete(TASK).text == "an answer"
    assert backend.used == 8
    with pytest.raises(BudgetExhausted) as refused:
        backend.complete(TASK)
    assert refused.value.cause == "budget"
    assert raw.count() == 3
    assert backend.used == 8


def test_a_refused_call_reaches_no_raw_call_is_not_counted_and_reports_nothing():
    raw, fake, log = ScriptedBackend(ok), FakeClock(), []
    backend = budgeted(raw, fake, limit=4, used=4, log=log)
    with pytest.raises(BudgetExhausted):
        backend.complete(TASK)
    assert (raw.count(), backend.used, log) == (0, 4, [])


def test_a_failed_attempt_is_counted_before_the_raw_call_and_its_error_propagates_unchanged():
    error = CallError("exit 1")
    seen_used = []
    fake = FakeClock()

    def script(_call: Call) -> Exception:
        seen_used.append(backend.used)
        return error

    backend = budgeted(ScriptedBackend(script), fake, limit=10, used=2)
    with pytest.raises(CallError) as raised:
        backend.complete(TASK)
    assert raised.value is error
    assert seen_used == [3]
    assert backend.used == 3


@pytest.mark.parametrize("error", [SessionNotLockedDown("tools"), ValueError("bad"), KeyError("x")])
def test_any_raw_exception_is_counted_and_propagates_unchanged(error: Exception):
    backend = budgeted(ScriptedBackend(lambda _c: error), FakeClock(), limit=10, used=0)
    with pytest.raises(type(error)) as raised:
        backend.complete(TASK)
    assert raised.value is error
    assert backend.used == 1


def test_the_deadline_refuses_a_call_once_the_clock_reaches_it():
    raw, fake = ScriptedBackend(ok), FakeClock()
    backend = budgeted(raw, fake, deadline=100.0)
    fake.advance(99.5)
    backend.complete(TASK)
    fake.advance(0.5)
    with pytest.raises(BudgetExhausted) as refused:
        backend.complete(TASK)
    assert refused.value.cause == "clock"
    assert (raw.count(), backend.used) == (1, 1)


def test_the_clock_wins_when_the_limit_and_the_deadline_are_both_reached():
    fake = FakeClock()
    backend = budgeted(ScriptedBackend(ok), fake, limit=3, used=3, deadline=50.0)
    fake.advance(50.0)
    with pytest.raises(BudgetExhausted) as refused:
        backend.complete(TASK)
    assert refused.value.cause == "clock"


def test_the_deadline_is_measured_on_the_carried_clock_of_a_resumed_run():
    fake = FakeClock()
    clock = Clock(now=fake.now, elapsed=2025.0)
    backend = BudgetedBackend(ScriptedBackend(ok), limit=10, used=0, clock=clock, deadline=2025.0)
    with pytest.raises(BudgetExhausted) as refused:
        backend.complete(TASK)
    assert refused.value.cause == "clock"


def test_raise_limit_opens_the_final_share_of_calls_and_clock():
    raw, fake = ScriptedBackend(ok), FakeClock()
    backend = budgeted(raw, fake, limit=5, used=5, deadline=60.0)
    fake.advance(60.0)
    with pytest.raises(BudgetExhausted):
        backend.complete(TASK)
    backend.raise_limit(7, deadline=80.0)
    assert (backend.limit, backend.deadline) == (7, 80.0)
    backend.complete(TASK)
    backend.complete(TASK)
    with pytest.raises(BudgetExhausted) as refused:
        backend.complete(TASK)
    assert refused.value.cause == "budget"
    fake.advance(20.0)
    backend.raise_limit(9, deadline=80.0)
    with pytest.raises(BudgetExhausted) as late:
        backend.complete(TASK)
    assert late.value.cause == "clock"
    assert raw.count() == 2


def test_on_call_reports_the_count_and_the_elapsed_time_after_every_attempt():
    fake, log = FakeClock(), []
    answers = iter(["fine", CallError("timeout"), "fine again"])
    raw = ScriptedBackend(lambda _c: next(answers), duration_s=4.0, clock=fake)
    backend = budgeted(raw, fake, limit=10, used=5, log=log)
    backend.complete(TASK)
    with pytest.raises(CallError):
        backend.complete(TASK)
    backend.complete(TASK)
    assert log == [(6, 4.0), (7, 4.0), (8, 8.0)]


# --- ResilientBackend (SPEC R24: two retries, three failed calls per role and model end the run) -

JUDGE = Call(role="judge", model="claude-opus-5-5", user="{}")
REFLECT = Call(role="reflect", model="claude-opus-5-5", user="reflect on this")


class Switch:
    """A raw script whose answer per call key the test flips between success and failure."""

    def __init__(self) -> None:
        self.failing: set[tuple[str, str]] = set()
        self.raise_next: list[Exception] = []

    def __call__(self, call: Call) -> str | Exception:
        if self.raise_next:
            return self.raise_next.pop(0)
        if (call.role, call.model) in self.failing:
            return CallError(f"{call.role} is down")
        return "ok"


def test_a_success_returns_at_once_after_one_attempt():
    raw = ScriptedBackend(ok)
    assert ResilientBackend(raw).complete(TASK).text == "an answer"
    assert raw.count() == 1


def test_a_call_error_is_retried_and_a_later_success_is_returned():
    answers = iter([CallError("exit 1"), CallError("timeout"), "third time"])
    raw = ScriptedBackend(lambda _c: next(answers))
    assert ResilientBackend(raw).complete(TASK).text == "third time"
    assert raw.count() == 1 + CALL_RETRIES == 3


def test_a_call_that_fails_every_attempt_raises_call_failed_from_the_last_error():
    errors = iter([CallError("first"), CallError("second"), CallError("last")])
    raw = ScriptedBackend(lambda _c: next(errors))
    with pytest.raises(CallFailed, match="last") as failed:
        ResilientBackend(raw).complete(TASK)
    assert isinstance(failed.value.__cause__, CallError)
    assert raw.count() == 3
    assert all(c is TASK for c in raw.calls)


def test_the_third_consecutive_failed_call_of_one_role_and_model_ends_the_run():
    switch = Switch()
    switch.failing.add(("judge", JUDGE.model))
    raw = ScriptedBackend(switch)
    backend = ResilientBackend(raw)
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        with pytest.raises(CallFailed):
            backend.complete(JUDGE)
    with pytest.raises(BackendError, match="judge") as ended:
        backend.complete(JUDGE)
    assert JUDGE.model in str(ended.value)
    assert raw.count() == 3 * MAX_CONSECUTIVE_FAILURES
    with pytest.raises(BackendError):
        backend.complete(JUDGE)


def test_successes_of_another_role_or_model_do_not_reset_the_count():
    switch = Switch()
    switch.failing.add(("judge", JUDGE.model))
    backend = ResilientBackend(ScriptedBackend(switch))
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        with pytest.raises(CallFailed):
            backend.complete(JUDGE)
        backend.complete(TASK)
        backend.complete(REFLECT)
    with pytest.raises(BackendError):
        backend.complete(JUDGE)


def test_the_same_role_on_another_model_is_counted_apart():
    switch = Switch()
    other = Call(role="judge", model="claude-sonnet-5-5", user="{}")
    switch.failing.update({("judge", JUDGE.model), ("judge", other.model)})
    backend = ResilientBackend(ScriptedBackend(switch))
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        for call in (JUDGE, other):
            with pytest.raises(CallFailed):
                backend.complete(call)


def test_a_success_of_the_same_role_and_model_resets_the_count():
    switch = Switch()
    backend = ResilientBackend(ScriptedBackend(switch))
    for _ in range(3):
        switch.failing.add(("judge", JUDGE.model))
        for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
            with pytest.raises(CallFailed):
                backend.complete(JUDGE)
        switch.failing.clear()
        backend.complete(JUDGE)


def test_a_key_at_the_limit_stays_there_until_it_succeeds():
    switch = Switch()
    switch.failing.add(("judge", JUDGE.model))
    backend = ResilientBackend(ScriptedBackend(switch))
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        with pytest.raises((CallFailed, BackendError)):
            backend.complete(JUDGE)
    switch.failing.clear()
    backend.complete(JUDGE)
    switch.failing.add(("judge", JUDGE.model))
    with pytest.raises(CallFailed):
        backend.complete(JUDGE)


PASS_THROUGH = [
    BudgetExhausted("limit", cause="budget"),
    BudgetExhausted("deadline", cause="clock"),
    SessionNotLockedDown("plugins"),
    BackendError("down"),
    CallFailed("already failed"),
    ValueError("bad reply"),
    KeyError("missing"),
]


@pytest.mark.parametrize("error", PASS_THROUGH, ids=lambda e: type(e).__name__)
def test_anything_but_call_error_propagates_at_once_unretried(error: Exception):
    raw = ScriptedBackend(lambda _c: error)
    with pytest.raises(type(error)) as raised:
        ResilientBackend(raw).complete(JUDGE)
    assert raised.value is error
    assert raw.count() == 1


@pytest.mark.parametrize("error", PASS_THROUGH, ids=lambda e: type(e).__name__)
def test_other_exceptions_neither_count_as_failures_nor_reset_the_count(error: Exception):
    switch = Switch()
    switch.failing.add(("judge", JUDGE.model))
    backend = ResilientBackend(ScriptedBackend(switch))
    with pytest.raises(CallFailed):
        backend.complete(JUDGE)
    switch.raise_next.append(error)
    with pytest.raises(type(error)):
        backend.complete(JUDGE)
    with pytest.raises(CallFailed):  # a counted exception would make this the third failure
        backend.complete(JUDGE)
    switch.raise_next.append(error)
    with pytest.raises(type(error)):
        backend.complete(JUDGE)
    with pytest.raises(BackendError):  # a reset would make this only the first failure
        backend.complete(JUDGE)


def test_budget_exhausted_between_retries_stops_the_retries():
    answers = iter([CallError("exit 1"), BudgetExhausted("limit")])
    raw = ScriptedBackend(lambda _c: next(answers))
    with pytest.raises(BudgetExhausted):
        ResilientBackend(raw).complete(TASK)
    assert raw.count() == 2


def test_every_attempt_of_a_failing_call_is_counted_by_the_budget_below():
    raw = ScriptedBackend(lambda _c: CallError("exit 1"))
    budget = budgeted(raw, FakeClock(), limit=10, used=0)
    with pytest.raises(CallFailed):
        ResilientBackend(budget).complete(TASK)
    assert budget.used == raw.count() == 3
