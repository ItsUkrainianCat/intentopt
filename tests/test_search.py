"""The GEPA seam in parts: the search meter, the run state and its stopper, the batch evaluator
adapter and the reflection wrapper (SPEC R15a, R17, R22, R24; ADR-004, ADR-006, ADR-008).
The whole search on the pinned GEPA, and its replay after a cut, are in `test_search_resume.py`."""

from collections.abc import Callable

import pytest
from fakes import ScriptedBackend, reflection_reply

from autoimprover.search import (
    Abort,
    EvaluatorAdapter,
    Example,
    ReflectionWrapper,
    RunState,
    SearchMeter,
    SkipProposal,
)
from autoimprover.types import (
    INSTRUCTION_BEGIN,
    INSTRUCTION_END,
    BackendError,
    BudgetExhausted,
    Call,
    CallFailed,
    Scenario,
    SessionNotLockedDown,
)

LIMITS = {
    "calls_left_at_start": 20,
    "iter_cost": 8,
    "search_start": (30, 100.0),
    "wall_clock_s": 1000.0,
    "clock_share": 0.75,
}


def charge(state: RunState, *durations: float) -> None:
    """Issue one new distinct call per duration."""
    for duration in durations:
        state.meter.issue(f"call-{state.meter.calls}", duration)


# --- the meter (ADR-004: each distinct call once, the first duration) ----------------------------


def test_the_meter_charges_each_distinct_call_once_with_its_first_duration():
    meter = SearchMeter()
    assert (meter.calls, meter.seconds, meter.seconds_per_call) == (0, 0.0, 0.0)
    meter.issue("a", 4.0)
    meter.issue("b", 2.0)
    meter.issue("a", 9.0)
    meter.issue("b", 0.0)
    assert (meter.calls, meter.seconds) == (2, 6.0)
    assert meter.seconds_per_call == pytest.approx(3.0)


# --- the stopper (ADR-004: reads the meter, the state and its arguments only) --------------------


def test_the_stopper_goes_on_while_one_more_iteration_fits_the_calls_left():
    state = RunState()
    stop = state.stopper(**LIMITS)
    charge(state, *[0.0] * 12)  # 20 - 12 = 8 calls left: one iteration still fits
    state.meter.issue("call-0", 0.0)  # a repeat is not charged
    assert stop(object()) is False and state.stop is None
    charge(state, 0.0)
    assert stop(object()) is True and state.stop == "budget"


def test_the_stopper_looks_one_iteration_ahead_on_the_clock_share():
    state = RunState()
    stop = state.stopper(**{**LIMITS, "calls_left_at_start": 100, "iter_cost": 2})
    charge(state, 162.0, 162.0)  # 100 + 324 + 2 x 162 = 748 < 750
    assert stop(object()) is False
    state = RunState()
    stop = state.stopper(**{**LIMITS, "calls_left_at_start": 100, "iter_cost": 2})
    charge(state, 162.5, 162.5)  # 100 + 325 + 2 x 162.5 = 750: the share is reached
    assert stop(object()) is True and state.stop == "clock"


def test_a_search_that_starts_past_its_clock_share_stops_before_any_iteration():
    state = RunState()
    assert state.stopper(**{**LIMITS, "search_start": (30, 750.0)})(object()) is True
    assert state.stop == "clock"


def test_the_calls_condition_names_the_cause_when_both_conditions_hold():
    state = RunState()
    stop = state.stopper(**{**LIMITS, "calls_left_at_start": 1})
    charge(state, 1000.0)
    assert stop(object()) is True and state.stop == "budget"


def test_a_recorded_stop_or_abort_stops_the_search_and_keeps_its_cause():
    state = RunState()
    stop = state.stopper(**LIMITS)
    assert stop(object()) is False and state.iterations == 1
    state.record_stop("clock")
    assert stop(object()) is True and (state.stop, state.iterations) == ("clock", 1)
    aborted = RunState()
    aborted.record(ValueError("bug"))
    assert aborted.stopper(**LIMITS)(object()) is True
    assert (aborted.stop, aborted.iterations) == (None, 0)


# --- recording a failure, and the exit it maps to (ADR-004 items 3 and 4) ------------------------

ABORTS = [
    (BackendError("three in a row"), "backend"),
    (SessionNotLockedDown("tools"), "lockdown"),
    (ValueError("a bug"), "bug"),
    (KeyError("a bug"), "bug"),
]


@pytest.mark.parametrize(("error", "cause"), ABORTS, ids=[type(e).__name__ for e, _ in ABORTS])
def test_an_abort_keeps_its_cause_and_is_re_raised_as_the_original_exception(error, cause):
    state = RunState()
    state.record(error)
    assert state.abort == Abort(cause, error) and state.stop is None
    with pytest.raises(type(error)) as raised:
        state.raise_if_aborted()
    assert raised.value is error


@pytest.mark.parametrize("cause", ["budget", "clock"])
def test_budget_exhausted_is_a_stop_with_its_own_cause_never_an_abort(cause):
    state = RunState()
    state.record(BudgetExhausted("used up", cause=cause))
    assert (state.stop, state.abort) == (cause, None)
    state.raise_if_aborted()


def test_the_first_cause_of_each_kind_is_kept():
    state = RunState()
    first = SessionNotLockedDown("tools")
    state.record(BudgetExhausted("deadline", cause="clock"))
    state.record(BudgetExhausted("limit", cause="budget"))
    state.record(first)
    state.record(BackendError("later"))
    assert state.stop == "clock" and state.abort == Abort("lockdown", first)
    with pytest.raises(SessionNotLockedDown):
        state.raise_if_aborted()


def test_notes_follow_the_latest_proposal_until_the_instruction_is_first_completed():
    state = RunState()
    state.note("A", ("first",))
    state.note("A", ("second",))
    state.complete("A", 0.5)
    state.note("A", ("a later proposal of the same text",))
    state.completed.pop("A")
    state.note("A", ("after it was dropped",))
    assert state.notes == {"A": ("second",)} and state.completed == {}


def test_a_fresh_state_raises_nothing_and_is_not_stopping():
    state = RunState()
    state.raise_if_aborted()
    assert (state.abort, state.stop, state.stopping, state.completed) == (None, None, False, {})


# --- the batch evaluator adapter (ADR-004 item 3; SPEC R15a, R24) --------------------------------

S1, S2, S3, S4, S5 = (Scenario(id=f"s{i}", input=f"input {i}") for i in range(1, 6))
VAL = (S4, S5)
SEED = "the original prompt"
INCOMPLETE = "incomplete"
NEUTRAL = (0.0, {"incomplete": True})


class FakeEvaluator:
    """A BatchEvaluator scoring each scenario by `score(candidate, scenario)`: a number, INCOMPLETE
    (a failed call), or an exception raised for the whole batch."""

    def __init__(self, score: Callable[[str, Scenario], object] = lambda _c, _s: 0.5) -> None:
        self.score = score
        self.calls: list[tuple[str, list[Scenario]]] = []

    def __call__(self, candidate, scenarios):
        self.calls.append((candidate, list(scenarios)))
        entries = []
        for scenario in scenarios:
            result = self.score(candidate, scenario)
            if isinstance(result, BaseException):
                raise result
            if result == INCOMPLETE:
                entries.append((0.0, {"incomplete": True, "error": "judge down"}))
            else:
                entries.append((result, {"scenario": scenario.id, "scores": {"content": result}}))
        return entries


def minibatch(candidate: str, *scenarios: Scenario) -> list[tuple[str, Example]]:
    return [(candidate, Example(s, val=False)) for s in scenarios]


def valset(candidate: str, *scenarios: Scenario) -> list[tuple[str, Example]]:
    return [(candidate, Example(s, val=True)) for s in scenarios or VAL]


def adapter(evaluator: FakeEvaluator, state: RunState | None = None) -> EvaluatorAdapter:
    return EvaluatorAdapter(evaluator, state or RunState(), seed=SEED, val=VAL)


def test_pairs_are_scored_per_candidate_in_order_and_returned_aligned():
    evaluator = FakeEvaluator(lambda c, s: {"A": 0.1, "B": 0.2}[c] + int(s.id[1:]) / 100)
    pairs = minibatch("A", S1) + minibatch("B", S2) + minibatch("A", S3) + valset("B", S4)
    results = adapter(evaluator)(pairs)
    assert evaluator.calls == [("A", [S1, S3]), ("B", [S2, S4])]
    assert [score for score, _ in results] == pytest.approx([0.11, 0.22, 0.13, 0.24])
    assert [info["scenario"] for _, info in results] == ["s1", "s2", "s3", "s4"]


def test_a_full_clean_valset_pass_completes_a_candidate_with_its_mean_score():
    state = RunState()
    run = adapter(FakeEvaluator(lambda c, s: 1.0 if s == S4 else 0.5), state)
    run(valset(SEED) + valset("A"))
    assert state.completed == {SEED: 0.75, "A": 0.75}


def test_a_minibatch_or_a_partial_valset_pass_completes_nothing():
    state = RunState()
    run = adapter(FakeEvaluator(), state)
    run(minibatch("A", S4, S5))  # the valset's scenarios, but as a minibatch (n < 8)
    run(valset("B", S4))
    assert state.completed == {}


def test_a_candidate_with_a_failed_call_never_completes_and_a_completed_one_is_dropped():
    state = RunState()
    run = adapter(FakeEvaluator(lambda c, s: INCOMPLETE if c in "AB" and s == S1 else 0.5), state)
    run(valset("B"))
    assert state.completed == {"B": 0.5}
    run(minibatch("A", S1, S2) + minibatch("B", S1))
    run(valset("A"))
    assert state.completed == {} and state.abort is None and state.stop is None


def test_a_failed_call_on_the_seed_aborts_as_a_backend_failure_never_a_zero():
    state = RunState()
    evaluator = FakeEvaluator(lambda c, s: INCOMPLETE if s == S5 else 1.0)
    results = adapter(evaluator, state)(valset(SEED) + valset("A"))
    assert state.abort is not None and state.abort.cause == "backend"
    error = state.abort.error
    assert isinstance(error, BackendError) and "original prompt" in str(error)
    assert evaluator.calls == [(SEED, [S4, S5])]  # "A" gets no call
    assert results[2:] == [NEUTRAL, NEUTRAL] and state.completed == {}


def test_once_the_search_is_stopping_every_pair_is_neutral_without_a_call():
    for record in (BudgetExhausted("limit"), ValueError("a bug")):
        state = RunState()
        state.record(record)
        evaluator = FakeEvaluator()
        assert adapter(evaluator, state)(minibatch("A", S1) + valset("B")) == [NEUTRAL] * 3
        assert evaluator.calls == []


@pytest.mark.parametrize("cause", ["budget", "clock"])
def test_budget_exhausted_stops_the_search_and_keeps_what_was_completed(cause):
    state = RunState()
    stop = BudgetExhausted("used up", cause=cause)
    evaluator = FakeEvaluator(lambda c, s: stop if c == "B" else 0.5)
    results = adapter(evaluator, state)(valset("A") + valset("B") + valset("C"))
    assert (state.stop, state.abort) == (cause, None)
    assert results[2:] == [NEUTRAL] * 4 and [c for c, _ in evaluator.calls] == ["A", "B"]
    adapter(evaluator, state)(minibatch("A", S1))  # neutral later: "A" stays completed
    assert state.completed == {"A": 0.5}


@pytest.mark.parametrize(("error", "cause"), ABORTS, ids=[type(e).__name__ for e, _ in ABORTS])
def test_any_other_exception_aborts_with_its_cause_and_later_pairs_are_neutral(error, cause):
    state = RunState()
    evaluator = FakeEvaluator(lambda c, s: error if c == "A" else 0.5)
    assert adapter(evaluator, state)(valset("A") + valset("B")) == [NEUTRAL] * 4
    assert state.abort == Abort(cause, error) and state.completed == {}
    assert [c for c, _ in evaluator.calls] == ["A"]


def test_a_call_failed_raised_by_the_evaluator_marks_only_that_candidate_incomplete():
    state = RunState()
    evaluator = FakeEvaluator(lambda c, s: CallFailed("task down") if c == "A" else 0.5)
    results = adapter(evaluator, state)(valset("A") + valset("B"))
    assert [info.get("incomplete") for _, info in results] == [True, True, None, None]
    assert state.completed == {"B": 0.5} and not state.stopping
    adapter(evaluator, state)(valset(SEED + " "))  # not the seed: no abort
    evaluator.score = lambda c, s: CallFailed("task down")
    adapter(evaluator, state)(valset(SEED))
    assert state.abort is not None and state.abort.cause == "backend"


def test_an_interrupt_passes_through_the_adapter_and_records_nothing():
    state = RunState()
    with pytest.raises(KeyboardInterrupt):
        adapter(FakeEvaluator(lambda c, s: KeyboardInterrupt()), state)(valset("A"))
    assert (state.abort, state.stop, state.completed) == (None, None, {})


# --- the reflection wrapper (ADR-004 item 3, ADR-006, ADR-008 reflect row; SPEC R16, R24) --------

REFLECT = "claude-opus-5-5"
CODE = "Answer like this:\n```python\nprint('keep me')\n```\nThen stop.\n```\nlast block\n```"


def reflector(answer: object, state: RunState | None = None):
    """A wrapper over a raw backend that always answers `answer` (text or an exception)."""

    def script(_call: Call):
        if isinstance(answer, BaseException) and not isinstance(answer, Exception):
            raise answer
        return answer

    raw = ScriptedBackend(script)
    return ReflectionWrapper(raw, state or RunState(), model=REFLECT), raw


def test_the_reflection_call_carries_the_rendered_prompt_and_a_running_sample():
    wrapper, raw = reflector(reflection_reply("Be brief."))
    wrapper("first prompt")
    wrapper("second prompt")
    assert raw.calls == [
        Call(role="reflect", model=REFLECT, user="first prompt", system="", sample=0),
        Call(role="reflect", model=REFLECT, user="second prompt", system="", sample=1),
    ]


def test_the_instruction_comes_back_whole_in_one_outer_fence_with_its_inner_fences():
    state = RunState()
    reply = "Thinking first.\n" + reflection_reply(CODE, ("a", "b", "c", "d"))
    wrapper, _ = reflector(reply, state)
    assert wrapper("prompt") == "```\n" + CODE + "\n```"
    assert state.notes == {CODE: ("a", "b", "c")}


def test_the_delimiters_are_whole_lines_and_surrounding_blank_lines_are_dropped():
    reply = f"see {INSTRUCTION_BEGIN} inline\n{INSTRUCTION_BEGIN}  \n\n  Be brief.\n\n"
    reply += f"{INSTRUCTION_END}\n- one\nnot a note\n-  two  \n"
    state = RunState()
    wrapper, _ = reflector(reply, state)
    assert wrapper("prompt") == "```\nBe brief.\n```"
    assert state.notes == {"Be brief.": ("one", "two")}


def test_an_instruction_holding_the_marker_lines_itself_is_cut_whole():
    inner = f"Wrap the result:\n{INSTRUCTION_BEGIN}\nresult\n{INSTRUCTION_END}\nDone."
    wrapper, _ = reflector(reflection_reply(inner, ("kept the markers",)))
    assert wrapper("prompt") == "```\n" + inner + "\n```"


REJECTED = {
    "no delimiters": "```\nBe brief.\n```",
    "no end": f"{INSTRUCTION_BEGIN}\nBe brief.",
    "end first": f"{INSTRUCTION_END}\nBe brief.\n{INSTRUCTION_BEGIN}",
    "empty": f"{INSTRUCTION_BEGIN}\n  \n\n{INSTRUCTION_END}\n- why",
    "curr_param": reflection_reply("Keep <curr_param> here."),
    "side_info": reflection_reply("Use <side_info>."),
    "a failed call": CallFailed("reflect call failed 3 attempts"),
}


@pytest.mark.parametrize("answer", REJECTED.values(), ids=REJECTED.keys())
def test_an_unusable_reply_or_a_failed_call_skips_the_proposal_and_aborts_nothing(answer):
    state = RunState()
    wrapper, raw = reflector(answer, state)
    for _ in range(2):  # GEPA's retry asks again, under the next sample
        with pytest.raises(SkipProposal):
            wrapper("prompt")
    assert [call.sample for call in raw.calls] == [0, 1]
    assert (state.abort, state.stop, state.notes) == (None, None, {})


@pytest.mark.parametrize("cause", ["budget", "clock"])
def test_budget_exhausted_is_recorded_as_a_stop_and_re_raised(cause):
    state = RunState()
    error = BudgetExhausted("used up", cause=cause)
    wrapper, _ = reflector(error, state)
    with pytest.raises(BudgetExhausted) as raised:
        wrapper("prompt")
    assert raised.value is error and (state.stop, state.abort) == (cause, None)


@pytest.mark.parametrize(("error", "cause"), ABORTS, ids=[type(e).__name__ for e, _ in ABORTS])
def test_any_other_exception_is_recorded_as_an_abort_and_re_raised(error, cause):
    state = RunState()
    wrapper, _ = reflector(error, state)
    with pytest.raises(type(error)) as raised:
        wrapper("prompt")
    assert raised.value is error and state.abort == Abort(cause, error)


def test_once_the_search_is_stopping_the_wrapper_raises_without_a_call():
    for record in (BudgetExhausted("limit"), BackendError("down")):
        state = RunState()
        state.record(record)
        wrapper, raw = reflector(reflection_reply("Be brief."), state)
        with pytest.raises(SkipProposal):
            wrapper("prompt")
        assert raw.calls == []


def test_an_interrupt_passes_through_the_wrapper_and_records_nothing():
    state = RunState()
    wrapper, _ = reflector(KeyboardInterrupt(), state)
    with pytest.raises(KeyboardInterrupt):
        wrapper("prompt")
    assert (state.abort, state.stop) == (None, None)
