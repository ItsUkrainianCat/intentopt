"""The in-run calibration of the fast tiers (SPEC R25; ADR-011): a fast run fits the latency model
to the replies it has had, after stage A and again after stage B, robustly (medians, clamped to
half and one and a half times the constants, so one slow call cannot dominate), slows it by the
median ratio of real to planned output tokens (from 1 to 4), and spends the time that model leaves
within the plan's share of the clock, never past the deadline, on more pick scenarios: from the
scenarios at hand first, else from one more synthesis call. What no longer fits shrinks, by the
same model, scenarios first.

The bench of 2026-10-06 (checked tier, `--time 2m`) planned 98 s and used 33 to 75 s.
"""

import json

import pytest
from fakes import FakeClock, ScriptedBackend, intake_reply
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    MECHANICS_OVERHEAD_S,
    PROMPT,
    World,
    mechanics_latency_model,
    no_disk_flush,
    run,
    scenario_of,
)

from autoimprover import fastplan
from autoimprover.fast import MORE_SAMPLE
from autoimprover.fast_calibrate import Growth, Timed, fit, grow, planned_tokens
from autoimprover.fastplan import (
    FastPlan,
    Latency,
    misfit,
    second_stages,
    stage_a,
    synthesis_stage,
    tail,
)
from autoimprover.pairwise_text import PAIRWISE_BATCH_SYSTEM
from autoimprover.runner import count_tokens
from autoimprover.runstore import RunStore
from autoimprover.types import Call, CallError, CallFailed, Reply, Scenario

PROMPT_TOKENS = count_tokens(PROMPT)  # 7
PLAIN = "Answer the user's request plainly."  # a rewrite that runs no better
SECOND = "Answer the user's request [[better]] now."

# Every test here runs on the mechanics' constants (the autouse fixture): 2.4 s, 70 per second.
OH, V = MECHANICS_OVERHEAD_S, fastplan.TOKENS_PER_S


def model(overhead_s: float, tokens_per_s: float, tokens: list[int]) -> list[tuple[int, float]]:
    return [(t, overhead_s + t / tokens_per_s) for t in tokens]


# --- the fit --------------------------------------------------------------------------------------


def test_fewer_than_three_replies_fit_nothing():
    assert fit([]) is None
    assert fit(model(2.0, 90, [100, 400])) is None
    assert fit(model(2.0, 90, [100, 400, 700])) == pytest.approx(Latency(2.0, 90))


def test_the_fit_finds_the_fixed_seconds_and_the_speed_of_the_replies():
    assert fit(model(2.0, 90, [80, 150, 300, 420, 700])) == pytest.approx(Latency(2.0, 90))
    assert fit(model(3.0, 50, [80, 150, 300])) == pytest.approx(Latency(3.0, 50))


def test_one_slow_call_cannot_dominate_the_fit():
    """Medians: the slopes between every two replies, then each reply's fixed seconds."""
    samples = [*model(2.0, 90, [80, 120, 150, 200, 300, 420, 500, 700]), (300, 60.0)]
    assert fit(samples) == pytest.approx(Latency(2.0, 90))


@pytest.mark.parametrize(
    ("truth", "fitted"),
    [
        ((0.5, 400), (0.5 * OH, 1.5 * V)),  # far faster than the constants: 1.2 s, 105 per s
        ((20.0, 10), (1.5 * OH, 0.5 * V)),  # far slower: 3.6 s, 35 per s
        ((1.0, 90), (0.5 * OH, 90)),
    ],
)
def test_the_fit_stays_within_half_and_one_and_a_half_times_the_constants(truth, fitted):
    assert fit(model(*truth, [80, 150, 300, 420])) == pytest.approx(Latency(*fitted))


def test_replies_that_take_no_longer_with_more_tokens_have_the_top_speed():
    """No positive slope: tokens cost nothing measurable, so the speed is the clamp's top."""
    fitted = fit([(20, 3.0), (40, 3.0), (60, 2.95)])
    assert fitted == pytest.approx(Latency(3.0 - 40 / (1.5 * V), 1.5 * V))


def test_replies_of_one_length_keep_the_speed_of_the_constants():
    assert fit([(100, 3.0), (100, 3.2), (100, 3.4)]) == pytest.approx(Latency(3.2 - 100 / V, V))


def test_the_fit_does_not_depend_on_the_order_of_the_replies():
    samples = [*model(2.0, 90, [80, 150, 300]), (420, 9.5), (700, 8.0)]
    assert fit(samples) == fit(list(reversed(samples))) == fit(samples[2:] + samples[:2])


def test_the_clamp_follows_the_constants_the_planner_reads():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fastplan, "OVERHEAD_S", 3.4)
        assert fit(model(0.5, 400, [80, 150, 300])) == pytest.approx(Latency(1.7, 1.5 * V))


# --- what is measured -----------------------------------------------------------------------------

SHORT = 20  # prompt tokens: a task call is planned at 150 output tokens, a rewrite at 60


def call(user: str, role: str = "task") -> Call:
    return Call(role=role, model="m", user=user)


class Replies:
    """A backend whose reply to a call is set per user text: (tokens out, seconds)."""

    def __init__(self, timings: dict[str, tuple[int, float] | Exception]) -> None:
        self.timings = timings

    def complete(self, made: Call) -> Reply:
        timing = self.timings[made.user]
        if isinstance(timing, Exception):
            raise timing
        tokens, seconds = timing
        return Reply(text=made.user, tokens_out=tokens, duration_s=seconds)


def test_every_reply_with_a_duration_is_a_sample_and_passes_through():
    timed = Timed(Replies({"a": (100, 3.0), "b": (300, 5.0), "c": (50, 0.0)}), SHORT)
    replies = [timed.complete(call(user)) for user in ("a", "b", "c")]
    assert [r.text for r in replies] == ["a", "b", "c"]
    assert sorted(timed.samples()) == [(100, 3.0), (300, 5.0)]  # no duration: not measured


def test_a_call_asked_twice_is_one_sample():
    """A repeat is the cache's reply to the same call, with the same stored duration."""
    timed = Timed(Replies({"a": (100, 3.0)}), SHORT)
    for _ in range(3):
        timed.complete(call("a"))
    timed.complete(call("a", role="judge"))
    assert timed.samples() == [(100, 3.0), (100, 3.0)]


def test_a_failed_call_is_no_sample_and_its_error_passes_through():
    timed = Timed(Replies({"a": CallFailed("down"), "b": CallError("x")}), SHORT)
    for user, error in (("a", CallFailed), ("b", CallError)):
        with pytest.raises(error):
            timed.complete(call(user))
    assert timed.samples() == []


def test_the_fake_backends_replies_carry_no_duration_so_no_test_is_calibrated_by_accident():
    timed = Timed(ScriptedBackend(lambda _call: "answer"), SHORT)
    timed.complete(call("a"))
    assert timed.samples() == []
    clocked = Timed(
        ScriptedBackend(lambda _call: "answer", duration_s=2.5, clock=FakeClock()), SHORT
    )
    clocked.complete(call("a"))
    assert clocked.samples() == [(len("answer"), 2.5)]


def judge(names: list[str], system: str = "") -> Call:
    user = json.dumps({"scenarios": [{"scenario": name} for name in names]})
    return Call(role="judge", model="m", user=user, system=system)


def test_each_call_has_the_output_tokens_the_planner_assumes_for_it():
    synth = Call(role="synth", model="m", user=json.dumps({"prompt": "p", "count": 3}))
    assert planned_tokens(Call(role="intake", model="m", user="p"), SHORT) == 380
    assert planned_tokens(synth, SHORT) == 3 * 45
    assert planned_tokens(Call(role="reflect", model="m", user="p"), SHORT) == 60
    assert planned_tokens(Call(role="reflect", model="m", user="p"), 500) == 600
    assert planned_tokens(call("a"), SHORT) == 150 and planned_tokens(call("a"), 148) == 444
    assert planned_tokens(judge(["s1", "s2"], PAIRWISE_BATCH_SYSTEM), SHORT) == 2 * 40
    assert planned_tokens(judge(["contract-1", "contract-2", "contract-3"]), SHORT) == 3 * 75
    assert planned_tokens(judge(["s1", "s2", "s3", "s4"]), SHORT) == 4 * 75  # stage E
    assert planned_tokens(Call(role="judge", model="m", user="not json"), SHORT) is None


def timed_tasks(tokens: list[int], overhead_s: float = 2.0, speed: float = 90) -> Timed:
    """Task calls of a 20-token prompt (planned at 150 tokens each) whose replies follow the
    model `overhead_s` + tokens / `speed`."""
    timings = {f"t{n}": (t, overhead_s + t / speed) for n, t in enumerate(tokens)}
    timed = Timed(Replies(timings), SHORT)  # type: ignore[arg-type]
    for user in timings:
        timed.complete(call(user))
    return timed


def test_replies_longer_than_planned_slow_the_model_by_their_median_ratio():
    """300 of a planned 150 tokens is twice the plan: the planner's 150 tokens take what 300 do."""
    assert timed_tasks([200, 300, 400]).model() == pytest.approx(Latency(2.0, 90 / 2))


def test_replies_shorter_than_planned_never_speed_the_model_up():
    assert timed_tasks([50, 100, 120]).model() == pytest.approx(Latency(2.0, 90))


def test_the_token_ratio_stops_at_4():
    assert timed_tasks([1500, 1800, 2100]).model() == pytest.approx(Latency(2.0, 90 / 4))


def test_a_model_needs_three_timed_replies():
    assert timed_tasks([200, 300]).model() is None


# --- the re-plan after stage A --------------------------------------------------------------------

FAST = Latency(0.5 * OH, 1.5 * V)  # 1.2 s + tokens / 105


def rest(rewrites: int, scenarios: int, holdout: int, workers: int, extra: int, second: int = 0):
    stages = tail(rewrites, scenarios, holdout, workers, SHORT, FAST)
    if second:
        stages += fastplan.second_stages(second, scenarios, workers, SHORT, FAST)
    return ((synthesis_stage(extra, FAST),) if extra else ()) + stages


def seconds(stages) -> float:
    return sum(stage.seconds for stage in stages)


def grown(**given) -> Growth | None:
    fields = {
        "tier": "fast",
        "rewrites": 1,
        "scenarios": 2,
        "holdout": 0,
        "have": 2,
        "can_synthesise": True,
        "rewrites2": 0,
        "workers": 6,
        "prompt_tokens": SHORT,
        "model": FAST,
        "seconds_left": 100.0,
        "calls_left": 100,
    } | given
    return grow(**fields)


def test_time_left_buys_up_to_4_pick_scenarios_with_one_more_synthesis_call():
    assert grown() == Growth(scenarios=4, holdout=0, synthesise=2)
    assert grown(scenarios=3, have=3) == Growth(4, 0, 1)


def test_the_scenarios_at_hand_come_first_and_need_no_call():
    """The user's examples: no synthesis, and never more pick scenarios than there are."""
    assert grown(have=10, can_synthesise=False) == Growth(4, 0, 0)
    assert grown(have=3, can_synthesise=False) == Growth(3, 0, 0)
    assert grown(have=2, can_synthesise=False) is None


def test_the_largest_pick_count_whose_rest_fits_the_time_left():
    four, three = seconds(rest(1, 4, 0, 6, 2)), seconds(rest(1, 3, 0, 6, 1))
    assert three < four
    assert grown(seconds_left=four) == Growth(4, 0, 2)
    assert grown(seconds_left=four - 0.01) == Growth(3, 0, 1)
    assert grown(seconds_left=three - 0.01) is None


def test_a_second_generation_and_a_holdout_are_part_of_the_rest():
    """The checked tier keeps its holdout: the held-out scenarios are synthesised too."""
    stages = rest(2, 4, 2, 6, 2, second=1)
    assert grown(
        tier="checked", rewrites=2, holdout=2, have=4, rewrites2=1, seconds_left=seconds(stages)
    ) == Growth(4, 2, 2)
    shorter = rest(2, 3, 2, 6, 1, second=1)  # 29.1 s against 33.3 s
    assert seconds(shorter) < seconds(stages) - 1
    assert grown(
        tier="checked", rewrites=2, holdout=2, have=4, rewrites2=1, seconds_left=seconds(stages) - 1
    ) == Growth(3, 2, 1)


def test_the_calls_left_bound_the_growth_too():
    calls = sum(stage.calls for stage in rest(1, 4, 0, 6, 2))
    assert grown(calls_left=calls) == Growth(4, 0, 2)
    assert misfit(rest(1, 4, 0, 6, 2), 100.0, calls - 1) == "budget"
    assert grown(calls_left=calls - 1) == Growth(3, 0, 1)


def test_a_plan_already_at_4_pick_scenarios_has_nothing_to_grow():
    assert grown(scenarios=4, have=4) is None


# --- a fast run that calibrates -------------------------------------------------------------------
# On 1 worker every paced call advances the fake clock by `pace` seconds. Replies of one duration
# fit the top speed and the lowest fixed seconds of the clamp: with the mechanics' constants (2.4 s,
# 70 per s, the autouse fixture) that is 1.2 s + tokens / 105, so a scoring run of the 7-token
# PROMPT takes 2.629 s, a pairwise call over 4 scenarios 2.724 s and a synthesis of 2 2.057 s.


class Waves(World):
    """The world, with a second synthesis wave that writes other situations than the first."""

    def __call__(self, call: Call) -> str | Exception:
        if call.role == "synth" and call.sample >= MORE_SAMPLE:
            self.hook(call)
            count = json.loads(call.user)["count"]
            items = [{"id": f"s{i}", "input": f"more {i}"} for i in range(1, count + 1)]
            return json.dumps({"scenarios": items})
        return super().__call__(call)


def shape(time_s: int, rewrites: int, scenarios: int, rewrites2: int = 0) -> FastPlan:
    """A fast plan of this shape on 1 worker, from the planner's own stages; the runner reads only
    the shape."""
    stages = (
        stage_a(rewrites, scenarios, 1, PROMPT_TOKENS),
        *tail(rewrites, scenarios, 0, 1, PROMPT_TOKENS),
    )
    if rewrites2:
        stages += second_stages(rewrites2, scenarios, 1, PROMPT_TOKENS)
    seconds, calls = sum(s.seconds for s in stages), sum(s.calls for s in stages)
    generations = 2 if rewrites2 else 1
    return FastPlan(
        "fast", time_s, 1, rewrites, scenarios, 0, stages, seconds, calls, generations, rewrites2
    )


def counts(result) -> list[int]:
    return [json.loads(c.user)["count"] for c in result.calls("synth")]


def picked(result) -> set[str]:
    return {scenario_of(c) for c in result.calls("task")}


@pytest.mark.parametrize("pace", [0.2, 5.0])
def test_after_stage_a_the_time_the_fitted_model_leaves_buys_more_pick_scenarios(tmp_path, pace):
    """At 59 s the share is 50.15 s. Paced at 0.2 s, stage A took 0.6 s and the rest on 4
    scenarios needs a synthesis of 2 more, 12 scoring runs and 5 judge calls: 2.057 + 31.543 +
    13.619 = 47.22 s, which fits. Paced at 5 s nothing more fits, and the plan's 2 stand."""
    result = run(tmp_path, Waves(), shape(59, 1, 2), pace=pace)
    if pace == 5.0:
        assert counts(result) == [2] and picked(result) == {"situation 1", "situation 2"}
        assert "after stage A" in result.log and "more scenarios" not in result.log
        return
    assert counts(result) == [2, 2]
    assert [c.sample for c in result.calls("synth")] == [0, MORE_SAMPLE]
    assert picked(result) == {"situation 1", "situation 2", "more 1", "more 2"}
    assert (
        "fast: after stage A: a call takes 1.2 s plus its output tokens at 105 per s" in result.log
    )
    assert "fast: stage A2: 2 more scenarios, to pick on 4" in result.log
    assert result.outcome.prompt == BETTER


def test_replies_far_longer_than_planned_leave_no_time_for_more_scenarios(tmp_path):
    """The twin of the 0.2 s run above with a long intake and a long rewrite: of stage A's three
    replies two have more than 4 times their planned tokens, so the model's speed is a quarter,
    105 / 4 per s, and even 3 scenarios no longer fit."""
    long_rewrite = f"{BETTER} " + " ".join(["extraordinarily"] * 30)
    world = Waves(rewrites=(long_rewrite,), intake=intake_reply("task", goal="x" * 3000))
    result = run(tmp_path, world, shape(59, 1, 2), pace=0.2)
    assert counts(result) == [2] and picked(result) == {"situation 1", "situation 2"}
    assert "replies are 4.0 times the planned tokens" in result.log
    assert result.outcome.prompt == long_rewrite


def test_the_more_scenarios_get_ids_of_their_own_and_are_kept_in_the_run_folder(tmp_path):
    first = run(tmp_path, Waves(), shape(59, 1, 2), pace=0.2)
    store = RunStore.resume(tmp_path, first.run_id)
    try:
        saved = store.scenarios() or []
    finally:
        store.close()
    assert [s.input for s in saved] == ["situation 1", "situation 2", "more 1", "more 2"]
    assert len({s.id for s in saved}) == 4
    again = run(tmp_path, Waves(), shape(59, 1, 2), pace=0.2, run_id=first.run_id)
    assert again.calls("synth") == []  # a resumed run reads them from the run folder (SPEC R22)


def test_the_plans_share_of_the_clock_bounds_the_growth_not_the_deadline(tmp_path):
    """At 30 s the share is 25.5 s: 3 scenarios would need 37.0 s. The deadline is 1000 s."""
    result = run(tmp_path, Waves(), shape(30, 1, 2), pace=0.2, deadline=1000)
    assert counts(result) == [2] and picked(result) == {"situation 1", "situation 2"}


@pytest.mark.parametrize("pace", [0.0, 0.2])
def test_the_fitted_model_also_decides_what_still_fits_the_deadline(tmp_path, pace):
    """Deadline 20 s: by the constants 1 rewrite on 1 scenario needs 3 T + 5 J1 = 31.0 s, so no
    task runs; by the fitted model 3 x 2.629 + 5 x 1.914 = 17.5 s, and 19.4 s are left."""
    result = run(tmp_path, Waves(), shape(59, 1, 2), pace=pace, deadline=20)
    assert counts(result) == [2]
    assert picked(result) == (set() if pace == 0.0 else {"situation 1"})


def test_the_users_examples_are_picked_on_first_and_need_no_synthesis(tmp_path):
    """Two rewrites, so stage A has the three replies a fit needs (the second repeats the first's
    meaning words and is dropped)."""
    examples = [Scenario(id=f"e{i}", input=f"example {i}") for i in range(1, 7)]
    result = run(tmp_path, Waves(), shape(59, 2, 2), pace=0.2, examples=examples)
    assert result.calls("synth") == []
    assert picked(result) == {f"example {i}" for i in range(1, 5)}
    assert "fast: re-plan: pick on 4 scenarios" in result.log


def test_a_failed_second_synthesis_keeps_the_plans_scenarios(tmp_path):
    def hook(call: Call) -> None:
        if call.role == "synth" and call.sample >= MORE_SAMPLE:
            raise CallFailed("down")

    world = Waves(hook=hook)
    result = run(tmp_path, world, shape(59, 1, 2), pace=0.2)
    assert picked(result) == {"situation 1", "situation 2"}
    assert "more scenarios dropped: failed" in result.log
    assert result.outcome.prompt == BETTER


@pytest.mark.parametrize("pace", [0.0, 0.5])
def test_after_stage_b_the_fitted_model_decides_whether_the_second_generation_fits(tmp_path, pace):
    """Deadline 36 s on 1 worker. By the constants 1 rewrite fits on 1 scenario only (3 T + 5 J1
    = 31.0 s) and the second generation then needs 2 Rs + 2 T + 5 J2 = 38.3 s. Paced at 0.5 s,
    stages A to C took 7 s and by the fitted model the second generation on 2 scenarios needs 2 x
    1.771 + 4 x 2.629 + 5 x 2.629 = 27.2 s of the 29 s left."""
    world = Waves(rewrites=(PLAIN,), reflections=(SECOND,))
    result = run(tmp_path, world, shape(45, 1, 2, rewrites2=2), pace=pace, deadline=36)
    second = [c for c in result.calls("reflect") if c.sample >= 100]
    assert (result.outcome.prompt, bool(second)) == ((SECOND, True) if pace else (PROMPT, False))
    assert ("after stage B: a call takes 1.2 s" in result.log) is bool(pace)


@pytest.mark.parametrize("pace", [0.0, 0.2])
def test_a_stage_c_that_does_not_fit_shrinks_its_scenarios_by_the_fitted_model(tmp_path, pace):
    """1 rewrite on 4 scenarios; the last scoring run overruns and leaves 12 s. By the fitted
    model stage C on 4 needs 5 x 2.724 = 13.6 s and on 3 5 x 2.343 = 11.7 s: it judges 3. By the
    constants even 1 scenario needs 5 x 3.471 = 17.4 s, and only the last chance of one pairwise
    call on 1 scenario remains."""
    clock = FakeClock()

    def hook(call: Call) -> None:
        last = call.role == "task" and scenario_of(call) == "situation 4" and BETTER in call.user
        if last:
            clock.advance(1000 - 12 - clock.t - pace)

    result = run(tmp_path, Waves(hook=hook), shape(30, 1, 4), pace=pace, clock=clock)
    pairwise = [c for c in result.calls("judge") if c.system == PAIRWISE_BATCH_SYSTEM]
    sizes = {len(json.loads(c.user)["scenarios"]) for c in pairwise}
    assert sizes == ({3} if pace else {1})
