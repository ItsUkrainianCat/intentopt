"""The time tiers and the fast plan (SPEC R25; ADR-011): `--time` picks the tier, and the plan takes
the most rewrites, then the most scenarios, whose estimate fits 85 % of the time, from the latency
model of ADR-011 (2.4 s per call plus output tokens at 70 per second; one slowest call per wave of
`workers` calls). Stage A is one wave: the intake, the synthesis and the rewrites; stage C is a
judge call per prompt plus one contract check of every rewrite. The expected numbers below were
worked out by hand from that model.
"""

import pytest

from autoimprover.fastplan import (
    CHECKED_HOLDOUT,
    FastPlan,
    Stage,
    fast_plan,
    misfit,
    shrink,
    stage_a,
    tail,
    tier_for,
)

SHORT, LONG = 20, 500  # prompt tokens: a one-line prompt and a long system prompt


@pytest.mark.parametrize(
    ("time_s", "tier"),
    [(15, "quick"), (24, "quick"), (25, "fast"), (30, "fast"), (59, "fast")]
    + [(60, "checked"), (299, "checked"), (300, "deep"), (1200, "deep")],
)
def test_the_time_picks_the_tier(time_s, tier):
    assert tier_for(time_s) == tier


@pytest.mark.parametrize("time_s", [14, 0, -30])
def test_a_time_below_15_seconds_is_refused_naming_the_flag_and_the_minimum(time_s):
    with pytest.raises(ValueError, match=r"--time.*15 s"):
        tier_for(time_s)


# (time, workers, prompt tokens, user examples) -> (tier, rewrites, scenarios, holdout, seconds)
TABLE = [
    ((15, 1, SHORT, False), ("quick", 1, 0, 0, 19.128571)),
    ((15, 4, SHORT, False), ("quick", 1, 0, 0, 11.3)),
    ((15, 4, LONG, False), ("quick", 1, 0, 0, 14.442857)),
    ((24, 6, SHORT, False), ("quick", 1, 0, 0, 11.3)),
    ((25, 4, SHORT, False), ("fast", 1, 2, 0, 17.628571)),
    ((25, 6, SHORT, False), ("fast", 2, 2, 0, 17.628571)),
    ((25, 8, SHORT, False), ("fast", 3, 2, 0, 18.7)),
    ((30, 1, SHORT, False), ("fast", 1, 2, 0, 58.142857)),
    ((30, 4, SHORT, False), ("fast", 2, 2, 0, 22.885714)),
    ((30, 4, LONG, False), ("fast", 1, 2, 0, 20.771429)),
    ((30, 6, SHORT, False), ("fast", 3, 3, 0, 23.957143)),
    ((30, 6, LONG, False), ("fast", 2, 2, 0, 20.771429)),
    ((30, 8, SHORT, False), ("fast", 3, 4, 0, 25.028571)),
    ((45, 4, SHORT, False), ("fast", 3, 2, 0, 37.4)),
    ((45, 4, SHORT, True), ("fast", 3, 3, 0, 34.828571)),
    ((45, 4, LONG, False), ("fast", 2, 4, 0, 33.428571)),
    ((45, 6, SHORT, False), ("fast", 3, 4, 0, 30.285714)),
    ((59, 1, SHORT, False), ("fast", 1, 2, 0, 58.142857)),
    ((59, 4, SHORT, False), ("fast", 3, 4, 0, 50.057143)),
    ((60, 1, SHORT, False), ("checked", 1, 2, 4, 113.571429)),
    ((60, 4, SHORT, False), ("checked", 2, 4, 4, 47.485714)),
    ((60, 6, SHORT, False), ("checked", 4, 3, 4, 47.485714)),
    ((60, 8, SHORT, False), ("checked", 6, 4, 4, 49.628571)),
    ((120, 4, SHORT, False), ("checked", 6, 4, 4, 87.314286)),
    ((120, 6, SHORT, False), ("checked", 6, 4, 4, 76.8)),
    ((240, 1, SHORT, False), ("checked", 4, 2, 4, 195.085714)),
]


@pytest.mark.parametrize(("given", "expected"), TABLE, ids=[str(given) for given, _ in TABLE])
def test_the_plan_table(given, expected):
    plan = fast_plan(*given)
    tier, rewrites, scenarios, holdout, seconds = expected
    assert (plan.tier, plan.rewrites, plan.scenarios, plan.holdout) == (
        tier,
        rewrites,
        scenarios,
        holdout,
    )
    assert plan.est_seconds == pytest.approx(seconds, abs=1e-5)
    assert (plan.time_s, plan.workers) == given[:2]
    assert plan.est_seconds == pytest.approx(sum(stage.seconds for stage in plan.stages))
    assert plan.est_calls == sum(stage.calls for stage in plan.stages)


def stage_names(plan: FastPlan) -> list[str]:
    return [stage.name[:2] for stage in plan.stages]


def test_the_quick_tier_runs_the_intake_beside_one_rewrite_then_the_contract_check():
    plan = fast_plan(15, 4, SHORT, False)
    assert [(stage.name, stage.calls) for stage in plan.stages] == [
        ("A: intake and rewrite", 2),
        ("contract check", 1),
    ]
    assert plan.est_calls == 3
    assert plan.stages[1].seconds == pytest.approx(2.4 + 75 / 70)


def test_the_fast_stages_and_their_calls():
    plan = fast_plan(30, 6, SHORT, False)  # K=3, M=3: the default time and workers
    assert [(stage.name, stage.calls) for stage in plan.stages] == [
        ("A: intake, synthesis and rewrites", 5),
        ("B: task runs", 12),
        ("C: judge and contract checks", 5),
        ("D: free gates and pick", 0),
    ]
    assert plan.stages[2].seconds == pytest.approx(2.4 + 25 * 3 * 3 / 70)  # one wave
    assert plan.stages[3].seconds == 0.0


def test_with_the_users_examples_no_synthesis_is_planned():
    plan = fast_plan(30, 6, SHORT, True)
    assert plan.stages[0] == Stage("A: intake and rewrites", 4, pytest.approx(2.4 + 380 / 70))


def test_the_contract_check_of_many_rewrites_can_be_the_slowest_judge_call():
    """Stage C's slowest call: 3 checks on each of M outputs, or 3 questions on each of K."""
    k6m3 = fast_plan(60, 8, SHORT, False).stages[2]  # K=6, M=4: the contract check of 6
    assert k6m3 == Stage("C: judge and contract checks", 8, pytest.approx(2.4 + 25 * 3 * 6 / 70))


def test_the_checked_tier_synthesises_the_holdout_too_and_ends_with_stage_e():
    plan = fast_plan(60, 4, SHORT, False)  # K=2, M=4, H=4: synthesises 8
    assert plan.stages[0] == Stage(
        "A: intake, synthesis and rewrites", 4, pytest.approx(2.4 + 380 / 70)
    )
    assert plan.stages[-1].name == "E: held-out check on the target model"
    assert plan.stages[-1].calls == 2 * CHECKED_HOLDOUT + 2
    assert plan.est_calls == 4 + 12 + 4 + 0 + 10


def test_a_synthesis_of_many_scenarios_can_be_the_slowest_call_of_stage_a():
    """8 scenarios cost 7.5 s, under the intake's 7.8 s; the 10 of a longer holdout would not."""
    assert stage_a(1, 8, 4, SHORT).seconds == pytest.approx(2.4 + 380 / 70)
    assert stage_a(1, 10, 4, SHORT).seconds == pytest.approx(2.4 + 450 / 70)
    assert stage_a(1, 0, 4, SHORT) == Stage(
        "A: intake and rewrite", 2, pytest.approx(2.4 + 380 / 70)
    )


def test_a_long_prompt_makes_the_rewrites_the_slowest_calls_of_stage_a():
    assert stage_a(3, 4, 8, SHORT).seconds == pytest.approx(2.4 + 380 / 70)  # the intake
    assert stage_a(3, 4, 8, LONG).seconds == pytest.approx(2.4 + 600 / 70)  # a rewrite
    assert stage_a(3, 4, 8, 1000).seconds == pytest.approx(2.4 + 600 / 70)  # 1200, capped at 600
    assert stage_a(3, 4, 4, SHORT).seconds == pytest.approx(2 * (2.4 + 380 / 70))  # 2 waves


@pytest.mark.parametrize("workers", [1, 2, 4, 6, 8, 16])
@pytest.mark.parametrize("tokens", [0, SHORT, LONG, 5000])
@pytest.mark.parametrize("examples", [False, True])
def test_more_time_never_gives_a_smaller_plan(workers, tokens, examples):
    for start, end in ((25, 60), (60, 300)):
        shapes = [
            (plan.rewrites, plan.scenarios)
            for plan in (fast_plan(t, workers, tokens, examples) for t in range(start, end))
        ]
        assert shapes == sorted(shapes)


@pytest.mark.parametrize("time_s", [25, 30, 45, 59, 60, 120, 299])
@pytest.mark.parametrize("workers", [1, 4, 8])
def test_a_plan_fits_85_percent_of_the_time_unless_it_is_the_smallest(time_s, workers):
    plan = fast_plan(time_s, workers, SHORT, False)
    smallest = (plan.rewrites, plan.scenarios) == (1, 2)
    assert plan.est_seconds <= 0.85 * time_s or smallest


def test_the_plan_is_the_same_for_the_same_inputs():
    assert fast_plan(30, 4, SHORT, False) == fast_plan(30, 4, SHORT, False)


def test_the_deep_tier_is_not_a_fast_plan():
    with pytest.raises(ValueError, match="deep"):
        fast_plan(300, 4, SHORT, False)


@pytest.mark.parametrize(("workers", "tokens"), [(0, SHORT), (-1, SHORT), (4, -1)])
def test_a_plan_needs_a_worker_and_a_token_count(workers, tokens):
    with pytest.raises(ValueError, match="workers"):
        fast_plan(30, workers, tokens, False)


# --- what is left at run time ---------------------------------------------------------------------


def test_the_tail_of_a_shape_is_stages_b_to_d_then_e_for_a_holdout():
    assert tail(0, 0, 0, 4) == ()
    assert [s.name[:2] for s in tail(1, 2, 0, 4)] == ["B:", "C:", "D:"]
    assert [s.name[:2] for s in tail(0, 0, 4, 4)] == ["E:"]
    assert [s.name[:2] for s in tail(3, 2, 4, 4)] == ["B:", "C:", "D:", "E:"]
    assert tail(3, 2, 4, 4)[0] == Stage("B: task runs", 8, pytest.approx(2 * (2.4 + 200 / 70)))
    assert tail(3, 2, 4, 4)[1].calls == 3 + 2  # a judge call per prompt, one contract check


TWO = (Stage("x", 2, 3.0), Stage("y", 1, 2.0))  # 3 calls, 5 seconds


@pytest.mark.parametrize(
    ("seconds_left", "calls_left", "cause"),
    [(5.0, 3, None), (4.9, 3, "clock"), (5.0, 2, "budget"), (4.9, 2, "clock"), (-1.0, 9, "clock")],
)
def test_stages_misfit_by_the_clock_first_then_by_the_calls(seconds_left, calls_left, cause):
    assert misfit(TWO, seconds_left, calls_left) == cause


def cost(rewrites: int, scenarios: int) -> tuple[Stage, ...]:
    """10 s and 1 call per rewrite, 1 s and 10 calls per scenario."""
    return (Stage("r", rewrites, 10.0 * rewrites), Stage("s", 10 * scenarios, 1.0 * scenarios))


@pytest.mark.parametrize(
    ("seconds_left", "calls_left", "expected"),
    [
        (100.0, 100, ((3, 4), None)),  # all fits
        (33.0, 100, ((3, 3), "clock")),  # scenarios shrink first
        (31.0, 100, ((3, 1), "clock")),
        (30.9, 100, ((2, 4), "clock")),  # then rewrites
        (11.0, 100, ((1, 1), "clock")),
        (10.9, 100, (None, "clock")),
        (100.0, 23, ((3, 2), "budget")),
        (100.0, 10, (None, "budget")),
        (33.0, 30, ((3, 2), "clock")),  # the first cause: (3, 4) by the clock, (3, 3) by calls
    ],
)
def test_shrink_takes_the_largest_shape_that_fits_scenarios_first(
    seconds_left, calls_left, expected
):
    assert shrink(3, 4, cost, seconds_left, calls_left) == expected


def test_shrink_of_no_rewrite_is_nothing_and_no_cause():
    assert shrink(0, 4, cost, 100.0, 100) == (None, None)
