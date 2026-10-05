"""The time tiers and the fast plan (SPEC R25; ADR-011): `--time` picks the tier, and the plan takes
the most rewrites, then the most scenarios, whose estimate fits 85 % of the time, from the latency
model of ADR-011 (2.4 s per call plus output tokens at 70 per second; one slowest call per wave of
`workers` calls). The expected numbers below were worked out by hand from that model.
"""

import pytest

from autoimprover.fastplan import (
    CHECKED_HOLDOUT,
    FastPlan,
    Stage,
    fast_plan,
    misfit,
    shrink,
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
    ((25, 4, SHORT, False), ("fast", 1, 2, 0, 22.385714)),
    ((25, 8, SHORT, False), ("fast", 1, 2, 0, 22.385714)),
    ((30, 1, SHORT, False), ("fast", 1, 2, 0, 51.6)),
    ((30, 4, SHORT, False), ("fast", 1, 2, 0, 22.385714)),
    ((30, 4, LONG, False), ("fast", 1, 2, 0, 25.528571)),
    ((30, 4, SHORT, True), ("fast", 3, 2, 0, 23.957143)),
    ((30, 6, SHORT, False), ("fast", 2, 2, 0, 22.385714)),
    ((30, 6, LONG, False), ("fast", 1, 2, 0, 25.528571)),
    ((30, 8, SHORT, False), ("fast", 3, 2, 0, 22.385714)),
    ((45, 4, SHORT, False), ("fast", 3, 3, 0, 34.614286)),
    ((45, 4, LONG, False), ("fast", 3, 3, 0, 37.757143)),
    ((45, 6, SHORT, False), ("fast", 3, 4, 0, 36.328571)),
    ((45, 8, SHORT, False), ("fast", 3, 4, 0, 31.071429)),
    ((59, 1, SHORT, False), ("fast", 1, 2, 0, 51.6)),
    ((59, 4, SHORT, False), ("fast", 3, 4, 0, 41.585714)),
    ((60, 1, SHORT, False), ("checked", 1, 2, 4, 109.6)),
    ((60, 4, SHORT, False), ("checked", 3, 2, 4, 47.414286)),
    ((60, 6, SHORT, False), ("checked", 5, 2, 4, 47.414286)),
    ((60, 8, SHORT, False), ("checked", 6, 3, 4, 49.128571)),
    ((120, 4, SHORT, False), ("checked", 6, 4, 4, 92.714286)),
    ((120, 6, SHORT, False), ("checked", 6, 4, 4, 82.2)),
    ((240, 1, SHORT, False), ("checked", 4, 2, 4, 181.471429)),
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
    plan = fast_plan(45, 4, SHORT, False)  # K=3, M=3
    assert [(stage.name, stage.calls) for stage in plan.stages] == [
        ("A: intake and rewrites", 4),
        ("A: scenario synthesis", 1),
        ("B: task runs", 12),
        ("C: judge and contract checks", 4),
        ("D: free gates and pick", 0),
    ]
    assert plan.stages[1].seconds == pytest.approx(2.4 + 45 * 3 / 70)  # synthesises M scenarios
    assert plan.stages[4].seconds == 0.0


def test_with_the_users_examples_no_synthesis_is_planned():
    plan = fast_plan(30, 4, SHORT, True)
    assert "A: scenario synthesis" not in [stage.name for stage in plan.stages]
    assert stage_names(plan) == ["A:", "B:", "C:", "D:"]


def test_the_checked_tier_synthesises_the_holdout_too_and_ends_with_stage_e():
    plan = fast_plan(60, 4, SHORT, False)  # K=3, M=2, H=4
    synthesis = plan.stages[1]
    assert synthesis == Stage("A: scenario synthesis", 1, pytest.approx(2.4 + 45 * 6 / 70))
    assert plan.stages[-1].name == "E: held-out check on the target model"
    assert plan.stages[-1].calls == 2 * CHECKED_HOLDOUT + 2
    assert plan.est_calls == 4 + 1 + 8 + 4 + 0 + 10


def test_a_long_prompt_makes_the_rewrites_the_slowest_calls_of_stage_a():
    short, long = fast_plan(45, 4, SHORT, False), fast_plan(45, 4, LONG, False)
    assert short.stages[0].seconds == pytest.approx(2.4 + 380 / 70)  # the intake
    assert long.stages[0].seconds == pytest.approx(2.4 + 600 / 70)  # a rewrite, capped at 600
    longer = fast_plan(45, 4, 1000, False)
    assert longer.stages[0].seconds == pytest.approx(2.4 + 600 / 70)  # 1200 tokens, capped


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
