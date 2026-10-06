"""The time tiers and the fast plan (SPEC R25; ADR-011): `--time` picks the tier, and the plan takes
the most rewrites, then the most scenarios, whose estimate fits 85 % of the time, from the latency
model of ADR-011 (3.4 s per call, the start-up measured with the trims of ADR-009's amendment of
2026-10-06, plus output tokens at 70 per second; one slowest call per wave of `workers` calls).
Stage A is one wave: the intake, the synthesis and the rewrites; stage B runs the original twice
(for the noise) and every rewrite on the M scenarios; stage C is a judge call per run plus one
contract check of every rewrite. The expected numbers below were worked out by hand from that
model.
"""

import pytest

from autoimprover.fastplan import (
    CHECKED_HOLDOUT,
    FastPlan,
    Stage,
    fast_plan,
    misfit,
    second_stages,
    shrink,
    stage_a,
    tail,
    tier_for,
)

SHORT, LONG = 20, 500  # prompt tokens: a one-line prompt and a long system prompt


@pytest.mark.parametrize(
    ("time_s", "tier"),
    [(15, "quick"), (24, "quick"), (25, "fast"), (30, "fast"), (59, "fast")]
    + [(60, "checked"), (300, "checked"), (599, "checked"), (600, "deep"), (1200, "deep")],
)
def test_the_time_picks_the_tier(time_s, tier):
    assert tier_for(time_s) == tier


@pytest.mark.parametrize("time_s", [14, 0, -30])
def test_a_time_below_15_seconds_is_refused_naming_the_flag_and_the_minimum(time_s):
    with pytest.raises(ValueError, match=r"--time.*15 s"):
        tier_for(time_s)


# (time, workers, prompt tokens, user examples) -> (tier, rewrites, scenarios, holdout, generations,
# second-generation rewrites, seconds)
# Calls at 3.4 s + tokens / 70: intake I = 3.4 + 380/70 = 8.828571; rewrite of SHORT (60 tokens, the
# floor) Rs = 4.257143, of LONG (600) Rl = 11.971429; task T = 3.4 + 150/70 = 5.542857 (a scoring
# run asks for at most 120 words, SPEC R25); a judge call over n outputs or a contract check of n
# rewrites J(n) = 3.4 + 75 n/70: J1 4.471429, J2 5.542857 (= T), J3 6.614286, J4 7.685714, J5
# 8.757143, J6 9.828571. Stage A: 1 + K (+ 1 synthesis) calls of at most I (a synthesis of 8 is
# 8.542857). Stage B: (K + 2) M task runs, the original twice. Stage C: K + 3 calls (a judge call
# per run, one contract check) of J(max(M, K)). From 45 s a second generation: R, K2 reflections
# of a rewrite's length; B2, K2 M task runs; C2, K2 + 1 calls of J(max(M, K2)); at M = 2 and K2 = 2
# on 4 or more workers G2 = Rs + T + J2 = 15.342857, at M = 3 on 6 G3 = Rs + T + J3 = 16.414286.
# Stage E (H = 4): ceil(8/w) T + ceil(2/w) J4 = 59.714286 (w 1), 18.771429 (w 4, 6), 13.228571
# (w 8). A stage costs its slowest call once per wave of w calls; a plan fits when it is at most
# 0.85 x time; from 45 s the largest plan with two generations that fits wins, else the largest
# with one.
TABLE = [
    ((15, 1, SHORT, False), ("quick", 1, 0, 0, 1, 0, 22.128571)),  # 2 I + J1
    ((15, 4, SHORT, False), ("quick", 1, 0, 0, 1, 0, 13.3)),  # I + J1
    ((15, 4, LONG, False), ("quick", 1, 0, 0, 1, 0, 16.442857)),  # Rl + J1
    ((24, 6, SHORT, False), ("quick", 1, 0, 0, 1, 0, 13.3)),  # I + J1
    ((25, 4, SHORT, False), ("fast", 1, 2, 0, 1, 0, 25.457143)),  # none fits: I + 2 T + J2
    ((25, 6, SHORT, False), ("fast", 1, 2, 0, 1, 0, 19.914286)),  # I + T + J2; K=2: 25.457143
    ((25, 8, SHORT, False), ("fast", 2, 2, 0, 1, 0, 19.914286)),  # I + T + J2; K=3: 26.528571
    ((30, 1, SHORT, False), ("fast", 1, 2, 0, 1, 0, 81.914286)),  # none fits: 3 I + 6 T + 4 J2
    ((30, 4, SHORT, False), ("fast", 1, 2, 0, 1, 0, 25.457143)),  # I + 2 T + J2 <= 25.5
    ((30, 4, LONG, False), ("fast", 1, 2, 0, 1, 0, 28.6)),  # none fits: Rl + 2 T + J2
    ((30, 6, SHORT, False), ("fast", 2, 2, 0, 1, 0, 25.457143)),  # I + 2 T + J2; K=3: + J3 - J2
    ((30, 6, LONG, False), ("fast", 1, 2, 0, 1, 0, 23.057143)),  # Rl + T + J2; K=2: 28.6
    ((30, 8, SHORT, False), ("fast", 2, 2, 0, 1, 0, 19.914286)),  # I + T + J2; K=3: 26.528571
    ((45, 4, SHORT, False), ("fast", 2, 2, 0, 1, 0, 31.0)),  # I + 2 T + 2 J2; 2 gens: 40.8
    ((45, 4, LONG, False), ("fast", 2, 2, 0, 1, 0, 34.142857)),  # Rl + 2 T + 2 J2
    ((45, 6, SHORT, False), ("fast", 1, 2, 0, 2, 2, 35.257143)),  # I + T + J2 + G2
    ((45, 8, SHORT, False), ("fast", 2, 2, 0, 2, 2, 35.257143)),  # I + T + J2 + G2
    ((59, 1, SHORT, False), ("fast", 1, 2, 0, 1, 0, 81.914286)),  # none fits, as at 30 s
    ((59, 4, SHORT, False), ("fast", 2, 2, 0, 2, 2, 46.342857)),  # I + 2 T + 2 J2 + G2
    ((59, 4, SHORT, True), ("fast", 2, 2, 0, 2, 2, 46.342857)),  # the same: A is one wave
    ((59, 6, SHORT, False), ("fast", 3, 3, 0, 2, 2, 48.485714)),  # I + 3 T + J3 + G3
    ((60, 1, SHORT, False), ("checked", 1, 2, 4, 1, 0, 141.628571)),  # 3 I + 6 T + 4 J2 + E
    ((60, 4, SHORT, False), ("checked", 2, 2, 4, 1, 0, 49.771429)),  # I + 2 T + 2 J2 + E
    ((60, 6, SHORT, False), ("checked", 3, 3, 4, 1, 0, 50.842857)),  # I + 3 T + J3 + E
    ((60, 8, SHORT, False), ("checked", 2, 2, 4, 2, 2, 48.485714)),  # I + T + J2 + G2 + E
    ((90, 4, SHORT, False), ("checked", 2, 3, 4, 2, 1, 73.871429)),  # I + 3 T + 2 J3 + G1 + E
    ((90, 6, SHORT, False), ("checked", 4, 3, 4, 2, 2, 76.014286)),  # I + 3 T + 2 J4 + G3 + E
    ((120, 4, SHORT, False), ("checked", 5, 2, 4, 2, 2, 91.457143)),  # 2 I + 4 T + 2 J5 + G2 + E
    ((120, 6, SHORT, False), ("checked", 6, 3, 4, 2, 2, 94.671429)),  # 2 I + 4 T + 2 J6 + G3 + E
    ((240, 1, SHORT, False), ("checked", 2, 2, 4, 2, 1, 193.514286)),  # 4 I + 8 T + 5 J2 + G + E
]
# G1 at 90 s on 4 workers: one reflection on M = 3, Rs + T + J3 = 16.414286 (two would be Rs + 2 T
# + J3 = 21.957143, 79.414286 in all, over 76.5). At 240 s on 1 worker G = Rs + 2 T + 2 J2 =
# 26.428571 for one reflection; two (2 Rs + 4 T + 3 J2) would make 214.4, over 204.


@pytest.mark.parametrize(("given", "expected"), TABLE, ids=[str(given) for given, _ in TABLE])
def test_the_plan_table(given, expected):
    plan = fast_plan(*given)
    *shape, seconds = expected
    assert [
        plan.tier,
        plan.rewrites,
        plan.scenarios,
        plan.holdout,
        plan.generations,
        plan.rewrites2,
    ] == shape
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
    assert plan.stages[1].seconds == pytest.approx(3.4 + 75 / 70)


def test_the_fast_stages_and_their_calls():
    plan = fast_plan(30, 6, SHORT, False)  # K=2, M=2: the default time and workers
    assert [(stage.name, stage.calls) for stage in plan.stages] == [
        ("A: intake, synthesis and rewrites", 4),
        ("B: task runs", (2 + 2) * 2),  # the original twice, for the noise
        ("C: judge and contract checks", 2 + 3),  # a judge call per run, one contract check
        ("D: free gates and pick", 0),
    ]
    assert plan.stages[1].seconds == pytest.approx(2 * (3.4 + 150 / 70))  # two waves of 150 tokens
    assert plan.stages[2].seconds == pytest.approx(3.4 + 25 * 3 * 2 / 70)  # one wave
    assert plan.stages[3].seconds == 0.0
    assert fast_plan(59, 6, SHORT, False).stages[1].calls == (3 + 2) * 3  # K=3 on M=3


def test_with_the_users_examples_no_synthesis_is_planned():
    plan = fast_plan(30, 6, SHORT, True)  # K=2: the intake and 2 rewrites
    assert plan.stages[0] == Stage("A: intake and rewrites", 3, pytest.approx(3.4 + 380 / 70))


def test_the_contract_check_of_many_rewrites_can_be_the_slowest_judge_call():
    """Stage C's slowest call: 3 checks on each of M outputs, or 3 questions on each of K."""
    k6m3 = fast_plan(120, 6, SHORT, False).stages[2]  # K=6, M=3: the contract check of 6
    assert k6m3 == Stage(
        "C: judge and contract checks", 9, pytest.approx(2 * (3.4 + 25 * 3 * 6 / 70))
    )


def test_the_checked_tier_synthesises_the_holdout_too_and_ends_with_stage_e():
    plan = fast_plan(60, 4, SHORT, False)  # K=2, M=2, H=4: synthesises 6
    assert plan.stages[0] == Stage(
        "A: intake, synthesis and rewrites", 4, pytest.approx(3.4 + 380 / 70)
    )
    assert plan.stages[-1].name == "E: held-out check on the target model"
    assert plan.stages[-1].calls == 2 * CHECKED_HOLDOUT + 2  # the original once, held out
    assert plan.est_calls == 4 + 8 + 5 + 0 + 10


def test_a_synthesis_of_many_scenarios_can_be_the_slowest_call_of_stage_a():
    """8 scenarios cost 8.5 s, under the intake's 8.8 s; the 10 of a longer holdout would not."""
    assert stage_a(1, 8, 4, SHORT).seconds == pytest.approx(3.4 + 380 / 70)
    assert stage_a(1, 10, 4, SHORT).seconds == pytest.approx(3.4 + 450 / 70)
    assert stage_a(1, 0, 4, SHORT) == Stage(
        "A: intake and rewrite", 2, pytest.approx(3.4 + 380 / 70)
    )


def test_a_long_prompt_makes_the_rewrites_the_slowest_calls_of_stage_a():
    assert stage_a(3, 4, 8, SHORT).seconds == pytest.approx(3.4 + 380 / 70)  # the intake
    assert stage_a(3, 4, 8, LONG).seconds == pytest.approx(3.4 + 600 / 70)  # a rewrite
    assert stage_a(3, 4, 8, 1000).seconds == pytest.approx(3.4 + 600 / 70)  # 1200, capped at 600
    assert stage_a(3, 4, 4, SHORT).seconds == pytest.approx(2 * (3.4 + 380 / 70))  # 2 waves


@pytest.mark.parametrize("workers", [1, 2, 4, 6, 8, 16])
@pytest.mark.parametrize("tokens", [0, SHORT, LONG, 5000])
@pytest.mark.parametrize("examples", [False, True])
def test_more_time_never_gives_a_smaller_plan(workers, tokens, examples):
    for start, end in ((25, 60), (60, 600)):
        shapes = [
            (plan.generations, plan.rewrites, plan.scenarios, plan.rewrites2)
            for plan in (fast_plan(t, workers, tokens, examples) for t in range(start, end))
        ]
        assert shapes == sorted(shapes)


@pytest.mark.parametrize("time_s", [25, 30, 45, 59, 60, 120, 299, 599])
@pytest.mark.parametrize("workers", [1, 4, 8])
def test_a_plan_fits_85_percent_of_the_time_unless_it_is_the_smallest(time_s, workers):
    plan = fast_plan(time_s, workers, SHORT, False)
    smallest = (plan.rewrites, plan.scenarios) == (1, 2)
    assert plan.est_seconds <= 0.85 * time_s or smallest


def test_from_45_s_a_second_generation_follows_the_first():
    plan = fast_plan(45, 6, SHORT, False)  # K=1, M=2, then K2=2
    assert [(stage.name, stage.calls) for stage in plan.stages] == [
        ("A: intake, synthesis and rewrite", 3),
        ("B: task runs", 6),
        ("C: judge and contract checks", 4),
        ("R: reflection on the first generation", 2),
        ("B2: task runs of the second generation", 2 * 2),
        ("C2: judge and contract checks of the second generation", 2 + 1),
        ("D: free gates and pick", 0),
    ]
    reflection, task_runs, judging = (stage.seconds for stage in plan.stages[3:6])
    assert (reflection, task_runs, judging) == pytest.approx(
        (3.4 + 60 / 70, 3.4 + 150 / 70, 3.4 + 25 * 3 * 2 / 70)
    )
    assert plan.est_calls == 3 + 6 + 4 + 2 + 4 + 3


def test_a_long_prompts_reflection_takes_as_long_as_its_rewrite():
    plan = fast_plan(120, 6, LONG, False)
    [reflection] = [stage for stage in plan.stages if stage.name.startswith("R:")]
    assert reflection.seconds == pytest.approx(3.4 + 600 / 70)


def test_the_second_generations_contract_check_can_be_its_slowest_judge_call():
    """C2's slowest call: 3 checks on each of M outputs, or 3 questions on each of K2."""
    assert second_stages(3, 2, 8, SHORT)[2] == Stage(
        "C2: judge and contract checks of the second generation", 4, pytest.approx(3.4 + 225 / 70)
    )
    assert second_stages(2, 3, 8, SHORT)[2].seconds == pytest.approx(3.4 + 225 / 70)


@pytest.mark.parametrize("workers", [1, 4, 6, 8, 16])
def test_below_45_s_there_is_never_a_second_generation(workers):
    assert {fast_plan(t, workers, SHORT, False).generations for t in range(15, 45)} == {1}
    assert fast_plan(45, workers, SHORT, False).rewrites2 in (0, 1, 2)


def test_the_plan_is_the_same_for_the_same_inputs():
    assert fast_plan(30, 4, SHORT, False) == fast_plan(30, 4, SHORT, False)


def test_the_deep_tier_is_not_a_fast_plan():
    assert fast_plan(599, 4, SHORT, False).tier == "checked"
    with pytest.raises(ValueError, match="deep"):
        fast_plan(600, 4, SHORT, False)


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
    assert tail(3, 2, 4, 4)[0] == Stage("B: task runs", 10, pytest.approx(3 * (3.4 + 150 / 70)))
    assert tail(3, 2, 4, 4)[1].calls == 3 + 3  # a judge call per run, one contract check


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
