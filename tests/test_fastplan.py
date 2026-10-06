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

from autoimprover import fastplan
from autoimprover.fastplan import (
    CHECKED_HOLDOUT,
    FastPlan,
    Stage,
    fast_plan,
    misfit,
    scoring_stages,
    second_stages,
    shrink,
    stage_a,
    tail,
    tier_for,
)
from autoimprover.runner import count_tokens

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
# run asks for at most 120 words, SPEC R25), of LONG TL = 3.4 + 600/70 = 11.971429 (its output
# grows with the prompt, `task_tokens`); a pairwise judge call over n scenarios P(n) = 3.4 +
# 40 n/70: P2 4.542857, P3 5.114286, P4 5.685714 (ADR-012); a contract check of n rewrites, or an
# absolute judge call over n outputs (stage E), J(n) = 3.4 + 75 n/70: J1 4.471429, J2 5.542857
# (= T), J3 6.614286, J4 7.685714, J5 8.757143, J6 9.828571. Stage A: 1 + K (+ 1 synthesis) calls
# of at most I (a synthesis of 8 is 8.542857). Stage B: (K + 2) M task runs, the original twice.
# Stage C: 2 K + 3 calls (two orders per rewrite, two for the original's two runs, one contract
# check) of max(P(M), J(K)). From 45 s a second generation: R, K2 reflections of a rewrite's
# length; B2, K2 M task runs; C2, 2 K2 + 1 calls of max(P(M), J(K2)); at M = 2 and K2 = 2 on 6 or
# more workers G2 = Rs + T + J2 = 15.342857. Stage E on H held out: ceil(2H/w) T + ceil(2/w) J(H)
# (below the table). A stage costs its slowest call once per wave of w calls; a plan fits when it
# is at most 0.85 x time; from 45 s the largest plan with two generations that fits wins, else the
# largest with one.
TABLE = [
    ((15, 1, SHORT, False), ("quick", 1, 0, 0, 1, 0, 22.128571)),  # 2 I + J1
    ((15, 4, SHORT, False), ("quick", 1, 0, 0, 1, 0, 13.3)),  # I + J1
    ((15, 4, LONG, False), ("quick", 1, 0, 0, 1, 0, 16.442857)),  # Rl + J1
    ((24, 6, SHORT, False), ("quick", 1, 0, 0, 1, 0, 13.3)),  # I + J1
    ((25, 4, SHORT, False), ("fast", 1, 2, 0, 1, 0, 29.0)),  # none fits: I + 2 T + 2 P2
    ((25, 6, SHORT, False), ("fast", 1, 2, 0, 1, 0, 18.914286)),  # I + T + P2; M=3: 25.03
    ((25, 8, SHORT, False), ("fast", 2, 2, 0, 1, 0, 19.914286)),  # I + T + J2; M=3: 25.46
    ((30, 1, SHORT, False), ("fast", 1, 2, 0, 1, 0, 82.457143)),  # none fits: 3 I + 6 T + 5 P2
    ((30, 4, SHORT, False), ("fast", 1, 2, 0, 1, 0, 29.0)),  # none fits: I + 2 T + 2 P2
    ((30, 4, LONG, False), ("fast", 1, 2, 0, 1, 0, 45.0)),  # none fits: Rl + 2 TL + 2 P2
    ((30, 6, SHORT, False), ("fast", 1, 3, 0, 1, 0, 25.028571)),  # I + 2 T + P3; K=2: 31.0
    ((30, 6, LONG, False), ("fast", 1, 2, 0, 1, 0, 28.485714)),  # none fits: Rl + TL + P2
    ((30, 8, SHORT, False), ("fast", 2, 3, 0, 1, 0, 25.457143)),  # I + 2 T + J2; M=4: 25.6
    ((45, 4, SHORT, False), ("fast", 2, 3, 0, 1, 0, 36.542857)),  # I + 3 T + 2 J2
    ((45, 4, LONG, False), ("fast", 1, 2, 0, 1, 0, 45.0)),  # none fits: Rl + 2 TL + 2 P2
    ((45, 6, SHORT, False), ("fast", 1, 2, 0, 2, 2, 34.257143)),  # I + T + P2 + G2
    ((45, 8, SHORT, False), ("fast", 2, 2, 0, 2, 2, 35.257143)),  # I + T + J2 + G2
    ((59, 1, SHORT, False), ("fast", 1, 2, 0, 1, 0, 82.457143)),  # none fits, as at 30 s
    ((59, 4, SHORT, False), ("fast", 2, 2, 0, 2, 1, 45.342857)),  # I + 2 T + 2 J2 + G1
    ((59, 4, SHORT, True), ("fast", 2, 2, 0, 2, 1, 45.342857)),  # the same: A is one wave
    ((59, 6, SHORT, False), ("fast", 3, 2, 0, 2, 2, 48.485714)),  # I + 2 T + 2 J3 + G2
    ((60, 1, SHORT, False), ("checked", 1, 2, 2, 1, 0, 115.714286)),  # none fits: see below
    ((60, 4, SHORT, False), ("checked", 1, 4, 2, 1, 0, 47.914286)),  # I + 3 T + 2 P4 + E2
    ((60, 6, SHORT, False), ("checked", 2, 4, 3, 1, 0, 48.985714)),  # I + 3 T + 2 P4 + E3
    ((60, 8, SHORT, False), ("checked", 3, 4, 3, 1, 0, 50.842857)),  # I + 3 T + 2 J3 + E3
    ((90, 4, SHORT, False), ("checked", 2, 4, 3, 2, 1, 75.557143)),  # I + 4 T + 2 P4 + G + E3
    ((90, 6, SHORT, False), ("checked", 4, 4, 3, 2, 1, 74.014286)),  # I + 4 T + 2 J4 + G + E3
    ((120, 4, SHORT, False), ("checked", 4, 4, 2, 2, 1, 100.542857)),  # see below
    ((120, 6, SHORT, False), ("checked", 5, 4, 3, 2, 1, 99.285714)),  # see below
    ((240, 1, SHORT, False), ("checked", 1, 4, 2, 2, 1, 198.171429)),  # see below
]
# G1 at 59 s on 4 workers: one reflection on M = 2, Rs + T + P2 = 14.342857 (two would need 2 J2 in
# C2: 20.885714, 51.885714 in all, over 50.15). The checked tier picks on 4 scenarios and holds out
# 4, 3 or 2 when such a plan fits, else on 3 or 2 with 2 held out; stage E on H held out is EH =
# ceil(2H/w) T + ceil(2/w) J(H): E2 = 33.257143 (w 1), 11.085714 (w 4 or more); E3 = 17.7 (w 4),
# 12.157143 (w 6, 8). With 4 picked a second generation of one reflection is G = Rs + T + P4 =
# 15.485714. At 60 s on 1 worker nothing fits: K=1 on M=2 with 2 held out, 3 I + 6 T + 5 P2 + E2
# = 26.485714 + 33.257143 + 22.714286 + 33.257143. At 120 s on 4 (K=4): 2 I + 6 T + 3 J4 + G + E2
# = 17.657143 + 33.257143 + 23.057143 + 15.485714 + 11.085714 = 100.542857 (K=5: 104.6, over 102);
# on 6 (K=5): 2 I + 5 T + 3 J5 + G + E3 = 17.657143 + 27.714286 + 26.271429 + 15.485714 +
# 12.157143 = 99.285714 (two reflections: 104.8, over 102). At 240 s on 1 worker (K=1): 3 I +
# 12 T + 5 P4 + (Rs + 4 T + 3 P4) + E2 = 26.485714 + 66.514286 + 28.428571 + 43.485714 +
# 33.257143 = 198.171429 (two reflections: 237.6, over 204).


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
    plan = fast_plan(30, 6, SHORT, False)  # K=1, M=3: the default time and workers
    assert [(stage.name, stage.calls) for stage in plan.stages] == [
        ("A: intake, synthesis and rewrite", 3),
        ("B: task runs", (1 + 2) * 3),  # the original twice, for the noise
        ("C: pairwise judge and contract checks", 2 * 1 + 2 + 1),  # two orders per pair, one check
        ("D: free gates and pick", 0),
    ]
    assert plan.stages[1].seconds == pytest.approx(2 * (3.4 + 150 / 70))  # two waves of 150 tokens
    assert plan.stages[2].seconds == pytest.approx(3.4 + 40 * 3 / 70)  # one pairwise wave on 3
    assert plan.stages[3].seconds == 0.0
    assert fast_plan(59, 6, SHORT, False).stages[1].calls == (3 + 2) * 2  # K=3 on M=2


def test_with_the_users_examples_no_synthesis_is_planned():
    plan = fast_plan(30, 6, SHORT, True)  # K=1 on M=3: the intake and 1 rewrite
    assert plan.stages[0] == Stage("A: intake and rewrite", 2, pytest.approx(3.4 + 380 / 70))


def test_the_contract_check_of_many_rewrites_can_be_the_slowest_judge_call():
    """Stage C's slowest call: a pairwise call over the M scenarios, or 3 questions on each of K."""
    k6m2 = scoring_stages(6, 2, 6, SHORT)[1]  # K=6, M=2: the contract check of 6
    assert k6m2 == Stage(
        "C: pairwise judge and contract checks", 15, pytest.approx(3 * (3.4 + 25 * 3 * 6 / 70))
    )


@pytest.mark.parametrize("time_s", [90, 120, 300, 599])
@pytest.mark.parametrize("workers", [4, 6, 8])
def test_the_checked_tier_picks_on_4_scenarios_and_holds_out_at_least_2_when_the_time_allows(
    time_s, workers
):
    """The bench of 2026-10-06 picked among six rewrites on 2 scenarios and held out 4, so the
    pick was noise: a checked plan that picks on 4 and holds out 2 to 4 comes before any that
    picks on fewer, and every checked plan holds out at least 2."""
    plan = fast_plan(time_s, workers, SHORT, False)
    assert plan.scenarios == 4 and 2 <= plan.holdout <= CHECKED_HOLDOUT


def test_a_checked_plan_with_no_time_for_4_picks_holds_out_2():
    """60 s on 1 worker fits nothing: the smallest checked plan, K=1 on 2 scenarios, 2 held out."""
    plan = fast_plan(60, 1, SHORT, False)
    assert (plan.rewrites, plan.scenarios, plan.holdout) == (1, 2, 2)


def test_the_checked_tier_synthesises_the_holdout_too_and_ends_with_stage_e():
    plan = fast_plan(60, 4, SHORT, False)  # K=1, M=4, H=2: synthesises 6
    assert plan.stages[0] == Stage(
        "A: intake, synthesis and rewrite", 3, pytest.approx(3.4 + 380 / 70)
    )
    assert plan.stages[-1].name == "E: held-out check on the target model"
    assert plan.stages[-1].calls == 2 * 2 + 2  # the original once, held out
    assert plan.est_calls == 3 + 12 + 5 + 0 + 6


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
    """Smaller in the planner's order: a checked plan that picks on 4 scenarios before one that
    picks on fewer, then two generations, rewrites, scenarios, held out, second rewrites."""
    for start, end in ((25, 60), (60, 600)):
        shapes = [
            (
                plan.tier == "checked" and plan.scenarios == 4,
                plan.generations,
                plan.rewrites,
                plan.scenarios,
                plan.holdout,
                plan.rewrites2,
            )
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
        ("C: pairwise judge and contract checks", 2 * 1 + 2 + 1),
        ("R: reflection on the first generation", 2),
        ("B2: task runs of the second generation", 2 * 2),
        ("C2: pairwise judge and contract checks of the second generation", 2 * 2 + 1),
        ("D: free gates and pick", 0),
    ]
    reflection, task_runs, judging = (stage.seconds for stage in plan.stages[3:6])
    assert (reflection, task_runs, judging) == pytest.approx(
        (3.4 + 60 / 70, 3.4 + 150 / 70, 3.4 + 25 * 3 * 2 / 70)
    )
    assert plan.est_calls == 3 + 6 + 5 + 2 + 4 + 5


def test_a_long_prompts_reflection_takes_as_long_as_its_rewrite():
    plan = fast_plan(120, 6, LONG, False)
    [reflection] = [stage for stage in plan.stages if stage.name.startswith("R:")]
    assert reflection.seconds == pytest.approx(3.4 + 600 / 70)


# The long-1 prompt of the bench (2026-10-06): it asks for a full deliverable, and its scoring runs
# took 6.7 to 13.9 s each although each asked for at most 120 words; the plan was K=1 on M=3, so
# stage B took two waves and stage C never ran.
LONG1 = (
    "I run a small bakery with two shops in the same town. We sell bread, pastries and cakes, and "
    "about a third of our income now comes from cake orders that people place by phone or in the "
    "shop.\n\nThe order process is all paper: a form in a binder, copied into a spreadsheet at the "
    "end of the day. Orders get lost, we sometimes bake the same cake twice, and customers call to "
    "check on their order because they have no confirmation.\n\nI want a simple way to take cake "
    "orders online, with a confirmation for the customer and one list of orders for both shops. "
    "We have no developer and a small budget. Suggest two or three ways to do this, with rough "
    "costs and what each would need from us."
)
T148 = 3.4 + 444 / 70  # a scoring run of a 148-token prompt: 150 + 3 x (148 - 50) tokens


def test_the_task_output_estimate_grows_with_the_prompt():
    """150 tokens up to a 50-token prompt (the 120-word answer binds), 3 more per prompt token
    above it, at most 600 (from 200 prompt tokens)."""
    estimate = fastplan.task_tokens
    assert [estimate(p) for p in (0, 20, 50)] == [150, 150, 150]
    assert [estimate(p) for p in (51, 100, 148, 199)] == [153, 300, 444, 597]
    assert [estimate(p) for p in (200, 400, 5000)] == [600, 600, 600]


def test_the_long_1_prompt_plans_two_scenarios_in_one_wave_of_task_runs():
    """At 30 s on 6 workers: M=3 needs two waves of task runs, I + 2 T148 + P3 = 33.43 s, over
    25.5; M=2 needs one, I + T148 + P2 = 23.114 s."""
    assert count_tokens(LONG1) == 148
    plan = fast_plan(30, 6, count_tokens(LONG1), False)
    assert (plan.rewrites, plan.scenarios, plan.generations) == (1, 2, 1)
    assert plan.stages[1] == Stage("B: task runs", 6, pytest.approx(T148))
    assert plan.est_seconds == pytest.approx(8.828571 + T148 + 4.542857, abs=1e-5)


def test_every_stage_of_task_runs_takes_the_prompts_estimate():
    """Stage B at run time, stage B2 and stage E (two waves of 8 runs on 6 workers, then J4)."""
    assert tail(1, 2, 0, 6, 148)[0] == Stage("B: task runs", 6, pytest.approx(T148))
    assert second_stages(2, 2, 6, 148)[1].seconds == pytest.approx(T148)
    assert fastplan.holdout_stage(4, 6, 148).seconds == pytest.approx(2 * T148 + 3.4 + 300 / 70)
    assert tail(1, 2, 0, 6, SHORT)[0].seconds == pytest.approx(3.4 + 150 / 70)


def test_the_second_generations_contract_check_can_be_its_slowest_judge_call():
    """C2's slowest call: a pairwise call over the M scenarios, or 3 questions on each of K2."""
    name = "C2: pairwise judge and contract checks of the second generation"
    assert second_stages(3, 2, 8, SHORT)[2] == Stage(name, 7, pytest.approx(3.4 + 225 / 70))
    assert second_stages(1, 4, 8, SHORT)[2] == Stage(name, 3, pytest.approx(3.4 + 160 / 70))


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
    assert tail(0, 0, 0, 4, SHORT) == ()
    assert [s.name[:2] for s in tail(1, 2, 0, 4, SHORT)] == ["B:", "C:", "D:"]
    assert [s.name[:2] for s in tail(0, 0, 4, 4, SHORT)] == ["E:"]
    assert [s.name[:2] for s in tail(3, 2, 4, 4, SHORT)] == ["B:", "C:", "D:", "E:"]
    b, c, *_ = tail(3, 2, 4, 4, SHORT)
    assert b == Stage("B: task runs", 10, pytest.approx(3 * (3.4 + 150 / 70)))
    assert c.calls == 2 * 3 + 2 + 1  # two orders per pair, one contract check


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
