"""The fast plan when every example carries a reference (SPEC R11, R25; WP21): stage C is one
absolute judge call per prompt run (the original's two and each rewrite's, K + 2) over its
scenarios plus the contract check, each judge call longer than a pairwise call since it quotes per
check; stage C2 is K2 + 1 calls; stage E runs the original twice and the winner (3 H task runs,
3 judge calls). A plan with references uses the examples (WP23): the largest pick it can price, up
to REFERENCE_MAX_SCENARIOS (8), and in the checked tier at least REFERENCE_MIN_HOLDOUT (3) held
out, as long as the examples last; a judge call holds at most JUDGE_BATCH_MAX (6) examples, so a
run's judging of 7 or 8 is two calls, one after the other. Numbers by hand from the latency model
of ADR-011 as in test_fastplan.py: I = 8.828571, Rs = 4.257143, T = 5.542857, a contract check of
n rewrites J(n) = 3.4 + 75 n / 70, a reference judge call over n scenarios of c checks F(n) = 3.4
+ (20 + 25 c) n / 70 (c = 1: F2 4.685714, F3 5.328571, F4 5.971429, F5 6.614286, F6 7.257143)."""

import pytest

from autoimprover import fastplan
from autoimprover.fastplan import (
    PAIRWISE_TOKENS_PER_SCENARIO,
    Reference,
    fast_plan,
    holdout_stage,
    last_chance,
    pair_seconds,
    reference_seconds,
    scoring_stages,
    second_stages,
    tail,
)

SHORT = 20
ONE = Reference(examples=18, checks=1)


@pytest.mark.parametrize("checks", [1, 2, 5])
@pytest.mark.parametrize("scenarios", [1, 4, 6])
def test_a_reference_judge_call_is_longer_than_a_pairwise_call(scenarios, checks):
    assert reference_seconds(scenarios, checks) > pair_seconds(scenarios)
    tokens = (fastplan.REFERENCE_TOKENS_PER_SCENARIO + 25 * checks) * scenarios
    assert reference_seconds(scenarios, checks) == pytest.approx(3.4 + tokens / 70)
    assert fastplan.REFERENCE_TOKENS_PER_SCENARIO + 25 > PAIRWISE_TOKENS_PER_SCENARIO


def test_stage_c_with_references_is_a_judge_call_per_run_and_the_contract_check():
    b, c, d = scoring_stages(3, 4, 6, SHORT, ref=ONE)
    assert (b.calls, c.calls, d.calls) == ((3 + 2) * 4, 3 + 2 + 1, 0)
    assert c.name == "C: reference judge and contract checks"
    # one wave of 6 calls, the slowest the contract check of 3 (J3 6.614 > F4 5.971)
    assert c.seconds == pytest.approx(3.4 + 75 * 3 / 70)
    _b, c1, _d = scoring_stages(1, 6, 6, SHORT, ref=ONE)
    assert c1.seconds == pytest.approx(3.4 + 45 * 6 / 70)  # F6 7.257 > J1 4.471
    assert scoring_stages(3, 4, 6, SHORT) == scoring_stages(3, 4, 6, SHORT, ref=None)


def test_the_second_generation_with_references():
    r, b2, c2 = second_stages(2, 6, 6, SHORT, ref=ONE)
    assert (r.calls, b2.calls, c2.calls) == (2, 12, 3)
    assert c2.seconds == pytest.approx(3.4 + 45 * 6 / 70)


def test_stage_e_with_references_runs_the_original_twice_and_the_winner():
    e = holdout_stage(4, 6, SHORT, ref=ONE)
    assert e.calls == 3 * 4 + 3
    assert e.seconds == pytest.approx(2 * (3.4 + 150 / 70) + 3.4 + 45 * 4 / 70)
    assert tail(0, 0, 4, 6, SHORT, ref=ONE) == (e,)
    assert holdout_stage(4, 6, SHORT).calls == 2 * 4 + 2  # pairwise plans keep their stage E


def test_the_last_chance_with_references_is_one_reference_call():
    judging, held = last_chance(4, 6, SHORT, ref=ONE)
    assert judging.seconds == pytest.approx(reference_seconds(1, 1))
    assert held == holdout_stage(4, 6, SHORT, ref=ONE)


def test_two_minutes_and_18_examples_pick_on_8_and_hold_out_4():
    """WP23: the largest pick the plan can price comes first, up to 8, with a second generation.
    K=2, M=8, H=4 and one reflection: I' + 6 T + (F6 + F2) + (Rs + 2 T + (F6 + F2)) + (2 T + F4)
    = 10.542857 + 33.257143 + 11.942857 + 27.285714 + 17.057143 = 100.085714 <= 102, where I' =
    3.4 + (380 + 120) / 70 is the intake of a plan with references (WP22, ADR-013) and a run's
    judging of 8 examples is two calls, of 6 and 2, one after the other; K=2 with 5 held out
    needs 106.3, K=3 with 3 held out 105.0."""
    plan = fast_plan(120, 6, SHORT, True, ONE)
    assert (plan.tier, plan.rewrites, plan.scenarios, plan.holdout) == ("checked", 2, 8, 4)
    assert (plan.generations, plan.rewrites2, plan.reference) == (2, 1, ONE)
    assert plan.est_seconds == pytest.approx(100.085714, abs=1e-5)
    assert [s.name[:2] for s in plan.stages] == ["A:", "B:", "C:", "R:", "B2", "C2", "D:", "E:"]
    assert plan.est_calls == 3 + 32 + 9 + 1 + 8 + 3 + 15 == sum(s.calls for s in plan.stages)


def test_the_plan_never_asks_for_more_examples_than_there_are():
    """8 examples cannot give 8 picked and 3 held out: 5 picked and 3 held out, K=5 and two
    reflections: I' + 6 T + 2 J5 + (Rs + 2 T + F5) + (2 T + F3) = 99.685714 <= 102."""
    plan = fast_plan(120, 6, SHORT, True, Reference(8, 1))
    assert (plan.rewrites, plan.scenarios, plan.holdout, plan.rewrites2) == (5, 5, 3, 2)
    assert plan.est_seconds == pytest.approx(99.685714, abs=1e-5)
    assert plan.stages[2].name == "C: reference judge and contract checks"


@pytest.mark.parametrize("examples", range(5, 20))
def test_a_checked_plan_with_references_holds_out_at_least_3(examples):
    plan = fast_plan(120, 6, SHORT, True, Reference(examples, 1))
    assert plan.holdout >= fastplan.REFERENCE_MIN_HOLDOUT == 3
    assert plan.scenarios + plan.holdout <= examples
    assert plan.scenarios == min(fastplan.REFERENCE_MAX_SCENARIOS, examples - 3)
    assert fastplan.REFERENCE_MAX_SCENARIOS == 8


def test_eleven_examples_pick_on_8_and_hold_out_3():
    plan = fast_plan(120, 6, SHORT, True, Reference(11, 1))
    assert (plan.rewrites, plan.scenarios, plan.holdout, plan.rewrites2) == (2, 8, 3, 1)


def test_a_judge_call_holds_at_most_6_examples_so_8_take_two_calls_one_after_the_other():
    """A run's judging of 8 examples is one job of two calls, F6 + F2 = 11.942857 s; the stage
    is a wave of the runs' jobs and the contract check."""
    _b, c, _d = scoring_stages(1, 8, 6, SHORT, ref=ONE)
    assert (c.calls, c.seconds) == (3 * 2 + 1, pytest.approx(11.942857, abs=1e-5))
    _r, _b2, c2 = second_stages(2, 8, 6, SHORT, ref=ONE)
    assert (c2.calls, c2.seconds) == (2 * 2 + 1, pytest.approx(11.942857, abs=1e-5))
    _b, six, _d = scoring_stages(1, 6, 6, SHORT, ref=ONE)
    assert (six.calls, six.seconds) == (3 + 1, pytest.approx(reference_seconds(6, 1)))
    _b, many, _d = scoring_stages(4, 8, 2, SHORT, ref=ONE)  # 7 jobs on 2 workers: 4 waves
    assert (many.calls, many.seconds) == (6 * 2 + 1, pytest.approx(4 * 11.942857, abs=1e-5))


@pytest.mark.parametrize("examples", [6, 12])
def test_the_fast_tier_picks_on_up_to_6_examples_when_they_fit(examples):
    plan = fast_plan(59, 16, SHORT, True, Reference(examples, 1))
    assert plan.tier == "fast" and plan.scenarios == 6 and plan.holdout == 0


def test_a_plan_without_references_is_the_pairwise_plan():
    assert fast_plan(120, 6, SHORT, True) == fast_plan(120, 6, SHORT, True, None)
    assert fast_plan(120, 6, SHORT, True).reference is None
    assert fast_plan(120, 6, SHORT, True).scenarios == 4
