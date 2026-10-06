"""The report of the time tiers (SPEC R2, R25): every outcome names its tier and the seconds the run
took on its clock; a quick or fast result says in its verified line, its meaning, its notices and
its JSON that it is not verified on held-out scenarios, and labels its pairwise preference as taken
on the scenarios it was picked on, never as a holdout score (ADR-012) (no row reads as success
without being one, ARCHITECTURE section 8); a checked result says what verified it. Nothing of the
GEPA search is said of a run that did not search."""

import dataclasses
import io
import json

import pytest

from autoimprover import cli_plan, report, runner
from autoimprover.types import DEFAULT_MODELS, Efforts, Outcome, Plan

ORIGINAL = "Answer the user's request."
BETTER = "Answer the user's request in three short steps."
PLAN = Plan(models=DEFAULT_MODELS, wall_clock_s=30, tier="fast", budget=66)
RUN = "/state/autoimprover/runs/20261004-120000-abcdef12"
FAST_LABEL = (
    "fast check: scored on the same few scenarios it was picked on, not verified on held-out "
    "scenarios, no noise measured"
)
QUICK_LABEL = (
    "quick check: passed the contract check and the free gates, not scored on any scenario, not "
    "verified on held-out scenarios, no noise measured"
)
FAST = Outcome(
    status="improved",
    prompt=BETTER,
    reason=f"tier fast: {FAST_LABEL}",
    reason_code="improved",
    verified=False,
    changes=("tightened: removed redundancy and filler, kept the meaning",),
    score_before=0.25,
    score_after=1.0,
    margin=0.65,
    length_ratio=1.2,
    calls_used=22,
    run_dir=RUN,
    mode="fast",
)
QUICK = dataclasses.replace(
    FAST,
    reason=f"tier quick: {QUICK_LABEL}",
    score_before=None,
    score_after=None,
    margin=None,
    mode="quick",
)
CHECKED = dataclasses.replace(
    FAST,
    reason="tier checked: beats the original on 4 held-out scenarios, on the target model, by more "
    "than 0.05; no noise measured",
    verified=True,
    score_before=0.5,
    score_after=1.0,
    search_score_before=0.25,
    search_score_after=1.0,
    margin=0.45,
    mode="checked",
)
KEPT = Outcome(
    status="unchanged",
    prompt=ORIGINAL,
    reason="tier fast: the clock ran out before a rewrite passed every gate",
    reason_code="unconfirmed_out_of_budget",
    stop="clock",
    score_before=0.25,
    calls_used=9,
    run_dir=RUN,
    mode="fast",
)
DEEP_ONLY = ("search ended", "GEPA", "never saw", "measured noise", "--trust-search", "valset")


def lines_of(text: str, start: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(start)]


@pytest.mark.parametrize(("outcome", "label"), [(FAST, FAST_LABEL), (QUICK, QUICK_LABEL)])
def test_a_quick_or_fast_result_says_it_is_not_verified_in_its_verified_line(outcome, label):
    text = report.render(outcome, ORIGINAL, None, PLAN, elapsed_s=27.4)
    [verified] = lines_of(text, "verified: ")
    assert verified.startswith("verified: no. NOT VERIFIED") and label in verified
    assert "verified: yes" not in text and "holdout" not in text
    assert report.REASON_LINES["improved"] not in text
    assert not [word for word in DEEP_ONLY if word in text]
    [meaning] = lines_of(text, "meaning: ")
    assert "not verified on held-out scenarios" in meaning and "no noise" in meaning


def test_the_report_names_the_tier_and_the_seconds_on_the_runs_clock():
    text = report.render(FAST, ORIGINAL, None, PLAN, elapsed_s=27.4)
    assert lines_of(text, "mode: ") == ["mode: fast (27 s)"]
    deep = dataclasses.replace(FAST, mode="deep", verified=True)
    assert lines_of(report.render(deep, ORIGINAL, None, PLAN, 1800.0), "mode: ") == [
        "mode: deep (1800 s)"
    ]
    assert not lines_of(
        report.render(dataclasses.replace(FAST, mode=None), ORIGINAL, None, PLAN), "mode: "
    )


def test_a_fast_preference_is_labelled_as_taken_on_the_scenarios_the_rewrite_was_picked_on():
    text = report.render(FAST, ORIGINAL, None, PLAN, elapsed_s=27.0)
    [score] = lines_of(text, "preference on the scenarios it was picked on")
    assert score.endswith("the original won 0.25, the rewrite 1.00")
    assert DEFAULT_MODELS.judge in score and "not held out" in score
    [margin] = lines_of(text, "margin: ")
    assert "0.65" in margin and "0.1" in margin


def test_a_checked_result_is_verified_on_the_holdout_and_its_pick_scores_are_labelled():
    text = report.render(CHECKED, ORIGINAL, None, PLAN, elapsed_s=200.0)
    assert lines_of(text, "verified: ") == [
        f"verified: yes, on the holdout, on the target model {DEFAULT_MODELS.target}"
    ]
    assert lines_of(text, "holdout score (target model")[0].endswith("0.50 before, 1.00 after")
    [picked] = lines_of(text, "preference on the scenarios it was picked on")
    assert picked.endswith("the original won 0.25, the rewrite 1.00")
    [margin] = lines_of(text, "margin: ")
    assert "0.45" in margin and f"{runner.MIN_THRESHOLD}" in margin and "no noise" in margin
    assert "held-out scenarios" in margin
    assert "search ended" not in text and "GEPA" not in text


def test_a_fast_run_cut_by_the_clock_says_so_in_its_own_words():
    text = report.render(KEPT, ORIGINAL, None, PLAN, elapsed_s=30.0)
    [meaning] = lines_of(text, "meaning: ")
    assert "--time" in meaning and "--budget" not in meaning and "holdout" not in meaning
    assert lines_of(text, "stages: ") and lines_of(text, "notice: ")
    assert "clock" in lines_of(text, "notice: ")[0]
    assert not [word for word in ("search", "GEPA", "noise") if word in text]


@pytest.mark.parametrize("code", ["no_reliable_improvement", "no_holdout"])
def test_every_fast_unchanged_reason_has_its_own_line_without_deep_words(code):
    kept = dataclasses.replace(KEPT, reason_code=code, reason=f"tier fast: {code}", stop=None)
    text = report.render(kept, ORIGINAL, None, PLAN, elapsed_s=12.0)
    [meaning] = lines_of(text, "meaning: ")
    assert meaning.removeprefix("meaning: ") == report.FAST_REASON_LINES[code]
    assert "noise" not in meaning and "--trust-search" not in meaning
    assert report.REASON_LINES[code] not in text


def test_with_json_a_fast_result_carries_its_mode_and_the_notice_says_not_verified():
    obj = report.outcome_object(FAST, ORIGINAL, None)
    assert (obj["mode"], obj["verified"], obj["status"]) == ("fast", False, "improved")
    out, err = io.StringIO(), io.StringIO()
    report.Emitter(out, err, json_mode=True).outcome(FAST, ORIGINAL, None, PLAN, 27.0)
    assert json.loads(out.getvalue())["mode"] == "fast"
    [notice] = [line for line in err.getvalue().splitlines() if "NOT VERIFIED" in line]
    assert notice.startswith("notice: ") and FAST_LABEL in notice and "--trust-search" not in notice


def test_a_deep_result_only_gains_the_mode_line():
    deep = dataclasses.replace(FAST, mode="deep", verified=True, reason="r", stop="budget")
    with_mode = report.render(deep, ORIGINAL, None, PLAN, elapsed_s=100.0).splitlines()
    without = report.render(dataclasses.replace(deep, mode=None), ORIGINAL, None, PLAN).splitlines()
    assert [line for line in with_mode if line not in without] == ["mode: deep (100 s)"]
    assert len(with_mode) == len(without) + 1


def test_the_dry_objects_of_every_tier_share_one_key_order():
    costs = runner.fixed_costs(Plan(models=DEFAULT_MODELS), 12, synthesising=True)
    deep = cli_plan.plan_object(cli_plan.PlanView(Plan(models=DEFAULT_MODELS), 12, True, costs))
    assert tuple(deep) == cli_plan.DRY_KEYS
    assert (deep["tier"], deep["workers"], deep["efforts"]) == (
        "deep",
        6,
        {"task": None, "judge": None, "reflect": None},
    )
    assert (deep["rewrites"], deep["stages"], deep["est_calls"], deep["est_seconds"]) == (
        None,
        None,
        None,
        None,
    )
    assert tuple(cli_plan.dry_object(status="dry")) == cli_plan.DRY_KEYS


def test_the_deep_dry_object_carries_the_plans_workers_and_efforts():
    plan = Plan(models=DEFAULT_MODELS, workers=3, efforts=Efforts("high", None, "low"))
    costs = runner.fixed_costs(plan, 12, synthesising=True)
    obj = cli_plan.plan_object(cli_plan.PlanView(plan, 12, True, costs))
    assert (obj["workers"], obj["efforts"]) == (
        3,
        {"task": "high", "judge": None, "reflect": "low"},
    )
