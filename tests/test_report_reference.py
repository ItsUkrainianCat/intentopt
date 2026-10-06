"""The report of a reference-scored result (SPEC R2, R25; WP21): when every example carries a
reference, the fast and checked tiers decide by agreement with it, and the report, its JSON and the
plan of `--dry` say "reference-scored" in place of the pairwise preference: the two numbers are
mean reference scores, the noise is the difference of the original's two runs, the margin is the
gain over that noise, and nothing reads as a pairwise preference."""

import dataclasses
import json

import pytest

from autoimprover import reference_text, report
from autoimprover.cli_plan import FastView
from autoimprover.fastplan import Reference, fast_plan
from autoimprover.types import DEFAULT_MODELS, Outcome, Plan

ORIGINAL = "Assign a priority to this support ticket."
BETTER = "Assign a priority (P1 to P4) to this support ticket."
PLAN = Plan(models=DEFAULT_MODELS, wall_clock_s=30, tier="fast", budget=66)
RUN = "/state/autoimprover/runs/20261006-120000-abcdef12"
FAST = Outcome(
    status="improved",
    prompt=BETTER,
    reason=f"tier fast: {reference_text.FAST_LABEL}",
    reason_code="improved",
    changes=("clarified: stated the request the prompt implied, as a direct instruction",),
    score_before=0.25,
    score_after=1.0,
    noise=0.5,
    margin=0.25,
    length_ratio=1.4,
    calls_used=20,
    run_dir=RUN,
    mode="fast",
)
CHECKED = dataclasses.replace(
    FAST,
    reason="tier checked: "
    + reference_text.HELD_WIN.format(held="4 held-out examples, on the target model"),
    verified=True,
    score_before=0.0,
    score_after=0.75,
    noise=0.25,
    margin=0.5,
    search_score_before=0.25,
    search_score_after=1.0,
    mode="checked",
)
NO_WIN = Outcome(
    status="unchanged",
    prompt=ORIGINAL,
    reason=f"tier fast: {reference_text.NO_WIN}",
    reason_code="no_reliable_improvement",
    noise=0.0,
    calls_used=20,
    run_dir=RUN,
    mode="fast",
)
UNGATED = dataclasses.replace(
    FAST,
    reason=f"tier fast: {reference_text.UNGATED_LABEL}",
    reason_code="ungated_best_candidate",
    score_before=None,
    score_after=None,
    search_score_before=0.5,
    search_score_after=0.75,
    noise=0.5,
    margin=-0.25,
)
PAIRWISE_WORDS = ("preference", "pairwise", "won", "lead", "had a winner")


def text_of(outcome: Outcome, plan: Plan = PLAN) -> str:
    return report.render(outcome, ORIGINAL, None, plan, 21.0)


def line(text: str, start: str) -> str:
    (found,) = [found for found in text.splitlines() if found.startswith(start)]
    return found


def test_a_fast_reference_result_reports_mean_reference_scores_not_a_preference():
    text = text_of(FAST)
    judge = DEFAULT_MODELS.judge
    assert line(text, "reference score") == (
        f"reference score on the scenarios it was picked on (judge {judge}, not held out): "
        "0.25 before, 1.00 after"
    )
    assert line(text, "noise: ") == (
        "noise: 0.50 between the mean reference scores of the original's two runs"
    )
    assert line(text, "margin: ") == (
        "margin: gain 0.75 vs noise 0.50 in the mean reference score on the scenarios it was "
        "picked on"
    )
    assert line(text, "meaning: ") == f"meaning: {reference_text.MEANING_UNVERIFIED}"
    assert "reference-scored" in line(text, "verified: ")
    assert not [word for word in PAIRWISE_WORDS if word in text]


def test_the_json_of_a_fast_reference_result_says_so():
    found = report.outcome_object(FAST, ORIGINAL, None, PLAN, 21.0)
    assert "reference-scored" in found["reason"] and "reference-scored" in found["verified_text"]
    assert found["meaning"] == reference_text.MEANING_UNVERIFIED
    assert found["margin_text"].startswith("gain 0.75 vs noise 0.50 in the mean reference score")
    assert not [word for word in PAIRWISE_WORDS if word in json.dumps(found)]


def test_a_checked_reference_result_names_the_held_out_examples():
    plan = dataclasses.replace(PLAN, tier="checked", wall_clock_s=120)
    text = text_of(CHECKED, plan)
    assert line(text, "meaning: ") == f"meaning: {reference_text.MEANING_VERIFIED}"
    assert line(text, "holdout score") == (
        f"holdout score (target model {DEFAULT_MODELS.target}): 0.00 before, 0.75 after"
    )
    assert line(text, "noise: ") == (
        "noise: 0.25 between the mean reference scores of the original's two held-out runs"
    )
    assert line(text, "margin: ") == (
        "margin: gain 0.75 vs noise 0.25 in the mean reference score on the held-out examples"
    )
    assert line(text, "reference score").endswith("0.25 before, 1.00 after")
    assert not [word for word in PAIRWISE_WORDS if word in text]


def test_no_reference_win_says_what_was_compared():
    text = text_of(NO_WIN)
    assert line(text, "meaning: ") == f"meaning: {reference_text.MEANING_NO_WIN}"
    assert line(text, "noise: ").endswith(
        "of the original's two runs; a rewrite had to gain more than 0.00"
    )


def test_an_ungated_reference_result_reports_its_gain_it_did_not_need():
    text = text_of(UNGATED)
    assert line(text, "margin: ") == (
        "margin: gain 0.25 vs noise 0.50 in the mean reference score on the scenarios it was "
        "picked on; ungated, so no gain was required"
    )
    assert line(text, "reference score").endswith("0.50 before, 0.75 after")
    assert line(text, "meaning: ") == f"meaning: {report.REASON_LINES['ungated_best_candidate']}"


@pytest.mark.parametrize(("time_s", "tier"), [(30, "fast"), (120, "checked")])
def test_the_dry_plan_says_its_evidence_is_reference_scored(time_s, tier):
    fplan = fast_plan(time_s, 6, 10, True, Reference(18, 1))
    view = FastView(dataclasses.replace(PLAN, tier=tier, wall_clock_s=time_s), fplan, False)
    assert f"evidence: {reference_text.EVIDENCE[tier]}" in view.text()
    assert "pairwise" not in view.text()
    names = [stage["name"] for stage in view.object()["stages"]]
    assert "C: reference judge and contract checks" in names
    plain = FastView(view.plan, fast_plan(time_s, 6, 10, True), False)
    assert "pairwise" in plain.text()
    assert "C: pairwise judge and contract checks" in [s["name"] for s in plain.object()["stages"]]
