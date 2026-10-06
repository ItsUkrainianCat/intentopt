"""The report of a fast result, decided by pairwise preference (SPEC R2, R25; ADR-012): the judge
compared the rewrite's answers with the original's on the scenarios it was picked on, so the report
gives the share of those scenarios each side won (a preference, never a score), the share where the
original's two runs had a winner (the noise) and, for a returned rewrite, its lead against that
noise; it never says that no noise was measured, and still says the result is not verified on
held-out scenarios. A deep result keeps its holdout wording. The `--json` object carries the run's
seconds and the report's own words (`meaning`, `verified_text`, `margin_text`: each the report's
line without its label), so the mod shows exactly what the command line prints."""

import dataclasses
import io
import json

import pytest

from autoimprover import report
from autoimprover.types import DEFAULT_MODELS, Outcome, Plan

ORIGINAL = "what do i need to make prompt improver app"
BETTER = "List what I need to build a prompt improver app."
PLAN = Plan(models=DEFAULT_MODELS, wall_clock_s=30, tier="fast", budget=66)
LABEL = (
    "fast check: preferred over the original by a pairwise judge on the same few scenarios it was "
    "picked on, noise measured by comparing the original with itself, not verified on held-out "
    "scenarios"
)
FAST = Outcome(
    status="improved",
    prompt=BETTER,
    reason=f"tier fast: {LABEL}",
    reason_code="improved",
    verified=False,
    score_before=0.25,  # the original won 1 of 4 scenarios
    score_after=0.75,  # the rewrite won 3
    noise=0.25,  # the original's two runs had a winner on 1
    margin=0.25,  # a lead of 0.5 above the noise
    calls_used=13,
    run_dir="/state/autoimprover/runs/20261006-220000-abcdef12",
    mode="fast",
)
KEPT = dataclasses.replace(
    FAST,
    status="unchanged",
    prompt=ORIGINAL,
    reason="tier fast: no rewrite kept the contract and was preferred over the original",
    reason_code="no_reliable_improvement",
    score_before=None,
    score_after=None,
    margin=None,
    noise=0.5,
)
PICKED = "the scenarios it was picked on"


def lines_of(text: str, start: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(start)]


def test_a_fast_win_names_the_preference_its_noise_and_its_lead_against_the_noise():
    text = report.render(FAST, ORIGINAL, None, PLAN, elapsed_s=21.0)
    assert lines_of(text, "preference ") == [
        f"preference on {PICKED} (judge {PLAN.models.judge}, not held out): the original won "
        "0.25, the rewrite 0.75"
    ]
    assert lines_of(text, "noise: ") == [
        f"noise: 0.25 of {PICKED} had a winner between the original's two runs"
    ]
    assert lines_of(text, "margin: ") == [f"margin: lead 0.50 vs noise 0.25 on {PICKED}"]
    assert not lines_of(text, "score on") and "no noise" not in text and "holdout" not in text
    [meaning] = lines_of(text, "meaning: ")
    assert "not verified on held-out scenarios" in meaning and "noise" in meaning
    assert "preferred" in meaning


def test_without_noise_a_lead_of_one_scenario_is_enough():
    calm = dataclasses.replace(FAST, score_before=0.0, score_after=0.25, noise=0.0, margin=0.25)
    [margin] = lines_of(report.render(calm, ORIGINAL, None, PLAN, 20.0), "margin: ")
    assert margin == f"margin: lead 0.25 vs noise 0.00 on {PICKED}"


def test_a_kept_fast_result_says_what_a_rewrite_had_to_lead_by():
    text = report.render(KEPT, ORIGINAL, None, PLAN, elapsed_s=21.0)
    assert lines_of(text, "noise: ") == [
        f"noise: 0.50 of {PICKED} had a winner between the original's two runs; a rewrite had to "
        "lead by more than 0.50"
    ]
    assert lines_of(text, "margin: ") == [] and lines_of(text, "preference ") == []


DEEP = dataclasses.replace(FAST, mode="deep", verified=True, margin=0.1, noise=0.1)


def test_a_fast_result_without_noise_and_a_deep_result_read_as_before():
    old = dataclasses.replace(FAST, noise=None, margin=0.65)
    text = report.render(old, ORIGINAL, None, PLAN, elapsed_s=21.0)
    assert lines_of(text, "noise: ") == [] and "(no noise measured)" in text
    [noise] = lines_of(report.render(DEEP, ORIGINAL, None, PLAN, 900.0), "noise: ")
    assert noise == (
        "noise: 0.10 between the original's two holdout runs; the result cleared the bar of 0.20 "
        "by 0.10"
    )


def said(text: str, label: str) -> str | None:
    """The report's line that starts with `label`, without the label; None when there is none."""
    found = lines_of(text, label)
    return found[0].removeprefix(label) if found else None


@pytest.mark.parametrize("outcome", [FAST, KEPT, DEEP], ids=["fast", "kept", "deep"])
def test_the_json_carries_the_seconds_and_the_reports_own_words(outcome):
    obj = report.outcome_object(outcome, ORIGINAL, None, PLAN, 21.5)
    assert list(obj)[-5:] == ["mode", "elapsed_s", "meaning", "verified_text", "margin_text"]
    text = report.render(outcome, ORIGINAL, None, PLAN, 21.5)
    assert obj["elapsed_s"] == 21.5 and obj["noise"] == outcome.noise
    assert obj["meaning"] == said(text, "meaning: ")
    assert obj["verified_text"] == said(text, "verified: ")
    margin = said(text, "margin: ")
    if outcome is DEEP:  # deep keeps the bar in its noise line
        assert margin is None and obj["margin_text"] == "the result cleared the bar of 0.20 by 0.10"
    else:
        assert obj["margin_text"] == margin


def test_a_kept_result_has_no_verified_or_margin_words():
    obj = report.outcome_object(KEPT, ORIGINAL, None, PLAN, 21.5)
    assert (obj["verified_text"], obj["margin_text"]) == (None, None)


def test_the_emitter_gives_the_json_the_runs_clock_and_plan():
    out, err = io.StringIO(), io.StringIO()
    report.Emitter(out, err, json_mode=True).outcome(FAST, ORIGINAL, None, PLAN, 33.25)
    obj = json.loads(out.getvalue())
    assert obj == report.outcome_object(FAST, ORIGINAL, None, PLAN, 33.25)
    assert obj["elapsed_s"] == 33.25 and obj["verified_text"].startswith("no. NOT VERIFIED")
