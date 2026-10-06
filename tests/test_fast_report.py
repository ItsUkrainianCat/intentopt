"""The report of a fast result whose noise was measured (SPEC R2, R25): the original ran twice on
the scenarios the rewrite was picked on, so the report says how far apart those two runs were and,
for a returned rewrite, its gain against the gain required, max(0.1, 2 x noise); it never says
that no noise was measured, and still says the result is not verified on held-out scenarios. A
deep result keeps its holdout wording. The `--json` object carries the run's seconds and the
report's own words (`meaning`, `verified_text`, `margin_text`: each the report's line without its
label), so the mod shows exactly what the command line prints."""

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
    "fast check: scored on the same few scenarios it was picked on, noise measured from two runs "
    "of the original on those scenarios, not verified on held-out scenarios"
)
FAST = Outcome(
    status="improved",
    prompt=BETTER,
    reason=f"tier fast: {LABEL}",
    reason_code="improved",
    verified=False,
    score_before=0.55,
    score_after=0.8,
    noise=0.1,
    margin=0.05,  # a gain of 0.25 above the bar of 0.2
    calls_used=13,
    run_dir="/state/autoimprover/runs/20261006-220000-abcdef12",
    mode="fast",
)
KEPT = dataclasses.replace(
    FAST,
    status="unchanged",
    prompt=ORIGINAL,
    reason="tier fast: no rewrite kept the contract and beat the original",
    reason_code="no_reliable_improvement",
    score_after=None,
    margin=None,
    noise=0.15,
)


def lines_of(text: str, start: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(start)]


def test_a_fast_win_names_its_noise_and_its_gain_against_the_gain_required():
    text = report.render(FAST, ORIGINAL, None, PLAN, elapsed_s=21.0)
    assert lines_of(text, "noise: ") == [
        "noise: 0.10 between the original's two runs on the scenarios it was picked on"
    ]
    assert lines_of(text, "margin: ") == [
        "margin: gain 0.25 vs required 0.20 on the scenarios it was picked on"
    ]
    assert "no noise" not in text and "holdout" not in text
    [meaning] = lines_of(text, "meaning: ")
    assert "not verified on held-out scenarios" in meaning and "noise" in meaning


def test_the_gain_required_is_the_least_gain_when_twice_the_noise_is_smaller():
    small = dataclasses.replace(FAST, noise=0.02, margin=0.15)  # max(0.1, 0.04) = 0.1
    [margin] = lines_of(report.render(small, ORIGINAL, None, PLAN, 20.0), "margin: ")
    assert margin == "margin: gain 0.25 vs required 0.10 on the scenarios it was picked on"


def test_a_kept_fast_result_says_what_a_rewrite_had_to_gain():
    text = report.render(KEPT, ORIGINAL, None, PLAN, elapsed_s=21.0)
    assert lines_of(text, "noise: ") == [
        "noise: 0.15 between the original's two runs on the scenarios it was picked on; a "
        "result had to gain more than 0.30"
    ]
    assert lines_of(text, "margin: ") == []
    quiet = dataclasses.replace(KEPT, noise=0.02)  # max(0.1, 0.04): the least gain, not 0.05
    [noise] = lines_of(report.render(quiet, ORIGINAL, None, PLAN, 21.0), "noise: ")
    assert noise.endswith("; a result had to gain more than 0.10")


DEEP = dataclasses.replace(FAST, mode="deep", verified=True, margin=0.1)


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
