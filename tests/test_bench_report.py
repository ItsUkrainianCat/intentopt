"""What `autoimprover bench` prints (SPEC R26, R2, R4): the summary as text or one JSON object
(prompts, improved rate, win/tie/loss and the win rate among improved prompts, the naive
baseline's, contract violations, median and 90th percentile seconds, total calls, a row per prompt)
and the plan of `--dry` (prompts, calls, seconds). Both hold ids, codes and numbers, never a
prompt."""

import json

import pytest

from autoimprover.bench import Measured, Row
from autoimprover.bench_judge import Comparison
from autoimprover.bench_report import DryRow, DryView, Summary, percentile
from autoimprover.types import Models

MODELS = Models(
    task="claude-haiku-4-5-20251001",
    judge="claude-opus-5-5",
    reflect="claude-sonnet-5-5",
    target="claude-fable-5-1",
)
WIN, TIE, LOSS = Comparison("win", 3, 1, 0), Comparison("tie", 1, 2, 1), Comparison("loss", 0, 1, 3)
UNCHANGED = Comparison("tie", 0, 0, 0)
FAILED = Comparison("error", 0, 0, 0, "no scenario was judged in both orders")


def row(i: str, status: str = "improved", tool=WIN, naive=None, seconds=20.0, **kw) -> Row:
    fields = {
        "id": i,
        "status": status,
        "reason_code": {"improved": "improved", "unchanged": "no_reliable_improvement"}.get(status),
        "verified": False if status != "error" else None,
        "seconds": seconds,
        "calls": 17,
        "bench_calls": 17 if status == "improved" else 0,
        "noise": 0.05 if status != "error" else None,
        "tool": tool if status != "error" else None,
        "naive": naive,
        "error": "backend failure: down" if status == "error" else "",
    } | kw
    return Row(**fields)


ROWS = (
    row("p1", tool=WIN, naive=TIE, seconds=10.0),
    row("p2", tool=LOSS, naive=WIN, seconds=30.0),
    row("p3", "unchanged", UNCHANGED, naive=TIE, seconds=20.0),
    row("p4", "unchanged", UNCHANGED, naive=LOSS, seconds=40.0),
    row("p5", tool=FAILED, naive=FAILED, seconds=50.0),
    row("p6", "error", seconds=None, calls=6),
)


def summary(rows=ROWS, interrupted=False, baseline=True, prompts=None) -> Summary:
    return Summary(
        measured=Measured(tuple(rows), interrupted),
        prompts=len(rows) if prompts is None else prompts,
        tier="fast",
        time_s=30,
        baseline=baseline,
        folder="/state/autoimprover/bench/20261007-010203-0123abcd",
    )


# --- the numbers --------------------------------------------------------------------------------


def test_the_summary_object_holds_the_measure():
    found = summary().object()
    assert found["status"] == "bench" and found["interrupted"] is False
    assert (found["prompts"], found["measured"], found["errors"]) == (6, 5, 1)
    assert (found["improved"], found["improved_rate"]) == (3, 3 / 5)
    # the tool against the original: p1 win, p2 loss, p3 and p4 unchanged ties, p5 not compared
    assert (found["wins"], found["ties"], found["losses"]) == (1, 2, 1)
    assert found["compare_errors"] == 1
    assert found["win_rate"] == 1 / 4 and found["win_rate_of_improved"] == 1 / 2
    assert found["baseline"] == {
        "kind": "naive",
        "wins": 1,
        "ties": 2,
        "losses": 1,
        "errors": 1,
        "win_rate": 1 / 4,
    }
    assert found["contract_violations"] is None
    assert found["seconds_median"] == 30.0 and found["seconds_p90"] == 50.0
    assert (found["calls"], found["run_calls"], found["bench_calls"]) == (
        5 * 17 + 6 + 3 * 17,
        5 * 17 + 6,
        3 * 17,
    )
    assert found["tier"] == "fast" and found["time_s"] == 30


def test_a_row_per_prompt_with_its_verdicts():
    rows = summary().object()["rows"]
    assert [r["id"] for r in rows] == ["p1", "p2", "p3", "p4", "p5", "p6"]
    assert rows[0] | {} == {
        "id": "p1",
        "status": "improved",
        "reason_code": "improved",
        "verified": False,
        "seconds": 10.0,
        "calls": 17,
        "bench_calls": 17,
        "noise": 0.05,
        "verdict": "win",
        "votes": {"wins": 3, "ties": 1, "losses": 0},
        "naive_verdict": "tie",
        "naive_votes": {"wins": 1, "ties": 2, "losses": 1},
        "error": None,
    }
    assert rows[5]["status"] == "error" and rows[5]["verdict"] is None
    assert rows[5]["error"] == "backend failure: down"
    assert rows[4]["verdict"] == "error" and rows[4]["error"] == FAILED.why


def test_without_a_baseline_and_without_data_the_rates_are_null():
    found = summary([row("p1", "error", seconds=None)], baseline=False).object()
    assert found["baseline"] is None
    assert found["improved_rate"] is None and found["win_rate"] is None
    assert found["win_rate_of_improved"] is None
    assert found["seconds_median"] is None and found["seconds_p90"] is None


def test_an_interrupted_bench_says_how_many_it_measured():
    found = summary(ROWS[:2], interrupted=True, prompts=20)
    assert found.object()["interrupted"] is True and found.object()["prompts"] == 20
    assert "interrupted" in found.text() and "2 of 20" in found.text()


@pytest.mark.parametrize(
    ("values", "p90"),
    [
        ([5.0], 5.0),
        ([1.0, 2.0], 2.0),
        ([float(i) for i in range(1, 11)], 9.0),
        ([3.0, 1.0, 2.0], 3.0),
    ],
)
def test_the_90th_percentile_is_the_nearest_rank(values, p90):
    assert percentile(values, 90) == p90


def test_the_object_is_one_line_of_json():
    text = json.dumps(summary().object(), allow_nan=False)
    assert json.loads(text)["status"] == "bench"


# --- the text -----------------------------------------------------------------------------------


def test_the_text_summary_holds_the_numbers_and_a_table_of_ids():
    text = summary().text()
    for said in (
        "5 of 6 prompts measured",
        "3 improved",
        "1 win, 2 ties, 1 loss",
        "25%",  # win rate of the compared
        "50% of the 2 improved compared",
        "naive",
        "not measured",
        "median 30.0 s",
        "p90 50.0 s",
        "calls: 142",
    ):
        assert said in text, said
    lines = text.splitlines()
    assert sum(line.split()[:1] == [f"p{i}"] for i in range(1, 7) for line in lines) == 6


# --- the dry plan ---------------------------------------------------------------------------------


def dry(refusal=None, baseline=False) -> DryView:
    return DryView(
        path="bench/prompts.jsonl",
        tier="fast",
        time_s=30,
        workers=6,
        models=MODELS,
        baseline=baseline,
        scenarios=4,
        rows=(DryRow("p1", 17, 25.5, 17, 35.0), DryRow("p2", 14, 20.6, 17, 35.0)),
        refusal=refusal,
    )


def test_the_dry_plan_adds_up_the_calls_and_seconds():
    found = dry().object()
    assert found["status"] == "dry" and found["prompts"] == 2
    assert (found["run_calls"], found["bench_calls"], found["est_calls"]) == (31, 34, 65)
    assert found["est_seconds"] == pytest.approx(25.5 + 20.6 + 70.0)
    assert found["baseline"] == "none" and found["scenarios"] == 4 and found["refusal"] is None
    assert found["per_prompt"][0] == {
        "id": "p1",
        "run_calls": 17,
        "run_seconds": 25.5,
        "bench_calls": 17,
        "bench_seconds": 35.0,
    }


def test_the_dry_text_says_the_calls_the_time_and_a_refusal():
    text = dry("the state folder is not writable", baseline=True).text()
    assert "2 prompts" in text and "65 calls" in text and "about 2 min (116 s)" in text
    assert "baseline: naive" in text and MODELS.judge in text and MODELS.target in text
    assert "a real bench would refuse: the state folder is not writable" in text
