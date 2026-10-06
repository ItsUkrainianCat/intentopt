"""Example-driven bench items (SPEC R26; WP21): an item with `eval_from` gives the run its examples
before that index and keeps the rest hidden; the bench scores the original and the returned prompt
on the hidden examples (`bench_hidden`) in place of the blind pairwise judge, reports per item the
pass counts and in the summary the pass rates over all hidden examples, text and JSON. Items without
`eval_from` behave as before. `autoimprover bench --prompts bench/examples-set.jsonl --time 2m`
runs end to end on the scripted world."""

import json
import math
from pathlib import Path

import pytest
from fakes import FakeClock, ScriptedBackend
from test_bench_report import ROWS, row
from test_bench_report import summary as plain_summary
from test_bench_runner import Pipeline, Raw, bench_plan, improved, unchanged
from test_bench_world import BETTER, ORIGINAL, BenchWorld, is_pairwise

from autoimprover import cli
from autoimprover.bench import SHIPPED_PROMPTS, BenchPrompt, load_prompts, run_bench
from autoimprover.bench_hidden import Hidden, Passed, hidden_calls
from autoimprover.bench_judge import Comparison
from autoimprover.bench_report import Summary
from autoimprover.cli_fast import plan_for
from autoimprover.fast_prompts import FAST_TASK_SUFFIX
from autoimprover.types import Scenario

EXAMPLES_SET = SHIPPED_PROMPTS.parent / "examples-set.jsonl"


def examples(n: int, **extra) -> list[dict]:
    return [{"input": f"ticket {i}", "expected": f"P{i % 4 + 1}", **extra} for i in range(1, n + 1)]


def write_set(tmp_path: Path, *entries: dict) -> Path:
    path = tmp_path / "set.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


# --- the set ------------------------------------------------------------------------------------


def test_eval_from_splits_the_examples_into_the_runs_and_the_hidden_ones(tmp_path):
    path = write_set(tmp_path, {"id": "a", "prompt": "p", "examples": examples(3), "eval_from": 1})
    (item,) = load_prompts(path)
    assert item.eval_from == 1
    assert [s.input for s in item.given or ()] == ["ticket 1"]
    assert [s.input for s in item.hidden] == ["ticket 2", "ticket 3"]
    plain = BenchPrompt(id="b", prompt="p", examples=(Scenario("e1", "x"),))
    assert plain.given == plain.examples and plain.hidden == ()


@pytest.mark.parametrize("value", [0, 3, -1, "2", True, 1.5, None])
def test_eval_from_is_a_whole_number_from_1_to_one_less_than_the_examples(tmp_path, value):
    entry = {"id": "a", "prompt": "p", "examples": examples(3), "eval_from": value}
    path = write_set(tmp_path, {"id": "ok", "prompt": "q"}, entry)
    with pytest.raises(
        ValueError, match=r"^line 2: `eval_from` must be a whole number from 1 to 2"
    ):
        load_prompts(path)


def test_eval_from_needs_examples_with_references_after_it(tmp_path):
    path = write_set(tmp_path, {"id": "a", "prompt": "p", "eval_from": 1})
    with pytest.raises(ValueError, match=r"^line 1: `eval_from` needs `examples`"):
        load_prompts(path)
    hidden_bare = [*examples(2), {"input": "ticket 3"}]
    path = write_set(tmp_path, {"id": "a", "prompt": "p", "examples": hidden_bare, "eval_from": 1})
    with pytest.raises(ValueError, match=r"^line 1: examples\[2\] is hidden by `eval_from`"):
        load_prompts(path)


def test_the_shipped_examples_set_loads():
    items = load_prompts(EXAMPLES_SET)
    assert [item.id for item in items] == ["ex-priority", "ex-refund", "ex-escalate"]
    assert all(item.eval_from == 8 and len(item.hidden) == 10 for item in items)
    assert all(len(item.given or ()) == 8 for item in items)


# --- the run of a bench ---------------------------------------------------------------------------


def hidden_item(n_hidden: int = 4) -> BenchPrompt:
    scenarios = tuple(
        Scenario(f"e{i}", f"ticket {i}", expected="P1") for i in range(1, 3 + n_hidden)
    )
    return BenchPrompt(id="a", prompt=ORIGINAL, examples=scenarios, eval_from=2)


def measure(tmp_path, outcome, item: BenchPrompt, raw: Raw, baseline=False):
    root = tmp_path / "state" / "bench" / "20261007-010203-0123abcd"
    return run_bench(
        [item],
        bench_plan(baseline),
        root,
        run_one=Pipeline({item.id: outcome}),
        make_raw=raw.make,
        now=FakeClock().now,
        note=[].append,
    ).rows[0]


def test_an_improved_item_is_scored_on_its_hidden_examples_not_pairwise(tmp_path):
    raw = Raw(BenchWorld())
    found = measure(tmp_path, improved(), hidden_item(5), raw)
    assert found.hidden == Hidden(Passed(0, 5), Passed(5, 5), None)
    assert found.tool == Comparison("win", 5, 0, 0)
    assert found.bench_calls == len(raw.calls()) == hidden_calls(True, False, 5)
    assert raw.calls("synth") == [] and not [c for c in raw.calls() if is_pairwise(c)]
    assert {c.sample for c in raw.calls()} == {7000}


def test_an_unchanged_item_is_a_tie_and_still_scores_the_original(tmp_path):
    raw = Raw(BenchWorld())
    found = measure(tmp_path, unchanged(), hidden_item(4), raw)
    assert found.tool == Comparison("tie", 0, 4, 0)
    assert found.hidden == Hidden(Passed(0, 4), Passed(0, 4), None)
    assert found.bench_calls == 4 + math.ceil(4 / 6)


def test_an_item_without_eval_from_is_judged_as_before(tmp_path):
    raw = Raw(BenchWorld())
    found = measure(tmp_path, improved(), BenchPrompt(id="a", prompt=ORIGINAL), raw)
    assert found.hidden is None and [c for c in raw.calls() if is_pairwise(c)]


# --- the summary ---------------------------------------------------------------------------------


HIDDEN_ROWS = (
    row("ex-a", hidden=Hidden(Passed(1, 10), Passed(9, 10)), tool=Comparison("win", 8, 2, 0)),
    row(
        "ex-b",
        "unchanged",
        Comparison("tie", 0, 9, 0),
        hidden=Hidden(Passed(4, 9), Passed(4, 9)),
    ),
    row("p", tool=Comparison("loss", 0, 2, 2)),
)


def summary(rows=HIDDEN_ROWS) -> Summary:
    return plain_summary(rows, baseline=False)


def test_the_summary_object_has_the_pass_rates_over_all_hidden_examples():
    found = summary().object()
    assert found["hidden"] == {
        "items": 2,
        "original_passed": 5,
        "original_of": 19,
        "original_pass_rate": 5 / 19,
        "returned_passed": 13,
        "returned_of": 19,
        "returned_pass_rate": 13 / 19,
    }
    first, second, third = found["rows"]
    assert first["hidden"] == {
        "original": {"passed": 1, "of": 10},
        "returned": {"passed": 9, "of": 10},
        "naive": None,
    }
    assert second["hidden"]["returned"] == {"passed": 4, "of": 9}
    assert "hidden" not in third  # an item without eval_from has the row of before
    assert (found["wins"], found["ties"], found["losses"]) == (1, 1, 1)


def test_without_example_items_the_summary_is_as_before():
    found = plain_summary().object()
    assert found["hidden"] is None and all("hidden" not in r for r in found["rows"])
    assert len(ROWS) == len(found["rows"])


def test_the_summary_text_says_the_pass_counts():
    text = summary().text()
    assert (
        "hidden examples of 2 items with eval_from: the original passed 5 of 19 (26%), the "
        "returned prompts 13 of 19 (68%)"
    ) in text
    assert "  ex-a: original 1 of 10, returned 9 of 10" in text
    assert "  ex-b: original 4 of 9, returned 4 of 9" in text
    assert "hidden examples" not in plain_summary().text()


# --- autoimprover bench --prompts bench/examples-set.jsonl --time 2m ----------------------------


def test_the_examples_set_runs_end_to_end_on_the_scripted_world(capsys):
    raw = ScriptedBackend(BenchWorld())
    argv = ["bench", "--json", "--prompts", str(EXAMPLES_SET), "--time", "2m"]
    code = cli.main(argv, backend=raw, now=FakeClock().now)
    out, err = capsys.readouterr()
    assert code == 0, err
    found = json.loads(out)
    assert (found["measured"], found["improved"], found["wins"]) == (3, 3, 3)
    assert found["hidden"]["original_pass_rate"] == 0.0
    assert found["hidden"]["returned_pass_rate"] == 1.0 and found["hidden"]["returned_of"] == 30
    assert all(r["hidden"]["returned"] == {"passed": 10, "of": 10} for r in found["rows"])
    items = load_prompts(EXAMPLES_SET)
    given = {s.input for item in items for s in item.given or ()}
    hidden = {s.input for item in items for s in item.hidden}
    ran = [c for c in raw.calls if c.role == "task"]
    in_runs = {c.user.split("\n\n")[0] for c in ran if c.user.endswith(FAST_TASK_SUFFIX)}
    in_bench = {c.user.split("\n\n")[0] for c in ran if c.sample == 7000}
    assert in_runs <= given and in_bench == hidden
    # every run decided by the references of its 8 examples (its stage lines on stderr)
    assert "reference judge calls" in err and "pairwise judge calls" not in err


def test_the_examples_set_dry_plans_the_hidden_scoring(capsys):
    argv = ["bench", "--dry", "--json", "--prompts", str(EXAMPLES_SET), "--time", "2m"]
    assert cli.main(argv, backend=ScriptedBackend(lambda c: AssertionError("no call"))) == 0
    found = json.loads(capsys.readouterr().out)
    items = load_prompts(EXAMPLES_SET)
    for item, planned in zip(items, found["per_prompt"], strict=True):
        assert planned["bench_calls"] == hidden_calls(True, False, 10)
        plan = plan_for(120, 6, item.prompt, list(item.given or ()))
        assert planned["run_calls"] == plan.est_calls and plan.reference is not None
    assert BETTER  # the world's rewrite, used by the end-to-end test above
