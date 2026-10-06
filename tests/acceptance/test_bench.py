"""Acceptance tests for SPEC R26 (`autoimprover bench`), written from the SPEC: the repository's set
loads and a bad line is refused naming it; `--dry` makes no call, writes nothing and prints the
calls and the time; each prompt runs the ordinary pipeline and a returned rewrite is compared with
the original by a blind pairwise judge in both orders, so a judge that always prefers the first
position yields ties, never wins, while one that prefers the better answer yields wins (twins); an
unchanged prompt is a tie with no comparison call; `--baseline naive` reports the naive rewrite's
win/tie/loss; `--json` is one object; Ctrl-C is exit 130 (SPEC R2) with the summary of the prompts
measured after the `error:` line and in `bench/<id>/summary.json` (the lead's decision); the bench
exits 0 even when the tool loses, writes under the state folder (`bench/<id>/`) and never prints a
prompt it did not write.

The model is scripted by role: the pipeline's rewrites propose `rewrite`, a task answer is GOOD
when the prompt it ran holds MARKER, the pipeline's judge passes GOOD answers and every contract
check, and the pairwise judge (a judge call whose schema asks for a `winner`, with `answer_A` and
`answer_B` in its JSON) is the test's own function."""

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from fakes import (
    MARKER,
    FakeClock,
    ScriptedBackend,
    intake_reply,
    judge_reply,
    pairwise_reply,
    reflection_reply,
    synth_reply,
)

from autoimprover import cli
from autoimprover.pairwise_text import PAIRWISE_BATCH_SYSTEM
from autoimprover.types import Call

TRIP = "Plan a weekend trip to the mountains for two people."
NOTE = "Write a short note to the team about the new release schedule."
BETTER = f"Plan this well. {MARKER}"
PLAIN = "Plan this, please, with care."  # a rewrite whose answers are no better


def is_pairwise(call: Call) -> bool:
    """The bench's own pairwise call (one scenario), not the fast tiers' batched one."""
    batched = call.system == PAIRWISE_BATCH_SYSTEM
    return call.role == "judge" and '"winner"' in (call.json_schema or "") and not batched


def pair(call: Call) -> tuple[str, str]:
    request = json.loads(call.user)
    return request["answer_A"], request["answer_B"]


def verdict(name: str) -> str:
    return json.dumps({"winner": name, "reason": "because"})


def first_position(_call: Call) -> str:
    return verdict("A")


def better_answer(call: Call) -> str:
    a, b = pair(call)
    if a.startswith("GOOD") == b.startswith("GOOD"):
        return verdict("tie")
    return verdict("A" if a.startswith("GOOD") else "B")


def worse_answer(call: Call) -> str:
    """A judge that prefers the answer the pipeline scored lower: the tool loses."""
    a, b = pair(call)
    if a.startswith("GOOD") == b.startswith("GOOD"):
        return verdict("tie")
    return verdict("B" if a.startswith("GOOD") else "A")


def model(
    rewrite: str = BETTER,
    pairwise: Callable[[Call], str] = better_answer,
    naive: str = PLAIN,
    hook: Callable[[Call], None] = lambda _call: None,
) -> ScriptedBackend:
    def script(call: Call) -> str:
        hook(call)
        if call.role == "intake":
            return intake_reply("task")
        if call.role == "synth":
            return synth_reply(json.loads(call.user)["count"])
        if call.role == "reflect":
            return reflection_reply(
                naive if call.system.startswith("Improve this prompt.") else rewrite
            )
        if call.role == "task":
            return "GOOD answer" if MARKER in call.system + call.user else "BAD answer"
        if is_pairwise(call):
            return pairwise(call)
        if call.system == PAIRWISE_BATCH_SYSTEM:
            return pairwise_reply(call)
        return judge_reply(
            call,
            lambda scenario, _c, out: scenario.startswith("contract") or out.startswith("GOOD"),
        )

    return ScriptedBackend(script)


@pytest.fixture
def bench(capsys) -> Callable:
    """`autoimprover bench <argv>` through `cli.main` with a scripted raw model and a fake clock."""

    def run(argv, backend=None):
        backend = backend or ScriptedBackend(lambda call: AssertionError(f"{call.role} call"))
        capsys.readouterr()
        code = cli.main(["bench", *argv], backend=backend, now=FakeClock().now)
        out, err = capsys.readouterr()
        return code, out, err, backend

    return run


@pytest.fixture
def prompts(tmp_path: Path) -> Callable[..., str]:
    def write(*items: tuple[str, str]) -> str:
        path = tmp_path / "set.jsonl"
        path.write_text("".join(json.dumps({"id": i, "prompt": p}) + "\n" for i, p in items))
        return str(path)

    return write


def one_object(out: str) -> dict:
    text = out.strip()
    obj, end = json.JSONDecoder().raw_decode(text)
    assert end == len(text) and isinstance(obj, dict), out
    return obj


def state() -> Path:
    return Path(os.environ["XDG_STATE_HOME"])


def pairwise_calls(backend: ScriptedBackend) -> list[Call]:
    return [call for call in backend.calls if is_pairwise(call)]


# --- the set and the plan ---------------------------------------------------------------------


def test_the_repository_set_loads_and_dry_makes_no_call_and_writes_nothing(bench):
    code, out, err, backend = bench(["--dry", "--json"])
    plan = one_object(out)
    assert (code, err, backend.calls) == (0, "", [])
    assert plan["prompts"] == 20 and len({row["id"] for row in plan["per_prompt"]}) == 20
    assert plan["est_calls"] > 0 and plan["est_seconds"] > 0
    assert not (state() / "autoimprover").exists()


def test_dry_text_prints_the_calls_and_the_time(bench):
    code, out, _err, backend = bench(["--dry", "--limit", "2"])
    assert code == 0 and backend.calls == []
    assert "calls" in out and " s)" in out and "2 prompts" in out


@pytest.mark.parametrize(
    "line",
    [
        '{"prompt": "no id"}',
        '{"id": "b"}',
        '{"id": "b", "prompt": ""}',
        '{"id": "a", "prompt": "the same id again"}',
        "not json",
    ],
)
def test_a_bad_line_is_refused_naming_it(bench, tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt": "fine"}) + "\n" + line + "\n")
    code, out, err, backend = bench(["--prompts", str(path)])
    assert (code, out, backend.calls) == (2, "", [])
    assert "line 2" in err


def test_more_than_25_prompts_are_refused(bench, tmp_path):
    code, out, err, backend = bench(["--limit", "26"])
    assert (code, out, backend.calls) == (2, "", []) and "--limit" in err and "25" in err
    path = tmp_path / "many.jsonl"
    path.write_text("".join(json.dumps({"id": f"p{i}", "prompt": "p"}) + "\n" for i in range(26)))
    code, out, err, backend = bench(["--dry", "--prompts", str(path)])
    assert (code, out, backend.calls) == (2, "", []) and "line 26" in err
    code, out, _err, _backend = bench(["--dry", "--json", "--limit", "25", "--prompts", str(path)])
    assert code == 0 and one_object(out)["prompts"] == 25


# --- the blind pairwise judge -------------------------------------------------------------------


def test_a_judge_that_prefers_the_first_position_yields_ties_never_wins(bench, prompts):
    code, out, _err, backend = bench(
        ["--json", "--prompts", prompts(("trip", TRIP))], model(pairwise=first_position)
    )
    found = one_object(out)
    assert code == 0 and found["rows"][0]["status"] == "improved"
    assert (found["wins"], found["losses"]) == (0, 0) and found["rows"][0]["verdict"] == "tie"
    assert pairwise_calls(backend)  # it was asked


def test_twin_a_judge_that_prefers_the_better_answer_yields_a_win(bench, prompts):
    code, out, _err, backend = bench(
        ["--json", "--prompts", prompts(("trip", TRIP))], model(pairwise=better_answer)
    )
    found = one_object(out)
    assert code == 0 and found["wins"] == 1 and found["win_rate_of_improved"] == 1.0
    orders = {pair(call) for call in pairwise_calls(backend)}
    assert orders == {("BAD answer", "GOOD answer"), ("GOOD answer", "BAD answer")}  # both orders


def test_the_judge_never_sees_the_rewrite_or_which_answer_is_the_original(bench, prompts):
    _code, _out, _err, backend = bench(["--prompts", prompts(("trip", TRIP))], model())
    assert pairwise_calls(backend)
    for call in pairwise_calls(backend):
        assert BETTER not in call.user and BETTER not in call.system
        assert "original" not in json.dumps(sorted(json.loads(call.user)))


def test_an_unchanged_prompt_is_a_tie_with_no_comparison_call(bench, prompts):
    code, out, _err, backend = bench(
        ["--json", "--prompts", prompts(("trip", TRIP))], model(rewrite=PLAIN)
    )
    found = one_object(out)
    assert code == 0 and found["rows"][0]["status"] == "unchanged"
    assert found["rows"][0]["verdict"] == "tie" and found["ties"] == 1
    assert pairwise_calls(backend) == [] and found["improved"] == 0


def test_the_bench_exits_0_when_the_tool_loses(bench, prompts):
    code, out, _err, _backend = bench(
        ["--json", "--prompts", prompts(("trip", TRIP))], model(pairwise=worse_answer)
    )
    found = one_object(out)
    assert code == 0 and found["losses"] == 1 and found["rows"][0]["verdict"] == "loss"


# --- the naive baseline, the summary, Ctrl-C, the folder ------------------------------------------


def test_the_naive_baseline_is_reported(bench, prompts):
    code, out, _err, backend = bench(
        ["--json", "--baseline", "naive", "--prompts", prompts(("trip", TRIP), ("note", NOTE))],
        model(naive=f"Improved: {MARKER}"),
    )
    found = one_object(out)
    assert code == 0
    assert found["baseline"]["wins"] == 2 and found["baseline"]["win_rate"] == 1.0
    assert [row["naive_verdict"] for row in found["rows"]] == ["win", "win"]
    naive = [c for c in backend.calls if c.system.startswith("Improve this prompt.")]
    assert len(naive) == 2 and {c.effort for c in naive} == {"low"}


def test_json_is_one_object_with_the_measure(bench, prompts):
    code, out, _err, _backend = bench(
        ["--json", "--prompts", prompts(("trip", TRIP), ("note", NOTE))], model()
    )
    found = one_object(out)
    assert code == 0
    for key in (
        "prompts",
        "improved_rate",
        "wins",
        "ties",
        "losses",
        "win_rate_of_improved",
        "baseline",
        "contract_violations",
        "seconds_median",
        "seconds_p90",
        "calls",
        "rows",
    ):
        assert key in found, key


def test_ctrl_c_is_exit_130_and_saves_the_summary_of_the_prompts_measured(bench, prompts):
    def hook(call: Call) -> None:
        if NOTE in call.user:
            raise KeyboardInterrupt

    code, out, err, _backend = bench(
        ["--json", "--prompts", prompts(("trip", TRIP), ("note", NOTE))], model(hook=hook)
    )
    # SPEC R2: exit 130, stdout only the error object, the `error:` line first on stderr
    error = one_object(out)
    assert code == 130 and (error["status"], error["code"]) == ("error", 130)
    lines = err.splitlines()
    error_line = next(n for n, line in enumerate(lines) if line.startswith("error: interrupted"))
    row_line = next(n for n, line in enumerate(lines) if line.split()[:2] == ["trip", "improved"])
    assert error_line < row_line  # the partial summary follows the error line
    (folder,) = (state() / "autoimprover" / "bench").iterdir()
    found = json.loads((folder / "summary.json").read_text())
    assert found["interrupted"] is True
    assert [row["id"] for row in found["rows"]] == ["trip"]


def test_the_bench_writes_under_the_state_folder_and_prints_no_prompt(bench, prompts):
    code, out, err, _backend = bench(
        ["--baseline", "naive", "--prompts", prompts(("trip", TRIP), ("note", NOTE))], model()
    )
    assert code == 0 and "trip" in out and "note" in out
    for text in (TRIP, NOTE, BETTER, PLAIN):
        assert text not in out and text not in err
    (folder,) = (state() / "autoimprover" / "bench").iterdir()
    assert {child.name for child in folder.iterdir()} == {"trip", "note"}
    assert oct(folder.stat().st_mode & 0o777) == oct(0o700)
