"""`autoimprover bench` through `cli.main` (SPEC R26, R2, R4, R14, R17, R23, R25): the flags, the
checks before the first paid call, `--dry`, the run of every prompt through the real pipeline in a
run folder under `<state>/autoimprover/bench/<id>/<prompt id>/`, and the one result on stdout. The
raw model is the content-addressed world of `test_bench_world.py`."""

import json
import os
import re
from pathlib import Path

import pytest
from fakes import FakeClock, ScriptedBackend
from test_bench_world import BETTER, ORIGINAL, BenchWorld, first_position, is_pairwise

from autoimprover import cli
from autoimprover.bench import SHIPPED_PROMPTS, load_prompts
from autoimprover.bench_judge import PAIRWISE_SCENARIOS, pairwise_calls
from autoimprover.fastplan import fast_plan
from autoimprover.runner import count_tokens
from autoimprover.types import RUN_ID_PATTERN, BackendError, Call, CallError

S = PAIRWISE_SCENARIOS
OTHER = "Write a short note to the team about the new release schedule."


def state() -> Path:
    return Path(os.environ["XDG_STATE_HOME"]) / "autoimprover"


def prompt_set(tmp_path: Path, *prompts: tuple[str, str], name: str = "set.jsonl") -> str:
    path = tmp_path / name
    lines = [json.dumps({"id": i, "prompt": p}) for i, p in prompts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main(capsys, argv, raw=None, clock=None):
    raw = raw if raw is not None else ScriptedBackend(lambda c: AssertionError(f"{c.role} call"))
    capsys.readouterr()
    code = cli.main(["bench", *argv], backend=raw, now=(clock or FakeClock()).now)
    out, err = capsys.readouterr()
    return code, out, err, raw


def one_object(out: str) -> dict:
    lines = out.splitlines()
    assert len(lines) == 1, out
    return json.loads(lines[0])


# --- the flags and the checks before any call ---------------------------------------------------


def test_dry_makes_no_call_and_writes_nothing(capsys):
    code, out, err, raw = main(capsys, ["--dry"])
    assert (code, err, raw.calls) == (0, "", [])
    assert out.startswith("dry run: no model call made, nothing written\n")
    assert "bench: 20 prompts" in out
    assert not state().exists()


def test_dry_json_plans_the_calls_and_seconds_of_every_prompt(capsys):
    code, out, _err, raw = main(capsys, ["--dry", "--json", "--baseline", "naive"])
    found = one_object(out)
    assert (code, raw.calls, found["status"], found["prompts"]) == (0, [], "dry", 20)
    prompts = load_prompts(SHIPPED_PROMPTS)
    plans = [fast_plan(30, 6, count_tokens(p.prompt), False) for p in prompts]
    assert found["run_calls"] == sum(plan.est_calls for plan in plans)
    assert found["bench_calls"] == 20 * pairwise_calls(True, True)
    assert found["est_calls"] == found["run_calls"] + found["bench_calls"]
    assert [row["id"] for row in found["per_prompt"]] == [p.id for p in prompts]
    assert found["tier"] == "fast" and found["time_s"] == 30 and found["refusal"] is None


def test_limit_takes_the_first_prompts(capsys):
    _code, out, _err, _raw = main(capsys, ["--dry", "--json", "--limit", "3"])
    assert [row["id"] for row in one_object(out)["per_prompt"]] == ["vague-1", "vague-2", "vague-3"]


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        (["--limit", "26"], "--limit: a whole number from 1 to 25"),
        (["--limit", "0"], "--limit: a whole number from 1 to 25"),
        (["--limit", "x"], "--limit: a whole number from 1 to 25"),
        (["--baseline", "smart"], "--baseline: invalid choice"),
        (["--seed", "-1"], "--seed: a whole number from 0 to 999"),
        (["--seed", "1000"], "--seed: a whole number from 0 to 999"),
        (["--limit", "3", "--time", "10s"], "--time"),
        # the default target model (SPEC R14)
        (["--limit", "3", "--judge-model", "sonnet"], "--judge-model"),
        (["--limit", "3", "--trust-search"], "unrecognized arguments: --trust-search"),
        (["--limit", "3", "a prompt"], "unrecognized arguments: a prompt"),
    ],
)
def test_bad_usage_is_exit_2_naming_the_flag_with_no_call(capsys, argv, says):
    code, out, err, raw = main(capsys, ["--dry", *argv])
    assert (code, out, raw.calls) == (2, "", [])
    assert err.startswith("error: ") and says in err


def test_a_bad_line_of_the_set_is_exit_2_naming_the_line(capsys, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "prompt": "fine"}\n{"id": "a", "prompt": "again"}\n')
    code, out, err, _raw = main(capsys, ["--json", "--prompts", str(path)])
    assert code == 2 and "line 2" in err and "--prompts" in err
    assert one_object(out)["status"] == "error"


def test_bench_must_come_first(capsys):
    capsys.readouterr()
    code = cli.main(["--json", "bench"], backend=ScriptedBackend(lambda c: "x"))
    out, err = capsys.readouterr()
    assert code == 2 and "bench" in err and json.loads(out)["status"] == "error"


def test_help_lists_the_flags(capsys):
    code, out, _err, _raw = main(capsys, ["--help"])
    for flag in ("--prompts", "--time", "--limit", "--baseline", "--judge-model", "--seed"):
        assert flag in out
    assert code == 0


def test_the_main_help_names_the_bench(capsys):
    capsys.readouterr()
    assert cli.main(["--help"]) == 0
    assert "autoimprover bench" in capsys.readouterr().out


def test_a_plan_that_would_refuse_is_said_by_dry_and_refused_before_any_call(capsys):
    # the deep tier at 10 minutes: 50 calls afford fewer than 4 search iterations (SPEC R4)
    code, out, _err, _raw = main(capsys, ["--dry", "--time", "10m", "--limit", "2"])
    assert code == 0 and "a real bench would refuse: prompt vague-1:" in out
    code, out, err, raw = main(capsys, ["--time", "10m", "--limit", "2"])
    assert (code, out, raw.calls) == (2, "", []) and "iterations" in err
    assert not state().exists()


def test_a_later_prompt_whose_plan_would_refuse_stops_the_bench_before_any_call(capsys, tmp_path):
    # the quick tier at 15 s fits a short prompt, not one whose rewrite takes 600 tokens (R25)
    path = prompt_set(tmp_path, ("short", "Write a haiku about rain."), ("long", "word " * 600))
    code, out, err, raw = main(
        capsys, ["--time", "15s", "--prompts", path], ScriptedBackend(BenchWorld())
    )
    assert (code, out, raw.calls) == (2, "", []) and "prompt long:" in err
    assert not state().exists()


def test_an_unusable_state_folder_is_refused_before_any_call(capsys, monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a folder")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocked))
    code, out, _err, _raw = main(capsys, ["--dry", "--limit", "1"])
    assert code == 0 and "a real bench would refuse:" in out
    code, _out, err, raw = main(capsys, ["--limit", "1"])
    assert (code, raw.calls) == (2, []) and "XDG_STATE_HOME" in err


# --- a real bench ---------------------------------------------------------------------------------


def test_each_prompt_runs_the_pipeline_in_its_own_folder_and_is_judged(capsys, tmp_path):
    path = prompt_set(tmp_path, ("trip", ORIGINAL), ("note", OTHER))
    code, out, err, raw = main(capsys, ["--json", "--prompts", path], ScriptedBackend(BenchWorld()))
    found = one_object(out)
    assert code == 0 and found["status"] == "bench" and found["prompts"] == 2
    assert [row["id"] for row in found["rows"]] == ["trip", "note"]
    assert all(row["status"] == "improved" for row in found["rows"])  # BETTER holds the marker
    assert [row["verdict"] for row in found["rows"]] == ["win", "win"]
    assert (found["wins"], found["win_rate_of_improved"]) == (2, 1.0)
    (bench_dir,) = (state() / "bench").iterdir()
    assert found["folder"] == str(bench_dir) and re.fullmatch(RUN_ID_PATTERN, bench_dir.name)
    for prompt_id in ("trip", "note"):
        (run_dir,) = (bench_dir / prompt_id).iterdir()
        assert re.fullmatch(RUN_ID_PATTERN, run_dir.name) and (run_dir / "manifest.json").exists()
    assert not (state() / "runs").exists()  # a bench never writes among the user's runs
    pairwise = [call for call in raw.calls if is_pairwise(call)]
    assert len(pairwise) == 2 * 2 * S
    assert found["calls"] == len(raw.calls)
    assert "bench: prompt 1 of 2: trip" in err


def test_the_pairwise_answers_come_from_the_target_model_without_the_suffix(capsys, tmp_path):
    path = prompt_set(tmp_path, ("trip", ORIGINAL))
    _code, _out, _err, raw = main(capsys, ["--prompts", path], ScriptedBackend(BenchWorld()))
    target = "claude-sonnet-5-5"
    plain = [c for c in raw.calls if c.role == "task" and c.model == target]
    assert len(plain) == 2 * S
    assert all("120 words" not in c.user for c in plain)
    assert {c.user.split("\n\n", 1)[1] for c in plain} == {ORIGINAL, BETTER}


def test_a_position_biased_judge_yields_ties_and_its_twin_wins(capsys, tmp_path):
    path = prompt_set(tmp_path, ("trip", ORIGINAL))
    biased = ScriptedBackend(BenchWorld(pairwise=first_position))
    _code, out, _err, _raw = main(capsys, ["--json", "--prompts", path], biased)
    assert one_object(out)["rows"][0]["verdict"] == "tie"
    _code, out, _err, _raw = main(
        capsys, ["--json", "--prompts", path], ScriptedBackend(BenchWorld())
    )
    assert one_object(out)["rows"][0]["verdict"] == "win"


def test_the_text_summary_names_ids_and_numbers_never_a_prompt(capsys, tmp_path):
    path = prompt_set(tmp_path, ("trip", ORIGINAL), ("note", OTHER))
    code, out, err, _raw = main(
        capsys, ["--prompts", path, "--baseline", "naive"], ScriptedBackend(BenchWorld())
    )
    assert code == 0 and "trip" in out and "note" in out and "naive baseline" in out
    for text in (ORIGINAL, OTHER, BETTER, "Plan a weekend", "release schedule"):
        assert text not in out
    assert ORIGINAL not in err and OTHER not in err


def test_ctrl_c_prints_the_summary_of_the_prompts_measured(capsys, tmp_path):
    path = prompt_set(tmp_path, ("trip", ORIGINAL), ("note", OTHER))
    world = BenchWorld()

    def script(call: Call) -> str | Exception:
        if OTHER in call.user:
            raise KeyboardInterrupt
        return world(call)

    code, out, err, _raw = main(capsys, ["--json", "--prompts", path], ScriptedBackend(script))
    found = one_object(out)
    assert code == 0 and found["interrupted"] is True
    assert [row["id"] for row in found["rows"]] == ["trip"] and found["prompts"] == 2
    assert "interrupted" in err


def test_a_failed_prompt_is_an_error_and_the_bench_goes_on(capsys, tmp_path):
    path = prompt_set(tmp_path, ("note", OTHER), ("trip", ORIGINAL))
    world = BenchWorld()

    def script(call: Call) -> str | Exception:
        return CallError("down") if call.role == "intake" and OTHER in call.user else world(call)

    code, out, _err, _raw = main(capsys, ["--json", "--prompts", path], ScriptedBackend(script))
    found = one_object(out)
    assert code == 0 and [row["status"] for row in found["rows"]] == ["error", "improved"]
    assert found["errors"] == 1 and "intake" in found["rows"][0]["error"]
    plan = fast_plan(30, 6, count_tokens(OTHER), False)
    assert found["rows"][0]["calls"] == 3 + 1 + plan.rewrites  # intake x3, synthesis, rewrites


def test_every_prompt_failing_is_a_backend_failure(capsys, tmp_path):
    path = prompt_set(tmp_path, ("note", OTHER), ("trip", ORIGINAL))
    code, out, err, _raw = main(
        capsys, ["--json", "--prompts", path], ScriptedBackend(lambda _c: CallError("down"))
    )
    assert code == 3 and one_object(out)["status"] == "error" and "every prompt" in err


def test_a_deep_run_that_keeps_the_original_without_a_folder_is_a_tie_with_no_call(
    capsys, tmp_path
):
    # the deep tier with 3 examples has no holdout, so it keeps the original before any call
    path = tmp_path / "examples.jsonl"
    examples = [{"input": f"case {i}"} for i in range(3)]
    path.write_text(json.dumps({"id": "trip", "prompt": ORIGINAL, "examples": examples}) + "\n")
    argv = ["--json", "--time", "20m", "--baseline", "naive", "--prompts", str(path)]
    code, out, _err, raw = main(capsys, argv, ScriptedBackend(BenchWorld()))
    row = one_object(out)["rows"][0]
    assert (code, raw.calls) == (0, [])
    assert (row["status"], row["reason_code"], row["verdict"]) == ("unchanged", "no_holdout", "tie")
    assert row["naive_verdict"] is None and row["calls"] == 0


def test_a_backend_failure_of_the_judge_is_reported_and_the_bench_exits_0(capsys, tmp_path):
    path = prompt_set(tmp_path, ("trip", ORIGINAL))
    world = BenchWorld(pairwise=lambda _c: BackendError("judge down"))
    code, out, _err, _raw = main(capsys, ["--json", "--prompts", path], ScriptedBackend(world))
    row = one_object(out)["rows"][0]
    assert code == 0 and row["verdict"] == "error" and "judge down" in row["error"]
