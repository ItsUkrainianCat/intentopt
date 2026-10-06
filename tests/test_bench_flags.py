"""`autoimprover bench` takes the run's model, effort, worker and strictness flags (SPEC R26, with
R14 and R25): the same names, values and refusals as a run, in the plan `--dry` prints, in every
prompt's run, and in the bench's own calls, where the target model answers the fresh scenarios at
the task role's effort, the judge model judges them, and the reflection model writes them. The
raw model is the content-addressed world of `test_bench_world.py`."""

import json

import pytest
from fakes import FakeClock, ScriptedBackend
from test_bench_cli import main, one_object, prompt_set, state
from test_bench_world import ORIGINAL, BenchWorld, is_pairwise

from autoimprover import cli
from autoimprover.bench_judge import BENCH_SAMPLE, PAIRWISE_SCENARIOS
from autoimprover.types import Call

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
JUDGE = "claude-test-judge"  # an unknown name passes through as its own id (SPEC R14)
SHARED = (
    "--task-model",
    "--judge-model",
    "--reflect-model",
    "--target-model",
    "--effort",
    "--task-effort",
    "--judge-effort",
    "--reflect-effort",
    "--workers",
    "--strictness",
)


def run_main(capsys, argv):
    """`autoimprover <argv>`, a run, with a raw model that fails the test when called."""
    capsys.readouterr()
    raw = ScriptedBackend(lambda c: AssertionError(f"{c.role} call"))
    code = cli.main(argv, backend=raw, now=FakeClock().now)
    out, err = capsys.readouterr()
    return code, out, err, raw


def line_of(text: str, start: str) -> str:
    (found,) = [line for line in text.splitlines() if line.startswith(start)]
    return found


# --- the plan ---------------------------------------------------------------------------------


def test_the_dry_plan_of_a_weak_target_names_the_models_it_will_use(capsys, tmp_path):
    # the command of the brief: a bench whose runs and comparisons answer on Haiku
    path = prompt_set(tmp_path, ("trip", ORIGINAL))
    argv = ["--dry", "--prompts", path, "--time", "2m", "--task-model", HAIKU]
    code, out, err, raw = main(capsys, [*argv, "--target-model", HAIKU])
    assert (code, err, raw.calls) == (0, "", [])
    models = line_of(out, "models: ")
    assert f"task {HAIKU}," in models and f"target {HAIKU} (the pairwise answers)" in models
    assert f"judge {OPUS} (also the pairwise judge)" in models
    assert "tier checked (--time 120 s)" in out


def test_the_dry_plan_shows_every_flag_given(capsys):
    argv = ["--dry", "--limit", "1", "--target-model", "haiku", "--reflect-model", "opus"]
    argv += ["--judge-model", JUDGE, "--effort", "medium", "--judge-effort", "high"]
    argv += ["--strictness", "bold", "--workers", "3"]
    code, out, _err, _raw = main(capsys, argv)
    assert code == 0
    models = line_of(out, "models: ")
    assert f"task {HAIKU}, judge {JUDGE}" in models and f"reflection {OPUS}" in models
    assert line_of(out, "effort: ").startswith("effort: task medium, judge high, reflection medium")
    assert line_of(out, "strictness: ").startswith("strictness: bold")
    assert "3 calls at a time" in out
    code, out, _err, _raw = main(capsys, [*argv, "--json"])
    found = one_object(out)
    assert found["models"] == {"task": HAIKU, "judge": JUDGE, "reflect": OPUS, "target": HAIKU}
    assert found["efforts"] == {"task": "medium", "judge": "high", "reflect": "medium"}
    assert (code, found["strictness"], found["workers"]) == (0, "bold", 3)


def test_without_flags_the_plan_shows_the_tier_defaults(capsys):
    code, out, _err, _raw = main(capsys, ["--dry", "--json", "--limit", "1"])
    found = one_object(out)
    assert found["efforts"] == {"task": "low", "judge": "low", "reflect": "low"}
    assert (code, found["strictness"], found["models"]["target"]) == (0, "balanced", SONNET)
    code, out, _err, _raw = main(capsys, ["--dry", "--json", "--limit", "1", "--time", "20m"])
    found = one_object(out)
    assert found["efforts"] == {"task": None, "judge": None, "reflect": None}
    assert (code, found["strictness"]) == (0, "conservative")


def test_the_help_lists_the_flags_of_a_run_it_takes(capsys):
    code, out, _err, _raw = main(capsys, ["--help"])
    words = out.split()  # a flag name is never broken across lines
    assert code == 0
    assert set(SHARED) <= {word.strip(",;.") for word in words}
    assert "low, medium, high, xhigh, max, default." in " ".join(words)


# --- the same refusals as a run -----------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ["--target-model", "opus", "--judge-model", "opus"],  # the judge is the target (R14)
        ["--task-model", "opus"],  # the default judge is now the task model (R14)
        ["--target-model", " "],
        ["--effort", "huge"],
        ["--reflect-effort", "extreme"],
        ["--workers", "0"],
        ["--workers", "17"],
        ["--strictness", "wild"],
        ["--task-model"],
    ],
)
def test_a_bad_shared_flag_is_refused_as_a_run_refuses_it(capsys, flags):
    code, out, err, raw = main(capsys, ["--dry", "--limit", "1", *flags])
    run_code, run_out, run_err, _raw = run_main(capsys, ["--dry", "a prompt", *flags])
    assert (code, out, raw.calls) == (2, "", [])
    assert (run_code, run_out) == (2, "") and run_err.startswith("error: ")
    assert err == run_err


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (["--budget", "10"], "--budget"),
        (["--deep"], "--deep"),
        (["--examples", "x.jsonl"], "--examples"),
        (["--kind", "task"], "--kind"),
        (["--resume", "20261006-120000-abcdef12"], "--resume"),
    ],
)
def test_a_run_flag_the_bench_does_not_take_is_refused_alone(capsys, flags, named):
    code, out, err, raw = main(capsys, ["--dry", "--task-model", "sonnet", *flags])
    assert (code, out, raw.calls) == (2, "", [])
    assert err.startswith("error: unrecognized arguments: ") and named in err
    assert "--task-model" not in err  # the shared flag beside it is not the problem


# --- a real bench -------------------------------------------------------------------------------


def test_every_run_and_the_bench_calls_take_the_chosen_models_and_efforts(capsys, tmp_path):
    path = prompt_set(tmp_path, ("trip", ORIGINAL))
    argv = ["--json", "--prompts", path, "--baseline", "naive"]
    argv += ["--task-model", "sonnet", "--target-model", "haiku", "--reflect-model", "opus"]
    argv += ["--judge-model", JUDGE, "--effort", "medium", "--judge-effort", "high"]
    argv += ["--reflect-effort", "max", "--strictness", "bold", "--workers", "5"]
    code, out, _err, raw = main(capsys, argv, ScriptedBackend(BenchWorld()))
    row = one_object(out)["rows"][0]
    assert (code, row["status"], row["verdict"]) == (0, "improved", "win")

    def used(calls: list[Call]) -> set[tuple[str, str | None]]:
        assert calls
        return {(call.model, call.effort) for call in calls}

    tasks = [call for call in raw.calls if call.role == "task"]
    scored = [call for call in tasks if "120 words" in call.user]  # the run's scoring runs
    answers = [call for call in tasks if "120 words" not in call.user]  # the bench's
    assert used(scored) == {(SONNET, "medium")}
    assert used(answers) == {(HAIKU, "medium")} and len(answers) == 3 * PAIRWISE_SCENARIOS
    judges = [call for call in raw.calls if call.role == "judge"]
    assert used(judges) == {(JUDGE, "high")} and any(is_pairwise(call) for call in judges)
    writers = [call for call in raw.calls if call.role in ("intake", "synth", "reflect")]
    naive = [call for call in writers if call.system.startswith("Improve this prompt.")]
    assert used(naive) == {(OPUS, "low")}  # the naive baseline is low effort (SPEC R26)
    others = [call for call in writers if call not in naive]
    assert used(others) == {(OPUS, "max")}
    assert any(call.role == "synth" and call.sample == BENCH_SAMPLE for call in others)
    (run_dir,) = (state() / "bench").glob("*/trip/*")
    plan = json.loads((run_dir / "manifest.json").read_text())["plan"]
    assert plan["models"] == {"task": SONNET, "judge": JUDGE, "reflect": OPUS, "target": HAIKU}
    assert plan["efforts"] == {"task": "medium", "judge": "high", "reflect": "max"}
    assert (plan["strictness"], plan["workers"]) == ("bold", 5)
