"""The flags of the time tiers (SPEC R25; ADR-011 "Customisation"): `--time` picks the tier and is
the run's clock, `--deep` is `--time 20m`, `--workers` the calls at a time, `--effort` and the
per-role effort flags the `claude --effort` of each role; a flag given always wins over the tier's
default, and every bad value is a usage error naming its flag (exit 2)."""

import pytest

from autoimprover.cli_options import UsageError, budget, duration, parse, settings
from autoimprover.types import Efforts, default_models


def chosen(*argv: str):
    return settings(parse([*argv, "a prompt"]))


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("15s", 15), ("30s", 30), ("1m", 60), ("45m", 2700), ("2h", 7200), ("3h", 10800)],
)
def test_a_time_is_a_whole_number_of_seconds_minutes_or_hours(text, seconds):
    assert duration(text) == seconds


@pytest.mark.parametrize(
    "text", ["30", "1.5m", "-5s", "+5s", "s", "5 m", "5M", "5d", "1h30m", "５s", "", "0x1s"]
)
def test_anything_else_is_refused_naming_the_flag_and_the_form(text):
    with pytest.raises(UsageError, match=r"--time: .*30s, 5m or 1h"):
        duration(text)


def test_without_flags_the_run_is_the_30_second_fast_tier_with_its_defaults():
    run = chosen()
    assert (run.tier, run.time_s, run.workers) == ("fast", 30, 6)
    assert run.efforts == Efforts("low", "low", "low")
    assert run.models == default_models(None, "fast")
    assert run.models.reflect == "claude-sonnet-5-5"


@pytest.mark.parametrize(
    ("argv", "tier", "seconds"),
    [
        (["--time", "15s"], "quick", 15),
        (["--time", "24s"], "quick", 24),
        (["--time", "59s"], "fast", 59),
        (["--time", "1m"], "checked", 60),
        (["--time", "9m"], "checked", 540),
        (["--time", "10m"], "deep", 600),
        (["--time", "3h"], "deep", 10800),
        (["--deep"], "deep", 1200),
    ],
)
def test_the_time_picks_the_tier(argv, tier, seconds):
    run = chosen(*argv)
    assert (run.tier, run.time_s) == (tier, seconds)


def test_the_deep_tier_keeps_the_models_own_effort_and_the_search_models():
    run = chosen("--deep")
    assert run.efforts == Efforts() and run.models == default_models(None, "deep")
    assert run.models.reflect == "claude-opus-5-5"


@pytest.mark.parametrize(
    ("argv", "words"),
    [
        (["--time", "14s"], ["--time", "15 s"]),
        (["--time", "0s"], ["--time", "15 s"]),
        (["--time", "181m"], ["--time", "3h"]),
        (["--time", "4h"], ["--time", "3h"]),
        (["--deep", "--time", "20m"], ["--deep", "--time"]),
        (["--workers", "0"], ["--workers", "1 to 16"]),
        (["--workers", "17"], ["--workers", "1 to 16"]),
        (["--workers", "x"], ["--workers", "1 to 16"]),
        (["--workers", "2.5"], ["--workers", "1 to 16"]),
        (["--effort", "fast"], ["--effort", "low, medium, high, xhigh, max, default"]),
        (["--task-effort", "LOW"], ["--task-effort"]),
        (["--judge-effort", ""], ["--judge-effort"]),
        (["--reflect-effort", "none"], ["--reflect-effort"]),
        (["--merge"], ["--merge", "deep"]),
        (["--trust-search"], ["--trust-search", "deep"]),
        (["--force-low-budget"], ["--force-low-budget", "deep"]),
        (["--time", "9m", "--merge"], ["--merge", "deep"]),
    ],
)
def test_a_bad_value_or_combination_is_a_usage_error_naming_the_flag(argv, words):
    with pytest.raises(UsageError) as refused:
        chosen(*argv)
    assert all(word in str(refused.value) for word in words), str(refused.value)


@pytest.mark.parametrize("flag", ["--merge", "--trust-search", "--force-low-budget"])
def test_the_deep_only_flags_are_accepted_with_the_deep_tier(flag):
    assert chosen("--deep", flag).tier == "deep"
    assert chosen("--time", "10m", flag).tier == "deep"


def test_workers_from_1_to_16():
    assert chosen("--workers", "1").workers == 1
    assert chosen("--workers", "16").workers == 16


@pytest.mark.parametrize(
    ("argv", "efforts"),
    [
        (["--effort", "high"], Efforts("high", "high", "high")),
        (["--effort", "high", "--judge-effort", "low"], Efforts("high", "low", "high")),
        (["--task-effort", "max"], Efforts("max", "low", "low")),
        (["--reflect-effort", "xhigh"], Efforts("low", "low", "xhigh")),
        (["--effort", "default"], Efforts()),
        (["--task-effort", "default"], Efforts(None, "low", "low")),
        (["--deep", "--effort", "medium"], Efforts("medium", "medium", "medium")),
        (["--deep", "--judge-effort", "low"], Efforts(None, "low", None)),
        (["--effort", "low", "--task-effort", "default"], Efforts(None, "low", "low")),
    ],
)
def test_a_given_effort_flag_wins_over_the_tiers_default_a_role_flag_over_effort(argv, efforts):
    assert chosen(*argv).efforts == efforts


def test_a_given_model_flag_wins_over_the_tiers_default():
    fast = chosen("--reflect-model", "opus", "--task-model", "sonnet", "--target-model", "haiku")
    assert (fast.models.reflect, fast.models.task, fast.models.target) == (
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-4-5-20251001",
    )
    assert chosen("--deep", "--reflect-model", "sonnet").models.reflect == "claude-sonnet-5-5"


def test_a_target_that_is_the_default_judge_gets_the_fallback_judge_in_every_tier():
    for argv in ([], ["--deep"], ["--time", "15s"]):
        assert chosen(*argv, "--target-model", "opus").models.judge == "claude-sonnet-5-5"


def test_the_judge_rule_names_the_flag_in_every_tier():
    for argv in ([], ["--deep"]):
        with pytest.raises(UsageError, match="--judge-model"):
            chosen(*argv, "--judge-model", "haiku")


def test_parse_takes_any_value_and_settings_checks_it():
    """The values of --time, --workers and the effort flags are checked after parsing, as the
    model names are, so every error names its flag the same way."""
    opts = parse(["--time", "x", "--workers", "y", "--effort", "z", "p"])
    assert {"time", "workers", "effort"} <= opts.given
    with pytest.raises(UsageError, match="--time"):
        settings(opts)


@pytest.mark.parametrize(
    ("tier", "seconds", "calls"),
    [("deep", 600, 50), ("deep", 1200, 100), ("deep", 10800, 100), ("deep", 120, 20)],
)
def test_the_deep_budget_is_one_call_per_12_seconds_from_20_to_100(tier, seconds, calls):
    assert budget(parse(["p"]), tier, seconds, est_calls=0) == calls


def test_the_fast_budget_is_three_times_the_estimate_at_most_the_ceiling_and_a_flag_wins():
    assert budget(parse(["p"]), "fast", 30, est_calls=22) == 66
    assert budget(parse(["p"]), "checked", 540, est_calls=150) == 300
    assert budget(parse(["--budget", "7", "p"]), "fast", 30, est_calls=22) == 7
    assert budget(parse(["--budget", "7", "p"]), "deep", 1200, est_calls=0) == 7
