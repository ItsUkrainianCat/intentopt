"""The strictness of the time tiers (SPEC R25 "Quality of the rewrites"; ADR-011 amendment of
2026-10-06): the quick, fast and checked tiers default to `balanced`, the deep tier keeps
`conservative`, a `--strictness` given always wins, the plan carries the level and `--dry` shows
it with its length cap."""

import json

import pytest

from autoimprover import cli


def dry(capsys, *argv: str) -> dict:
    assert cli.main(["--dry", "--json", *argv, "Summarise the notes."]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("argv", "level"),
    [
        ((), "balanced"),
        (("--time", "15s"), "balanced"),
        (("--time", "2m"), "balanced"),
        (("--deep",), "conservative"),
        (("--strictness", "conservative"), "conservative"),
        (("--strictness", "bold", "--deep"), "bold"),
        (("--time", "15s", "--strictness", "conservative"), "conservative"),
    ],
)
def test_the_tier_sets_the_strictness_a_flag_given_wins(capsys, argv, level):
    assert dry(capsys, *argv)["plan"]["strictness"] == level


def test_dry_shows_the_level_and_its_length_cap(capsys):
    assert cli.main(["--dry", "Summarise the notes."]) == 0
    out = capsys.readouterr().out
    assert "strictness: balanced, length cap 1.5x the original's tokens" in out
