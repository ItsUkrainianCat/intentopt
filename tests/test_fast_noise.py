"""The noise of the fast tiers (SPEC R12 in spirit, R25): stage B runs the original twice (samples 0
and 1, so two calls, never a cache hit), stage C judges each run; the noise is the difference of
the two runs' mean scores, the baseline their mean, and a rewrite must beat the baseline by MORE
than max(FAST_MARGIN, 2 x noise) and win more scenarios than it loses against the baseline's
per-scenario mean. And a rewrite that changes no meaning word of the original (case, punctuation,
whitespace, single letters, articles) is dropped like an identical one, before it costs a call.

The judge passes the first n of the 10 judged checks of a scenario (8 from the contract, 2 from
the example's criteria), n read from a table by prompt (O and O1 are the original's two runs) and
scenario, so every score is a tenth. Every "not returned" test has a twin that is.
"""

import pytest
from fakes import intake_reply
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2,
    K1M2_EXAMPLES,
    MODELS,
    PROMPT,
    QUICK,
    World,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    runs_of,
    scenario_of,
    scoring_judges,
)

from autoimprover.fast_stages import FAST_MARGIN, meaning_words
from autoimprover.types import BackendError, Call, CallError, Scenario

CHECKS = [
    {"id": f"c{n}", "group": "content", "text": f"check {n}", "rule": None, "arg": None}
    for n in range(1, 9)
]
SENT = [f"c:c{n}" for n in range(1, 9)] + ["s:crit-1", "s:crit-2"]
EXAMPLES = [
    Scenario(id=f"e{n}", input=f"example {n}", criteria=("criterion one", "criterion two"))
    for n in (1, 2)
]
A = "Answer the user's request now. [A]"


def tag(call: Call) -> str:
    """O and O1 for the original's two runs, A for the rewrite."""
    if "[A]" in prompt_of(call):
        return "A"
    return "O1" if call.sample == 1 else "O"


def table_world(passed: dict[str, tuple[int, int]]) -> World:
    """`passed[tag]`: how many checks the outputs of that run pass on e1 and e2."""

    def task(call: Call) -> str:
        return f"answer {tag(call)} on e{scenario_of(call).split()[-1]}"

    def passes(scenario: str, check_id: str, output: str) -> bool:
        return SENT.index(check_id) < passed[output.split()[1]][int(scenario[1:]) - 1]

    return World(rewrites=(A,), intake=intake_reply(checks=CHECKS), task=task, passes=passes)


def outcome(tmp_path, passed):
    return run(tmp_path, table_world(passed), K1M2_EXAMPLES, examples=EXAMPLES).outcome


def test_the_original_runs_twice_as_two_calls_the_rewrite_once(tmp_path):
    result = run(tmp_path, World(), K1M2)
    assert runs_of(result) == [(PROMPT, 0)] * 2 + [(PROMPT, 1)] * 2 + [(BETTER, 0)] * 2
    assert [c.sample for c in scoring_judges(result)] == [0, 1, 0]
    assert result.outcome.noise == 0.0


@pytest.mark.parametrize(
    ("runs", "rewrite", "returned", "noise"),
    [
        (((5, 5), (5, 5)), (7, 6), True, 0.0),  # +0.15 > max(0.1, 0)
        (((5, 5), (5, 5)), (6, 6), False, 0.0),  # +0.1: not MORE than 0.1 (0.6 - 0.5 in floats)
        (((5, 5), (6, 6)), (8, 8), True, 0.1),  # baseline 0.55, +0.25 > 2 x 0.1
        (((5, 5), (6, 6)), (8, 7), False, 0.1),  # +0.2: not more than 2 x 0.1
        (((5, 5), (6, 6)), (7, 7), False, 0.1),  # +0.15: enough without noise, not with it
        (((5, 5), (5, 6)), (8, 6), True, 0.05),  # 2 x 0.05 = 0.1 is the bar: +0.15 clears it
        (((5, 5), (5, 5)), (10, 3), False, 0.0),  # +0.15, but wins 1 and loses 1
    ],
)
def test_a_rewrite_must_beat_the_baseline_by_more_than_the_margin_or_twice_the_noise(
    tmp_path, runs, rewrite, returned, noise
):
    result = outcome(tmp_path, {"O": runs[0], "O1": runs[1], "A": rewrite})
    assert (result.prompt == A) is returned
    baseline = (sum(runs[0]) + sum(runs[1])) / 40
    assert result.noise == pytest.approx(noise) and result.score_before == pytest.approx(baseline)
    if returned:
        gain = sum(rewrite) / 20 - baseline
        assert result.score_after == pytest.approx(sum(rewrite) / 20)
        assert result.margin == pytest.approx(gain - max(FAST_MARGIN, 2 * noise))
    else:
        assert result.reason_code == "no_reliable_improvement"


def test_wins_and_losses_count_against_the_baselines_mean_per_scenario(tmp_path):
    """Baseline per scenario (0.5, 0.65): the rewrite's (0.8, 0.6) wins e1 and loses e2, so it
    is not returned though it gains 0.125 on the mean; (0.8, 0.7) wins both and is."""
    lost = outcome(tmp_path / "lost", {"O": (5, 6), "O1": (5, 7), "A": (8, 6)})
    won = outcome(tmp_path / "won", {"O": (5, 6), "O1": (5, 7), "A": (8, 7)})
    assert (lost.prompt, won.prompt) == ("Answer the user's request.", A)


def test_a_fast_result_says_its_noise_is_measured_and_it_is_not_held_out(tmp_path):
    reason = outcome(tmp_path, {"O": (5, 5), "O1": (5, 5), "A": (8, 8)}).reason
    assert "noise measured from two runs of the original" in reason
    assert "not verified on held-out scenarios" in reason and "no noise" not in reason


@pytest.mark.parametrize("run_index", [0, 1])
def test_an_original_run_whose_task_calls_all_fail_ends_the_run(tmp_path, run_index):
    def task(call: Call) -> str:
        if prompt_of(call) == PROMPT and call.sample == run_index:
            raise CallError("down")
        return "GOOD answer" if "[[better]]" in prompt_of(call) else "BAD answer"

    with pytest.raises(BackendError, match="every task run of one of the original's two runs"):
        run(tmp_path, World(task=task), K1M2)  # before stage C spends a call


@pytest.mark.parametrize("fails", [False, True])
def test_an_original_run_whose_judge_call_fails_ends_the_run(tmp_path, fails):
    def hook(call: Call) -> None:
        if fails and call.role == "judge" and call.sample == 1 and "BAD answer" in call.user:
            raise CallError("down")

    if not fails:
        assert run(tmp_path, World(hook=hook), K1M2).outcome.prompt == BETTER
        return
    with pytest.raises(BackendError, match="original"):
        run(tmp_path, World(hook=hook), K1M2)


def test_the_checked_tier_measures_noise_where_it_picks_and_holds_out_the_original_once(tmp_path):
    result = run(tmp_path, World(), CHECKED)
    picked = [(prompt_of(c), c.sample) for c in result.calls("task") if c.model == MODELS.task]
    assert sorted(picked) == sorted([(PROMPT, 0)] * 2 + [(PROMPT, 1)] * 2 + [(BETTER, 0)] * 2)
    on_target = [(prompt_of(c), c.sample) for c in result.calls("task") if c.model == MODELS.target]
    assert sorted(on_target) == sorted([(PROMPT, 0)] * 4 + [(BETTER, 0)] * 4)
    assert (result.outcome.verified, result.outcome.noise) == (True, None)  # no held-out noise


def test_the_quick_tier_stays_unscored_and_measures_no_noise(tmp_path):
    result = run(tmp_path, World(), QUICK)
    assert result.calls("task") == [] and result.outcome.noise is None


# --- a rewrite that changes no meaning word -------------------------------------------------------

REAL = "what do i need to make prompt improver app"


@pytest.mark.parametrize(
    ("original", "rewrite", "same"),
    [
        (REAL, "what do i need to make a prompt improver app?", True),  # the live run's rewrite
        (REAL, "What do I need to make the prompt-improver app", True),
        (REAL, "  what, do i need... to make an prompt improver app!  ", True),
        (REAL, "what do I need to make x prompt improver app", True),  # one more single letter
        (REAL, "what do i need to make my prompt improver app", False),  # "my" is a word
        (REAL, "what do i need to make prompt improver apps", False),
        (REAL, "what do i need to make prompt app improver", False),  # the order changed
        ("Summarise in 5 bullets.", "summarise in 6 bullets", False),  # a digit is meaning
        ("Fasse den Text zusammen.", "fasse den Text zusammen", True),
        ("Fasse den Text zusammen.", "Fasse den ganzen Text zusammen.", False),
    ],
)
def test_meaning_words_ignore_case_punctuation_single_letters_and_articles(original, rewrite, same):
    assert (meaning_words(original) == meaning_words(rewrite)) is same


@pytest.mark.parametrize(
    ("rewrite", "dropped"),
    [
        ("what do i need to make a prompt improver app?", True),
        ("what do i need to build a prompt improver app?", False),
    ],
)
def test_a_rewrite_with_no_change_in_meaning_words_is_dropped_before_any_call(
    tmp_path, rewrite, dropped
):
    result = run(tmp_path, World(rewrites=(rewrite,)), K1M2, prompt=REAL)
    assert ("rewrite 0 dropped: no change in meaning words" in result.log) is dropped
    assert any(prompt_of(c) == rewrite for c in result.calls("task")) is not dropped
    if dropped:
        assert result.outcome.reason_code == "no_reliable_improvement"
