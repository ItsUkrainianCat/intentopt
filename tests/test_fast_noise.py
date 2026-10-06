"""The noise of the fast tiers (SPEC R12 in spirit, R25; ADR-012): stage B runs the original twice
(samples 0 and 1, so two calls, never a cache hit), and stage C compares the two runs' answers
with the pairwise judge in both orders; the noise is the number of scenarios where they have an
agreed winner. A rewrite, compared with the original's run 0, must win more scenarios than it loses
by more than that noise. And a rewrite that changes no meaning word of the original (case,
punctuation, whitespace, single letters, articles) is dropped like an identical one, before it
costs a call.

The judge prefers the answer of higher quality, read from a table by run (O and O1 are the
original's two runs, A the rewrite) and scenario, and calls equal qualities a tie. Every "not
returned" test has a twin that is.
"""

import dataclasses
import json

import pytest
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2,
    K1M2_EXAMPLES,
    MODELS,
    PROMPT,
    QUICK,
    World,
    is_pairwise,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    runs_of,
    scenario_of,
)

from autoimprover.fast_stages import meaning_words
from autoimprover.types import BackendError, Call, CallError, Scenario

EXAMPLES = [Scenario(id=f"e{n}", input=f"example {n}") for n in (1, 2, 3, 4)]
FOUR = dataclasses.replace(K1M2_EXAMPLES, scenarios=4)  # K=1 on the four examples
A = "Answer the user's request now. [A]"


def tag(call: Call) -> str:
    """O and O1 for the original's two runs, A for the rewrite."""
    if "[A]" in prompt_of(call):
        return "A"
    return "O1" if call.sample == 1 else "O"


def table_world(quality: dict[str, tuple[int, int, int, int]]) -> World:
    """`quality[tag]`: how good that run's answers are on e1 to e4; the higher wins a pair."""

    def task(call: Call) -> str:
        return f"{tag(call)} on {scenario_of(call).split()[-1]}"

    def judge(call: Call) -> str:
        results = []
        for item in json.loads(call.user)["scenarios"]:
            n = int(item["scenario"][1:]) - 1
            a, b = (quality[item[key].split()[0]][n] for key in ("answer_A", "answer_B"))
            winner = "A" if a > b else "B" if b > a else "tie"
            results.append({"scenario": item["scenario"], "winner": winner, "reason": "r"})
        return json.dumps({"results": results})

    return World(rewrites=(A,), task=task, pairwise=judge)


def outcome(tmp_path, quality):
    return run(tmp_path, table_world(quality), FOUR, examples=EXAMPLES).outcome


def test_the_original_runs_twice_as_two_calls_the_rewrite_once(tmp_path):
    """Each answer names its run's sample, so the noise pair's two orders are two calls (with
    equal answers they would be one call, and the cache would answer the second)."""

    def task(call: Call) -> str:
        return f"{'GOOD' if '[[better]]' in prompt_of(call) else 'BAD'} answer {call.sample}"

    result = run(tmp_path, World(task=task), K1M2)
    assert runs_of(result) == [(PROMPT, 0)] * 2 + [(PROMPT, 1)] * 2 + [(BETTER, 0)] * 2
    pairwise = [c for c in result.calls("judge") if is_pairwise(c)]
    assert len(pairwise) == 2 + 2  # the noise pair and the rewrite's pair, each in both orders
    assert (result.outcome.noise, result.outcome.prompt) == (0.0, BETTER)


SAME = (1, 1, 1, 1)


@pytest.mark.parametrize(
    ("original", "noisy", "rewrite", "wins", "losses", "noise"),
    [
        (SAME, SAME, (2, 1, 1, 1), 1, 0, 0),  # a lead of 1 over no noise
        (SAME, SAME, SAME, 0, 0, 0),  # ties only
        (SAME, (2, 1, 1, 1), (2, 1, 1, 1), 1, 0, 1),  # a lead of 1 is not more than 1
        (SAME, (2, 1, 1, 1), (2, 2, 1, 1), 2, 0, 1),  # 2 - 0 > 1
        (SAME, SAME, (2, 2, 0, 1), 2, 1, 0),  # more wins than losses
        (SAME, SAME, (2, 0, 1, 1), 1, 1, 0),  # as many losses as wins
        (SAME, (0, 2, 1, 1), (2, 2, 2, 1), 3, 0, 2),  # noise in both directions counts: 3 > 2
        (SAME, (0, 2, 1, 1), (2, 2, 1, 1), 2, 0, 2),  # 2 is not more than 2
    ],
)
def test_a_rewrite_must_lead_the_original_by_more_scenarios_than_the_noise(
    tmp_path, original, noisy, rewrite, wins, losses, noise
):
    result = outcome(tmp_path, {"O": original, "O1": noisy, "A": rewrite})
    returned = wins - losses > noise
    assert (result.prompt == A) is returned and result.noise == noise / 4
    if returned:
        assert (result.score_before, result.score_after) == (losses / 4, wins / 4)
        assert result.margin == pytest.approx((wins - losses - noise) / 4)
    else:
        assert result.reason_code == "no_reliable_improvement"


def test_a_fast_result_says_its_noise_is_measured_and_it_is_not_held_out(tmp_path):
    reason = outcome(tmp_path, {"O": SAME, "O1": SAME, "A": (2, 2, 2, 2)}).reason
    assert "noise measured by comparing the original with itself" in reason
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
def test_a_noise_pair_the_judge_cannot_compare_ends_the_run(tmp_path, fails):
    """The noise pair is the one whose answers are all the original's (BAD) on both sides."""

    def hook(call: Call) -> None:
        if fails and is_pairwise(call) and "GOOD" not in call.user:
            raise CallError("down")

    if not fails:
        assert run(tmp_path, World(hook=hook), K1M2).outcome.prompt == BETTER
        return
    with pytest.raises(BackendError, match="could not compare the original's two runs"):
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
