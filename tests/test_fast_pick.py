"""Stage D's pick (SPEC R25): a rewrite is returned only when it beats the original's mean judged
score by at least 0.1 AND wins on more scenarios than it loses, compared on the scenarios both
completed; the highest gain wins and a tie goes to the shorter rewrite, then to the earlier one.

The judge passes the first n of the 10 judged checks of a scenario (8 from the contract, 2 from the
example's criteria), n read from a table by prompt and scenario, so every score is a tenth.
"""

import dataclasses

import pytest
from fakes import MARKER, intake_reply
from test_fast_world import (  # noqa: F401  (no_disk_flush is an autouse fixture)
    BETTER,
    CHECKED,
    K1M2_EXAMPLES,
    K3M2_EXAMPLES,
    MODELS,
    PROMPT,
    World,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
)

from autoimprover.fastplan import FastPlan
from autoimprover.types import Call, CallError, Scenario

CHECKS = [
    {"id": f"c{n}", "group": "content", "text": f"check {n}", "rule": None, "arg": None}
    for n in range(1, 9)
]
SENT = [f"c:c{n}" for n in range(1, 9)] + ["s:crit-1", "s:crit-2"]
EXAMPLES = [
    Scenario(id=f"e{n}", input=f"example {n}", criteria=("criterion one", "criterion two"))
    for n in (1, 2, 3)
]
A = "Answer the user's request. [A]"
B = "Answer the user's request. [B]"
LONG_B = "Answer the user's request carefully. [B]"
SHORT_C = "Answer the request. [C]"


def tag(text: str) -> str:
    return next((t for t in "ABC" if f"[{t}]" in text), "O")


TEXTS = {"A": A, "B": LONG_B, "C": SHORT_C}


def world(
    passed: dict[str, tuple[int, ...]],
    fails: tuple[tuple[str, str], ...] = (),
    rewrites: list[str] | None = None,
) -> World:
    """`passed[tag]` is how many checks the outputs of that prompt pass, per example in order (O
    is the original); the rewrites are those of the other tags, in order, unless given; `fails`
    lists the (tag, example id) task runs that fail every attempt."""

    def task(call: Call) -> str:
        scenario = "e" + scenario_of(call).split()[-1]
        if (tag(prompt_of(call)), scenario) in fails:
            raise CallError("down")
        return f"answer {tag(prompt_of(call))} on {scenario}"

    def passes(scenario: str, check_id: str, output: str) -> bool:
        return SENT.index(check_id) < passed[output.split()[1]][int(scenario[1:]) - 1]

    texts = rewrites or [TEXTS[t] for t in passed if t != "O"]
    return World(rewrites=texts, intake=intake_reply(checks=CHECKS), task=task, passes=passes)


def pick(tmp_path, passed, fplan: FastPlan = K1M2_EXAMPLES, fails=()):
    return run(tmp_path, world(passed, fails), fplan, examples=EXAMPLES).outcome


@pytest.mark.parametrize(
    ("rewrite", "returned"),
    [
        ((6, 6), True),  # +0.1 exactly (0.6 - 0.5 is 0.0999... in floating point), wins 2
        ((6, 5), False),  # +0.05: below the margin, though it wins 1 and loses none
        ((7, 5), True),  # +0.1, wins 1, loses none: a tie counts for neither side
        ((9, 3), False),  # +0.1, but wins 1 and loses 1
    ],
)
def test_the_margin_and_the_win_count_at_their_boundaries(tmp_path, rewrite, returned):
    outcome = pick(tmp_path, {"O": (5, 5), "A": rewrite})
    assert (outcome.prompt == A) is returned
    assert outcome.score_before == pytest.approx(0.5)
    if returned:
        after = sum(rewrite) / 20
        assert outcome.score_after == pytest.approx(after)
        assert outcome.margin == pytest.approx(after - 0.5 - 0.1)
    else:
        assert outcome.reason_code == "no_reliable_improvement"


THREE = dataclasses.replace(K1M2_EXAMPLES, scenarios=3)


@pytest.mark.parametrize(
    ("rewrite", "returned"),
    [
        ((7, 7, 4), True),  # +0.1, wins 2, loses 1
        ((10, 4, 4), False),  # +0.1, wins 1, loses 2
        ((5, 5, 8), True),  # +0.1, wins 1, two ties
    ],
)
def test_more_wins_than_losses_over_three_scenarios(tmp_path, rewrite, returned):
    outcome = pick(tmp_path, {"O": (5, 5, 5), "A": rewrite}, THREE)
    assert (outcome.prompt == A) is returned


def test_the_highest_gain_wins_even_when_longer(tmp_path):
    passed = {"O": (5, 5), "A": (7, 5), "B": (10, 10), "C": (6, 6)}
    assert pick(tmp_path, passed, K3M2_EXAMPLES).prompt == LONG_B


def test_a_tie_goes_to_the_shorter_rewrite(tmp_path):
    passed = {"O": (5, 5), "A": (9, 9), "B": (9, 9), "C": (9, 9)}
    assert pick(tmp_path, passed, K3M2_EXAMPLES).prompt == SHORT_C


def test_a_tie_of_equal_length_goes_to_the_earlier_rewrite(tmp_path):
    ab = world({"O": (5, 5), "A": (9, 9), "B": (9, 9)}, rewrites=[A, B])
    assert run(tmp_path, ab, K3M2_EXAMPLES, examples=EXAMPLES).outcome.prompt == A


@pytest.mark.parametrize(
    ("fails", "returned"),
    [
        ((), False),  # (7, 1) against (5, 5): 0.4 against 0.5
        ((("A", "e2"),), True),  # the rewrite's e2 never ran: 0.7 against 0.5 on e1
        ((("O", "e2"),), True),  # the original's e2 never ran: the same comparison
    ],
)
def test_the_comparison_uses_only_the_scenarios_both_completed(tmp_path, fails, returned):
    outcome = pick(tmp_path, {"O": (5, 5), "A": (7, 1)}, fails=fails)
    assert (outcome.prompt == A) is returned
    if returned:
        assert (outcome.score_before, outcome.score_after) == pytest.approx((0.5, 0.7))


@dataclasses.dataclass
class InvalidFirstContractReply(World):
    """The first reply to a judge call that holds the contract scenario is not JSON."""

    def __call__(self, call: Call) -> str | Exception:
        if call.role == "judge" and call.sample == 0 and '"contract"' in call.user:
            return "not json"
        return super().__call__(call)


def test_an_invalid_judge_reply_is_asked_again_as_a_new_sample(tmp_path):
    """The evaluator's rule (ADR-008), kept for the call that carries the contract check."""
    retried = InvalidFirstContractReply(**vars(world({"O": (5, 5), "A": (7, 7)})))
    result = run(tmp_path, retried, K1M2_EXAMPLES, examples=EXAMPLES)
    judges = result.calls("judge")
    assert [c.sample for c in judges] == [0, 0, 1000] and result.outcome.prompt == A
    assert (judges[2].user, judges[2].system) == (judges[1].user, judges[1].system)


def test_the_original_is_never_shown_its_prompt(tmp_path):
    result = run(tmp_path, world({"O": (5, 5), "A": (7, 7)}), K1M2_EXAMPLES, examples=EXAMPLES)
    assert PROMPT not in result.calls("judge")[0].user


@pytest.mark.parametrize(("passed", "returned"), [(1, False), (2, True)])
def test_stage_e_needs_more_than_the_threshold_on_the_target(tmp_path, passed, returned):
    """On the target the winner passes `passed` of 5 checks on one of the 4 held-out scenarios
    and nothing else: a mean of 0.05 against 0.0 is not more than MIN_THRESHOLD, 0.1 is."""
    five = [dict(check, id=f"c{n}") for n, check in enumerate(CHECKS[:5], start=1)]

    def task(call: Call) -> str:
        return f"{'GOOD' if MARKER in prompt_of(call) else 'BAD'} on {call.model}"

    def passes(scenario: str, check_id: str, output: str) -> bool:
        if output.startswith("BAD"):
            return False
        if MODELS.task in output:
            return True
        return scenario == "s3" and int(check_id[3:]) <= passed

    world = World(intake=intake_reply(checks=five), task=task, passes=passes)
    outcome = run(tmp_path, world, CHECKED).outcome
    assert (outcome.prompt == BETTER, outcome.verified) == (returned, returned)
    assert outcome.score_before == 0.0
