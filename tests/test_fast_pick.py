"""Stage D's pick (SPEC R25 "Decision by pairwise preference"; ADR-012): a rewrite is returned
only when it kept the contract and the pairwise judge, both orders agreeing, gives it more of the
scenarios both runs completed than the original's run 0, by more scenarios than the noise; the
largest lead wins and a tie goes to the shorter rewrite, then to the earlier one. The original's
two runs agree here (no noise); the noise cases are in test_fast_noise.py.

The judge prefers the answer of higher quality, read from a table by prompt and scenario (O is the
original), and calls equal qualities a tie.
"""

import dataclasses
import json

import pytest
from fakes import MARKER, intake_reply
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2_EXAMPLES,
    K3M2_EXAMPLES,
    MODELS,
    PROMPT,
    World,
    is_contract_check,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
    scoring_judges,
)

from autoimprover.fastplan import FastPlan
from autoimprover.types import Call, CallError, Scenario

CHECKS = [
    {"id": f"c{n}", "group": "content", "text": f"check {n}", "rule": None, "arg": None}
    for n in range(1, 9)
]
EXAMPLES = [Scenario(id=f"e{n}", input=f"example {n}") for n in (1, 2, 3)]
A = "Answer the user's request now. [A]"  # "now": a rewrite needs a new meaning word
B = "Answer the user's request here. [B]"
LONG_B = "Answer the user's request carefully. [B]"
SHORT_C = "Answer the request. [C]"


def tag(text: str) -> str:
    return next((t for t in "ABC" if f"[{t}]" in text), "O")


TEXTS = {"A": A, "B": LONG_B, "C": SHORT_C}


def world(
    quality: dict[str, tuple[int, ...]],
    fails: tuple[tuple[str, str], ...] = (),
    rewrites: list[str] | None = None,
) -> World:
    """`quality[tag]` is how good the answers of that prompt are, per example in order (O is the
    original); the rewrites are those of the other tags, in order, unless given; `fails` lists the
    (tag, example id) task runs that fail every attempt."""

    def task(call: Call) -> str:
        scenario = "e" + scenario_of(call).split()[-1]
        if (tag(prompt_of(call)), scenario) in fails:
            raise CallError("down")
        return f"answer {tag(prompt_of(call))} on {scenario}"

    def judge(call: Call) -> str:
        results = []
        for item in json.loads(call.user)["scenarios"]:
            n = int(item["scenario"][1:]) - 1
            a, b = (quality[item[key].split()[1]][n] for key in ("answer_A", "answer_B"))
            winner = "A" if a > b else "B" if b > a else "tie"
            results.append({"scenario": item["scenario"], "winner": winner, "reason": "r"})
        return json.dumps({"results": results})

    texts = rewrites or [TEXTS[t] for t in quality if t != "O"]
    return World(rewrites=texts, intake=intake_reply(checks=CHECKS), task=task, pairwise=judge)


def pick(tmp_path, quality, fplan: FastPlan = K1M2_EXAMPLES, fails=()):
    return run(tmp_path, world(quality, fails), fplan, examples=EXAMPLES).outcome


@pytest.mark.parametrize(
    ("rewrite", "wins", "losses"),
    [
        ((5, 5), 0, 0),  # ties only
        ((6, 5), 1, 0),  # a lead of one scenario, a tie counting for neither side
        ((6, 4), 1, 1),  # as many losses as wins
        ((4, 4), 0, 2),
        ((6, 6), 2, 0),
    ],
)
def test_the_lead_and_the_win_count_at_their_boundaries(tmp_path, rewrite, wins, losses):
    outcome = pick(tmp_path, {"O": (5, 5), "A": rewrite})
    returned = wins > losses
    assert (outcome.prompt == A) is returned and outcome.noise == 0.0
    if returned:
        assert (outcome.score_before, outcome.score_after) == (losses / 2, wins / 2)
        assert outcome.margin == pytest.approx((wins - losses) / 2)
    else:
        assert outcome.reason_code == "no_reliable_improvement"


THREE = dataclasses.replace(K1M2_EXAMPLES, scenarios=3)


@pytest.mark.parametrize(
    ("rewrite", "returned"),
    [
        ((6, 6, 4), True),  # wins 2, loses 1
        ((6, 4, 4), False),  # wins 1, loses 2
        ((5, 5, 6), True),  # wins 1, two ties
    ],
)
def test_more_wins_than_losses_over_three_scenarios(tmp_path, rewrite, returned):
    outcome = pick(tmp_path, {"O": (5, 5, 5), "A": rewrite}, THREE)
    assert (outcome.prompt == A) is returned


def test_the_largest_lead_wins_even_when_longer(tmp_path):
    quality = {"O": (5, 5), "A": (6, 4), "B": (6, 6), "C": (6, 5)}
    assert pick(tmp_path, quality, K3M2_EXAMPLES).prompt == LONG_B


def test_a_tie_goes_to_the_shorter_rewrite(tmp_path):
    quality = {"O": (5, 5), "A": (6, 6), "B": (6, 6), "C": (6, 6)}
    assert pick(tmp_path, quality, K3M2_EXAMPLES).prompt == SHORT_C


def test_a_tie_of_equal_length_goes_to_the_earlier_rewrite(tmp_path):
    ab = world({"O": (5, 5), "A": (6, 6), "B": (6, 6)}, rewrites=[A, B])
    assert run(tmp_path, ab, K3M2_EXAMPLES, examples=EXAMPLES).outcome.prompt == A


@pytest.mark.parametrize(
    ("fails", "returned"),
    [
        ((), False),  # (6, 4) against (5, 5): one win, one loss
        ((("A", "e2"),), True),  # the rewrite's e2 never ran: one win on e1
        ((("O", "e2"),), True),  # the original's e2 never ran: the same comparison
    ],
)
def test_the_comparison_uses_only_the_scenarios_both_completed(tmp_path, fails, returned):
    outcome = pick(tmp_path, {"O": (5, 5), "A": (6, 4)}, fails=fails)
    assert (outcome.prompt == A) is returned
    if returned:  # shares of the 2 scenarios picked on
        assert (outcome.score_before, outcome.score_after) == (0.0, 0.5)


@dataclasses.dataclass
class InvalidFirstReplies(World):
    """The first reply to the contract check and to each of the rewrite's pairwise calls is not
    JSON."""

    def __call__(self, call: Call) -> str | Exception:
        first = call.role == "judge" and call.sample == 0
        if first and ('"contract-1"' in call.user or "answer A" in call.user):
            return "not json"
        return super().__call__(call)


def test_an_invalid_judge_reply_is_asked_again_as_a_new_sample(tmp_path):
    """`contract._ask`'s rule for the contract check and the pairwise calls alike (sample + 1;
    ADR-008), with the same messages. The original's two runs answer alike, so the noise pair's
    two orders are one call."""
    retried = InvalidFirstReplies(**vars(world({"O": (5, 5), "A": (6, 6)})))
    result = run(tmp_path, retried, K1M2_EXAMPLES, examples=EXAMPLES)
    judges = result.calls("judge")
    # the contract check twice, the noise pair once, each order of the rewrite's pair twice
    assert [c.sample for c in judges] == [0, 1, 0, 0, 1, 0, 1] and result.outcome.prompt == A
    for first, again in ((0, 1), (3, 4), (5, 6)):
        assert (judges[again].user, judges[again].system) == (
            judges[first].user,
            judges[first].system,
        )


def test_a_pairwise_call_sees_the_original_prompt_as_the_request_and_never_a_rewrite(tmp_path):
    """ADR-002, ADR-012: only the contract check, a veto, sees a rewrite's text."""
    result = run(tmp_path, world({"O": (5, 5), "A": (6, 6)}), K1M2_EXAMPLES, examples=EXAMPLES)
    grading = scoring_judges(result)
    assert len(grading) == 1 + 2  # the noise pair's two orders are one call, the rewrite's two
    assert all(json.loads(c.user)["request"] == PROMPT and A not in c.user for c in grading)
    assert [A in c.user for c in result.calls("judge") if is_contract_check(c)] == [True]


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
