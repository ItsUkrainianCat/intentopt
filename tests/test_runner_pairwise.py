"""The deep tier's final decision without references (SPEC R25; ADR-014): the finalists are run on
the holdout on the target model and judged against the original by the blind pairwise judge, in
both orders, against the noise of the original's two runs. The search is a stand-in, as in
`test_runner_prompts.py`, whose helpers this file shares; with references the absolute rule of
SPEC R3 and R12 holds (tested there)."""

import json
from collections.abc import Callable

import pytest
from fakes import ScriptedBackend
from test_runner_prompts import (
    HOLDOUT,
    MODELS,
    PROMPT,
    A,
    Searched,
    cand,
    decide,
    found,
    model,
    ran,
)

from autoimprover.pairwise_text import PAIRWISE_BATCH_SYSTEM
from autoimprover.types import BackendError, Call, CallError

MARK = "pairwise judge on the holdout"


def pairwise(call: Call) -> bool:
    return call.system == PAIRWISE_BATCH_SYSTEM


def good_on(original_run1: int, finalist: int) -> Callable[[Call], str]:
    """A scripted model whose target-model answers are good for the first `original_run1`
    holdout scenarios in the original's second run and for the first `finalist` ones in A's run;
    the original's first run is never good."""

    def inputs(count: int) -> set[str]:
        return {scenario.input for scenario in HOLDOUT[:count]}

    healthy = model()

    def script(call: Call) -> str:
        if call.role == "task" and call.model == MODELS.target:
            situation = call.user.split("\n\n", 1)[0]
            if ran(call) == A:
                return "GOOD answer" if situation in inputs(finalist) else "BAD answer"
            good = call.sample == 1 and situation in inputs(original_run1)
            return "GOOD answer" if good else "BAD answer"
        return healthy(call)

    return script


def test_a_finalist_the_judge_prefers_is_returned_verified_and_never_scored(tmp_path, monkeypatch):
    raw = ScriptedBackend(good_on(0, 4))
    outcome = decide(tmp_path, monkeypatch, Searched(found(cand(A, 0.5))), raw)
    assert (outcome.status, outcome.prompt, outcome.reason_code) == ("improved", A, "improved")
    assert outcome.verified and MARK in outcome.reason
    # the shares of the 4 holdout scenarios the original and the finalist won, and the noise
    assert (outcome.score_before, outcome.score_after, outcome.noise) == (0.0, 1.0, 0.0)
    assert outcome.margin == pytest.approx(1.0)
    pairs = [c for c in raw.calls if pairwise(c)]
    # the original against itself, the finalist against the original, each in two orders; the two
    # orders of two identical answers are one cached call
    assert len(pairs) == 3
    for call in pairs:
        assert json.loads(call.user)["request"] == PROMPT  # the judge never sees a candidate
        assert A not in call.user and call.model == MODELS.judge
    # judge calls: the original's two holdout runs scored, A's contract check, the pairs; the
    # finalist's run is not scored
    assert sum(c.role == "judge" for c in raw.calls) == 2 + 1 + len(pairs)


@pytest.mark.parametrize(("wins", "code"), [(2, "no_reliable_improvement"), (3, "improved")])
def test_the_finalist_must_lead_by_more_scenarios_than_the_originals_two_runs_disagree_on(
    tmp_path, monkeypatch, wins, code
):
    raw = ScriptedBackend(good_on(2, wins))  # the original's runs differ on 2 of 4 scenarios
    outcome = decide(tmp_path, monkeypatch, Searched(found(cand(A, 0.5))), raw)
    assert outcome.reason_code == code and MARK in outcome.reason
    assert outcome.noise == pytest.approx(0.5)
    if code == "improved":
        assert (outcome.score_before, outcome.score_after) == (0.0, 0.75)
        assert outcome.margin == pytest.approx((3 - 0 - 2) / 4)
    else:
        assert outcome.prompt == PROMPT and outcome.margin is None


def test_a_judge_that_prefers_a_position_makes_ties_never_wins(tmp_path, monkeypatch):
    healthy = good_on(0, 4)

    def script(call: Call) -> str:
        if not pairwise(call):
            return healthy(call)
        items = json.loads(call.user)["scenarios"]
        results = [{"scenario": i["scenario"], "winner": "A", "reason": "first"} for i in items]
        return json.dumps({"results": results})

    outcome = decide(tmp_path, monkeypatch, Searched(found(cand(A, 0.5))), ScriptedBackend(script))
    assert (outcome.status, outcome.reason_code) == ("unchanged", "no_reliable_improvement")
    assert MARK in outcome.reason and outcome.noise == 0.0


def test_a_failed_pairwise_call_ends_the_run_and_is_never_a_tie(tmp_path, monkeypatch):
    healthy = good_on(0, 4)

    def script(call: Call) -> str | Exception:
        return CallError("exit 1") if pairwise(call) else healthy(call)

    with pytest.raises(BackendError):
        decide(tmp_path, monkeypatch, Searched(found(cand(A, 0.5))), ScriptedBackend(script))


def test_no_finalist_means_no_pairwise_call(tmp_path, monkeypatch):
    raw = ScriptedBackend(good_on(0, 4))
    outcome = decide(tmp_path, monkeypatch, Searched(found()), raw)
    assert outcome.reason_code == "no_reliable_improvement"
    assert not any(pairwise(c) for c in raw.calls)
    assert outcome.score_before == 0.0  # the original's holdout score, measured before
