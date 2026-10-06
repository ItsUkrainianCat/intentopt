"""The fast pipeline end to end (SPEC R25, and R6, R7, R9, R11, R14a, R24 for what it returns): the
stages, their calls, the gates every returned rewrite passed, the tiers' evidence labels. Every test
that shows a rewrite is NOT returned has a twin on the same world showing one IS, so it cannot pass
for the wrong reason (ARCHITECTURE section 3).
"""

import dataclasses
import json

import pytest
from fakes import MARKER, ScriptedBackend, intake_reply
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K1M2,
    K1M2_EXAMPLES,
    K2M2,
    K3M3,
    MODELS,
    PLAN,
    PROMPT,
    QUICK,
    Verbatim,
    World,
    is_contract_check,
    is_pairwise,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    runs_of,
    scenario_of,
    scoring_judges,
    tagged,
)

from autoimprover.bench_judge import PAIRWISE_BATCH_SYSTEM
from autoimprover.fast_prompts import FAST_TASK_SUFFIX, STRATEGY_NOTES, FastEvaluator
from autoimprover.runner import count_tokens
from autoimprover.runstore import RunStore
from autoimprover.types import BackendError, Call, CallError, Scenario

FAST_LABEL = (
    "fast check: preferred over the original by a pairwise judge on the same few scenarios it was "
    "picked on, noise measured by comparing the original with itself, not verified on held-out "
    "scenarios"
)


def test_a_rewrite_that_wins_is_returned_unverified_with_its_report(tmp_path):
    result = run(tmp_path, World(), K3M3)
    outcome = result.outcome
    assert (outcome.status, outcome.prompt, outcome.reason_code) == ("improved", BETTER, "improved")
    assert (outcome.verified, outcome.stop, outcome.changes) == (
        False,
        None,
        (STRATEGY_NOTES["clarify"],),  # variant 0 clarifies (SPEC R25)
    )
    assert "tier fast" in outcome.reason and FAST_LABEL in outcome.reason
    assert (outcome.score_before, outcome.score_after, outcome.noise) == (0.0, 1.0, 0.0)
    assert outcome.margin == pytest.approx(1.0)  # won all 3 scenarios, lost none, no noise
    assert outcome.length_ratio == count_tokens(BETTER) / count_tokens(PROMPT)
    assert outcome.calls_used == len(result.raw.calls)


def test_the_stages_make_the_calls_of_the_plan(tmp_path):
    """K=3 rewrites that all propose the same text are scored once: stage A is the intake, the
    synthesis of M=3 and three rewrite calls; then the original twice and the rewrite on 3
    scenarios; then the contract check and one judge call per run."""
    result = run(tmp_path, World(), K3M3)
    roles = [call.role for call in result.raw.calls]
    assert roles == ["intake", "synth"] + ["reflect"] * 3 + ["task"] * 9 + ["judge"] * 4
    assert "contract" not in result.calls("synth")[0].user  # it runs beside the intake
    assert json.loads(result.calls("synth")[0].user)["count"] == 3
    assert runs_of(result) == [(PROMPT, 0)] * 3 + [(PROMPT, 1)] * 3 + [(BETTER, 0)] * 3
    tasks = result.calls("task")
    assert [scenario_of(c) for c in tasks] == [f"situation {i}" for i in (1, 2, 3)] * 3


def test_a_rewrite_that_scores_no_better_leaves_the_original(tmp_path):
    outcome = run(tmp_path, World(rewrites=("Answer the request well.",)), K3M3).outcome
    assert (outcome.status, outcome.prompt) == ("unchanged", PROMPT)
    assert (outcome.reason_code, outcome.verified, outcome.stop) == (
        "no_reliable_improvement",
        False,
        None,
    )
    assert (outcome.score_before, outcome.noise) == (None, 0.0) and "tier fast" in outcome.reason


# --- the free gates (SPEC R7, R9) -----------------------------------------------------------------

LITERAL_PROMPT = "Answer the user's request about {topic}."


@pytest.mark.parametrize(
    ("rewrite", "returned"),
    [
        (f"Answer the user's request {MARKER}.", False),  # lost {topic}
        (f"Answer the user's request about {{topic}} {MARKER}.", True),
    ],
)
def test_a_rewrite_that_drops_a_literal_is_never_run_or_returned(tmp_path, rewrite, returned):
    result = run(tmp_path, World(rewrites=(rewrite,)), K1M2, prompt=LITERAL_PROMPT)
    assert (result.outcome.prompt == rewrite) is returned
    assert any(prompt_of(c) == rewrite for c in result.calls("task")) is returned
    if not returned:
        assert result.outcome.reason_code == "no_reliable_improvement"
        assert result.calls("task") == [] and result.calls("judge") == []


@pytest.mark.parametrize(("extra", "returned"), [(41, False), (40, True)])
def test_a_rewrite_over_the_length_cap_is_never_run_or_returned(tmp_path, extra, returned):
    """The original has 7 tokens, so the conservative cap is 7 + 40 = 47 tokens."""
    rewrite = f"Answer the user's request {MARKER}" + " more" * (extra - 4)  # MARKER is 5 tokens
    assert count_tokens(rewrite) == 7 + extra
    result = run(tmp_path, World(rewrites=(rewrite,)), K1M2)
    assert (result.outcome.prompt == rewrite) is returned
    assert any(prompt_of(c) == rewrite for c in result.calls("task")) is returned


def test_allow_growth_lifts_the_length_cap(tmp_path):
    rewrite = f"Answer the user's request {MARKER}" + " more" * 60
    plan = dataclasses.replace(PLAN, allow_growth=True)
    assert run(tmp_path, World(rewrites=(rewrite,)), K1M2, plan=plan).outcome.prompt == rewrite


# --- the judge sees outputs only; one contract check vetoes the rewrites (SPEC R6; ADR-002) -------


@pytest.mark.parametrize("keeps", [False, True])
def test_a_rewrite_failing_the_contract_check_is_never_returned(tmp_path, keeps):
    outcome = run(tmp_path, World(contract_ok=lambda _text: keeps), K1M2).outcome
    assert (outcome.prompt == BETTER) is keeps
    assert outcome.status == ("improved" if keeps else "unchanged")


SECOND = f"{PROMPT} {MARKER} Be brief."


def test_the_judge_compares_answers_only_and_one_contract_check_holds_every_rewrite(tmp_path):
    """The pairwise calls see the original prompt as the request and the answers, never a
    rewrite's text (ADR-002, ADR-012)."""
    result = run(tmp_path, World(rewrites=(BETTER, SECOND), task=tagged), K3M3)
    scoring = scoring_judges(result)
    assert scoring and all(call.system == PAIRWISE_BATCH_SYSTEM for call in scoring)
    for call in scoring:
        sent = json.loads(call.user)
        assert sent["request"] == PROMPT and [s["scenario"] for s in sent["scenarios"]] == [
            "s1",
            "s2",
            "s3",
        ]
        assert not any(text in call.user for text in (BETTER, SECOND))
    (contract,) = [c for c in result.calls("judge") if is_contract_check(c)]
    sent = json.loads(contract.user)["scenarios"]
    assert [(s["scenario"], s["input"], s["output"]) for s in sent] == [
        ("contract-1", PROMPT, BETTER),
        ("contract-2", PROMPT, SECOND),
    ]
    assert [check["id"] for check in sent[0]["checks"]] == [
        "no-new-goal",
        "same-language",
        "same-format",
    ]
    assert {c.model for c in result.calls("judge")} == {MODELS.judge}


@dataclasses.dataclass
class QuotesTheOriginal(World):
    """Passes every contract question with a quote found in the original, not in the rewrite."""

    def __call__(self, call: Call) -> str | Exception:
        reply = super().__call__(call)
        if not is_contract_check(call):
            return reply
        body = json.loads(str(reply))
        for check in body["results"][0]["checks"]:
            check["quote"] = "request."
        return json.dumps(body)


@pytest.mark.parametrize(("world", "returned"), [(QuotesTheOriginal(), False), (World(), True)])
def test_a_contract_pass_without_a_quote_from_the_rewrite_is_a_fail(tmp_path, world, returned):
    assert (run(tmp_path, world, K1M2).outcome.prompt == BETTER) is returned


# --- failures (SPEC R24) --------------------------------------------------------------------------


def test_every_rewrite_dropped_leaves_the_original_without_scoring(tmp_path):
    world = World(rewrites=(Verbatim("no delimiters here"), Verbatim(""), "<side_info> x"))
    result = run(tmp_path, world, K3M3)
    assert (result.outcome.prompt, result.outcome.reason_code) == (
        PROMPT,
        "no_reliable_improvement",
    )
    assert [c.role for c in result.raw.calls] == [
        "intake",
        "synth",
        "reflect",
        "reflect",
        "reflect",
    ]
    assert "rewrite 0 dropped" in result.log


def test_failed_rewrite_calls_are_dropped_and_the_others_go_on(tmp_path):
    result = run(tmp_path, World(rewrites=(CallError("down"), BETTER)), K2M2)
    assert (result.outcome.prompt, result.outcome.changes) == (
        BETTER,
        (STRATEGY_NOTES["structure"],),
    )
    assert len(result.calls("reflect")) == 3 + 1  # three attempts of variant 0, one of variant 1


def test_three_failed_rewrite_calls_to_one_model_end_the_run(tmp_path):
    with pytest.raises(BackendError, match="3 consecutive reflect calls"):
        run(tmp_path, World(rewrites=(CallError("down"),)), K3M3)


@dataclasses.dataclass
class WrongCount(World):
    """Every synthesis reply holds no scenario."""

    def __call__(self, call: Call) -> str | Exception:
        return '{"scenarios": []}' if call.role == "synth" else super().__call__(call)


@pytest.mark.parametrize(
    ("world", "role"), [(World(intake="not json"), "intake"), (WrongCount(), "synth")]
)
def test_a_failed_intake_or_synthesis_ends_the_run(tmp_path, world, role):
    with pytest.raises(BackendError, match=role):
        run(tmp_path, world, K1M2)


def test_an_original_whose_runs_all_fail_ends_the_run(tmp_path):
    def task(call: Call) -> str:
        if prompt_of(call) == PROMPT and call.sample == 0:
            raise CallError("down")
        return "GOOD answer"

    with pytest.raises(BackendError, match="every task run of one of the original's two runs"):
        run(tmp_path, World(task=task), K1M2)


# --- the user's examples and the tiers (SPEC R11, R14a, R25) --------------------------------------

EXAMPLES = [Scenario(id=f"e{i}", input=f"example {i}") for i in range(1, 6)]


def test_the_users_examples_replace_the_synthesis(tmp_path):
    result = run(tmp_path, World(), K1M2_EXAMPLES, examples=EXAMPLES)
    assert result.calls("synth") == []
    assert {scenario_of(c) for c in result.calls("task")} == {"example 1", "example 2"}
    assert result.outcome.prompt == BETTER
    store = RunStore.resume(tmp_path, result.run_id)
    try:
        assert store.scenarios() == EXAMPLES
    finally:
        store.close()


@pytest.mark.parametrize("keeps", [False, True])
def test_the_quick_tier_makes_no_scoring_call(tmp_path, keeps):
    result = run(tmp_path, World(contract_ok=lambda _text: keeps), QUICK)
    assert [c.role for c in result.raw.calls] == ["intake", "reflect", "judge"]
    assert judged_scenarios(result.calls("judge")[0]) == ["contract"]
    outcome = result.outcome
    assert (outcome.prompt == BETTER, outcome.verified) == (keeps, False)
    assert (outcome.score_before, outcome.score_after) == (None, None)
    assert "tier quick" in outcome.reason
    if keeps:
        assert "not scored on any scenario" in outcome.reason


def test_the_checked_tier_confirms_the_winner_on_the_held_out_scenarios_on_the_target(tmp_path):
    result = run(tmp_path, World(), CHECKED)
    outcome = result.outcome
    assert (outcome.prompt, outcome.verified, outcome.reason_code) == (BETTER, True, "improved")
    assert (outcome.score_before, outcome.score_after) == (0.0, 1.0)
    assert (outcome.search_score_before, outcome.search_score_after) == (0.0, 1.0)
    assert outcome.margin == pytest.approx(0.95) and "tier checked" in outcome.reason
    assert json.loads(result.calls("synth")[0].user)["count"] == 6
    on_target = [c for c in result.calls("task") if c.model == MODELS.target]
    assert sorted((prompt_of(c), scenario_of(c)) for c in on_target) == sorted(
        (text, f"situation {i}") for text in (PROMPT, BETTER) for i in (3, 4, 5, 6)
    )
    picked = [c for c in result.calls("task") if c.model == MODELS.task]
    assert {scenario_of(c) for c in picked} == {"situation 1", "situation 2"}


def test_a_checked_winner_that_loses_on_the_target_is_not_returned(tmp_path):
    def task(call: Call) -> str:
        good = MARKER in prompt_of(call) and call.model == MODELS.task
        return "GOOD answer" if good else "BAD answer"

    outcome = run(tmp_path, World(task=task), CHECKED).outcome
    assert (outcome.prompt, outcome.verified) == (PROMPT, False)
    assert outcome.reason_code == "no_reliable_improvement"
    assert (outcome.score_before, outcome.score_after) == (0.0, None)
    assert outcome.search_score_before == 0.0


def test_a_checked_run_without_a_winner_reports_no_held_out_score(tmp_path):
    """The score fields of the checked tier are held-out scores on the target model (SPEC R14a);
    the preference on the scenarios it was picked on is set by a winner only (ADR-012)."""
    outcome = run(tmp_path, World(rewrites=("Answer the request well.",)), CHECKED).outcome
    assert (outcome.prompt, outcome.reason_code) == (PROMPT, "no_reliable_improvement")
    assert (outcome.score_before, outcome.search_score_before, outcome.noise) == (None, None, None)
    assert not [c for c in run(tmp_path / "b", World(), K1M2).raw.calls if c.model == MODELS.target]


def test_the_checked_tier_with_too_few_examples_keeps_the_original_before_any_call(tmp_path):
    result = run(tmp_path, World(), CHECKED, examples=EXAMPLES[:2])
    assert (result.outcome.reason_code, result.raw.calls) == ("no_holdout", [])


def test_the_kind_given_replaces_the_guess(tmp_path):
    """The intake guesses `task`; `--kind template` makes the prompt the system prompt."""
    tasks = run(tmp_path, World(), K1M2, kind="template").calls("task")
    assert [(c.system, c.user) for c in tasks] == [
        (text, f"situation {i}{FAST_TASK_SUFFIX}")
        for text in (PROMPT, PROMPT, BETTER)
        for i in (1, 2)
    ]


def test_the_run_folder_keeps_the_contract_and_the_synthesised_scenarios(tmp_path):
    """SPEC R5, R22: a resumed run reads both from the run folder."""
    result = run(tmp_path, World(), CHECKED)
    store = RunStore.resume(tmp_path, result.run_id)
    try:
        contract, scenarios = store.contract(), store.scenarios()
    finally:
        store.close()
    assert contract is not None and contract.goal == "answer the user's request well"
    assert [s.input for s in scenarios or []] == [f"situation {i}" for i in range(1, 7)]


NOBODY = {"id": "g", "group": "format", "text": "says XYZ", "rule": "contains", "arg": "XYZ"}


def test_the_contracts_scoring_checks_take_no_part_in_the_fast_pick(tmp_path):
    """ADR-012 as briefed for WP16: the pairwise judge decides and the contract check is the
    meaning floor; a check no answer passes neither vetoes nor costs a judge call."""
    result = run(tmp_path, World(intake=intake_reply(checks=[NOBODY])), K1M2)
    assert all(is_pairwise(c) or is_contract_check(c) for c in result.calls("judge"))
    outcome = result.outcome
    assert (outcome.prompt, outcome.score_before, outcome.score_after) == (BETTER, 0.0, 1.0)


def test_stage_b_asks_the_very_task_calls_of_the_evaluator(tmp_path):
    """The same calls, so the same cache keys, as the fast tiers' `FastEvaluator` (the evaluator's
    own task call with the 120-word suffix) makes for these prompts (R22, R25)."""
    result = run(tmp_path, World(), K1M2)
    store = RunStore.resume(tmp_path, result.run_id)
    try:
        contract, scenarios = store.contract(), store.scenarios() or []
    finally:
        store.close()
    assert contract is not None
    model = ScriptedBackend(World())
    for text, sample in ((PROMPT, 0), (PROMPT, 1), (BETTER, 0)):
        FastEvaluator(model, contract, MODELS.task, MODELS.judge, sample)(text, scenarios[:2])
    assert result.calls("task") == [c for c in model.calls if c.role == "task"]


MODE_CASES = {
    "fast improved": (World(), K3M3, {}),
    "fast unchanged": (World(rewrites=("Answer the request well.",)), K3M3, {}),
    "fast out of time": (World(), K3M3, {"deadline": 0}),
    "quick improved": (World(), QUICK, {}),
    "quick unchanged": (World(contract_ok=lambda _text: False), QUICK, {}),
    "checked improved": (World(), CHECKED, {}),
    "checked without a holdout": (World(), CHECKED, {"examples": EXAMPLES[:2]}),
}


@pytest.mark.parametrize("case", MODE_CASES)
def test_every_outcome_names_the_tier_of_the_run(tmp_path, case):
    world, fplan, extra = MODE_CASES[case]
    assert run(tmp_path, world, fplan, **extra).outcome.mode == fplan.tier
