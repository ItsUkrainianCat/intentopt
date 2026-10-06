"""The fast and checked picks when every example carries a reference (SPEC R11, R25; WP21): stage C
scores each run against the references, one batched judge call per run (the original's two runs
and each rewrite's) with only the references' checks, beside the contract check; a rewrite wins iff
its summed score exceeds the mean of the original's two sums by more than their difference, it
improves more scenarios than it worsens and at least one; stage C2 reuses the original's scores;
the reflection reads the failing scenarios with their references; stage E compares the same way on
the held-out examples on the target model. Mixed references keep the pairwise pick.

The task output says whether it is GOOD, by (prompt tag, example, sample, model); the judge passes a
check of a GOOD output only."""

import json
from collections.abc import Callable

import pytest
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    CHECKED,
    K1M2_EXAMPLES,
    K3M2_EXAMPLES,
    MODELS,
    PROMPT,
    TWO_K1,
    World,
    is_contract_check,
    is_pairwise,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
    scoring_judges,
)

from autoimprover.types import BackendError, Call, CallError, Scenario

A = "Answer the user's request now. [A]"
B = "Answer the user's request here. [B]"
C = "Answer the request, please. [C]"


def refs(n: int, **extra) -> list[Scenario]:
    return [
        Scenario(f"e{i}", f"example {i}", expected=f"reference {i}", **extra)
        for i in range(1, n + 1)
    ]


def tag(text: str) -> str:
    return next((t for t in "ABC" if f"[{t}]" in text), "O")


Good = Callable[[str, str, int, str], bool]


def world(good: Good, rewrites=(A,), reflections=(), **fields) -> World:
    """`good(tag, example id, sample, model)` says whether that task run answers GOOD."""

    def task(call: Call) -> str:
        name, scenario = tag(prompt_of(call)), "e" + scenario_of(call).split()[-1]
        mark = "GOOD" if good(name, scenario, call.sample, call.model) else "BAD"
        return f"{mark} {name} {scenario} {call.sample} {call.model}"

    return World(rewrites=list(rewrites), reflections=list(reflections), task=task, **fields)


def only(*goods: tuple[str, str]) -> Good:
    """GOOD exactly for the (tag, example) pairs given, on every sample and model."""
    return lambda name, scenario, _sample, _model: (name, scenario) in goods


def checks_sent(call: Call) -> list[list[str]]:
    return [[c["id"] for c in item["checks"]] for item in json.loads(call.user)["scenarios"]]


def ran(call: Call) -> set[str]:
    """The tags of the prompts whose outputs a judge call grades."""
    return {item["output"].split()[1] for item in json.loads(call.user)["scenarios"]}


# --- stage C and D ------------------------------------------------------------------------------


def test_stage_c_is_one_reference_judge_call_per_run_and_no_pairwise_call(tmp_path):
    result = run(tmp_path, world(only(("A", "e1"), ("A", "e2"))), K1M2_EXAMPLES, examples=refs(3))
    outcome = result.outcome
    assert outcome.prompt == A and outcome.reason_code == "improved"
    assert "reference-scored" in outcome.reason
    assert not [c for c in result.raw.calls if is_pairwise(c)]
    grading = scoring_judges(result)
    assert [c.sample for c in grading] == [0, 1, 0]  # the original twice, then the rewrite
    assert all(checks_sent(c) == [["s:expected"], ["s:expected"]] for c in grading)
    assert all(judged_scenarios(c) == ["e1", "e2"] for c in grading)  # the M=2 picked
    assert [is_contract_check(c) for c in result.calls("judge")].count(True) == 1
    assert (outcome.score_before, outcome.score_after, outcome.noise) == (0.0, 1.0, 0.0)
    assert outcome.margin == pytest.approx(1.0)


def test_mixed_references_keep_the_pairwise_pick(tmp_path):
    mixed = [*refs(2), Scenario("e3", "example 3")]
    result = run(tmp_path, world(only(("A", "e1"), ("A", "e2"))), K1M2_EXAMPLES, examples=mixed)
    assert result.outcome.prompt == A and "reference-scored" not in result.outcome.reason
    assert [c for c in result.raw.calls if is_pairwise(c)]
    assert all(
        "s:expected" not in str(checks_sent(c)) for c in result.calls("judge") if not is_pairwise(c)
    )


@pytest.mark.parametrize(
    ("goods", "returned"),
    [
        ((("A", "e1"),), True),  # one improved, none worsened, gain 1 above no noise
        ((), False),  # ties on all go to the original
        ((("O", "e1"), ("A", "e2")), False),  # one improved, one worsened
        ((("O", "e1"), ("A", "e1"), ("A", "e2")), True),
    ],
)
def test_the_reference_pick_at_its_boundaries(tmp_path, goods, returned):
    outcome = run(tmp_path, world(only(*goods)), K1M2_EXAMPLES, examples=refs(2)).outcome
    assert (outcome.prompt == A) is returned
    if not returned:
        assert outcome.reason_code == "no_reliable_improvement"
        assert "reference-scored" in outcome.reason


@pytest.mark.parametrize(("goods", "returned"), [({"e1", "e2"}, True), ({"e2"}, False)])
def test_the_noise_of_the_originals_two_runs_is_the_bar(tmp_path, goods, returned):
    """The original answers e1 well on its run 0 only: means 0.5 and 0, noise 1. A rewrite good on
    both gains 1.5; one good on e2 only gains 0.5, below the noise."""

    def good(name: str, scenario: str, sample: int, _model: str) -> bool:
        if name == "O":
            return scenario == "e1" and sample == 0
        return scenario in goods

    outcome = run(tmp_path, world(good), K1M2_EXAMPLES, examples=refs(2)).outcome
    assert (outcome.prompt == A) is returned
    assert outcome.noise == pytest.approx(0.5)  # the summed difference 1 over the 2 picked
    if returned:
        assert (outcome.score_before, outcome.score_after) == (0.25, 1.0)
        assert outcome.margin == pytest.approx(0.75 - 0.5)


def test_the_largest_gain_wins_and_a_tie_goes_to_the_shorter(tmp_path):
    goods = only(("A", "e1"), ("B", "e1"), ("B", "e2"), ("C", "e1"), ("C", "e2"))
    result = run(tmp_path, world(goods, rewrites=(A, B, C)), K3M2_EXAMPLES, examples=refs(2))
    assert result.outcome.prompt == C  # B and C gain 2, C is shorter


def test_a_failed_reference_call_of_a_rewrite_drops_only_that_rewrite(tmp_path):
    goods = only(("A", "e1"), ("B", "e1"), ("B", "e2"))

    def judge_down_for_b(call: Call) -> None:
        if call.role == "judge" and "GOOD B" in call.user:
            raise CallError("judge down")

    fails = world(goods, rewrites=(A, B), hook=judge_down_for_b)
    result = run(tmp_path, fails, K3M2_EXAMPLES, examples=refs(2))
    assert result.outcome.prompt == A and "reference-scored" in result.outcome.reason
    assert "reference judge 3 dropped" not in result.log  # the evaluator kept B unscored


def test_the_judge_failing_on_the_originals_runs_ends_the_run(tmp_path):
    def judge_down_for_original(call: Call) -> None:
        if call.role == "judge" and "BAD O" in call.user:
            raise CallError("judge down")

    fails = world(only(("A", "e1")), hook=judge_down_for_original)
    with pytest.raises(BackendError, match="original"):
        run(tmp_path, fails, K1M2_EXAMPLES, examples=refs(2))


# --- the second generation ------------------------------------------------------------------------


def test_the_reflection_reads_the_failing_scenarios_with_their_references(tmp_path):
    goods = only(("A", "e1"), ("B", "e1"), ("B", "e2"))
    result = run(tmp_path, world(goods, reflections=(B, C)), TWO_K1, workers=4, examples=refs(2))
    assert result.outcome.prompt == B  # gain 2 against A's 1
    reflections = [c for c in result.calls("reflect") if c.sample >= 100]
    assert len(reflections) == 2
    sent = json.loads(reflections[0].user)
    (parent,) = sent["candidates"]
    assert parent["prompt"] == A
    (failing,) = parent["scenarios"]
    assert failing["input"] == "example 2" and failing["expected"] == "reference 2"
    assert failing["output"].startswith("BAD A e2")
    assert failing["failed"] == [
        "the output agrees with the reference answer in substance: reference 2"
    ]
    assert "reference" in reflections[0].system
    # C2 judges the two reflections only: the original's scores of stage C stand (each stage's
    # calls run side by side, so each stage's are compared in any order)
    tags = [min(ran(c)) for c in scoring_judges(result)]
    assert (sorted(tags[:3]), sorted(tags[3:])) == (["A", "O", "O"], ["B", "C"])


def test_with_no_rewrite_kept_the_reflection_reads_the_originals_failures(tmp_path):
    goods = only(("B", "e1"), ("B", "e2"))
    result = run(
        tmp_path,
        world(goods, reflections=(B, C), contract_ok=lambda text: "[A]" not in text),
        TWO_K1,
        workers=4,
        examples=refs(2),
    )
    sent = json.loads(next(c for c in result.calls("reflect") if c.sample >= 100).user)
    (parent,) = sent["candidates"]
    assert parent["prompt"] == PROMPT and len(parent["scenarios"]) == 2
    assert result.outcome.prompt == B


# --- stage E -------------------------------------------------------------------------------------


@pytest.mark.parametrize("on_target", [True, False])
def test_stage_e_compares_by_the_references_on_the_target(tmp_path, on_target):
    """K=1 on e1, e2; e3 to e6 held out. The rewrite is GOOD everywhere on the task model, and on
    the target model only when `on_target`."""

    def good(name: str, _scenario: str, _sample: int, model: str) -> bool:
        return name == "A" and (model == MODELS.task or on_target)

    result = run(tmp_path, world(good), CHECKED, examples=refs(6))
    outcome = result.outcome
    assert (outcome.prompt == A, outcome.verified) == (on_target, on_target)
    held = [c for c in result.calls("task") if c.model == MODELS.target]
    assert sorted((tag(prompt_of(c)), c.sample) for c in held) == sorted(
        [("O", 0)] * 4 + [("O", 1)] * 4 + [("A", 0)] * 4
    )
    targets = [c for c in scoring_judges(result) if MODELS.target in c.user]
    assert len(targets) == 3
    assert all(checks_sent(c) == [["s:expected"]] * 4 for c in targets)
    assert "reference-scored" in outcome.reason
    if on_target:
        assert (outcome.score_before, outcome.score_after, outcome.noise) == (0.0, 1.0, 0.0)
        assert outcome.search_score_before == 0.0 and outcome.search_score_after == 1.0
    else:
        assert outcome.reason_code == "no_reliable_improvement"


# --- --ungated -----------------------------------------------------------------------------------


def test_ungated_ranks_by_the_reference_gain(tmp_path):
    """Neither rewrite clears the noise (1, the original GOOD on e1 in run 0 only); B gains more."""

    def good(name: str, scenario: str, sample: int, _model: str) -> bool:
        if name == "O":
            return scenario == "e1" and sample == 0
        return name == "B" and scenario == "e2"

    result = run(
        tmp_path, world(good, rewrites=(A, B)), K3M2_EXAMPLES, examples=refs(2), ungated=True
    )
    outcome = result.outcome
    assert outcome.reason_code == "ungated_best_candidate" and outcome.prompt == B
    assert "reference-scored" in outcome.reason
