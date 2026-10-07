"""End to end, the rewrites learn a hidden policy from the user's examples (SPEC R6, R7, R25;
ADR-013, WP22). The prompt "Assign a priority to this support ticket." hides a policy that only the
references show. In this world the task model answers with the policy's label only when the prompt
it runs states RULE, and "P2" otherwise; the reference judge passes an output that names the
example's label; the rewrite model writes RULE into its rewrite only when its call carries the
examples and it `learns` from them (else a rewrite that knows no policy); the contract judge reads
the `no-new-goal` question and passes RULE or UNRELATED only when that question names it.

So a reference-mode run returns the induced rewrite (verified in the checked tier), and the twins
that cannot learn the policy keep the original; only pick examples ever reach a generating call
(intake, rewrite, reflection, synthesis); an unrelated requirement is still vetoed; a rewrite that
copies an example's input is dropped; the cap leaves room for the rule unless conservative."""

import dataclasses
import json
from dataclasses import dataclass

import pytest
from fakes import intake_reply
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    CHECKED,
    K1M2_EXAMPLES,
    K3M2_EXAMPLES,
    MODELS,
    PLAN,
    World,
    is_contract_check,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
)

from autoimprover import fast_prompts
from autoimprover.contract_text import INTAKE_SYSTEM
from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END, Call, Scenario

PROMPT = "Assign a priority to this support ticket."
RULE = "Use P1 when the service is down for everyone, P3 for billing, and P4 for anything else."
UNRELATED = "Also translate the ticket into French."
NO_POLICY = "Assign a priority to this support ticket, please."
LONG_RULE = (
    f"{RULE} When a ticket mentions both an outage and a billing problem, the outage decides the "
    "priority; when it is unclear whether the service is down for everyone or for one user only, "
    "ask how many users are affected before you choose P1."
)
TICKETS = {
    "e1": ("Nobody at our company can log in since 9am.", "P1"),
    "e2": ("Please send me last month's invoice again.", "P3"),
    "e3": ("The checkout page is down for every shopper.", "P1"),
    "e4": ("I was charged twice for my subscription renewal.", "P3"),
    "e5": ("Can you add a dark mode to the dashboard?", "P4"),
    "e6": ("The export button label has a typo in it.", "P4"),
}
LABELS = {text: label for text, label in TICKETS.values()}


def examples(n: int, references: bool = True) -> list[Scenario]:
    """The first `n` tickets; each reference is its label and a tag naming the ticket."""
    return [
        Scenario(sid, text, expected=f"{label} (ticket {sid})" if references else None)
        for sid, (text, label) in list(TICKETS.items())[:n]
    ]


def task(call: Call) -> str:
    return LABELS[scenario_of(call)] if RULE in prompt_of(call) else "P2"


def names_label(scenario: str, _check: str, output: str) -> bool:
    return output == TICKETS[scenario][1]


@dataclass
class Policy(World):
    """The world above; `induced` is what a rewrite that learns from the examples writes."""

    learns: bool = True
    induced: str = f"{PROMPT} {RULE}"

    def __call__(self, call: Call) -> str | Exception:
        if call.role == "reflect":
            return f"{INSTRUCTION_BEGIN}\n{self.rewrite(call)}\n{INSTRUCTION_END}"
        if is_contract_check(call):
            return contract_reply(call)
        return super().__call__(call)

    def rewrite(self, call: Call) -> str:
        if call.sample >= fast_prompts.REFLECT_SAMPLE:
            return f"{PROMPT} {RULE} Answer with the label only."
        if call.sample == 0 and self.learns and "examples" in json.loads(call.user):
            return self.induced
        return NO_POLICY if call.sample == 0 else f"{PROMPT} Variant {call.sample}."


def contract_reply(call: Call) -> str:
    results = []
    for item in json.loads(call.user)["scenarios"]:
        output, checks = item["output"], []
        for asked in item["checks"]:
            added = [s for s in (RULE, UNRELATED) if s in output]
            ok = asked["id"] != "no-new-goal" or all(s in asked["text"] for s in added)
            checks.append({"id": asked["id"], "pass": ok, "quote": output[:20]})
        results.append({"scenario": item["scenario"], "checks": checks})
    return json.dumps({"results": results})


def policy(**fields) -> Policy:
    return Policy(
        task=task,
        passes=names_label,
        intake=intake_reply("task", from_examples=[RULE]),
        **fields,
    )


def generating(calls: list[Call]) -> list[Call]:
    return [c for c in calls if c.role in ("intake", "reflect", "synth")]


# --- the hidden policy is learned and confirmed ---------------------------------------------------


@pytest.mark.parametrize("learns", [True, False])
def test_a_fast_run_with_references_returns_the_rewrite_that_states_the_policy(tmp_path, learns):
    result = run(
        tmp_path, policy(learns=learns), K1M2_EXAMPLES, prompt=PROMPT, examples=examples(2)
    )
    outcome = result.outcome
    assert "reference-scored" in outcome.reason
    if learns:
        assert (outcome.status, outcome.prompt) == ("improved", f"{PROMPT} {RULE}")
        assert (outcome.score_before, outcome.score_after) == (0.0, 1.0)
        assert outcome.changes == (fast_prompts.INDUCE_NOTE,)
    else:
        assert (outcome.status, outcome.reason_code) == ("unchanged", "no_reliable_improvement")
    (rewrite,) = [c for c in result.calls("reflect")]
    assert fast_prompts.INDUCE_VARIANT in rewrite.system
    (intake,) = result.calls("intake")
    assert [e["input"] for e in json.loads(intake.user)["examples"]] == [
        TICKETS["e1"][0],
        TICKETS["e2"][0],
    ]


@pytest.mark.parametrize("learns", [True, False])
def test_a_checked_run_verifies_the_learned_policy_on_the_held_out_examples(tmp_path, learns):
    result = run(tmp_path, policy(learns=learns), CHECKED, prompt=PROMPT, examples=examples(6))
    outcome = result.outcome
    assert (outcome.status == "improved", outcome.verified) == (learns, learns)
    if learns:
        assert outcome.prompt == f"{PROMPT} {RULE}" and "held-out examples" in outcome.reason
        assert (outcome.score_before, outcome.score_after) == (0.0, 1.0)
    held = {scenario_of(c) for c in result.calls("task") if c.model == MODELS.target}
    assert held == ({TICKETS[s][0] for s in ("e3", "e4", "e5", "e6")} if learns else set())


def test_without_references_nothing_reaches_the_rewrites_and_the_original_stays(tmp_path):
    result = run(
        tmp_path, policy(), K1M2_EXAMPLES, prompt=PROMPT, examples=examples(2, references=False)
    )
    assert result.outcome.prompt == PROMPT and "reference-scored" not in result.outcome.reason
    assert all("examples" not in json.loads(c.user) for c in result.calls("reflect"))
    assert result.calls("intake")[0].system == INTAKE_SYSTEM


def test_from_k2_the_first_rewrite_induces_and_every_rewrite_sees_the_examples(tmp_path):
    result = run(tmp_path, policy(), K3M2_EXAMPLES, prompt=PROMPT, examples=examples(2))
    rewrites = sorted(result.calls("reflect"), key=lambda c: c.sample)
    assert [c.sample for c in rewrites] == [0, 1, 2]
    assert fast_prompts.INDUCE_VARIANT in rewrites[0].system
    assert fast_prompts.REWRITE_VARIANTS["clarify"] in rewrites[1].system
    assert fast_prompts.REWRITE_VARIANTS["structure"] in rewrites[2].system
    assert all(len(json.loads(c.user)["examples"]) == 2 for c in rewrites)
    assert result.outcome.prompt == f"{PROMPT} {RULE}"


# --- only pick examples reach a generating call (ADR-013) -----------------------------------------


def test_no_held_out_example_reaches_any_generating_call(tmp_path):
    """Checked, two generations: e1 and e2 are picked on, e3 to e6 held out. The pick examples
    reach the intake and every rewrite; no held-out input or reference reaches any call that
    writes text (intake, rewrite, reflection, synthesis)."""
    fplan = dataclasses.replace(CHECKED, generations=2, rewrites2=2)
    given = examples(6)
    result = run(tmp_path, policy(), fplan, prompt=PROMPT, examples=given, workers=4)
    calls = generating(result.raw.calls)
    reflections = [c for c in calls if c.sample >= fast_prompts.REFLECT_SAMPLE]
    assert len(reflections) == 2 and result.outcome.verified
    for call in calls:
        text = call.user + call.system
        for held in given[2:]:
            assert held.input not in text and str(held.expected) not in text, (call.role, held.id)
    for call in [c for c in calls if c not in reflections]:
        for picked in given[:2]:
            assert picked.input in call.user and str(picked.expected) in call.user


# --- the meaning check, the copy gate and the length cap ------------------------------------------


@pytest.mark.parametrize(
    ("added", "returned"),
    [(RULE, True), (f"{RULE} {UNRELATED}", False)],
    ids=["learned rule", "and an unrelated requirement"],
)
def test_an_unrelated_requirement_is_still_vetoed(tmp_path, added, returned):
    induced = f"{PROMPT} {added}"
    world = policy(induced=induced)
    outcome = run(tmp_path, world, K1M2_EXAMPLES, prompt=PROMPT, examples=examples(2)).outcome
    assert (outcome.prompt == induced) is returned
    if not returned:
        assert outcome.reason_code == "no_reliable_improvement"


@pytest.mark.parametrize(
    ("example", "dropped"),
    [(TICKETS["e1"][0], True), ("nobody can log in", False)],
    ids=["an example's input", "a paraphrase"],
)
def test_a_rewrite_that_copies_an_examples_input_is_dropped(tmp_path, example, dropped):
    induced = f"{PROMPT} {RULE} For example, '{example}' is P1."
    result = run(
        tmp_path, policy(induced=induced), K1M2_EXAMPLES, prompt=PROMPT, examples=examples(2)
    )
    assert ("fast: rewrite 0 dropped: copies an example" in result.log) is dropped
    assert (result.outcome.prompt == induced) is not dropped
    if dropped:
        assert "copies of an example" in result.outcome.reason


@pytest.mark.parametrize(("strictness", "returned"), [("balanced", True), ("conservative", False)])
def test_the_cap_leaves_room_for_the_policy_unless_conservative(tmp_path, strictness, returned):
    induced = f"{PROMPT} {LONG_RULE}"  # 74 tokens: over 8 + 40, within 8 + 150
    plan = dataclasses.replace(PLAN, strictness=strictness)
    result = run(
        tmp_path,
        policy(induced=induced),
        K1M2_EXAMPLES,
        prompt=PROMPT,
        plan=plan,
        examples=examples(2),
    )
    assert (result.outcome.prompt == induced) is returned
    assert ("rewrite 0 dropped: it is longer than the length cap" in result.log) is not returned
    if returned:
        assert result.outcome.length_ratio == pytest.approx(74 / 8)
    (rewrite,) = result.calls("reflect")
    assert f"at most {158 if returned else 48} tokens" in rewrite.system
