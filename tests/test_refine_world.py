"""End to end, the reflection rounds refine the learned rules from the failures (SPEC R16, R25;
ADR-013, WP23). The prompt "Decide what to do with this refund request." hides a policy of rules
that only the references show. In this world each example is decided by one rule (or by none:
approve); the task model answers an example's label only when the prompt it runs states that
example's rule, and "approve" otherwise; the reference judge passes an output that is the label.
The first generation's induce rewrite states the first rule only. A reflection that reads the best
candidate's failing examples states the rule of the first one its candidate lacks (its second
reflection of a round adds nothing that helps); a reflection that reads no failure learns nothing.

So a reference-mode run returns the prompt with both rules (verified in the checked tier), and its
twin whose reflections never see the failures does not; the rounds go on while a round gains, at
most three, and stop at a round without gain or when the clock leaves no room for another; no
held-out example reaches any call but those of stage E; the deadline is never passed."""

import dataclasses
import json
from dataclasses import dataclass

import pytest
from fakes import FakeClock, intake_reply
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    CHECKED,
    MODELS,
    PLAN,
    World,
    is_contract_check,
    judged_scenarios,
    mechanics_latency_model,
    no_disk_flush,
    prompt_of,
    run,
    scenario_of,
)

from autoimprover import fast_prompts
from autoimprover.strata import ordered
from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END, Backend, Call, Reply, Scenario

PROMPT = "Decide what to do with this refund request."
PRICE = "Escalate a purchase of $200 or more."
DAYS = "Deny a request made more than 30 days after the purchase."
DIGITAL = "Deny a downloaded digital product returned for a change of mind."
LEGAL = "Escalate a request that threatens legal action."
GIFT = "Deny a refund for a gift card."
# (input, label, the rule that decides it or None)
TWO = {
    "e1": ("Headphones for $59, bought 10 days ago: the left earbud is dead.", "approve", None),
    "e2": ("A laptop for $1,200, bought 14 days ago: the screen flickers.", "escalate", PRICE),
    "e3": ("A jacket for $45, bought 45 days ago: the zipper broke.", "deny", DAYS),
    "e4": ("A monitor for $350 that never arrived, ordered 20 days ago.", "escalate", PRICE),
    "e5": ("A blender for $70, bought 5 days ago: it arrived cracked.", "approve", None),
    "e6": ("A lamp for $90, bought 60 days ago: it flickers now.", "deny", DAYS),
    "e7": ("A camera for $500, bought 3 days ago: the lens is scratched.", "escalate", PRICE),
    "e8": ("Shoes for $30, bought 40 days ago: the sole came off.", "deny", DAYS),
    "e9": ("A mug for $20, bought 2 days ago: it arrived chipped.", "approve", None),
}
FIVE = {
    "f1": ("A sofa for $900, bought 9 days ago: a leg broke off.", "escalate", PRICE),
    "f2": ("A kettle for $25, bought 50 days ago: it leaks.", "deny", DAYS),
    "f3": ("An ebook for $9, bought yesterday and downloaded: I changed my mind.", "deny", DIGITAL),
    "f4": (
        "A toaster for $40, bought 4 days ago: refund it or I call my lawyer.",
        "escalate",
        LEGAL,
    ),
    "f5": ("A gift card for $50, bought 2 days ago: I do not want it.", "deny", GIFT),
    "f6": ("A scarf for $15, bought 3 days ago: it has a hole.", "approve", None),
}
ALL = {**TWO, **FIVE}
RULE_OF = {text: rule for text, _label, rule in ALL.values()}
LABEL_OF = {text: label for text, label, _rule in ALL.values()}
BOTH = f"{PROMPT} {PRICE} {DAYS}"
BALANCED = dataclasses.replace(PLAN, strictness="balanced", budget=300)


def examples(table: dict[str, tuple[str, str, str | None]]) -> list[Scenario]:
    return [Scenario(sid, text, expected=label) for sid, (text, label, _rule) in table.items()]


def task(call: Call) -> str:
    text = scenario_of(call)
    rule = RULE_OF[text]
    return LABEL_OF[text] if rule is None or rule in prompt_of(call) else "approve"


def is_label(scenario: str, _check: str, output: str) -> bool:
    return output == ALL[scenario][1]


@dataclass
class Refunds(World):
    """The world above. In stage A the induce rewrite states the first rule and a second rewrite
    no rule; a reflection writes from the first candidate it reads."""

    def __call__(self, call: Call) -> str | Exception:
        if call.role == "reflect":
            return f"{INSTRUCTION_BEGIN}\n{self.write(call)}\n{INSTRUCTION_END}"
        return super().__call__(call)

    def write(self, call: Call) -> str:
        if call.sample < fast_prompts.REFLECT_SAMPLE:  # stage A
            return f"{PROMPT} {PRICE}" if call.sample == 0 else f"{PROMPT} Decide it fairly."
        best = json.loads(call.user)["candidates"][0]
        failing = [item["input"] for item in best["scenarios"]]
        if call.sample % 2:  # the second reflection of a round
            return f"{best['prompt']} Be fair."
        lacks = [RULE_OF[text] for text in failing if RULE_OF[text] not in best["prompt"]]
        learned = next((rule for rule in lacks if rule is not None), "Be kind.")
        return f"{best['prompt']} {learned}"


def refunds() -> Refunds:
    return Refunds(task=task, passes=is_label, intake=intake_reply("task", from_examples=[PRICE]))


class Blind:
    """A layer above the run's stack that takes the failing examples out of every reflection
    call before it is sent: the twin whose reflections never see the failures."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner

    def complete(self, call: Call) -> Reply:
        if call.role == "reflect" and call.sample >= fast_prompts.REFLECT_SAMPLE:
            sent = json.loads(call.user)
            sent["candidates"] = [{**c, "scenarios": []} for c in sent["candidates"]]
            call = dataclasses.replace(call, user=json.dumps(sent))
        return self._inner.complete(call)


def plan(tier: str, holdout: int, time_s: int = 300):
    """K=1 on 6 pick examples, `holdout` held out, then rounds of 2 reflections: the runner reads
    only the shape; 300 s leaves room for every round the tests allow."""
    shape = {"tier": tier, "scenarios": 6, "holdout": holdout, "time_s": time_s}
    return dataclasses.replace(CHECKED, generations=2, rewrites2=2, **shape)


def reflections(result) -> list[Call]:
    return [c for c in result.calls("reflect") if c.sample >= fast_prompts.REFLECT_SAMPLE]


# --- the failures teach the missing rule ----------------------------------------------------------


@pytest.mark.parametrize("sees", [True, False], ids=["reads the failures", "blind"])
def test_a_fast_run_learns_the_day_rule_from_the_failures(tmp_path, sees):
    result = run(
        tmp_path,
        refunds(),
        plan("fast", 0),
        prompt=PROMPT,
        plan=BALANCED,
        examples=examples(TWO)[:6],
        wrap=None if sees else Blind,
    )
    outcome = result.outcome
    assert outcome.status == "improved" and not outcome.verified
    assert "reference-scored" in outcome.reason
    if sees:
        assert outcome.prompt == BOTH
        assert (outcome.score_before, outcome.score_after) == (pytest.approx(2 / 6), 1.0)
        assert "fast: round 1: best passes 6 of 6 pick examples, was 4" in result.log
    else:
        assert outcome.prompt == f"{PROMPT} {PRICE}" and DAYS not in outcome.prompt
        assert "fast: round 1: best passes 4 of 6 pick examples, was 4" in result.log


def test_the_reflection_reads_only_the_best_candidates_failures_with_their_references(tmp_path):
    """Two rewrites keep the contract, the rule and one that adds none: the reflection reads the
    best one only, with the examples it failed."""
    two = dataclasses.replace(plan("fast", 0), rewrites=2)
    result = run(tmp_path, refunds(), two, prompt=PROMPT, plan=BALANCED, examples=examples(TWO)[:6])
    assert f"{PROMPT} Decide it fairly." in [prompt_of(c) for c in result.calls("task")]
    first = reflections(result)[0]
    (best,) = json.loads(first.user)["candidates"]
    assert best["prompt"] == f"{PROMPT} {PRICE}"
    assert [s["input"] for s in best["scenarios"]] == [TWO["e3"][0], TWO["e6"][0]]
    assert [s["expected"] for s in best["scenarios"]] == ["deny", "deny"]
    assert [s["output"] for s in best["scenarios"]] == ["approve", "approve"]
    assert all(s["failed"] == [f"{_AGREES}deny"] for s in best["scenarios"])
    assert json.loads(first.user)["contract"]["from_examples"] == [PRICE]
    assert fast_prompts.REFLECT_SAMPLE == first.sample


_AGREES = "the output agrees with the reference answer in substance: "


@pytest.mark.parametrize("sees", [True, False], ids=["reads the failures", "blind"])
def test_a_checked_run_verifies_both_rules_on_the_held_out_examples(tmp_path, sees):
    result = run(
        tmp_path,
        refunds(),
        plan("checked", 3),
        prompt=PROMPT,
        plan=BALANCED,
        examples=examples(TWO),
        wrap=None if sees else Blind,
    )
    outcome = result.outcome
    assert outcome.verified and outcome.status == "improved"
    assert (outcome.prompt == BOTH) is sees
    if sees:
        assert (outcome.score_before, outcome.score_after) == (pytest.approx(1 / 3), 1.0)
        assert outcome.reason.endswith("rounds: 1; pick examples passed: 6 of 6 (original 2 of 6)")
    else:
        assert DAYS not in outcome.prompt
        assert outcome.reason.endswith("rounds: 1; pick examples passed: 4 of 6 (original 2 of 6)")


# --- the rounds -----------------------------------------------------------------------------------


def test_the_rounds_go_on_while_they_gain_and_stop_after_three(tmp_path):
    """Five rules: the first generation states one, each round learns one more, and the fourth
    round that the failures left would need is never asked. The pick is in the label order
    (WP25: f1 f2 f6 f4 f3 f5), so the failures, and the rules learned, come f2 f4 f3."""
    result = run(
        tmp_path, refunds(), plan("fast", 0), prompt=PROMPT, plan=BALANCED, examples=examples(FIVE)
    )
    assert [c.sample for c in reflections(result)] == [100, 101, 110, 111, 120, 121]
    for number, (now, was) in enumerate([(3, 2), (4, 3), (5, 4)], start=1):
        line = f"fast: round {number}: best passes {now} of 6 pick examples, was {was}"
        assert line in result.log
    outcome = result.outcome
    assert outcome.prompt == f"{PROMPT} {PRICE} {DAYS} {LEGAL} {DIGITAL}"
    assert outcome.reason.endswith("rounds: 3; pick examples passed: 5 of 6 (original 1 of 6)")
    assert outcome.changes and "round 3" in outcome.changes[0]


def test_a_round_without_gain_stops_the_rounds(tmp_path):
    result = run(
        tmp_path,
        refunds(),
        plan("fast", 0),
        prompt=PROMPT,
        plan=BALANCED,
        examples=examples(FIVE),
        wrap=Blind,
    )
    assert [c.sample for c in reflections(result)] == [100, 101]
    assert "fast: round 1: best passes 2 of 6 pick examples, was 2" in result.log
    assert "round 2" not in result.log
    assert result.outcome.prompt == f"{PROMPT} {PRICE}"


def test_a_best_candidate_with_no_failure_left_ends_the_rounds(tmp_path):
    result = run(
        tmp_path, refunds(), plan("fast", 0), prompt=PROMPT, plan=BALANCED, examples=examples(TWO)
    )
    assert [c.sample for c in reflections(result)] == [100, 101]
    assert "fast: round 1: best passes 6 of 6 pick examples, was 4" in result.log
    assert result.outcome.prompt == BOTH


def test_no_held_out_example_reaches_any_call_but_those_of_stage_e(tmp_path):
    """The extension of the WP22 test to the rounds: the checked run picks on the first six of
    the label order (WP25: f1 f2 f6 f4 f3 e9) and holds out the other three (e7 f5 e8); every
    generating call of every round (intake, rewrites, reflections) and every call of the pick
    sees the pick examples only, and a held-out example reaches only the task runs on the target
    model and their judge calls."""
    given = examples({**FIVE, **{k: TWO[k] for k in ("e7", "e8", "e9")}})
    result = run(
        tmp_path, refunds(), plan("checked", 3), prompt=PROMPT, plan=BALANCED, examples=given
    )
    assert len(reflections(result)) == 6 and result.outcome.verified
    order = ordered(given)
    pick, held = order[:6], order[6:]
    assert [e.id for e in pick] == ["f1", "f2", "f6", "f4", "f3", "e9"]
    assert [e.id for e in held] == ["e7", "f5", "e8"]
    for call in result.raw.calls:
        text = call.user + call.system
        if not any(h.input in text for h in held):
            continue
        stage_e = call.role == "task" and call.model == MODELS.target
        judged = call.role == "judge" and set(judged_scenarios(call)) <= {h.id for h in held}
        assert stage_e or judged, (call.role, call.sample)
    for call in [c for c in result.raw.calls if c.role in ("intake", "reflect", "synth")]:
        for example in held:
            assert example.input not in call.user + call.system
    # WP24: the contract check of stage C and of every round's C2 sees the pick examples only
    checks = [c for c in result.raw.calls if is_contract_check(c)]
    assert len(checks) == 4
    for call in checks:
        shown = json.loads(call.user)["examples"]
        assert [e["input"] for e in shown] == [e.input for e in pick]
        assert not any(h.input in call.user + call.system for h in held)


# --- the clock ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("deadline", "rounds"), [(170.0, 1), (1000.0, 3)])
def test_a_round_starts_only_when_it_and_stage_e_fit_and_the_deadline_is_never_passed(
    tmp_path, deadline, rounds
):
    """Every call takes 2 s on the fake clock: the first generation and its scoring take 48 s,
    a round 32 s (the judge call of the reflection that answers as its parent did is the
    parent's, from the cache), stage E 24 s. With 170 s, after the first round the replan
    (fitted: 1.9 s a call plus its tokens at 105 per s) leaves no room for a second one and stage
    E within 0.85 of the clock, and the run ends at 104 s; with 1000 s the rounds stop after
    three, at 168 s."""
    clock = FakeClock()
    fplan = plan("checked", 3, time_s=int(deadline))
    given = examples({**FIVE, **{k: TWO[k] for k in ("e7", "e8", "e9")}})
    result = run(
        tmp_path,
        refunds(),
        fplan,
        prompt=PROMPT,
        plan=BALANCED,
        examples=given,
        clock=clock,
        deadline=deadline,
        pace=2.0,
    )
    assert len(reflections(result)) == 2 * rounds
    assert clock.now() == (104.0 if rounds == 1 else 168.0) <= deadline
    assert result.outcome.verified and result.outcome.stop is None
    if rounds == 1:
        assert "checked: round 2: no time left within 0.85 of the clock" in result.log
