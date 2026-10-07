"""End to end, the meaning check accepts a rule the pick examples support (SPEC R6, R25; ADR-013
amendment of 2026-10-07, WP24). The refund world of `test_refine_world`: the intake learns the price
rule from the examples, the first generation states it, and a reflection that reads the failures
adds the day rule, which is not in `from_examples` (the live run of 2026-10-07 vetoed exactly that).

The contract judge here reads its call: `no-new-goal` passes a rule of the policy when the question
names it (`from_examples`, WP22) or when a shown example is decided by that rule, quoting that
example's input; an unrelated requirement never passes; with examples shown, a pass quotes an
example's input as the check is told to. So a reference-mode run returns the prompt with both
rules; its twin whose check is shown no example of the day rule keeps the price rule only; a
reflection that adds an unrelated requirement is vetoed with the examples shown; the examples ride
in the one contract check of each stage C and C2, no call is added."""

import dataclasses
import json
from dataclasses import dataclass

import pytest
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    QUICK,
    is_contract_check,
    mechanics_latency_model,
    no_disk_flush,
    run,
)
from test_refine_world import (
    ALL,
    BALANCED,
    BOTH,
    DAYS,
    DIGITAL,
    GIFT,
    LEGAL,
    PRICE,
    PROMPT,
    RULE_OF,
    TWO,
    Refunds,
    examples,
    plan,
    reflections,
    refunds,
)

from autoimprover import fast_prompts
from autoimprover.contract_text import CONTRACT_MANY_SYSTEM, CONTRACT_SYSTEM, NO_NEW_GOAL_SUPPORT
from autoimprover.types import Backend, Call, Reply, Scenario

RULES = (PRICE, DAYS, DIGITAL, LEGAL, GIFT)
FRENCH = "Always answer in French."
LOYALTY = "Mention our loyalty program."


def meaning(call: Call) -> str:
    """The contract judge of this world (the module docstring)."""
    request = json.loads(call.user)
    shown = [example["input"] for example in request.get("examples", [])]
    results = []
    for item in request["scenarios"]:
        output, checks = item["output"], []
        for asked in item["checks"]:
            passed, quote = True, output[:20]
            if asked["id"] == "no-new-goal":
                added = [r for r in RULES if r in output and r not in asked["text"]]
                support = [next((i for i in shown if RULE_OF.get(i) == r), "") for r in added]
                passed = all(support) and not any(u in output for u in (FRENCH, LOYALTY))
                quote = next((s for s in support if s), shown[0]) if shown else quote
            checks.append({"id": asked["id"], "pass": passed, "quote": quote})
        results.append({"scenario": item["scenario"], "checks": checks})
    return json.dumps({"results": results})


@dataclass
class Meaning(Refunds):
    """The refund world with the contract judge above; a reflection that learns a rule adds
    `extra` after it."""

    extra: str = ""

    def __call__(self, call: Call) -> str | Exception:
        if is_contract_check(call):
            return meaning(call)
        return super().__call__(call)

    def write(self, call: Call) -> str:
        text = super().write(call)
        learned = call.sample >= fast_prompts.REFLECT_SAMPLE and not call.sample % 2
        return f"{text} {self.extra}" if learned and self.extra else text


def world(extra: str = "") -> Meaning:
    base = refunds()
    return Meaning(task=base.task, passes=base.passes, intake=base.intake, extra=extra)


class Unshown:
    """A layer above the run's stack that takes the examples decided by the day rule out of every
    contract check before it is sent: the twin whose check sees no example supporting it."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner

    def complete(self, call: Call) -> Reply:
        if is_contract_check(call):
            sent = json.loads(call.user)
            sent["examples"] = [e for e in sent["examples"] if RULE_OF[e["input"]] != DAYS]
            call = dataclasses.replace(call, user=json.dumps(sent))
        return self._inner.complete(call)


def contract_checks(result) -> list[Call]:
    return [c for c in result.calls("judge") if is_contract_check(c)]


@pytest.mark.parametrize("supported", [True, False], ids=["shown", "no example of the day rule"])
def test_the_rule_a_reflection_learns_passes_the_check_when_a_pick_example_supports_it(
    tmp_path, supported
):
    result = run(
        tmp_path,
        world(),
        plan("fast", 0),
        prompt=PROMPT,
        plan=BALANCED,
        examples=examples(TWO)[:6],
        wrap=None if supported else Unshown,
    )
    outcome = result.outcome
    assert outcome.status == "improved"
    assert outcome.prompt == (BOTH if supported else f"{PROMPT} {PRICE}")
    assert len(reflections(result)) == 2
    checks = contract_checks(result)
    assert len(checks) == 2  # stage C and the C2 of round 1: the examples add no call
    # the pick in the label order (WP25: approve, escalate, deny, then again)
    picked = ("e1", "e2", "e3", "e5", "e4", "e6") if supported else ("e1", "e2", "e5", "e4")
    shown = [ALL[s][0] for s in picked]  # as the raw model got them, after `Unshown`
    assert all([e["input"] for e in json.loads(c.user)["examples"]] == shown for c in checks)


def test_a_checked_run_verifies_the_supported_rule_on_the_held_out_examples(tmp_path):
    result = run(
        tmp_path,
        world(),
        plan("checked", 3),
        prompt=PROMPT,
        plan=BALANCED,
        examples=examples(TWO),
    )
    assert (result.outcome.prompt, result.outcome.verified) == (BOTH, True)


@pytest.mark.parametrize("extra", [FRENCH, LOYALTY], ids=["French", "loyalty program"])
def test_an_unrelated_requirement_a_reflection_adds_is_vetoed_with_the_examples_shown(
    tmp_path, extra
):
    result = run(
        tmp_path,
        world(extra),
        plan("fast", 0),
        prompt=PROMPT,
        plan=BALANCED,
        examples=examples(TWO)[:6],
    )
    assert result.outcome.prompt == f"{PROMPT} {PRICE}"
    (round_one,) = contract_checks(result)[1:]
    assert any(extra in s["output"] for s in json.loads(round_one.user)["scenarios"])


def test_outside_reference_mode_the_contract_check_is_shown_no_example(tmp_path):
    """A set with an example that has no reference decides pairwise, and the quick tier picks on
    no example: their contract checks carry no example and the instruction they always had."""
    mixed = [*examples(TWO)[:5], Scenario("e9", ALL["e9"][0])]
    pairwise = run(
        tmp_path / "pairwise",
        world(),
        plan("fast", 0),
        prompt=PROMPT,
        plan=BALANCED,
        examples=mixed,
    )
    quick = run(
        tmp_path / "quick", world(), QUICK, prompt=PROMPT, plan=BALANCED, examples=examples(TWO)
    )
    many, (one,) = contract_checks(pairwise), contract_checks(quick)
    assert len(many) == 2  # stages C and C2
    assert [c.system for c in [*many, one]] == [CONTRACT_MANY_SYSTEM] * 2 + [CONTRACT_SYSTEM]
    for call in [*many, one]:
        assert "examples" not in json.loads(call.user)
        assert NO_NEW_GOAL_SUPPORT not in call.user
