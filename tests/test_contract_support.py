"""The meaning check accepts what a pick example supports (SPEC R6; ADR-013 amendment of 2026-10-07,
WP24): in reference mode the contract check also receives the pick examples, input and reference,
as data (at most 8, each field cut to 300 characters), and `no-new-goal` passes a requirement the
candidate adds when one of them supports it; the pass counts only when its quote is the input of a
shown example, verbatim, which the code checks, not the model. A requirement no shown example
supports, or an unrelated one, is still vetoed. Without examples the call is byte for byte the call
it was before WP24."""

import hashlib
import json
from collections.abc import Callable

import pytest
from fakes import ScriptedBackend, judge_reply

from autoimprover import contract_text
from autoimprover.contract import Violation, check, check_many
from autoimprover.types import Call, Contract, Scenario

JUDGE = "claude-opus-5-5"
PROMPT = "Decide what to do with this refund request."
GOAL = "decide what to do with a refund request"
PRICE = "Escalate a purchase of $200 or more."
LEARNED = Contract(goal=GOAL, kind="task", from_examples=(PRICE,))
LATE = Scenario("e1", "A jacket for $45, bought 45 days ago: the zipper broke.", expected="deny")
FINE = Scenario("e2", "A mug for $20, bought 2 days ago: it arrived chipped.", expected="approve")
RUDE = Scenario("e3", "My order is late again.", criteria=("is polite", "offers a refund"))
HELD = Scenario("e9", "Shoes for $30, bought 40 days ago: the sole came off.", expected="deny")
PICK = (LATE, FINE, RUDE)
DAYS = "Deny a request made more than 30 days after the purchase."
SUPPORTED = f"{PROMPT} {DAYS}"
FRENCH = f"{PROMPT} Always answer in French."
LOYALTY = f"{PROMPT} Mention our loyalty program."


def quoting(no_new_goal: Callable[[Call], str], passes: bool = True) -> ScriptedBackend:
    """A contract judge that passes every check (each with a quote from its candidate, as the
    fakes do) and answers `no-new-goal` with `passes` and the quote `no_new_goal(call)`."""

    def script(call: Call) -> str:
        body = json.loads(judge_reply(call))
        for result in body["results"]:
            for item in result["checks"]:
                if item["id"] == "no-new-goal":
                    item.update({"pass": passes, "quote": no_new_goal(call)})
        return json.dumps(body)

    return ScriptedBackend(script)


def sent(call: Call) -> list[dict]:
    return json.loads(call.user)["examples"]


# --- (a) a requirement a pick example supports passes, quoting its input --------------------------


def test_a_requirement_a_pick_example_supports_passes_with_that_examples_input_as_the_quote():
    backend = quoting(lambda _call: LATE.input)
    assert check(backend, JUDGE, LEARNED, PROMPT, SUPPORTED, examples=PICK) == []
    assert check_many(backend, JUDGE, LEARNED, PROMPT, [SUPPORTED, PROMPT], examples=PICK) == [
        None,
        None,
    ]
    for call in backend.calls:
        assert [e["input"] for e in sent(call)] == [LATE.input, FINE.input, RUDE.input]


def test_the_quote_is_compared_after_whitespace_normalisation():
    spaced = "  " + LATE.input.replace(" ", "\n  ") + "\n"
    backend = quoting(lambda _call: spaced)
    assert check_many(backend, JUDGE, LEARNED, PROMPT, [SUPPORTED], examples=PICK) == [None]


def test_the_quote_is_compared_without_case():
    for quote in (LATE.input.upper(), LATE.input.lower()):
        backend = quoting(lambda _call, quote=quote: quote)
        assert check(backend, JUDGE, LEARNED, PROMPT, SUPPORTED, examples=PICK) == []


# --- (b) the same clause with any other quote fails -----------------------------------------------

OTHER_QUOTES = {
    "a passage of the candidate": DAYS,
    "a paraphrase of the input": "a refund asked for 45 days after buying a jacket",
    "a part of the input": LATE.input[:30],
    "the input of a held-out example": HELD.input,
    "the reference of the example": "deny",
    "blank": "   ",
}


@pytest.mark.parametrize("quote", OTHER_QUOTES.values(), ids=OTHER_QUOTES.keys())
def test_a_pass_of_no_new_goal_whose_quote_is_no_shown_input_is_a_fail(quote):
    backend = quoting(lambda _call: quote)
    (violation,) = check(backend, JUDGE, LEARNED, PROMPT, SUPPORTED, examples=PICK)
    assert violation.check_id == "no-new-goal"
    assert violation.text.startswith("no shown example's input as the quote: ")
    found = check_many(backend, JUDGE, LEARNED, PROMPT, [SUPPORTED], examples=PICK)
    assert found == [violation]


def test_the_other_checks_still_quote_the_candidate():
    """Only `no-new-goal` quotes an example: a keep check that quotes one is still a fail."""
    keeping = Contract(goal=GOAL, kind="task", keep=("refund",))

    def script(call: Call) -> str:
        body = json.loads(judge_reply(call))
        for item in body["results"][0]["checks"]:
            if item["id"] in ("keep-1", "no-new-goal"):
                item["quote"] = LATE.input
        return json.dumps(body)

    (violation,) = check(ScriptedBackend(script), JUDGE, keeping, PROMPT, SUPPORTED, examples=PICK)
    assert violation.check_id == "keep-1" and "no verbatim quote" in violation.text


# --- (c) an unrelated requirement is still vetoed -------------------------------------------------


@pytest.mark.parametrize("candidate", [FRENCH, LOYALTY], ids=["French", "loyalty program"])
def test_an_unrelated_requirement_is_vetoed_even_when_examples_are_shown(candidate):
    """The judge fails it (no example supports it); a judge that passes it anyway, quoting the
    candidate, is overruled by the code."""
    failed = quoting(lambda _call: candidate[-20:], passes=False)
    (violation,) = check_many(failed, JUDGE, LEARNED, PROMPT, [candidate], examples=PICK)
    assert violation is not None and violation.check_id == "no-new-goal"
    assert violation.text.startswith("failed: ")
    lenient = quoting(lambda _call: candidate[-20:])
    (violation,) = check_many(lenient, JUDGE, LEARNED, PROMPT, [candidate], examples=PICK)
    assert violation is not None and violation.check_id == "no-new-goal"


def test_the_question_says_what_a_shown_example_supports_and_what_still_fails():
    backend = quoting(lambda _call: LATE.input)
    check_many(backend, JUDGE, LEARNED, PROMPT, [SUPPORTED], examples=PICK)
    (call,) = backend.calls
    (item,) = json.loads(call.user)["scenarios"]
    question = next(c["text"] for c in item["checks"] if c["id"] == "no-new-goal")
    assert question.endswith(contract_text.NO_NEW_GOAL_SUPPORT)
    for words in ("supports", "reference answer", "quote", "input", "still is"):
        assert words in contract_text.NO_NEW_GOAL_SUPPORT
    for words in ("unrelated", "topic", "fact", "tone", "format"):
        assert words in contract_text.NO_NEW_GOAL_SUPPORT
    assert PRICE in question  # the rules of from_examples stay in the question (WP22)
    assert call.system == contract_text.CONTRACT_MANY_SYSTEM + contract_text.CONTRACT_EXAMPLES
    for words in ("examples", "data", "verbatim", "no-new-goal", "counts as a fail"):
        assert words in contract_text.CONTRACT_EXAMPLES
    assert not any(e.input in call.system for e in PICK)  # SPEC R18: user text on stdin only


# --- what the call carries ------------------------------------------------------------------------


def test_the_examples_ride_in_the_user_json_as_input_and_reference():
    backend = quoting(lambda _call: LATE.input)
    check(backend, JUDGE, LEARNED, PROMPT, SUPPORTED, examples=PICK)
    (call,) = backend.calls
    assert sent(call) == [
        {"input": LATE.input, "expected": "deny", "criteria": []},
        {"input": FINE.input, "expected": "approve", "criteria": []},
        {"input": RUDE.input, "expected": None, "criteria": ["is polite", "offers a refund"]},
    ]
    assert call.system == contract_text.CONTRACT_SYSTEM + contract_text.CONTRACT_EXAMPLES


def test_at_most_8_examples_are_shown_each_field_cut_to_300_characters():
    many = [Scenario(f"e{n}", f"ticket {n} " + "x" * 400, expected="y" * 400) for n in range(10)]
    many[0] = Scenario("e0", "z" * 400, criteria=("c" * 400, "short"))
    backend = quoting(lambda call: sent(call)[0]["input"])
    assert check_many(backend, JUDGE, LEARNED, PROMPT, [SUPPORTED], examples=many) == [None]
    shown = sent(backend.calls[0])
    assert len(shown) == 8 and [e["input"][:8] for e in shown[1:3]] == ["ticket 1", "ticket 2"]
    assert shown[0] == {"input": "z" * 300, "expected": None, "criteria": ["c" * 300, "short"]}
    assert all(len(e["input"]) == 300 and e["expected"] == "y" * 300 for e in shown[1:])
    late = quoting(lambda _call: many[8].input[:300])  # the ninth was never shown
    (violation,) = check_many(late, JUDGE, LEARNED, PROMPT, [SUPPORTED], examples=many)
    assert violation is not None and violation.check_id == "no-new-goal"


def test_a_candidate_that_lost_a_literal_is_still_not_sent():
    original = f"{PROMPT} Reply with `deny` or `approve`."
    backend = quoting(lambda _call: LATE.input)
    found = check_many(backend, JUDGE, LEARNED, original, [SUPPORTED], examples=PICK)
    assert found == [Violation("literal", "`deny`")] and backend.calls == []


# --- (e) outside reference mode the call is the call it was ---------------------------------------

PLAIN = Contract(
    goal="summarise a bug report",
    kind="template",
    keep=("Acme",),
    constraints=("under 100 words",),
    output_format="markdown list",
    language="en",
)
ORIGINAL = "Summarise the report for {audience} in `three bullets`. Mention Acme by name."
FIRST = "Summarise this report for {audience} in `three bullets`, and name Acme."
SECOND = "For {audience}: summarise the report in `three bullets`; name Acme."
# sha256 of system, user and schema joined by NUL, from the calls of commit 3da612e (before WP24):
# (check_many of FIRST and SECOND, check of FIRST) for PLAIN and for LEARNED.
BEFORE = {
    "plain": (
        "9399133e483ec2933ba45e927a8bcbdeb7b778681c86ca4a2e92200b618fcb44",
        "ae7322ef11af40b604ecb46eae67849109686862aa08ace4138d609058e09127",
    ),
    "learned": (
        "2ca1ea91b00320fa4fbb961d625a098b403fefc964f61181bb13f9f886cc93af",
        "0d7175a8f961d7ce5cb38d97793e165339b1af6b100ac442fe597c4fc5b84908",
    ),
}


def digest(call: Call) -> str:
    joined = "\0".join((call.system, call.user, call.json_schema or ""))
    return hashlib.sha256(joined.encode()).hexdigest()


@pytest.mark.parametrize("name", BEFORE)
def test_without_examples_the_check_calls_are_byte_for_byte_those_before_wp24(name):
    contract = {"plain": PLAIN, "learned": LEARNED}[name]
    for given in ({}, {"examples": ()}):
        many = ScriptedBackend(lambda call: judge_reply(call))
        check_many(many, JUDGE, contract, ORIGINAL, [FIRST, SECOND], **given)
        one = ScriptedBackend(lambda call: judge_reply(call))
        check(one, JUDGE, contract, ORIGINAL, FIRST, **given)
        assert (digest(many.calls[0]), digest(one.calls[0])) == BEFORE[name]


def test_without_examples_a_pass_still_quotes_the_candidate():
    backend = quoting(lambda _call: LATE.input)  # no example was shown, so no input to quote
    (violation,) = check(backend, JUDGE, LEARNED, PROMPT, SUPPORTED)
    assert violation.check_id == "no-new-goal" and "no verbatim quote" in violation.text
    assert check(quoting(lambda _call: DAYS), JUDGE, LEARNED, PROMPT, SUPPORTED) == []
