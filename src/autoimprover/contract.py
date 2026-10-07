"""The intent contract of a prompt, and the check that a rewrite keeps it (SPEC R5, R6, R9, R10b).

`extract_contract` asks one intake call for the contract (SPEC R5); given the pick examples of a
run whose examples all carry a reference, it sends them as data (`induce.example_data`) and the
contract records `from_examples`, what they show and the prompt leaves unsaid, which the
`no-new-goal` check then counts as intent (ADR-013). `check` vetoes a candidate that lost a literal
of the original (SPEC R9), and otherwise asks one judge call whether it still keeps the contract, a
pass counting only with a verbatim quote from the candidate (SPEC R6, R10b); shown the pick
examples of such a run, `no-new-goal` passes a requirement one of them supports, and its pass
counts only with the input of a shown example as its quote (the ADR-013 amendment of 2026-10-07);
`check_many` does the same for up to JUDGE_BATCH_MAX candidates in one call (SPEC R25). `literals`
finds the spans a rewrite must keep verbatim: code blocks, inline code, placeholders,
URLs, file paths and quoted strings (SPEC R9; `autoimprover.literals`, re-exported here).

Prompts and replies are untrusted data: a prompt travels as the user message or inside its JSON,
never in the fixed system instructions (`contract_text`), and a reply is parsed as JSON and never
evaluated. A reply that is not valid is asked again as a new sample, so the call cache cannot
serve the bad reply back (ADR-004, ADR-008); whatever the backend raises propagates unchanged. An
intake reply with one of GEPA's template tokens in any of its texts is not valid, because the
reflection template could not carry it (ADR-006).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast, get_args

from autoimprover.contract_text import (
    CONTRACT_EXAMPLES,
    CONTRACT_MANY_SYSTEM,
    CONTRACT_SYSTEM,
    INTAKE_EXAMPLES_SYSTEM,
    INTAKE_SYSTEM,
    NO_NEW_GOAL,
    NO_NEW_GOAL_EXAMPLES,
    NO_NEW_GOAL_SUPPORT,
)
from autoimprover.induce import INTAKE_EXAMPLES_SCHEMA, check_data, example_data
from autoimprover.literals import literals as literals
from autoimprover.literals import literals_preserved as literals_preserved
from autoimprover.types import (
    CALL_RETRIES,
    INTAKE_SCHEMA,
    JUDGE_BATCH_MAX,
    JUDGE_SCHEMA,
    Backend,
    Call,
    CallFailed,
    Check,
    CheckGroup,
    Contract,
    Kind,
    Scenario,
)

_MAX_CHECKS = 8
_CHECK_KEYS = ("id", "group", "text", "rule", "arg")
# GEPA renders the reflection template by plain replacement of these tokens, so a contract text
# holding one would have feedback spliced into it (ADR-006). The same tuple as in runner.py, kept
# here because runner imports this module.
_GEPA_TOKENS = ("<curr_param>", "<side_info>")

_CONTRACT_SCENARIO = "contract"
_LITERAL = "literal"  # the check id of a lost literal
_NO_NEW_GOAL = "no-new-goal"
_NO_INPUT = "no shown example's input as the quote: "  # a pass of no-new-goal the code refuses
_VIOLATION_TEXT_MAX = 80


@dataclass(frozen=True)
class Violation:
    """One way a candidate breaks the contract: `check_id` is "literal" for a literal it lost
    (`text` is the literal), else the id of the contract check it did not satisfy (SPEC R6)."""

    check_id: str
    text: str


def extract_contract(
    backend: Backend,
    model: str,
    prompt: str,
    kind: Kind | None = None,
    examples: Sequence[Scenario] = (),
) -> Contract:
    """The intent contract of `prompt` from one intake call to `model`, the reflection model
    (SPEC R5; ADR-008). A `kind` given (`--kind`) replaces the model's guess; the call is the same
    either way. With `examples` (the pick examples of a run whose examples all carry a reference)
    the user message is JSON with the prompt and the examples as data, and the contract records
    `from_examples` (ADR-013); a reply without the key records none. An invalid reply is asked
    again as `sample + 1`, at most CALL_RETRIES times, then CallFailed."""
    if kind is not None and kind not in get_args(Kind):
        raise ValueError(f"unknown prompt kind {kind!r}; allowed: {get_args(Kind)}")
    call = Call(
        role="intake",
        model=model,
        user=prompt,
        system=INTAKE_SYSTEM,
        json_schema=json.dumps(INTAKE_SCHEMA),
    )
    if examples:
        user = json.dumps({"prompt": prompt, "examples": example_data(examples)})
        schema = json.dumps(INTAKE_EXAMPLES_SCHEMA)
        call = dataclasses.replace(
            call, user=user, system=INTAKE_EXAMPLES_SYSTEM, json_schema=schema
        )
    contract = _ask(backend, call, lambda text: _contract(text, bool(examples)))
    return contract if kind is None else dataclasses.replace(contract, kind=kind)


def check(
    backend: Backend,
    judge_model: str,
    contract: Contract,
    original: str,
    candidate: str,
    examples: Sequence[Scenario] = (),
) -> list[Violation]:
    """How `candidate` breaks the contract of `original`; [] only when it breaks nothing (SPEC R6).

    Literals first, free: each literal of `original` missing from `candidate` is a violation
    (SPEC R9), and then no model is called, because the candidate already fails. Otherwise one
    judge call to `judge_model` asks the contract checks (ADR-008). A check holds only when the
    judge passes it with a quote that is not blank and occurs in the candidate after whitespace is
    normalised (SPEC R10b); a check failed, without such a quote, or not answered is a violation,
    because the contract check is a veto. With `examples`, the pick examples of a run whose
    examples all carry a reference, the call also carries them (`induce.check_data`) and
    `no-new-goal` passes a requirement one of them supports, but holds only when its quote is,
    case and whitespace runs ignored, the whole input of one example shown (ADR-013 amendment);
    without, the call is the one it always was. An invalid reply is asked again as `sample + 1`,
    at most CALL_RETRIES times, then CallFailed."""
    lost = [literal for literal in literals(original) if literal not in candidate]
    if lost:
        return [Violation(_LITERAL, _shorten(literal)) for literal in lost]
    shown = check_data(examples)
    checks = _contract_checks(contract, bool(shown))
    scenario = {
        "scenario": _CONTRACT_SCENARIO,
        "input": original,
        "output": candidate,
        "checks": [{"id": check_id, "text": text} for check_id, text in checks],
    }
    call = Call(
        role="judge",
        model=judge_model,
        user=_request([scenario], shown),
        system=CONTRACT_SYSTEM + (CONTRACT_EXAMPLES if shown else ""),
        json_schema=json.dumps(JUDGE_SCHEMA),
    )
    asked = [check_id for check_id, _ in checks]
    found = _ask(backend, call, lambda text: _many_verdicts(text, [_CONTRACT_SCENARIO], asked))
    return _violations(checks, found[_CONTRACT_SCENARIO], candidate, _inputs(shown))


def check_many(
    backend: Backend,
    judge_model: str,
    contract: Contract,
    original: str,
    candidates: Sequence[str],
    examples: Sequence[Scenario] = (),
) -> list[Violation | None]:
    """Per candidate, in order, its first violation of the contract of `original` or None, as
    `check` decides (SPEC R6, R9, R10b), from one judge call for all (SPEC R25): one scenario
    each, named contract-1, contract-2, ... A candidate that lost a literal is not sent (its
    violation is that literal); nothing to send, no call; more than JUDGE_BATCH_MAX, ValueError.
    `examples` are shown and decide `no-new-goal`'s quote as in `check`. Invalid replies are
    retried as in `check`."""
    kept = literals(original)
    found: list[Violation | None] = []
    for candidate in candidates:
        lost = next((literal for literal in kept if literal not in candidate), None)
        found.append(None if lost is None else Violation(_LITERAL, _shorten(lost)))
    sent = [index for index, violation in enumerate(found) if violation is None]
    if len(sent) > JUDGE_BATCH_MAX:
        raise ValueError(f"one contract check holds at most {JUDGE_BATCH_MAX} candidates")
    if not sent:
        return found
    shown = check_data(examples)
    checks = _contract_checks(contract, bool(shown))
    names = [f"{_CONTRACT_SCENARIO}-{n}" for n in range(1, len(sent) + 1)]
    items = [{"id": check_id, "text": text} for check_id, text in checks]
    request = [
        {"scenario": name, "input": original, "output": candidates[index], "checks": items}
        for name, index in zip(names, sent, strict=True)
    ]
    call = Call(
        role="judge",
        model=judge_model,
        user=_request(request, shown),
        system=CONTRACT_MANY_SYSTEM + (CONTRACT_EXAMPLES if shown else ""),
        json_schema=json.dumps(JUDGE_SCHEMA),
    )
    asked = [check_id for check_id, _ in checks]
    answers = _ask(backend, call, lambda text: _many_verdicts(text, names, asked))
    for name, index in zip(names, sent, strict=True):
        violations = _violations(checks, answers.get(name, {}), candidates[index], _inputs(shown))
        found[index] = violations[0] if violations else None
    return found


def _request(scenarios: list[dict[str, Any]], shown: list[dict[str, Any]]) -> str:
    """The user JSON of a contract check: the scenarios, and the examples when some are shown."""
    return json.dumps({"scenarios": scenarios, **({"examples": shown} if shown else {})})


def _inputs(shown: list[dict[str, Any]]) -> frozenset[str]:
    """The inputs of the examples shown, whitespace normalised and case folded: the quotes
    `no-new-goal` may give."""
    return frozenset(flat for example in shown if (flat := _flat(example["input"]).casefold()))


def _violations(
    checks: list[tuple[str, str]],
    verdicts: dict[str, tuple[bool, str]],
    candidate: str,
    inputs: frozenset[str] = frozenset(),
) -> list[Violation]:
    """The checks failed, unanswered or passed without a quote from the candidate (R6, R10b); with
    the `inputs` of shown examples, a pass of `no-new-goal` needs one of them as its quote."""
    output = _flat(candidate)
    violations = []
    for check_id, text in checks:
        if check_id not in verdicts:
            violations.append(Violation(check_id, f"not answered: {text}"))
        elif not verdicts[check_id][0]:
            violations.append(Violation(check_id, f"failed: {text}"))
        elif check_id == _NO_NEW_GOAL and inputs:
            if _flat(verdicts[check_id][1]).casefold() not in inputs:
                violations.append(Violation(check_id, f"{_NO_INPUT}{text}"))
        elif not (quote := _flat(verdicts[check_id][1])) or quote not in output:
            violations.append(Violation(check_id, f"no verbatim quote from the candidate: {text}"))
    return violations


# --- replies -------------------------------------------------------------------------------------


def _ask[T](backend: Backend, call: Call, parse: Callable[[str], T]) -> T:
    """The parsed reply to `call`. A reply `parse` refuses (ValueError) is asked again under a new
    sample, a new cache key, at most CALL_RETRIES times; then CallFailed. Backend errors are not
    caught. The reply text is never echoed."""
    problem = ""
    for attempt in range(1 + CALL_RETRIES):
        if attempt:
            call = dataclasses.replace(call, sample=call.sample + 1)
        reply = backend.complete(call)
        try:
            return parse(reply.text)
        except ValueError as e:
            problem = str(e)
    raise CallFailed(
        f"{call.role} call to {call.model}: no valid reply in {1 + CALL_RETRIES} attempts, "
        f"last: {problem}"
    )


def _loads(text: str) -> Any:
    """The JSON value of `text`; ValueError if it is not JSON, also when it nests deeper than the
    parser can recurse (a RecursionError otherwise)."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as e:
        raise ValueError(f"not valid JSON ({type(e).__name__})") from None


def _text(value: object, where: str, *, blank: bool = True) -> str:
    """`value` if it is a string that can travel on to later calls: no NUL, no lone surrogate, no
    GEPA template token, and not blank unless `blank`; else ValueError naming `where`."""
    if not isinstance(value, str):
        raise ValueError(f"{where} is not a string")
    if not blank and not value.strip():
        raise ValueError(f"{where} is blank")
    if "\0" in value:
        raise ValueError(f"{where} contains a NUL character")
    if token := next((token for token in _GEPA_TOKENS if token in value), None):
        raise ValueError(f"{where} contains GEPA's template token {token}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{where} contains a lone surrogate") from None
    return value


def _texts(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"`{where}` is not a list")
    return tuple(_text(item, f"an item of `{where}`", blank=False) for item in value)


def _contract(text: str, examples: bool = False) -> Contract:
    """The contract an intake reply holds; ValueError when it is not valid for INTAKE_SCHEMA or
    for Check and Contract, has a blank goal, keep item, constraint, check id or check text, or
    has a GEPA template token in any text. Keys the schema does not name are ignored; with
    `examples`, `from_examples` is read (absent: none) and checked as the keep items are."""
    reply = _loads(text)
    if not isinstance(reply, dict):
        raise ValueError("not a JSON object")
    learned = reply.get("from_examples", []) if examples else []
    return Contract(
        goal=_text(reply.get("goal"), "`goal`", blank=False),
        kind=cast(Kind, _text(reply.get("kind"), "`kind`")),  # Contract checks the value
        keep=_texts(reply.get("keep"), "keep"),
        constraints=_texts(reply.get("constraints"), "constraints"),
        output_format=_text(reply.get("output_format"), "`output_format`"),
        language=_text(reply.get("language"), "`language`"),
        tone=_text(reply.get("tone"), "`tone`"),
        checks=_checks(reply.get("checks")),
        from_examples=_texts(learned, "from_examples"),
    )


def _checks(value: object) -> tuple[Check, ...]:
    if not isinstance(value, list):
        raise ValueError("`checks` is not a list")
    if not 1 <= len(value) <= _MAX_CHECKS:
        raise ValueError(f"not 1 to {_MAX_CHECKS} checks: the evaluator needs one to score")
    found: list[Check] = []
    for item in value:
        if not isinstance(item, dict) or not all(key in item for key in _CHECK_KEYS):
            raise ValueError(f"a check is not an object with {', '.join(_CHECK_KEYS)}")
        rule, arg = item["rule"], item["arg"]
        if rule is None and arg == "":
            arg = None  # ADR-008: "arg": "" on a judged check is read as null
        found.append(
            Check(
                id=_text(item["id"], "a check id", blank=False),
                group=cast(CheckGroup, _text(item["group"], "a check group")),  # Check checks it
                text=_text(item["text"], "a check text", blank=False),
                rule=None if rule is None else _text(rule, "a check rule"),
                arg=None if arg is None else _text(arg, "a check arg"),
            )
        )
    if len({check.id for check in found}) != len(found):
        raise ValueError("two checks share an id")
    return tuple(found)


def _contract_checks(contract: Contract, supported: bool = False) -> list[tuple[str, str]]:
    """(id, text) of the contract checks: one per keep item and per constraint, then three fixed
    ones (no new goal, same language, same output format). The ids are stable. The `no-new-goal`
    question names the rules the examples show, when the contract has some (ADR-013), and, when
    the check is `supported` by shown examples, what one of them supports (its amendment)."""
    learned = "; ".join(contract.from_examples)
    shown = NO_NEW_GOAL_EXAMPLES.format(rules=learned) if learned else ""
    shown += NO_NEW_GOAL_SUPPORT if supported else ""
    checks = [
        (f"keep-{n}", f"the candidate still keeps this, verbatim or with the same meaning: {item}")
        for n, item in enumerate(contract.keep, start=1)
    ]
    checks += [
        (f"constraint-{n}", f"the candidate still sets this constraint: {item}")
        for n, item in enumerate(contract.constraints, start=1)
    ]
    return [
        *checks,
        (
            _NO_NEW_GOAL,
            NO_NEW_GOAL + _aside("the original's goal: ", contract.goal) + shown,
        ),
        (
            "same-language",
            "the candidate is written in the same language as the original"
            + _aside("", contract.language),
        ),
        (
            "same-format",
            "the candidate asks for the same output format as the original"
            + _aside("", contract.output_format),
        ),
    ]


def _many_verdicts(
    text: str, names: list[str], asked: list[str]
) -> dict[str, dict[str, tuple[bool, str]]]:
    """Scenario name -> check id -> (pass, quote) of a contract-check reply; ValueError when it is
    not valid for JUDGE_SCHEMA, has no result, or answers a scenario or a check that was not
    asked, or one twice. A scenario or a check left out is simply absent."""
    reply = _loads(text)
    results = reply.get("results") if isinstance(reply, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError("not an object with a non-empty list of results")
    found: dict[str, dict[str, tuple[bool, str]]] = {}
    for result in results:
        name = result.get("scenario") if isinstance(result, dict) else None
        if not isinstance(name, str) or name not in names:
            raise ValueError("a result is not an object for a scenario that was asked")
        if name in found:
            raise ValueError("the reply answers a scenario twice")
        items = cast(dict[str, Any], result).get("checks")
        if not isinstance(items, list):
            raise ValueError("`checks` is not a list")
        verdicts = found[name] = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("a check is not an object")
            check_id, passed, quote = item.get("id"), item.get("pass"), item.get("quote")
            if not (
                isinstance(check_id, str) and isinstance(passed, bool) and isinstance(quote, str)
            ):
                raise ValueError("a check lacks a string id, a boolean pass or a string quote")
            if check_id not in asked:
                raise ValueError("the reply answers a check that was not asked")
            if check_id in verdicts:
                raise ValueError("the reply answers a check twice")
            verdicts[check_id] = (passed, quote)
    return found


def _shorten(text: str) -> str:
    if len(text) <= _VIOLATION_TEXT_MAX:
        return text
    return text[: _VIOLATION_TEXT_MAX - 3] + "..."


def _flat(text: str) -> str:
    """`text` with every run of whitespace made one space, and none at either end."""
    return " ".join(text.split())


def _aside(label: str, value: str) -> str:
    return f" ({label}{value})" if value.strip() else ""
