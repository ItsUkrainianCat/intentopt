"""Test doubles shared by unit and acceptance tests; import them as `from fakes import ...`.

Owned by the lead. `ScriptedBackend` implements the `Backend` protocol and stands in for the raw
`ClaudeCli` layer (ADR-004): it answers from a function, records every call and can inject
failures, so no test needs a real model.
"""

import json
from collections.abc import Callable, Mapping, Sequence

from autoimprover.types import Call, CallError, Reply

Script = Callable[[Call], "str | Exception"]


class ScriptedBackend:
    def __init__(self, script: Script) -> None:
        self._script = script
        self.calls: list[Call] = []

    def complete(self, call: Call) -> Reply:
        self.calls.append(call)
        answer = self._script(call)
        if isinstance(answer, Exception):
            raise answer
        return Reply(text=answer, tokens_in=len(call.user), tokens_out=len(answer))

    def count(self, role: str | None = None) -> int:
        return sum(1 for c in self.calls if role is None or c.role == role)


def by_role(replies: Mapping[str, str | Exception | Sequence[str | Exception]]) -> ScriptedBackend:
    """Answer per role: a constant, or a list served in order (the last entry repeats)."""
    served: dict[str, int] = {}

    def script(call: Call) -> str | Exception:
        answer = replies[call.role]
        if isinstance(answer, str | Exception):
            return answer
        i = served.get(call.role, 0)
        served[call.role] = i + 1
        return answer[min(i, len(answer) - 1)]

    return ScriptedBackend(script)


def failing(message: str = "backend down") -> ScriptedBackend:
    """A raw backend whose every attempt fails with CallError, the way ClaudeCli does (SPEC R24).
    Tests inject it with `cli.main(argv, backend=...)`, which replaces the raw layer only, so the
    retry and failure counting of `Resilient` still run."""
    return ScriptedBackend(lambda _call: CallError(message))


# --- Reply builders (wire formats: docs/adr/ADR-008, schemas: autoimprover.types) ---------------

MARKER = "[[better]]"  # a candidate containing it makes `happy_backend` produce good output


def intake_reply(kind: str = "task", checks: Sequence[dict] | None = None, **fields: object) -> str:
    """An intake reply (INTAKE_SCHEMA). Default: one judged content check."""
    body: dict = {
        "goal": "answer the user's request well",
        "kind": kind,
        "keep": [],
        "constraints": [],
        "output_format": "",
        "language": "en",
        "tone": "neutral",
        "checks": list(
            checks
            if checks is not None
            else [
                {
                    "id": "c1",
                    "group": "content",
                    "text": "answers the request",
                    "rule": None,
                    "arg": None,
                }
            ]
        ),
    }
    body.update(fields)
    return json.dumps(body)


def synth_reply(n: int = 12) -> str:
    """A synthesis reply (SYNTH_SCHEMA) with n scenarios s1..sn."""
    return json.dumps(
        {"scenarios": [{"id": f"s{i}", "input": f"situation {i}"} for i in range(1, n + 1)]}
    )


def judge_reply(call: Call, passes: Callable[[str, str, str], bool] = lambda *_: True) -> str:
    """Answer a judge request (JSON in `call.user`, see ADR-008) with a JUDGE_SCHEMA reply.

    `passes(scenario_id, check_id, output)` decides each check; the quote is a slice of the output,
    so the verbatim-quote rule (R10b) holds.
    """
    request = json.loads(call.user)
    results = []
    for item in request["scenarios"]:
        output = item["output"]
        checks = [
            {
                "id": c["id"],
                "pass": passes(item["scenario"], c["id"], output),
                "quote": output[:20],
            }
            for c in item["checks"]
        ]
        results.append({"scenario": item["scenario"], "checks": checks})
    return json.dumps({"results": results})


def reflection_reply(instruction: str, why: Sequence[str] = ("tightened the wording",)) -> str:
    """A reflection reply: the new instruction in one fenced block, then bullet lines (ADR-008)."""
    return "```\n" + instruction + "\n```\n" + "\n".join(f"- {line}" for line in why)


def happy_backend(improved_prompt: str, kind: str = "task", n: int = 12) -> ScriptedBackend:
    """A complete scripted model for end-to-end tests.

    The task model's output is good exactly when the prompt text it was given contains MARKER;
    the judge passes a check exactly when the output is good; reflection always proposes
    `improved_prompt`. So `improved_prompt` containing MARKER is returned as an improvement, and
    one without it is not: use both as twins so a test cannot pass for the wrong reason.
    """

    def script(call: Call) -> str:
        if call.role == "intake":
            return intake_reply(kind=kind)
        if call.role == "synth":
            return synth_reply(n)
        if call.role == "task":
            return "GOOD answer" if MARKER in call.system + call.user else "BAD answer"
        if call.role == "judge":
            return judge_reply(call, lambda _s, _c, output: output.startswith("GOOD"))
        return reflection_reply(improved_prompt)

    return ScriptedBackend(script)
