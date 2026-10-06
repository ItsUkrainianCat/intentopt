"""Test doubles shared by unit and acceptance tests; import them as `from fakes import ...`.

Owned by the lead. `ScriptedBackend` implements the `Backend` protocol and stands in for the raw
`ClaudeCli` layer (ADR-004): it answers from a function, records every call and can inject
failures, so no test needs a real model.
"""

import json
from collections.abc import Callable, Mapping, Sequence

from autoimprover.pairwise_text import PAIRWISE_BATCH_SYSTEM
from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END, Call, CallError, Reply

Script = Callable[[Call], "str | Exception"]


class FakeClock:
    """A monotonic clock a test advances by hand; pass `clock.now` as the `now` seam."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class ScriptedBackend:
    def __init__(self, script: Script, duration_s: float = 0.0, clock: FakeClock | None = None):
        self._script = script
        self._duration_s = duration_s
        self._clock = clock
        self.calls: list[Call] = []

    def complete(self, call: Call) -> Reply:
        self.calls.append(call)
        answer = self._script(call)
        if isinstance(answer, Exception):
            raise answer
        if self._clock is not None:
            self._clock.advance(self._duration_s)
        return Reply(
            text=answer,
            tokens_in=len(call.user),
            tokens_out=len(answer),
            duration_s=self._duration_s,
        )

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


def reflection_reply(
    instruction: str,
    why: Sequence[str] = ("tightened the wording", "kept the output format", "kept every literal"),
) -> str:
    """A reflection reply: the new instruction between delimiter lines, then bullet lines
    (ADR-008). The instruction may contain fenced code blocks; they stay intact."""
    bullets = "\n".join(f"- {line}" for line in why)
    return f"{INSTRUCTION_BEGIN}\n{instruction}\n{INSTRUCTION_END}\n{bullets}"


def pairwise_reply(
    call: Call, better: Callable[[str], bool] = lambda answer: answer.startswith("GOOD")
) -> str:
    """A batched pairwise reply (`pairwise_text.PAIRWISE_BATCH_SCHEMA`): per scenario the answer
    `better` holds for wins over one it does not, else a tie."""
    results = []
    for item in json.loads(call.user)["scenarios"]:
        a, b = better(item["answer_A"]), better(item["answer_B"])
        winner = "A" if a and not b else "B" if b and not a else "tie"
        results.append(
            {"scenario": item["scenario"], "winner": winner, "reason": f"{winner} answers better"}
        )
    return json.dumps({"results": results})


def happy_backend(improved_prompt: str, kind: str = "task", n: int = 12) -> ScriptedBackend:
    """A complete scripted model for end-to-end tests.

    The task model's output is good exactly when the prompt text it was given contains MARKER;
    the judge passes a scoring check exactly when the output is good and passes every check of
    the R6 contract check (scenario `contract`, whose "output" is the candidate prompt); reflection
    always proposes
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
        if call.role == "judge" and call.system == PAIRWISE_BATCH_SYSTEM:
            return pairwise_reply(call)
        if call.role == "judge":
            return judge_reply(
                call,
                lambda scenario, _c, output: (
                    scenario.startswith("contract") or output.startswith("GOOD")
                ),
            )
        return reflection_reply(improved_prompt)

    return ScriptedBackend(script)
