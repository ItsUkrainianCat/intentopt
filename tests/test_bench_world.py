"""The scripted model of the bench tests (SPEC R26), shared by the `test_bench*.py` files: it
answers by the content of each call, never by the order of the calls, so the pipeline's parallel
stages and the bench's parallel waves see the same replies whatever thread runs first. The tests
here pin the world itself."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from fakes import MARKER, intake_reply, judge_reply, pairwise_reply, synth_reply

from autoimprover.pairwise_text import PAIRWISE_BATCH_SYSTEM
from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END, Call

ORIGINAL = "Plan a weekend trip to the mountains for two people."
BETTER = f"Plan a weekend trip to the mountains for two people. {MARKER}"
NAIVE = "Plan a relaxing weekend trip to the mountains for two people."


def is_pairwise(call: Call) -> bool:
    """A call of the bench's pairwise judge (one scenario; not the fast tiers' batched call)."""
    batched = call.system == PAIRWISE_BATCH_SYSTEM
    return call.role == "judge" and '"winner"' in (call.json_schema or "") and not batched


def answers(call: Call) -> tuple[str, str]:
    request = json.loads(call.user)
    return request["answer_A"], request["answer_B"]


def winner(name: str, reason: str = "short reason") -> str:
    return json.dumps({"winner": name, "reason": reason})


def first_position(_call: Call) -> str:
    """A judge that always prefers the answer shown first: pure position bias."""
    return winner("A")


def by_content(call: Call) -> str:
    """A judge that prefers a GOOD answer to a BAD one, wherever it is shown."""
    a, b = answers(call)
    if a.startswith("GOOD") == b.startswith("GOOD"):
        return winner("tie")
    return winner("A" if a.startswith("GOOD") else "B")


@dataclass
class BenchWorld:
    """The raw model: the pipeline's rewrites propose `rewrite`, the naive baseline `naive`; a task
    output is GOOD when the prompt it ran holds MARKER (`task` replaces that); the pipeline's
    judge passes GOOD outputs and every contract check; the pairwise judge is `pairwise`."""

    rewrite: str = BETTER
    naive: str = NAIVE
    kind: str = "task"
    pairwise: Callable[[Call], str | Exception] = by_content
    task: Callable[[Call], str | Exception] = field(
        default=lambda call: "GOOD answer" if MARKER in call.system + call.user else "BAD answer"
    )

    def __call__(self, call: Call) -> str | Exception:
        if call.role == "intake":
            return intake_reply(self.kind)
        if call.role == "synth":
            return synth_reply(json.loads(call.user)["count"])
        if call.role == "reflect":
            naive = call.system.startswith("Improve this prompt.")
            text = self.naive if naive else self.rewrite
            return f"{INSTRUCTION_BEGIN}\n{text}\n{INSTRUCTION_END}"
        if call.role == "task":
            return self.task(call)
        if is_pairwise(call):
            return self.pairwise(call)
        if call.system == PAIRWISE_BATCH_SYSTEM:
            return pairwise_reply(call)
        return judge_reply(
            call,
            lambda scenario, _c, output: (
                scenario.startswith("contract") or output.startswith("GOOD")
            ),
        )


def test_the_world_answers_each_role_by_content():
    world = BenchWorld()
    synth = Call(role="synth", model="m", user=json.dumps({"prompt": "p", "count": 3}))
    assert len(json.loads(str(world(synth)))["scenarios"]) == 3
    good = Call(role="task", model="m", user=f"situation\n\n{BETTER}")
    bad = Call(role="task", model="m", user=f"situation\n\n{ORIGINAL}")
    assert (world(good), world(bad)) == ("GOOD answer", "BAD answer")
    naive = Call(role="reflect", model="m", user=ORIGINAL, system="Improve this prompt. Reply...")
    rewrite = Call(role="reflect", model="m", user="x", system="You rewrite a prompt")
    assert NAIVE in str(world(naive)) and BETTER in str(world(rewrite))


def test_the_two_pairwise_judges():
    pair = {"request": "r", "situation": "s", "answer_A": "BAD x", "answer_B": "GOOD y"}
    call = Call(role="judge", model="m", user=json.dumps(pair), json_schema='{"winner": 1}')
    assert is_pairwise(call)
    assert json.loads(first_position(call))["winner"] == "A"
    assert json.loads(by_content(call))["winner"] == "B"
