"""The scripted world of the fast pipeline tests (SPEC R25), shared by the `test_fast*.py` files:
a run folder under tmp_path, the real stack `Cached(Resilient(Budgeted(raw)))` over a raw model
that answers by the content of each call (role, sample, text), never by the order of the calls,
and `improve_fast` run on it. The tests in this file pin the world itself.
"""

import io
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend, intake_reply, judge_reply, synth_reply

from autoimprover import fastplan
from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.bench_judge import PAIRWISE_BATCH_SYSTEM
from autoimprover.fast import improve_fast
from autoimprover.fast_prompts import FAST_TASK_SUFFIX
from autoimprover.fastplan import FastPlan, fast_plan
from autoimprover.runner import count_tokens
from autoimprover.runstore import RunStore
from autoimprover.types import (
    INSTRUCTION_BEGIN,
    INSTRUCTION_END,
    Backend,
    Call,
    Kind,
    Models,
    Outcome,
    Plan,
    Scenario,
)

PROMPT = "Answer the user's request."
BETTER = f"Answer the user's request {MARKER}."
MODELS = Models(
    task="claude-haiku-4-5-20251001",
    judge="claude-opus-5-5",
    reflect="claude-sonnet-5-5",
    target="claude-fable-5-1",
)
PLAN = Plan(models=MODELS, wall_clock_s=30)
SHORT = count_tokens(PROMPT)
WAIT = 10.0  # seconds; only a broken implementation ever waits this long
# Mechanics tests are calibration-independent; the calibration is pinned in test_fastplan.py. They
# keep the per-call start-up they were written against: the plans below and every estimate the
# fast runner makes while a test runs (the autouse fixture `mechanics_latency_model`).
MECHANICS_OVERHEAD_S = 2.4


def mechanics_plan(time_s: int, workers: int, prompt_tokens: int, examples: bool) -> FastPlan:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fastplan, "OVERHEAD_S", MECHANICS_OVERHEAD_S)
        return fast_plan(time_s, workers, prompt_tokens, examples)


def two_generations(time_s: int, workers: int, rewrites: int, scenarios: int) -> FastPlan:
    """A fast plan of `rewrites` on `scenarios`, then 2 reflections, built from the planner's own
    stages: the planner never picks this shape for a prompt of SHORT tokens (at every time a plan
    with more rewrites or scenarios fits first; `test_fastplan.py` pins its choices), and the
    runner reads only the shape."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fastplan, "OVERHEAD_S", MECHANICS_OVERHEAD_S)
        b, c, d = fastplan.scoring_stages(rewrites, scenarios, workers)
        stages = (
            fastplan.stage_a(rewrites, scenarios, workers, SHORT),
            b,
            c,
            *fastplan.second_stages(2, scenarios, workers, SHORT),
            d,
        )
    seconds, calls = sum(s.seconds for s in stages), sum(s.calls for s in stages)
    return FastPlan("fast", time_s, workers, rewrites, scenarios, 0, stages, seconds, calls, 2, 2)


# Plans by shape: K rewrites, M scenarios, H held out, at MECHANICS_OVERHEAD_S: intake I = 7.829,
# task T = 4.543, rewrite Rs = 3.257, pairwise judge P(n) = 2.4 + 40 n/70 (P2 3.543), contract
# check J(n) = 2.4 + 75 n/70 (J1 3.471, J2 4.543, J3 5.614); stage B (K + 2) M task runs, stage C
# 2 K + 3 calls of max(P(M), J(K)).
QUICK = mechanics_plan(15, 4, SHORT, False)  # K=1
K1M2 = mechanics_plan(25, 4, SHORT, False)  # the smallest: synthesises 2
K2M2 = mechanics_plan(25, 8, SHORT, False)  # synthesises 2
K3M3 = mechanics_plan(30, 9, SHORT, False)  # I + 2 T + J3 = 22.529 <= 25.5: synthesises 3
K3M2_EXAMPLES = mechanics_plan(34, 6, SHORT, True)  # I + 2 T + 2 J3 = 28.143 <= 28.9
K1M2_EXAMPLES = mechanics_plan(25, 4, SHORT, True)  # the smallest
CHECKED = mechanics_plan(60, 1, SHORT, False)  # K=1, M=2, H=4 (the smallest): synthesises 6
# Two generations (from 45 s): the first, then K2=2 reflections; on 6 workers the second costs
# Rs + T + J2 = 3.257 + 4.543 + 4.543 = 12.343 s, on 4 Rs + T + 2 J2 = 16.886 s (C2: 5 calls).
TWO_K1 = two_generations(45, 4, 1, 2)  # K=1, M=2: I + 2 T + 2 P2 = 24.0, 40.886 in all
TWO_K3 = mechanics_plan(48, 6, SHORT, False)  # K=3, M=2: I + 2 T + 2 J3 + 12.343 <= 40.8


class Verbatim(str):
    """A rewrite reply sent as it is, without the delimiter lines around it."""


def good_by_marker(call: Call) -> str:
    return "GOOD answer" if MARKER in call.system + call.user else "BAD answer"


def by_quality(call: Call) -> str:
    """A pairwise judge that reads the answers: per scenario the one starting "GOOD" wins over one
    that does not, else a tie; its reason names the side and the scenario."""
    results = []
    for item in json.loads(call.user)["scenarios"]:
        a, b = (item[key].startswith("GOOD") for key in ("answer_A", "answer_B"))
        winner = "A" if a and not b else "B" if b and not a else "tie"
        results.append(
            {
                "scenario": item["scenario"],
                "winner": winner,
                "reason": f"{winner} for {item['scenario']}",
            }
        )
    return json.dumps({"results": results})


def position_a(call: Call) -> str:
    """A pairwise judge that always prefers the first answer."""
    names = [item["scenario"] for item in json.loads(call.user)["scenarios"]]
    return json.dumps(
        {"results": [{"scenario": n, "winner": "A", "reason": "first"} for n in names]}
    )


def is_pairwise(call: Call) -> bool:
    return call.role == "judge" and call.system == PAIRWISE_BATCH_SYSTEM


def tagged(call: Call) -> str:
    """As good_by_marker, but each prompt's outputs differ (by its length), so no two prompts'
    judge calls are the same call (the cache would answer the second)."""
    return f"{good_by_marker(call)} {len(prompt_of(call))}"


@dataclass
class World:
    """The raw model. `rewrites[i]` is the new prompt of rewrite variant i (the last one repeats),
    an Exception to raise, or a Verbatim reply; `reflections[i]` the same for reflection i of a
    second generation (sample 100 + i; without them a reflection answers as a rewrite would); the
    task output is `task(call)`; a pairwise judge call is answered by `pairwise(call)`; a scoring
    check passes when `passes(scenario, check id, output)` and a contract check when
    `contract_ok(candidate)`; `hook(call)` runs first (barriers, clock jumps)."""

    rewrites: Sequence[str | Exception] = (BETTER,)
    reflections: Sequence[str | Exception] = ()
    intake: str = field(default_factory=lambda: intake_reply("task"))
    task: Callable[[Call], str] = good_by_marker
    passes: Callable[[str, str, str], bool] = lambda _s, _c, output: output.startswith("GOOD")
    contract_ok: Callable[[str], bool] = lambda _candidate: True
    hook: Callable[[Call], None] = lambda _call: None
    pairwise: Callable[[Call], str | Exception] = by_quality

    def __call__(self, call: Call) -> str | Exception:
        self.hook(call)
        if call.role == "intake":
            return self.intake
        if call.role == "synth":
            return synth_reply(json.loads(call.user)["count"])
        if call.role == "reflect" and call.sample >= 100 and self.reflections:
            reply = self.reflections[min(call.sample - 100, len(self.reflections) - 1)]
            if isinstance(reply, Exception | Verbatim):
                return reply
            return f"{INSTRUCTION_BEGIN}\n{reply}\n{INSTRUCTION_END}"
        if call.role == "reflect":
            reply = self.rewrites[min(call.sample, len(self.rewrites) - 1)]
            if isinstance(reply, Exception | Verbatim):
                return reply
            return f"{INSTRUCTION_BEGIN}\n{reply}\n{INSTRUCTION_END}"
        if call.role == "task":
            return self.task(call)
        if is_pairwise(call):
            return self.pairwise(call)
        return judge_reply(
            call,
            lambda scenario, check_id, output: (
                self.contract_ok(output)
                if scenario.startswith("contract")
                else self.passes(scenario, check_id, output)
            ),
        )


@dataclass
class Result:
    outcome: Outcome
    raw: ScriptedBackend
    log: str
    run_id: str

    def calls(self, role: str) -> list[Call]:
        return [call for call in self.raw.calls if call.role == role]


def prompt_of(call: Call) -> str:
    """The prompt a task call ran: what follows the situation (kind task) or the system prompt,
    without the suffix every scoring run carries (`test_fast_suffix.py`)."""
    return call.system or call.user.removesuffix(FAST_TASK_SUFFIX).split("\n\n", 1)[1]


def runs_of(result: Result) -> list[tuple[str, int]]:
    """Each task call as (the prompt it ran, its sample), in order."""
    return [(prompt_of(c), c.sample) for c in result.calls("task")]


def scenario_of(call: Call) -> str:
    user = call.user.removesuffix(FAST_TASK_SUFFIX)
    return user if call.system else user.split("\n\n", 1)[0]


def judged_scenarios(call: Call) -> list[str]:
    return [item["scenario"] for item in json.loads(call.user)["scenarios"]]


def is_contract_check(call: Call) -> bool:
    return call.role == "judge" and judged_scenarios(call)[0].startswith("contract")


def scoring_judges(result: Result) -> list[Call]:
    """The judge calls that grade outputs, not the contract checks."""
    return [c for c in result.calls("judge") if not is_contract_check(c)]


def run(
    root: Path,
    world: World,
    fplan: FastPlan,
    *,
    workers: int = 1,
    prompt: str = PROMPT,
    examples: Sequence[Scenario] | None = None,
    plan: Plan = PLAN,
    deadline: float = 1000.0,
    clock: FakeClock | None = None,
    run_id: str = "",
    wrap: Callable[[Backend], Backend] | None = None,
    kind: Kind | None = None,
) -> Result:
    """One process: open the run folder (new unless `run_id`), build the real stack over a raw
    ScriptedBackend running `world`, with `deadline` on a fake clock, and run improve_fast."""
    store = RunStore.resume(root, run_id) if run_id else RunStore.open_or_create(root, plan, prompt)
    try:
        fake = clock or FakeClock()
        raw = ScriptedBackend(world)
        now = Clock(now=fake.now, elapsed=store.elapsed_s)
        budgeted = BudgetedBackend(
            raw, plan.budget, store.calls_used, now, deadline, on_call=store.save_progress
        )
        stack: Backend = CachedBackend(ResilientBackend(budgeted), store)
        log = io.StringIO()
        outcome = improve_fast(
            prompt,
            plan,
            fplan,
            backend=wrap(stack) if wrap else stack,
            budgeted=budgeted,
            clock=now,
            store=store,
            scenarios=examples,
            kind=kind,
            log=log,
            workers=workers,
        )
        assert outcome.calls_used == budgeted.used and outcome.run_dir == str(store.path)
        return Result(outcome, raw, log.getvalue(), store.run_id)
    finally:
        store.close()


@pytest.fixture(autouse=True)
def mechanics_latency_model(monkeypatch: pytest.MonkeyPatch):
    """Mechanics tests are calibration-independent; the calibration is pinned in
    test_fastplan.py. The fast runner's estimates use MECHANICS_OVERHEAD_S while a test runs."""
    monkeypatch.setattr(fastplan, "OVERHEAD_S", MECHANICS_OVERHEAD_S)


@pytest.fixture(autouse=True)
def no_disk_flush(monkeypatch: pytest.MonkeyPatch):
    """Writes stay atomic but are not flushed (durability is tested with the run store)."""
    monkeypatch.setattr(os, "fsync", lambda _fd: None)


# --- the world itself -----------------------------------------------------------------------------


def test_the_world_answers_by_content_not_by_order():
    world = World(rewrites=("first", ValueError("second"), Verbatim("third")))
    reflect = [Call(role="reflect", model="m", user="p", sample=i) for i in (3, 0, 2, 1)]
    answers = [world(call) for call in reflect]
    assert answers[0] == answers[2] == Verbatim("third")
    assert answers[1] == f"{INSTRUCTION_BEGIN}\nfirst\n{INSTRUCTION_END}"
    assert isinstance(answers[3], ValueError)
    synth = Call(role="synth", model="m", user=json.dumps({"count": 3}))
    assert len(json.loads(str(world(synth)))["scenarios"]) == 3


def test_the_world_passes_good_outputs_and_the_contract():
    world = World(contract_ok=lambda text: "ok" in text)
    request = {
        "scenarios": [
            {
                "scenario": "s1",
                "input": "i",
                "output": "GOOD x",
                "checks": [{"id": "a", "text": "t"}],
            },
            {
                "scenario": "s2",
                "input": "i",
                "output": "BAD x",
                "checks": [{"id": "a", "text": "t"}],
            },
            {
                "scenario": "contract-1",
                "input": "p",
                "output": "not",
                "checks": [{"id": "k", "text": "t"}],
            },
        ]
    }
    reply = json.loads(str(world(Call(role="judge", model="m", user=json.dumps(request)))))
    assert [r["checks"][0]["pass"] for r in reply["results"]] == [True, False, False]
