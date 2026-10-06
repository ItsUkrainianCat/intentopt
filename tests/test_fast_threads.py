"""The fast pipeline on threads (SPEC R22, R25; ADR-011): each stage's calls are in flight together
(barriers that break after a timeout only when they are not), the Outcome and the calls do not
depend on the number of workers, a resumed run replays every call from the cache, and the roles,
models and effort levels reach the raw layer exactly as given.
"""

import dataclasses
import threading
from collections import Counter

import pytest
from test_fast_world import (  # noqa: F401  (two autouse fixtures)
    BETTER,
    CHECKED,
    K3M2_EXAMPLES,
    K3M3,
    MODELS,
    QUICK,
    TWO_K1,
    WAIT,
    World,
    mechanics_latency_model,
    no_disk_flush,
    run,
    tagged,
)

from autoimprover.types import Backend, Call, Reply, Scenario

CLEAR = f"{BETTER} Be clear."
VERY_CLEAR = f"{BETTER} Be very clear."
THREE = (BETTER, CLEAR, VERY_CLEAR)


def barrier_world(roles: set[str], parties: int, rewrites=THREE) -> World:
    """Every call of `roles` waits until `parties` of them are in flight. Each prompt's outputs
    differ (`tagged`), so no two judge calls are one call the cache would answer."""
    barrier = threading.Barrier(parties)

    def hook(call: Call) -> None:
        if call.role in roles:
            barrier.wait(WAIT)  # BrokenBarrierError unless all `parties` calls are in flight

    return World(rewrites=rewrites, hook=hook, task=tagged)


def test_stage_a_runs_the_intake_the_synthesis_and_the_rewrites_in_one_wave(tmp_path):
    world = barrier_world({"intake", "synth", "reflect"}, 5)
    assert run(tmp_path, world, K3M3, workers=5).outcome.prompt == BETTER


def test_stage_b_runs_every_prompt_on_every_scenario_in_one_wave(tmp_path):
    """The original twice and 3 rewrites on 3 scenarios: 15 task calls in flight at once, not
    run after run."""
    result = run(tmp_path, barrier_world({"task"}, 15), K3M3, workers=15)
    assert result.outcome.prompt == BETTER


def test_stage_c_runs_the_judge_calls_and_the_contract_check_in_one_wave(tmp_path):
    """5 runs (the original twice): 5 judge calls and 1 contract check, K + 3 calls at once."""
    result = run(tmp_path, barrier_world({"judge"}, 6), K3M3, workers=6)
    assert result.outcome.prompt == BETTER


def test_stage_r_writes_the_reflections_side_by_side(tmp_path):
    barrier = threading.Barrier(2)

    def hook(call: Call) -> None:
        if call.role == "reflect" and call.sample >= 100:
            barrier.wait(WAIT)  # BrokenBarrierError unless both reflections are in flight

    world = World(rewrites=("Answer it.",), reflections=(BETTER, CLEAR), hook=hook)
    assert run(tmp_path, world, TWO_K1, workers=4).outcome.prompt == BETTER


EXAMPLES = [Scenario(id=f"e{n}", input=f"example {n}") for n in (1, 2)]
CASES = {
    "fast": (World(rewrites=THREE), K3M3, None),
    "loses": (World(rewrites=("Answer it.", "Answer it now.")), K3M3, None),
    "examples": (World(rewrites=THREE), K3M2_EXAMPLES, EXAMPLES),
    "quick": (World(), QUICK, None),
    "checked": (World(rewrites=THREE), CHECKED, None),
    "two generations": (World(rewrites=("Answer it.",), reflections=THREE), TWO_K1, None),
}


@pytest.mark.parametrize("case", CASES)
def test_the_outcome_and_the_calls_do_not_depend_on_the_workers(tmp_path, case):
    world, fplan, examples = CASES[case]
    one = run(tmp_path / "one", world, fplan, workers=1, examples=examples)
    four = run(tmp_path / "four", world, fplan, workers=4, examples=examples)
    assert dataclasses.replace(four.outcome, run_dir="") == dataclasses.replace(
        one.outcome, run_dir=""
    )
    assert Counter(four.raw.calls) == Counter(one.raw.calls)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("workers", [1, 4])
def test_a_resumed_run_replays_every_call_from_the_cache(tmp_path, case, workers):
    world, fplan, examples = CASES[case]
    first = run(tmp_path, world, fplan, workers=workers, examples=examples)
    again = run(tmp_path, world, fplan, workers=workers, examples=examples, run_id=first.run_id)
    assert again.raw.calls == []
    assert again.outcome == first.outcome


EFFORT = {"intake": "low", "synth": "medium", "reflect": "high", "task": "xhigh", "judge": "max"}


class EffortByRole:
    """The effort layer of ADR-011, above the cache: it sets a call's effort from its role when
    the call has none, and records the effort each call arrived with."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        self.arrived: list[str | None] = []

    def complete(self, call: Call) -> Reply:
        self.arrived.append(call.effort)
        if call.effort is None:
            call = dataclasses.replace(call, effort=EFFORT[call.role])
        return self._inner.complete(call)


def test_roles_models_and_effort_reach_the_raw_layer_as_given(tmp_path):
    layers: list[EffortByRole] = []

    def wrap(inner: Backend) -> Backend:
        layers.append(EffortByRole(inner))
        return layers[0]

    result = run(tmp_path, World(rewrites=THREE), CHECKED, workers=4, wrap=wrap)
    assert result.outcome.verified
    assert set(layers[0].arrived) == {None}  # the pipeline leaves the level to the role
    seen = {(c.role, c.model, c.effort) for c in result.raw.calls}
    assert seen == {
        ("intake", MODELS.reflect, "low"),
        ("reflect", MODELS.reflect, "high"),
        ("synth", MODELS.reflect, "medium"),
        ("task", MODELS.task, "xhigh"),
        ("task", MODELS.target, "xhigh"),
        ("judge", MODELS.judge, "max"),
    }


def test_stage_e_runs_the_winner_and_the_original_side_by_side(tmp_path):
    """4 workers: the two prompts at once, two of each one's held-out runs at a time."""
    barrier = threading.Barrier(4)

    def hook(call: Call) -> None:
        if call.role == "task" and call.model == MODELS.target:
            barrier.wait(WAIT)

    result = run(tmp_path, World(hook=hook), CHECKED, workers=4)
    assert result.outcome.verified
