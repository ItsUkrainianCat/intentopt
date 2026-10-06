"""`claude --effort` per role (SPEC R25; ADR-011 "Customisation"): `EffortBackend` sets a call's
effort from its role when the call has none, above the cache, so the effort is part of the cache
key and no module that builds calls changes."""

import pytest
from fakes import ScriptedBackend

from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.efforts import EffortBackend
from autoimprover.runstore import RunStore, runs_root
from autoimprover.types import DEFAULT_MODELS, Call, Efforts, Plan

EFFORTS = Efforts(task="low", judge="high", reflect="max")
ROLES = {"task": "low", "judge": "high", "intake": "max", "synth": "max", "reflect": "max"}


def echo() -> ScriptedBackend:
    return ScriptedBackend(lambda call: f"{call.role} at {call.effort}")


@pytest.mark.parametrize(("role", "level"), list(ROLES.items()))
def test_a_call_without_an_effort_gets_its_roles_level(role, level):
    inner = echo()
    reply = EffortBackend(inner, EFFORTS).complete(Call(role, "m", "hi"))
    assert reply.text == f"{role} at {level}"
    assert inner.calls == [Call(role, "m", "hi", effort=level)]


def test_a_call_that_names_its_effort_keeps_it():
    inner = echo()
    EffortBackend(inner, EFFORTS).complete(Call("task", "m", "hi", effort="xhigh"))
    assert inner.calls[0].effort == "xhigh"


def test_a_role_whose_level_is_none_keeps_the_models_own():
    inner = echo()
    wrapper = EffortBackend(inner, Efforts(task=None, judge="low", reflect=None))
    for role in ("task", "intake", "synth", "reflect"):
        wrapper.complete(Call(role, "m", "hi"))
    wrapper.complete(Call("judge", "m", "hi"))
    assert [call.effort for call in inner.calls] == [None, None, None, None, "low"]


def test_no_effort_at_all_passes_every_call_unchanged():
    inner = echo()
    calls = [Call(role, "m", "hi", sample=3) for role in ROLES]
    for call in calls:
        EffortBackend(inner, Efforts()).complete(call)
    assert inner.calls == calls


@pytest.fixture
def store():
    opened = RunStore.open_or_create(runs_root(), Plan(models=DEFAULT_MODELS), "prompt")
    yield opened
    opened.close()


def stack(raw: ScriptedBackend, store: RunStore, efforts: Efforts) -> EffortBackend:
    budgeted = BudgetedBackend(raw, limit=10, used=0, clock=Clock(lambda: 0.0), deadline=100.0)
    return EffortBackend(CachedBackend(ResilientBackend(budgeted), store), efforts)


def test_the_effort_is_part_of_the_cache_key_because_it_is_set_above_the_cache(store):
    raw = echo()
    stack(raw, store, Efforts("low", "low", "low")).complete(Call("task", "m", "hi"))
    again = stack(raw, store, Efforts("high", "high", "high")).complete(Call("task", "m", "hi"))
    hit = stack(raw, store, Efforts("low", "low", "low")).complete(Call("task", "m", "hi"))
    assert [call.effort for call in raw.calls] == ["low", "high"]
    assert (again.text, again.cached) == ("task at high", False)
    assert (hit.text, hit.cached) == ("task at low", True)
