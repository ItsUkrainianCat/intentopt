"""The disk cache above the other layers, and the whole stack `Cached(Resilient(Budgeted(raw)))`
(SPEC R17, R22, R24; ADR-004, ADR-007). A real run folder under tmp_path; the raw layer is
`tests/fakes.py`."""

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import FakeClock, ScriptedBackend

from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.runstore import RunStore, RunStoreError, cache_key
from autoimprover.types import (
    DEFAULT_MODELS,
    BackendError,
    BudgetExhausted,
    Call,
    CallError,
    CallFailed,
    Plan,
    Reply,
    SessionNotLockedDown,
)

PLAN = Plan(models=DEFAULT_MODELS)
NOW = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
BASE = Call(
    role="judge",
    model="claude-opus-5-5",
    user='{"scenarios": []}',
    system="judge the outputs",
    json_schema='{"type": "object"}',
    sample=0,
)
VARIANTS = {
    "role": dataclasses.replace(BASE, role="intake"),
    "model": dataclasses.replace(BASE, model="claude-sonnet-5-5"),
    "user": dataclasses.replace(BASE, user='{"scenarios": [1]}'),
    "system": dataclasses.replace(BASE, system="judge the outputs strictly"),
    "json_schema": dataclasses.replace(BASE, json_schema=None),
    "sample": dataclasses.replace(BASE, sample=1),
}


@pytest.fixture
def store(tmp_path: Path):
    s = RunStore.open_or_create(tmp_path / "runs", PLAN, "a prompt", utcnow=lambda: NOW)
    yield s
    s.close()


def never_called(call: Call) -> Exception:
    return AssertionError(f"the inner layer was called for {call.role}")


def log_entries(store: RunStore) -> list[dict]:
    path = store.path / "calls.jsonl"
    return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []


def test_the_key_covers_every_call_field():
    assert [f.name for f in dataclasses.fields(Call)] == list(VARIANTS)
    keys = {cache_key(BASE), *(cache_key(v) for v in VARIANTS.values())}
    assert len(keys) == 1 + len(VARIANTS)
    assert cache_key(dataclasses.replace(BASE)) == cache_key(BASE)


def test_a_call_differing_in_any_one_field_is_a_miss(store: RunStore):
    raw = ScriptedBackend(lambda c: f"reply to {c.role}/{c.sample}")
    cached = CachedBackend(raw, store)
    cached.complete(BASE)
    for n, variant in enumerate(VARIANTS.values(), start=2):
        assert cached.complete(variant).cached is False
        assert raw.count() == n
    cached.complete(BASE)
    assert raw.count() == 1 + len(VARIANTS)


def test_a_miss_is_answered_live_stored_with_its_duration_and_returned_unchanged(
    store: RunStore,
):
    replies: list[Reply] = []

    class Inner:
        def complete(self, call: Call) -> Reply:
            replies.append(Reply(text="the verdict", tokens_in=11, tokens_out=22, duration_s=7.5))
            return replies[-1]

    reply = CachedBackend(Inner(), store).complete(BASE)
    assert reply is replies[0] and reply.cached is False
    entry = store.cache_get(BASE)
    assert entry is not None and entry.outcome == "ok" and entry.duration_s == 7.5
    assert entry.reply is not None and entry.reply.text == "the verdict"


def test_a_hit_makes_no_inner_call_and_carries_the_stored_reply_and_duration(store: RunStore):
    CachedBackend(ScriptedBackend(lambda _c: "the verdict", duration_s=7.5), store).complete(BASE)
    reply = CachedBackend(ScriptedBackend(never_called), store).complete(BASE)
    assert reply == Reply(
        text="the verdict", cached=True, tokens_in=len(BASE.user), tokens_out=11, duration_s=7.5
    )


def test_the_second_of_two_identical_calls_in_one_run_is_a_hit(store: RunStore):
    raw = ScriptedBackend(lambda _c: "same", duration_s=3.0)
    cached = CachedBackend(raw, store)
    first, second = cached.complete(BASE), cached.complete(BASE)
    assert raw.count() == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text and second.duration_s == 3.0


def test_a_tombstone_replays_as_call_failed_without_an_inner_call(store: RunStore):
    store.record_failure(BASE, "timeout after 300 s", 300.0)
    raw = ScriptedBackend(never_called)
    with pytest.raises(CallFailed, match="timeout after 300 s"):
        CachedBackend(raw, store).complete(BASE)
    assert raw.count() == 0


ERRORS = [
    CallFailed("judge failed 3 attempts"),
    CallError("exit 1"),
    BudgetExhausted("limit", cause="budget"),
    BudgetExhausted("deadline", cause="clock"),
    BackendError("three in a row"),
    SessionNotLockedDown("tools"),
    ValueError("bug"),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: type(e).__name__)
def test_an_inner_exception_propagates_and_nothing_is_stored(store: RunStore, error: Exception):
    raw = ScriptedBackend(lambda _c: error)
    cached = CachedBackend(raw, store)
    with pytest.raises(type(error)) as raised:
        cached.complete(BASE)
    assert raised.value is error
    assert store.cache_get(BASE) is None
    assert list((store.path / "cache").iterdir()) == []
    with pytest.raises(type(error)):
        cached.complete(BASE)
    assert raw.count() == 2


def test_live_calls_and_hits_are_logged_with_their_cached_flag(store: RunStore):
    cached = CachedBackend(ScriptedBackend(lambda _c: "out", duration_s=2.5), store)
    cached.complete(BASE)
    cached.complete(BASE)
    entries = log_entries(store)
    assert [(e["role"], e["model"], e["cached"], e["duration_s"]) for e in entries] == [
        ("judge", BASE.model, False, 2.5),
        ("judge", BASE.model, True, 2.5),
    ]
    assert entries[0]["tokens_out"] == entries[1]["tokens_out"] == 3


# --- Reading an entry (ADR-007: damaged or foreign is a miss and is deleted, newer is refused) --

DROP = object()


def entry(kind: str = "ok", **change: object) -> bytes:
    doc = {"schema_version": 1, "key_fields": dataclasses.asdict(BASE), "outcome": kind}
    doc |= {"reply": {"text": "t", "tokens_in": 1, "tokens_out": 2}} if kind == "ok" else {}
    doc |= {"error": "down"} if kind == "failed" else {}
    doc |= {"duration_s": 1.0}
    for key, value in change.items():
        doc.pop(key) if value is DROP else doc.update({key: value})
    return json.dumps(doc).encode()


BAD_ENTRIES = {
    "truncated": b'{"schema_version": 1, "key_fie',
    "empty": b"",
    "a list": b"[1, 2]",
    "not utf-8": b'{"schema_version": 1, "x": "\xff\xfe"}',
    "deeply nested": b"[" * 100_000 + b"]" * 100_000,
    "key fields of another call": entry(key_fields=dataclasses.asdict(VARIANTS["sample"])),
    "no key fields": entry(key_fields=DROP),
    "no version": entry(schema_version=DROP),
    "old version": entry(schema_version=0),
    "version true": entry(schema_version=True),
    "no outcome": entry(outcome=DROP),
    "unknown outcome": entry("maybe"),
    "ok without reply": entry(reply=DROP),
    "reply a string": entry(reply="t"),
    "reply text not text": entry(reply={"text": 5, "tokens_in": 1, "tokens_out": 2}),
    "negative tokens": entry(reply={"text": "t", "tokens_in": -1, "tokens_out": 2}),
    "tokens a bool": entry(reply={"text": "t", "tokens_in": 1, "tokens_out": True}),
    "reply with an extra field": entry(
        reply={"text": "t", "tokens_in": 1, "tokens_out": 2, "x": 1}
    ),
    "duration text": entry(duration_s="1"),
    "duration nan": entry(duration_s=float("nan")),
    "duration negative": entry(duration_s=-1.0),
    "duration too big for a float": entry(duration_s=10**400),
    "ok with an error": entry(error="x"),
    "tombstone with a reply": entry("failed", reply={"text": "t"}),
    "tombstone error not text": entry("failed", error=None),
    "tombstone without error": entry("failed", error=DROP),
}


@pytest.mark.parametrize("raw", BAD_ENTRIES.values(), ids=BAD_ENTRIES.keys())
def test_a_damaged_or_foreign_entry_is_a_miss_deleted_and_answered_live(store: RunStore, raw):
    path = store.path / "cache" / f"{cache_key(BASE)}.json"
    path.write_bytes(raw)
    assert store.cache_get(BASE) is None
    assert not path.exists()
    path.write_bytes(raw)
    inner = ScriptedBackend(lambda _c: "fresh")
    assert CachedBackend(inner, store).complete(BASE).text == "fresh"
    assert inner.count() == 1
    assert json.loads(path.read_text())["reply"]["text"] == "fresh"


def test_well_formed_entries_are_read(store: RunStore):
    path = store.path / "cache" / f"{cache_key(BASE)}.json"
    path.write_bytes(entry())
    hit = CachedBackend(ScriptedBackend(never_called), store).complete(BASE)
    assert (hit.text, hit.cached, hit.duration_s) == ("t", True, 1.0)
    path.write_bytes(entry("failed"))
    with pytest.raises(CallFailed, match="down"):
        CachedBackend(ScriptedBackend(never_called), store).complete(BASE)


def test_a_symlinked_entry_is_a_miss_and_only_the_link_is_removed(store: RunStore, tmp_path):
    planted = tmp_path / "planted.json"
    planted.write_bytes(entry())
    link = store.path / "cache" / f"{cache_key(BASE)}.json"
    link.symlink_to(planted)
    assert store.cache_get(BASE) is None
    assert not link.is_symlink() and planted.read_bytes() == entry()


def test_an_entry_from_a_newer_version_is_refused_and_kept(store: RunStore):
    path = store.path / "cache" / f"{cache_key(BASE)}.json"
    path.write_bytes(entry(schema_version=2))
    with pytest.raises(RunStoreError, match="newer version"):
        CachedBackend(ScriptedBackend(never_called), store).complete(BASE)
    assert path.read_bytes() == entry(schema_version=2)


def test_a_cache_folder_removed_by_hand_is_recreated(store: RunStore):
    (store.path / "cache").rmdir()
    cached = CachedBackend(ScriptedBackend(lambda _c: "again"), store)
    cached.complete(BASE)
    assert cached.complete(BASE).cached is True


# --- The whole stack (ADR-004) -------------------------------------------------------------------


def stack(store: RunStore, raw: ScriptedBackend, fake: FakeClock, limit: int = 10):
    budgeted = BudgetedBackend(
        raw, limit=limit, used=store.calls_used, clock=Clock(now=fake.now), deadline=100.0
    )
    return CachedBackend(ResilientBackend(budgeted), store), budgeted


def test_hits_cost_nothing_and_every_live_attempt_is_counted(store: RunStore):
    answers = iter([CallError("exit 1"), "ok"])
    raw = ScriptedBackend(lambda _c: next(answers))
    backend, budgeted = stack(store, raw, FakeClock())
    backend.complete(BASE)
    assert budgeted.used == 2
    backend.complete(BASE)
    assert (budgeted.used, raw.count()) == (2, 2)


def test_a_resumed_run_replays_hits_and_tombstones_free_and_continues_the_count(
    store: RunStore, tmp_path: Path
):
    raw = ScriptedBackend(lambda c: "ok" if c.sample == 0 else CallError("down"))
    backend, budgeted = stack(store, raw, FakeClock(), limit=10)
    backend.complete(BASE)
    failing = dataclasses.replace(BASE, sample=1)
    with pytest.raises(CallFailed) as failed:
        backend.complete(failing)
    store.record_failure(failing, str(failed.value), 0.0)
    store.save_progress(budgeted.used, 0.0)
    store.close()
    resumed = RunStore.resume(tmp_path / "runs", store.run_id)
    replay_raw = ScriptedBackend(never_called)
    replay, replay_budget = stack(resumed, replay_raw, FakeClock(), limit=4)
    assert replay.complete(BASE).cached is True
    with pytest.raises(CallFailed):
        replay.complete(failing)
    assert (replay_raw.count(), replay_budget.used) == (0, 4)
    with pytest.raises(BudgetExhausted):
        replay.complete(dataclasses.replace(BASE, sample=2))
    resumed.close()


# --- Tombstones written by the cache while the search runs (ADR-004, SPEC R22) -------------------


def test_record_failures_stores_a_call_that_failed_every_attempt_as_a_tombstone(store: RunStore):
    raw = ScriptedBackend(lambda c: CallError("exit 1") if c.sample == 1 else "ok")
    backend, budgeted = stack(store, raw, FakeClock())
    assert backend.record_failures is False
    backend.record_failures = True
    failing = dataclasses.replace(BASE, sample=1)
    with pytest.raises(CallFailed) as failed:
        backend.complete(failing)
    entry = store.cache_get(failing)
    assert entry is not None and (entry.outcome, entry.error) == ("failed", str(failed.value))
    assert entry.duration_s == 0.0
    with pytest.raises(CallFailed):
        backend.complete(failing)
    assert (raw.count(), budgeted.used) == (3, 3)
    assert backend.complete(BASE).text == "ok" and store.cache_get(BASE) is not None


@pytest.mark.parametrize("error", ERRORS[1:], ids=lambda e: type(e).__name__)
def test_record_failures_stores_nothing_but_a_call_failed(store: RunStore, error: Exception):
    cached = CachedBackend(ScriptedBackend(lambda _c: error), store)
    cached.record_failures = True
    with pytest.raises(type(error)):
        cached.complete(BASE)
    assert list((store.path / "cache").iterdir()) == []


def test_the_third_consecutive_failure_raises_backend_error_and_is_not_stored(store: RunStore):
    raw = ScriptedBackend(lambda c: CallError("exit 1"))
    backend, _ = stack(store, raw, FakeClock(), limit=20)
    backend.record_failures = True
    calls = [dataclasses.replace(BASE, sample=n) for n in range(3)]
    for call in calls[:2]:
        with pytest.raises(CallFailed):
            backend.complete(call)
    with pytest.raises(BackendError):
        backend.complete(calls[2])
    assert [store.cache_get(c) is not None for c in calls] == [True, True, False]
