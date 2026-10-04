"""The files inside one run folder: atomic JSON writes, the checkpoint, contract and scenarios, the
call log and the call cache with its tombstones, and how partial, corrupt, foreign or newer state
files are read (SPEC R17, R22, R24; ADR-007); damaged cache entries are in test_backend_layers.py.
Real files under tmp_path."""

import dataclasses
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autoimprover.runstore import RunStore, RunStoreError
from autoimprover.types import DEFAULT_MODELS, Call, Check, Contract, Plan, Reply, Scenario

PROMPT = "Rewrite the paragraph in plain English."
PLAN = Plan(models=DEFAULT_MODELS, budget=80)
NOW = datetime(2026, 10, 4, 9, 0, 0, tzinfo=UTC)
CONTRACT = Contract(
    goal="rewrite in plain English",
    kind="template",
    keep=("the term 'GEPA'",),
    constraints=("no more than 3 sentences",),
    output_format="one paragraph",
    language="en",
    tone="friendly",
    checks=(
        Check(id="c1", group="format", text="short", rule="max_chars", arg="400"),
        Check(id="c2", group="content", text="keeps the meaning"),
        Check(id="c3", group="constraints", text="no jargon", rule="not_contains", arg="utilise"),
    ),
)
SCENARIOS = [
    Scenario(id="s1", input="The utilisation of resources was suboptimal."),
    Scenario(id="s2", input="x", expected="a plain sentence", criteria=("short", "polite")),
]
TASK = Call(role="task", model="claude-haiku-4-5-20251001", user="situation 1", system="be brief")


@pytest.fixture
def store(tmp_path: Path):
    s = RunStore.open_or_create(tmp_path / "runs", PLAN, PROMPT, {}, utcnow=lambda: NOW)
    yield s
    s.close()


def key_of(call: Call) -> str:
    text = json.dumps(
        {"schema_version": 1, "call": dataclasses.asdict(call)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


def cache_files(store: RunStore) -> list[Path]:
    return sorted((store.path / "cache").iterdir())


# --- Atomic writes (ADR-007) ---------------------------------------------------------------------


def test_every_json_file_is_fsynced_in_its_own_folder_and_moved_into_place(
    store: RunStore, monkeypatch: pytest.MonkeyPatch
):
    synced: set[int] = set()  # inodes fsynced so far
    moves: list[tuple[Path, Path, bool]] = []
    real_replace, real_fsync = os.replace, os.fsync

    def replace(src, dst):
        moves.append((Path(src), Path(dst), os.stat(src).st_ino in synced))
        real_replace(src, dst)

    def fsync(fd):
        synced.add(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "fsync", fsync)
    store.save_progress(3, 4.5)
    store.search_start_or_record(3, 4.5)
    store.save_contract(CONTRACT)
    store.save_scenarios(SCENARIOS)
    store.cache_put(TASK, Reply(text="ok", duration_s=1.0))
    store.record_failure(dataclasses.replace(TASK, sample=1), "timeout", 300.0)
    targets = [dst.name for _, dst, _ in moves]
    assert targets[:4] == ["manifest.json", "checkpoint.json", "contract.json", "scenarios.json"]
    assert len(targets) == 6
    for src, dst, was_synced in moves:
        assert src.parent == dst.parent, "the temporary file must live in the target's folder"
        assert src != dst and not src.exists() and dst.exists()
        assert was_synced, "the temporary file is fsynced before the move"


def test_a_failed_move_keeps_the_previous_file_whole(
    store: RunStore, monkeypatch: pytest.MonkeyPatch
):
    store.save_progress(5, 50.0)
    before = (store.path / "manifest.json").read_bytes()

    def broken_replace(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="No space"):
        store.save_progress(6, 60.0)
    monkeypatch.undo()
    assert (store.path / "manifest.json").read_bytes() == before
    assert sorted(p.name for p in store.path.iterdir()) == ["cache", "manifest.json", "run.lock"]


def test_progress_that_would_not_read_back_is_refused_and_nothing_is_written(store: RunStore):
    before = (store.path / "manifest.json").read_bytes()
    for calls_used, elapsed in ((-1, 0.0), (True, 0.0), (1, -2.0), (1, float("nan"))):
        with pytest.raises(ValueError):
            store.save_progress(calls_used, elapsed)
    assert (store.path / "manifest.json").read_bytes() == before
    assert (store.calls_used, store.elapsed_s) == (0, 0.0)


# --- Checkpoint (ADR-004: the search start is recorded once) -------------------------------------


def test_the_search_start_is_recorded_once_and_never_overwritten(store: RunStore):
    assert store.search_start_or_record(12, 95.5) == (12, 95.5)
    path = store.path / "checkpoint.json"
    first = path.read_bytes()
    assert json.loads(first) == {"schema_version": 1, "search_start": {"used": 12, "elapsed": 95.5}}
    assert store.search_start_or_record(40, 700.0) == (12, 95.5)
    assert path.read_bytes() == first


# --- Contract and scenarios ----------------------------------------------------------------------


def test_contract_and_scenarios_are_none_until_saved_then_round_trip(store: RunStore):
    assert store.contract() is None and store.scenarios() is None
    store.save_contract(CONTRACT)
    store.save_scenarios(SCENARIOS)
    assert store.contract() == CONTRACT
    assert store.scenarios() == SCENARIOS
    for name in ("contract.json", "scenarios.json"):
        assert json.loads((store.path / name).read_text())["schema_version"] == 1


def test_saving_scenarios_does_not_mutate_the_caller_list(store: RunStore):
    scenarios = list(SCENARIOS)
    store.save_scenarios(scenarios)
    returned = store.scenarios()
    assert returned is not None
    returned.append(Scenario(id="s9", input="added later"))
    assert scenarios == SCENARIOS and store.scenarios() == SCENARIOS


# --- Resume reads back what was saved (SPEC R22) ---------------------------------------------


def test_resume_restores_the_checkpoint_contract_and_scenarios(tmp_path: Path):
    root = tmp_path / "runs"
    store = RunStore.open_or_create(root, PLAN, PROMPT, utcnow=lambda: NOW)
    contract = Contract(goal="g", kind="task", keep=("k",))
    scenarios = [Scenario(id="s1", input="i", criteria=("c",))]
    store.search_start_or_record(9, 81.0)
    store.save_contract(contract)
    store.save_scenarios(scenarios)
    store.close()
    resumed = RunStore.resume(root, store.run_id)
    assert resumed.contract() == contract
    assert resumed.scenarios() == scenarios
    assert resumed.search_start_or_record(50, 900.0) == (9, 81.0)
    resumed.close()


def test_a_resumed_run_finds_the_cache_of_the_original(tmp_path: Path):
    root = tmp_path / "runs"
    store = RunStore.open_or_create(root, PLAN, PROMPT, utcnow=lambda: NOW)
    call = Call(role="task", model="m", user="u", sample=1)
    store.cache_put(call, Reply(text="cached", duration_s=3.0))
    store.close()
    resumed = RunStore.resume(root, store.run_id)
    entry = resumed.cache_get(call)
    assert entry is not None and entry.reply is not None and entry.reply.text == "cached"
    resumed.close()


def test_resume_writes_nothing_by_itself(tmp_path: Path):
    root = tmp_path / "runs"
    store = RunStore.open_or_create(root, PLAN, PROMPT, utcnow=lambda: NOW)
    store.save_contract(Contract(goal="g", kind="task"))
    store.close()
    before = {p: p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    RunStore.resume(root, store.run_id).close()
    after = {p: p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    assert after == before


# --- Call log (ADR-007: append-only JSON lines) --------------------------------------------------


def log_lines(store: RunStore) -> list[bytes]:
    return (store.path / "calls.jsonl").read_bytes().splitlines()


def test_the_call_log_appends_one_json_line_per_call(store: RunStore):
    store.log_call("task", "m1", 10, 20, False, 2.5)
    first = (store.path / "calls.jsonl").read_bytes()
    store.log_call("judge", "m2", 30, 40, True, 0.5)
    data = (store.path / "calls.jsonl").read_bytes()
    assert data.startswith(first) and data.endswith(b"\n")
    lines = [json.loads(line) for line in log_lines(store)]
    assert lines[0] == {
        "schema_version": 1,
        "role": "task",
        "model": "m1",
        "tokens_in": 10,
        "tokens_out": 20,
        "cached": False,
        "duration_s": 2.5,
    }
    assert (lines[1]["role"], lines[1]["model"], lines[1]["cached"]) == ("judge", "m2", True)


def test_a_partial_last_line_left_by_a_crash_is_not_glued_to_the_next(store: RunStore):
    store.log_call("task", "m1", 1, 2, False, 1.0)
    with open(store.path / "calls.jsonl", "ab") as log:
        log.write(b'{"role": "judge", "mod')
    store.log_call("reflect", "m3", 5, 6, False, 3.0)
    lines = log_lines(store)
    assert lines[1] == b'{"role": "judge", "mod'
    assert [json.loads(lines[i])["role"] for i in (0, 2)] == ["task", "reflect"]


# --- Cache files (ADR-007: one file per call, key fields, outcome, duration) ---------------------


def test_a_cached_reply_is_one_file_named_by_the_key_with_its_fields_and_duration(store: RunStore):
    store.cache_put(TASK, Reply(text="the answer", tokens_in=7, tokens_out=9, duration_s=12.5))
    [path] = cache_files(store)
    assert path.name == f"{key_of(TASK)}.json"
    assert json.loads(path.read_text()) == {
        "schema_version": 1,
        "key_fields": dataclasses.asdict(TASK),
        "outcome": "ok",
        "reply": {"text": "the answer", "tokens_in": 7, "tokens_out": 9},
        "duration_s": 12.5,
    }
    entry = store.cache_get(TASK)
    assert entry is not None
    assert (entry.outcome, entry.error, entry.duration_s) == ("ok", "", 12.5)
    assert entry.reply == Reply(
        text="the answer", cached=True, tokens_in=7, tokens_out=9, duration_s=12.5
    )


def test_a_tombstone_records_the_error_and_duration_without_a_reply(store: RunStore):
    store.record_failure(TASK, "timeout after 300 s", 300.0)
    [path] = cache_files(store)
    assert path.name == f"{key_of(TASK)}.json"
    assert json.loads(path.read_text()) == {
        "schema_version": 1,
        "key_fields": dataclasses.asdict(TASK),
        "outcome": "failed",
        "error": "timeout after 300 s",
        "duration_s": 300.0,
    }
    entry = store.cache_get(TASK)
    assert entry is not None
    assert (entry.outcome, entry.reply, entry.error, entry.duration_s) == (
        "failed",
        None,
        "timeout after 300 s",
        300.0,
    )


def test_a_call_that_was_never_stored_is_a_miss(store: RunStore):
    store.cache_put(TASK, Reply(text="a"))
    assert store.cache_get(dataclasses.replace(TASK, sample=1)) is None


def test_hostile_text_is_stored_and_read_back_exactly(store: RunStore):
    hostile = 'x"}\n\0\ud800 ../../etc/passwd $(rm -rf ~) \u202e'
    call = Call(role="judge", model="m", user=hostile, json_schema='{"type": "object"}')
    store.cache_put(call, Reply(text=hostile, duration_s=1.0))
    entry = store.cache_get(call)
    assert entry is not None and entry.reply is not None
    assert entry.reply.text == hostile
    assert [p.parent for p in cache_files(store)] == [store.path / "cache"]


# --- Reading rules (ADR-007: newer is refused, corrupt state refuses resume, a bad cache entry is
# a miss, foreign files are left alone) -----------------------------------------------------------


def full_run(root: Path) -> RunStore:
    """A closed run with every state file written."""
    s = RunStore.open_or_create(root, PLAN, PROMPT, {"kind": "template"}, utcnow=lambda: NOW)
    s.save_progress(4, 40.0)
    s.search_start_or_record(4, 40.0)
    s.save_contract(CONTRACT)
    s.save_scenarios(SCENARIOS)
    s.close()
    return s


def edit(path: Path, change) -> None:
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


STATE_FILES = ["manifest.json", "checkpoint.json", "contract.json", "scenarios.json"]


@pytest.mark.parametrize("name", STATE_FILES)
def test_a_state_file_from_a_newer_version_refuses_resume(tmp_path: Path, name: str):
    run = full_run(tmp_path / "runs")
    edit(run.path / name, lambda d: d.update(schema_version=2))
    with pytest.raises(RunStoreError, match=f"{name}.*newer version"):
        RunStore.resume(tmp_path / "runs", run.run_id)


RAW_CORRUPTIONS = {
    "truncated": b'{"schema_version": 1, "pro',
    "empty": b"",
    "a list": b"[1, 2]",
    "not utf-8": b'{"schema_version": 1, "x": "\xff\xfe"}',
    "deeply nested": b"[" * 100_000 + b"]" * 100_000,
    "no version": b'{"prompt": "p"}',
    "version true": b'{"schema_version": true}',
    "version as text": b'{"schema_version": "1"}',
}


@pytest.mark.parametrize("name", STATE_FILES)
@pytest.mark.parametrize("raw", RAW_CORRUPTIONS.values(), ids=RAW_CORRUPTIONS.keys())
def test_an_unreadable_state_file_refuses_resume_naming_it(tmp_path: Path, name: str, raw: bytes):
    run = full_run(tmp_path / "runs")
    (run.path / name).write_bytes(raw)
    with pytest.raises(RunStoreError, match=name):
        RunStore.resume(tmp_path / "runs", run.run_id)


def _set(*path_and_value):
    *keys, value = path_and_value

    def change(doc):
        for k in keys[:-1]:
            doc = doc[k]
        doc[keys[-1]] = value

    return change


def _drop(*keys):
    def change(doc):
        for k in keys[:-1]:
            doc = doc[k]
        del doc[keys[-1]]

    return change


FIELD_CORRUPTIONS = [
    ("manifest.json", _drop("prompt")),
    ("manifest.json", _set("prompt", 5)),
    ("manifest.json", _set("calls_used", "7")),
    ("manifest.json", _set("calls_used", -1)),
    ("manifest.json", _set("calls_used", True)),
    ("manifest.json", _set("calls_used", 1.5)),
    ("manifest.json", _set("elapsed_s", "x")),
    ("manifest.json", _set("elapsed_s", -1.0)),
    ("manifest.json", _set("elapsed_s", float("nan"))),
    ("manifest.json", _set("elapsed_s", float("inf"))),
    ("manifest.json", _set("elapsed_s", 10**400)),
    ("manifest.json", _set("opts", [])),
    ("manifest.json", _set("created", None)),
    ("manifest.json", _set("plan", {})),
    ("manifest.json", _set("plan", "budget=100")),
    ("manifest.json", _set("plan", "budget", 0)),
    ("manifest.json", _set("plan", "budget", "80")),
    ("manifest.json", _set("plan", "strictness", "wild")),
    ("manifest.json", _set("plan", "merge", "no")),
    ("manifest.json", _set("plan", "seed", 1.5)),
    ("manifest.json", _set("plan", "extra", 1)),
    ("manifest.json", _set("plan", "models", "task", 5)),
    ("manifest.json", _set("plan", "models", "judge", "claude-haiku-4-5-20251001")),
    ("manifest.json", _drop("plan", "models", "target")),
    ("checkpoint.json", _drop("search_start")),
    ("checkpoint.json", _set("search_start", [1, 2.0])),
    ("checkpoint.json", _set("search_start", "used", "4")),
    ("checkpoint.json", _set("search_start", "used", -4)),
    ("checkpoint.json", _set("search_start", "elapsed", None)),
    ("contract.json", _drop("contract")),
    ("contract.json", _set("contract", "kind", "poem")),
    ("contract.json", _set("contract", "goal", None)),
    ("contract.json", _set("contract", "keep", "abc")),
    ("contract.json", _set("contract", "keep", [1])),
    ("contract.json", _set("contract", "extra", "x")),
    ("contract.json", _set("contract", "checks", [{"id": "c1"}])),
    ("contract.json", _set("contract", "checks", 0, "rule", "regex")),
    ("contract.json", _set("contract", "checks", 0, "arg", 400)),
    ("contract.json", _set("contract", "checks", 0, "evil", 1)),
    ("contract.json", _set("contract", "checks", ["c1"])),
    ("scenarios.json", _set("scenarios", {})),
    ("scenarios.json", _drop("scenarios", 0, "input")),
    ("scenarios.json", _set("scenarios", 0, "id", 1)),
    ("scenarios.json", _set("scenarios", 1, "criteria", "short")),
    ("scenarios.json", _set("scenarios", 1, "expected", 3)),
    ("scenarios.json", _set("scenarios", 0, "extra", 1)),
    ("scenarios.json", _set("scenarios", ["s1"])),
]


@pytest.mark.parametrize(("name", "change"), FIELD_CORRUPTIONS)
def test_a_state_file_with_a_bad_field_refuses_resume_naming_it(tmp_path: Path, name, change):
    run = full_run(tmp_path / "runs")
    edit(run.path / name, change)
    with pytest.raises(RunStoreError, match=name):
        RunStore.resume(tmp_path / "runs", run.run_id)


def test_a_missing_manifest_refuses_resume_naming_it(tmp_path: Path):
    run = full_run(tmp_path / "runs")
    (run.path / "manifest.json").unlink()
    with pytest.raises(RunStoreError, match="manifest.json"):
        RunStore.resume(tmp_path / "runs", run.run_id)


def test_a_symlinked_state_file_is_not_followed(tmp_path: Path):
    run = full_run(tmp_path / "runs")
    outside = tmp_path / "elsewhere.json"
    outside.write_bytes((run.path / "contract.json").read_bytes())
    (run.path / "contract.json").unlink()
    (run.path / "contract.json").symlink_to(outside)
    with pytest.raises(RunStoreError, match="contract.json"):
        RunStore.resume(tmp_path / "runs", run.run_id)


def test_unknown_files_and_crash_leftovers_are_ignored_and_never_deleted(tmp_path: Path):
    run = full_run(tmp_path / "runs")
    foreign = {
        run.path / "notes.txt": b"mine",
        run.path / ".manifest.json.k3j2l1.tmp": b'{"schema_version": 1, "pro',
        run.path / "cache" / "foreign.json": b"not json",
        run.path / "cache" / "README": b"hello",
    }
    for path, data in foreign.items():
        path.write_bytes(data)
    resumed = RunStore.resume(tmp_path / "runs", run.run_id)
    assert (resumed.calls_used, resumed.contract()) == (4, CONTRACT)
    resumed.cache_put(TASK, Reply(text="x"))
    assert resumed.cache_get(TASK) is not None
    assert resumed.cache_get(dataclasses.replace(TASK, sample=3)) is None
    resumed.close()
    assert {p: p.read_bytes() for p in foreign} == foreign
