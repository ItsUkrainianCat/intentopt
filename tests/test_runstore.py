"""The run folder: where it lives, its id, modes and manifest, the lock, the safety rules for ids
and the state folder, `clean`, and resume (SPEC R17, R22, R23; ADR-007). Real files under
tmp_path, no mocks of the filesystem."""

import gc
import hashlib
import json
import os
import re
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autoimprover.runstore import RunStore, RunStoreError, runs_root
from autoimprover.types import (
    DEFAULT_MODELS,
    RUN_ID_PATTERN,
    Call,
    Contract,
    Plan,
    Reply,
    Scenario,
    default_models,
)

PROMPT = "Summarise the text in three bullet points."
PLAN = Plan(models=default_models("opus"), strictness="balanced", budget=120, seed=7)
DEFAULT_PLAN = Plan(models=DEFAULT_MODELS)
OPTS = {"kind": "task", "trust_search": False, "examples": None}
NOW = datetime(2026, 10, 4, 13, 25, 7, tzinfo=UTC)
DIGEST = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]


def create(root: Path, prompt: str = PROMPT, plan: Plan = PLAN, now: datetime = NOW) -> RunStore:
    return RunStore.open_or_create(root, plan, prompt, OPTS, utcnow=lambda: now)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "state" / "autoimprover" / "runs"


@pytest.fixture(params=[0o022, 0o077], ids=["umask022", "umask077"])
def umask(request: pytest.FixtureRequest):
    old = os.umask(request.param)
    yield request.param
    os.umask(old)


def mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


# --- Layout, id, manifest, modes (ADR-007) -------------------------------------------------------


def test_a_new_run_folder_is_named_by_utc_time_and_the_prompt_hash(root: Path):
    store = create(root)
    assert store.run_id == f"20261004-132507-{DIGEST}"
    assert store.path == root / store.run_id
    assert store.path.is_dir()
    assert {p.name for p in store.path.iterdir()} >= {"manifest.json", "run.lock", "cache"}
    store.close()


def test_a_colliding_id_gets_a_numbered_suffix(root: Path):
    stores = [create(root) for _ in range(3)]
    base = f"20261004-132507-{DIGEST}"
    assert [s.run_id for s in stores] == [base, f"{base}-2", f"{base}-3"]
    assert all(re.fullmatch(RUN_ID_PATTERN, s.run_id) for s in stores)
    for s in stores:
        s.close()


def test_the_id_never_carries_prompt_text(root: Path):
    hostile = "../../etc/passwd $(rm -rf ~) \0 `x` \ud800 " + "/" * 50
    store = create(root, prompt=hostile)
    assert store.path.parent == root
    digest = hashlib.sha256(hostile.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    assert store.run_id == f"20261004-132507-{digest}"
    assert store.prompt == hostile
    store.close()


def test_the_manifest_holds_what_a_resume_needs(root: Path):
    store = create(root)
    doc = json.loads((store.path / "manifest.json").read_text())
    assert set(doc) == {
        *("schema_version", "prompt", "plan", "opts", "created", "elapsed_s", "calls_used")
    }
    assert doc["schema_version"] == 1
    assert doc["prompt"] == PROMPT
    assert doc["opts"] == OPTS
    assert (doc["calls_used"], doc["elapsed_s"]) == (0, 0.0)
    assert doc["created"] == "2026-10-04T13:25:07+00:00"
    assert (store.prompt, store.plan, store.opts) == (PROMPT, PLAN, OPTS)
    assert (store.calls_used, store.elapsed_s) == (0, 0.0)
    store.close()


def test_save_progress_rewrites_the_saved_totals(root: Path):
    store = create(root)
    store.save_progress(17, 321.5)
    assert (store.calls_used, store.elapsed_s) == (17, 321.5)
    doc = json.loads((store.path / "manifest.json").read_text())
    assert (doc["calls_used"], doc["elapsed_s"], doc["prompt"]) == (17, 321.5, PROMPT)
    store.close()


def test_inputs_are_not_mutated_and_the_opts_are_a_copy(root: Path):
    opts = {"kind": "task", "nested": [1, 2]}
    store = RunStore.open_or_create(root, PLAN, PROMPT, opts, utcnow=lambda: NOW)
    assert opts == {"kind": "task", "nested": [1, 2]}
    store.opts["kind"] = "template"  # type: ignore[index]
    assert store.opts["kind"] == "task"
    store.close()


def test_opts_that_are_not_json_values_create_nothing(tmp_path: Path, root: Path):
    before = tree(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        RunStore.open_or_create(root, PLAN, PROMPT, {"examples": Path("/x")}, utcnow=lambda: NOW)
    assert tree(tmp_path) == before


def test_opts_default_to_an_empty_mapping(root: Path):
    store = RunStore.open_or_create(root, DEFAULT_PLAN, PROMPT, utcnow=lambda: NOW)
    assert store.opts == {}
    store.close()


def test_every_folder_is_0700_and_every_file_0600_whatever_the_umask(root: Path, umask: int):
    store = create(root)
    call = Call(role="task", model="m", user="u")
    store.save_progress(1, 1.0)
    store.search_start_or_record(1, 1.0)
    store.save_contract(Contract(goal="g", kind="task"))
    store.save_scenarios([Scenario(id="s1", input="i")])
    store.cache_put(call, Reply(text="t", duration_s=1.0))
    store.record_failure(Call(role="judge", model="m", user="u"), "down", 2.0)
    store.log_call("task", "m", 1, 2, False, 1.0)
    with store.open_log() as log:
        log.write("gepa says hi\n")
    assert store.cwd() == store.path / "cwd"
    store.close()
    for folder in (root.parent.parent, root.parent, root):
        assert mode(folder) == 0o700, folder
    for path in [store.path, *store.path.rglob("*")]:
        assert mode(path) == (0o700 if path.is_dir() else 0o600), path
    assert len(list((store.path / "cache").iterdir())) == 2
    assert list((store.path / "cwd").iterdir()) == []


def test_the_working_folder_of_the_child_process_is_empty_and_inside_the_run(root: Path):
    store = create(root)
    cwd = store.cwd()
    assert cwd == store.path / "cwd" and cwd.is_dir() and not cwd.is_symlink()
    assert list(cwd.iterdir()) == []
    assert store.cwd() == cwd
    store.close()


# --- Resume (SPEC R22: the same prompt, plan, budget and clock) ----------------------------------


def test_resume_restores_the_prompt_plan_opts_and_saved_totals(root: Path):
    store = create(root)
    store.save_progress(17, 321.5)
    run_id, path = store.run_id, store.path
    store.close()
    resumed = RunStore.resume(root, run_id)
    assert (resumed.run_id, resumed.path) == (run_id, path)
    assert resumed.prompt == PROMPT
    assert resumed.plan == PLAN
    assert resumed.opts == OPTS
    assert (resumed.calls_used, resumed.elapsed_s) == (17, 321.5)
    assert resumed.contract() is None and resumed.scenarios() is None
    resumed.close()


def test_a_resumed_run_continues_from_its_saved_totals_and_saves_on_top(root: Path):
    store = create(root)
    store.save_progress(42, 1000.0)
    store.close()
    resumed = RunStore.resume(root, store.run_id)
    assert resumed.calls_used == 42
    resumed.save_progress(43, 1010.0)
    resumed.close()
    again = RunStore.resume(root, store.run_id)
    assert (again.calls_used, again.elapsed_s, again.plan) == (43, 1010.0, PLAN)
    again.close()


@pytest.mark.parametrize("missing", ["20261004-132507-00000000", "20261004-132507-00000000-2"])
def test_resume_of_an_unknown_id_is_refused(root: Path, missing: str):
    create(root).close()
    with pytest.raises(RunStoreError, match=missing):
        RunStore.resume(root, missing)


def test_resume_of_an_id_that_is_a_file_is_refused(root: Path):
    create(root).close()
    (root / "20261004-132507-00000000").write_text("not a run")
    with pytest.raises(RunStoreError):
        RunStore.resume(root, "20261004-132507-00000000")


# --- The state folder (SPEC R23: writable, never inside a git repository) ------------------------


def tree(base: Path) -> list[str]:
    return sorted(str(p.relative_to(base)) for p in base.rglob("*"))


def test_runs_root_follows_xdg_state_home_else_the_home_state_folder(tmp_path: Path):
    assert runs_root({"XDG_STATE_HOME": "/x/state", "HOME": "/h"}) == Path(
        "/x/state/autoimprover/runs"
    )
    for env in (
        {"HOME": "/h"},
        {"HOME": "/h", "XDG_STATE_HOME": ""},
        {"HOME": "/h", "XDG_STATE_HOME": "rel"},
    ):
        assert runs_root(env) == Path("/h/.local/state/autoimprover/runs")
    assert runs_root() == Path(os.environ["XDG_STATE_HOME"]) / "autoimprover" / "runs"


@pytest.mark.parametrize("env", [{}, {"HOME": ""}, {"HOME": "relative/home"}])
def test_runs_root_without_a_usable_home_names_xdg_state_home(env: dict):
    with pytest.raises(RunStoreError, match="XDG_STATE_HOME"):
        runs_root(env)


def test_check_root_accepts_a_fresh_or_existing_writable_root_and_writes_nothing(
    tmp_path: Path, root: Path
):
    before = tree(tmp_path)
    RunStore.check_root(root)
    assert tree(tmp_path) == before
    create(root).close()
    before = tree(tmp_path)
    RunStore.check_root(root)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("dot_git", ["folder", "file"])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_a_root_inside_a_git_repository_is_refused_and_nothing_is_written(
    tmp_path: Path, dot_git: str, depth: int
):
    repo = tmp_path / "repo"
    root = repo.joinpath(*["sub"] * depth) if depth else repo
    root.mkdir(parents=True) if depth == 0 else repo.mkdir()
    if dot_git == "folder":
        (repo / ".git").mkdir()
    else:
        (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    before = tree(tmp_path)
    for attempt in (lambda: RunStore.check_root(root), lambda: create(root)):
        with pytest.raises(RunStoreError, match="git") as refused:
            attempt()
        assert "XDG_STATE_HOME" in str(refused.value)
    assert tree(tmp_path) == before


def test_a_root_reached_through_a_symlink_into_a_repository_is_refused(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "state").mkdir()
    (tmp_path / "state").symlink_to(repo / "state")
    with pytest.raises(RunStoreError, match="git"):
        RunStore.check_root(tmp_path / "state" / "autoimprover" / "runs")


@pytest.mark.parametrize("existing", [False, True])
def test_a_read_only_root_is_refused_naming_xdg_state_home(tmp_path: Path, existing: bool):
    locked = tmp_path / "locked"
    root = locked / "autoimprover" / "runs"
    root.mkdir(parents=True) if existing else locked.mkdir()
    (root if existing else locked).chmod(0o500)
    try:
        before = tree(tmp_path)
        for attempt in (lambda: RunStore.check_root(root), lambda: create(root)):
            with pytest.raises(RunStoreError, match="XDG_STATE_HOME"):
                attempt()
        assert tree(tmp_path) == before
    finally:
        (root if existing else locked).chmod(0o700)


def test_a_root_below_a_file_is_refused(tmp_path: Path):
    (tmp_path / "plain-file").write_text("x")
    with pytest.raises(RunStoreError, match="XDG_STATE_HOME"):
        RunStore.check_root(tmp_path / "plain-file" / "runs")


# --- Run ids are never paths (SPEC R23, ADR-007) -------------------------------------------------

BAD_IDS = [
    "",
    ".",
    "..",
    "../x",
    "/etc",
    "/tmp/20261004-132507-abcdef01",
    "20261004-132507-abcdef01/..",
    "20261004-132507-abcdef01/../../x",
    "20261004-132507-ABCDEF01",
    "20261004-132507-abcdef0",
    "20261004-132507-abcdef01\n",
    " 20261004-132507-abcdef01",
    "20261004-132507-abcdef01-",
    "20261004-132507-abcdef01-x",
    "20261004-132507-abcdef01\0",
    "٢٠٢٦١٠٠٤-132507-abcdef01",
]


@pytest.mark.parametrize("bad", BAD_IDS)
def test_only_a_run_id_is_accepted_by_resolve_run_resume_and_clean(root: Path, bad: str):
    create(root).close()
    before = tree(root)
    for attempt in (RunStore.resolve_run, RunStore.resume, RunStore.clean):
        with pytest.raises(RunStoreError, match="run id"):
            attempt(root, bad)
    assert tree(root) == before


def test_resolve_run_joins_a_run_id_to_the_root(root: Path):
    assert RunStore.resolve_run(root, "20261004-132507-abcdef01-12") == (
        root / "20261004-132507-abcdef01-12"
    )


def test_a_run_folder_that_is_a_symlink_is_refused(root: Path, tmp_path: Path):
    store = create(root)
    store.close()
    outside = tmp_path / "outside"
    shutil.copytree(store.path, outside)
    for target in (outside, store.path):
        link = root / "20261004-132507-0000000f"
        link.symlink_to(target)
        for attempt in (RunStore.resolve_run, RunStore.resume, RunStore.clean):
            with pytest.raises(RunStoreError):
                attempt(root, link.name)
        link.unlink()
    assert (outside / "manifest.json").exists() and (store.path / "manifest.json").exists()


# --- One live run per folder (ADR-007: flock on run.lock, no pid) -------------------------------


def test_a_live_run_cannot_be_opened_twice_until_it_is_closed(root: Path):
    store = create(root)
    with pytest.raises(RunStoreError, match="already running"):
        RunStore.resume(root, store.run_id)
    store.close()
    resumed = RunStore.resume(root, store.run_id)
    with pytest.raises(RunStoreError, match="already running"):
        RunStore.resume(root, store.run_id)
    resumed.close()
    resumed.close()
    RunStore.resume(root, store.run_id).close()


def test_the_lock_is_not_released_by_garbage_collection(root: Path):
    store = create(root)
    run_id = store.run_id
    gc.collect()
    with pytest.raises(RunStoreError, match="already running"):
        RunStore.resume(root, run_id)
    store.close()


def test_the_lock_is_taken_before_anything_is_read(root: Path):
    store = create(root)
    (store.path / "manifest.json").write_text("{damaged")
    with pytest.raises(RunStoreError, match="already running"):
        RunStore.resume(root, store.run_id)
    store.close()


def test_a_refused_resume_releases_the_lock(root: Path):
    store = create(root)
    store.close()
    manifest = store.path / "manifest.json"
    good = manifest.read_bytes()
    manifest.write_text("{damaged")
    with pytest.raises(RunStoreError, match="manifest.json"):
        RunStore.resume(root, store.run_id)
    manifest.write_bytes(good)
    RunStore.resume(root, store.run_id).close()


def test_the_lock_file_holds_no_pid_and_is_never_followed(root: Path, tmp_path: Path):
    store = create(root)
    assert (store.path / "run.lock").read_bytes() == b""
    store.close()
    outside = tmp_path / "victim"
    outside.write_text("keep")
    (store.path / "run.lock").unlink()
    (store.path / "run.lock").symlink_to(outside)
    with pytest.raises(RunStoreError, match="run.lock"):
        RunStore.resume(root, store.run_id)
    assert outside.read_text() == "keep"


# --- clean (SPEC R23) ----------------------------------------------------------------------------


def test_clean_removes_closed_runs_skips_a_live_one_and_leaves_everything_else(
    root: Path, tmp_path: Path
):
    done = [create(root, prompt=p) for p in ("one", "two")]
    for store in done:
        store.close()
    live = create(root, prompt="three")
    foreign = root / "my-notes"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("x")
    link = root / "20261004-132507-1111111f"
    link.symlink_to(outside)
    stray = root / "20261004-132507-2222222f"
    stray.write_text("a file, not a run")
    (done[0].path / "escape").symlink_to(outside)
    assert RunStore.clean(root) == (2, 1)
    assert not any(store.path.exists() for store in done)
    assert (live.path / "manifest.json").exists() and (foreign / "keep.txt").exists()
    assert link.is_symlink() and (outside / "precious.txt").exists() and stray.exists()
    live.close()
    assert RunStore.clean(root, live.run_id) == (1, 0)
    assert not live.path.exists() and foreign.exists()


def test_clean_of_one_id_leaves_the_other_runs(root: Path):
    first, second = create(root, prompt="a"), create(root, prompt="b")
    first.close()
    second.close()
    assert RunStore.clean(root, first.run_id) == (1, 0)
    assert not first.path.exists() and second.path.is_dir()


def test_clean_of_a_live_id_skips_it(root: Path):
    store = create(root)
    assert RunStore.clean(root, store.run_id) == (0, 1)
    assert store.path.is_dir()
    store.close()


def test_clean_with_nothing_to_remove_is_not_an_error(root: Path):
    assert RunStore.clean(root) == (0, 0)
    assert RunStore.clean(root, "20261004-132507-abcdef01") == (0, 0)
    assert not root.exists()
