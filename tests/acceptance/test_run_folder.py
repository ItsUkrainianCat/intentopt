"""Acceptance tests for SPEC R23 (run data under `$XDG_STATE_HOME/autoimprover/runs/<id>/`, mode
0700, `autoimprover clean [<id>]`, ids never paths) and the one-run-per-folder lock of SPEC section
4 and ADR-007. The refusals for a state folder inside git or not writable are in
test_dry_budget.py (they share the R4 table).
"""

import fcntl
import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fakes import MARKER, happy_backend

from autoimprover.types import RUN_ID_PATTERN

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."


def _run(run_cli, prompt: str = ORIGINAL) -> Path:
    r = run_cli(["--json", prompt], happy_backend(f"{prompt} {MARKER}"))
    assert r.code == 0, r.err
    return Path(r.json()["run_dir"])


@contextmanager
def _locked(folder: Path) -> Iterator[None]:
    """Hold `run.lock` the way a live run does (ADR-007: flock, exclusive)."""
    with open(folder / "run.lock", "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def test_run_data_is_stored_in_a_private_run_folder(run_cli, runs_dir):
    """R23, ADR-007: the prompt, contract, scenarios, outputs and call cache are stored under
    `$XDG_STATE_HOME/autoimprover/runs/<id>/`, the folder mode 0700 and its files 0600; the id is
    `YYYYMMDD-HHMMSS-<first 8 hex of the prompt's SHA-256>`."""
    folder = _run(run_cli)
    assert folder.parent == runs_dir
    assert re.fullmatch(RUN_ID_PATTERN, folder.name)
    assert folder.name.split("-")[2] == hashlib.sha256(ORIGINAL.encode()).hexdigest()[:8]
    assert folder.stat().st_mode & 0o777 == 0o700
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["prompt"] == ORIGINAL
    assert (folder / "contract.json").is_file()
    assert (folder / "scenarios.json").is_file()
    assert "situation 1" in (folder / "scenarios.json").read_text()
    cache = list((folder / "cache").glob("*.json"))
    assert cache
    assert any("GOOD answer" in p.read_text() for p in cache)  # the model outputs are kept
    for path in folder.rglob("*"):
        if path.is_file():
            assert path.stat().st_mode & 0o777 == 0o600, path


@pytest.mark.parametrize("proposal", [IMPROVED, f"{ORIGINAL} Be brief."], ids=["improved", "kept"])
def test_report_names_the_run_folder(run_cli, runs_dir, proposal):
    """R23: the prompt is the user's own data, so the report names the folder it is stored in."""
    r = run_cli([ORIGINAL], happy_backend(proposal))
    assert r.code == 0, r.err
    folders = list(runs_dir.iterdir())
    assert len(folders) == 1
    assert str(folders[0]) in r.err


def test_clean_with_an_id_removes_that_run_only(run_cli):
    """R23: `autoimprover clean <id>` removes that run folder (exit 0, nothing on stdout) and leaves
    the others; a second clean of the same id still exits 0 (nothing to remove)."""
    one = _run(run_cli, ORIGINAL)
    two = _run(run_cli, "Write a haiku about the sea.")
    r = run_cli(["clean", one.name])
    assert r.code == 0, r.err
    assert r.out == ""
    assert not one.exists()
    assert two.is_dir()
    again = run_cli(["clean", one.name])
    assert again.code == 0, again.err
    assert two.is_dir()


def test_clean_without_an_id_removes_every_run(run_cli, runs_dir):
    """R23: `autoimprover clean` removes all run folders; --json reports the count in one object."""
    _run(run_cli, ORIGINAL)
    _run(run_cli, "Write a haiku about the sea.")
    r = run_cli(["clean", "--json"])
    assert r.code == 0, r.err
    assert r.json() == {"status": "cleaned", "removed": 2, "skipped": 0}
    assert not runs_dir.exists() or not any(runs_dir.iterdir())


def test_clean_with_nothing_to_remove_exits_0(run_cli, state_home):
    """R23: clean exits 0 also when there is nothing to remove (no runs folder at all)."""
    r = run_cli(["clean"])
    assert r.code == 0, r.err
    assert not (state_home / "autoimprover" / "runs").exists() or not any(
        (state_home / "autoimprover" / "runs").iterdir()
    )


@pytest.mark.parametrize("form", ["parent", "absolute", "relative", "dot-dot", "glob"])
def test_clean_accepts_only_a_run_id_never_a_path(run_cli, form):
    """R23: a run id is the only form clean accepts; a path, `..` or a pattern exits 2 with an
    `error:` line and removes nothing."""
    folder = _run(run_cli)
    arg = {
        "parent": "..",
        "absolute": str(folder),
        "relative": f"runs/{folder.name}",
        "dot-dot": f"{folder.name}/..",
        "glob": "*",
    }[form]
    r = run_cli(["clean", arg])
    assert r.code == 2, r.err
    assert r.out == ""
    assert any(line.startswith("error: ") for line in r.err.splitlines())
    assert folder.is_dir()
    assert (folder / "manifest.json").is_file()


def test_a_run_folder_held_by_a_live_run_is_skipped_and_not_resumed(run_cli, cut_at):
    """SPEC section 4, ADR-007: one run per run folder. While a live run holds the folder's lock, a
    second `--resume` of that id is refused (exit 2, no call) and clean skips that folder while it
    removes the finished one; once the lock is released the resume works."""
    reference = happy_backend(IMPROVED)
    done = run_cli([ORIGINAL], reference).run_folder()
    k = next(i for i, c in enumerate(reference.calls) if c.role == "reflect")
    first = run_cli([ORIGINAL], cut_at(happy_backend(IMPROVED), k))
    assert first.code == 130
    folder = first.run_folder()
    second = happy_backend(IMPROVED)
    with _locked(folder):
        refused = run_cli(["--resume", folder.name], second)
        cleaned = run_cli(["clean", "--json"])
    assert refused.code == 2, refused.err
    assert second.calls == []
    assert cleaned.code == 0
    assert cleaned.json() == {"status": "cleaned", "removed": 1, "skipped": 1}
    assert not done.exists()
    assert folder.is_dir()
    resumed = run_cli(["--resume", folder.name], second)
    assert resumed.code == 0, resumed.err
    assert resumed.out == IMPROVED + "\n"
