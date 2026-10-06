"""The two runstore helpers the bench uses for its partial summary (SPEC R23, R26; ADR-007):
`make_dirs` creates a folder and its missing parents, each 0700, and `write_json` writes a JSON file
atomically, 0600, through a temporary file in the target's own folder. They are the run store's own
helpers under public names, so the bench's `summary.json` is written as every run-folder file is."""

import json
import os
import stat

from autoimprover import runstore
from autoimprover.runstore import make_dirs, write_json


def mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_they_are_the_run_stores_own_helpers():
    assert make_dirs is runstore._make_dirs and write_json is runstore._write_json


def test_make_dirs_creates_every_missing_folder_0700(tmp_path):
    old = os.umask(0o022)  # a usual umask, which Path.mkdir would leave at 0755
    try:
        target = tmp_path / "state" / "bench" / "20261007-010203-0123abcd"
        make_dirs(target)
    finally:
        os.umask(old)
    for folder in (tmp_path / "state", tmp_path / "state" / "bench", target):
        assert folder.is_dir() and mode(folder) == 0o700
    make_dirs(target)  # an existing folder is fine


def test_write_json_writes_a_whole_0600_file_and_replaces_an_old_one(tmp_path):
    folder = tmp_path / "bench-folder"
    folder.mkdir()
    path = folder / "summary.json"
    path.write_text("old")
    write_json(path, {"b": 1, "a": [1, 2], "status": "bench"})
    assert json.loads(path.read_text()) == {"a": [1, 2], "b": 1, "status": "bench"}
    assert mode(path) == 0o600
    assert [p.name for p in folder.iterdir()] == ["summary.json"]  # no temporary file left
