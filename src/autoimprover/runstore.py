"""The run folder, `$XDG_STATE_HOME/autoimprover/runs/<id>/` (ADR-007), and the only module that
opens files in it (SPEC R22, R23). Every JSON file carries `schema_version` and is written to a
temporary file in its own folder, fsynced and moved into place; folders are 0700, files 0600. A
newer file is refused, a damaged state file refuses the resume naming it, a damaged cache entry
is a miss and is deleted, and files the tool does not know are left alone.

The call cache of `CachedBackend` lives here too: one file per call, named by the SHA-256 of all
`Call` fields, holding the reply and its duration, or a tombstone for a call that failed inside
the search (SPEC R17, R24; ADR-004). Prompts and replies are data, never part of a path (SPEC R19).
The JSON formats and their reading rules are in `runformat.py`.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from autoimprover.runformat import (
    SCHEMA_VERSION,
    CacheEntry,
    cache_key,
    newer_version,
    parse_cache_entry,
    parse_checkpoint,
    parse_contract,
    parse_manifest,
    parse_scenarios,
)
from autoimprover.types import RUN_ID_PATTERN, Call, Contract, Plan, Reply, Scenario

_DIR_MODE, _FILE_MODE = 0o700, 0o600
_MANIFEST, _CHECKPOINT = "manifest.json", "checkpoint.json"
_CONTRACT, _SCENARIOS = "contract.json", "scenarios.json"
_CACHE, _CWD, _LOCK = "cache", "cwd", "run.lock"
# More runs than this started in one second from one prompt means something is wrong.
_MAX_ID_SUFFIX = 1000


class RunStoreError(Exception):
    """The run folder cannot be used as asked; the CLI exits 2 and prints this message, which says
    what to change (SPEC R23)."""


def runs_root(environ: Mapping[str, str] | None = None) -> Path:
    """`$XDG_STATE_HOME/autoimprover/runs`, by default `$HOME/.local/state/autoimprover/runs`; a
    relative XDG_STATE_HOME is ignored, as the XDG base directory specification asks."""
    env = os.environ if environ is None else environ
    state = env.get("XDG_STATE_HOME", "")
    if not os.path.isabs(state):
        home = env.get("HOME", "")
        if not os.path.isabs(home):
            raise RunStoreError("HOME is not set; set XDG_STATE_HOME to a folder for run data")
        state = os.path.join(home, ".local", "state")
    return Path(state) / "autoimprover" / "runs"


class RunStore:
    """One run folder, opened by `open_or_create` or `resume`."""

    def __init__(self, path: Path, lock: int, manifest: dict[str, Any], plan: Plan) -> None:
        self._path = path
        self._lock: int | None = lock  # the flock is held as long as this descriptor is open
        self._manifest = manifest
        self._plan = plan
        self._search_start: tuple[int, float] | None = None
        self._contract: Contract | None = None
        self._scenarios: list[Scenario] | None = None

    @classmethod
    def open_or_create(
        cls,
        root: Path,
        plan: Plan,
        prompt: str,
        opts: Mapping[str, object] | None = None,
        utcnow: Callable[[], datetime] | None = None,
    ) -> RunStore:
        """A new run folder under `root`, named `YYYYMMDD-HHMMSS-<8 hex of the prompt's SHA-256>`
        (`-2`, `-3`... on a collision); the time is used for the name and display only."""
        now = (utcnow or _utcnow)()
        now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "prompt": prompt,
            "plan": dataclasses.asdict(plan),
            "opts": dict(opts or {}),
            "created": now.isoformat(),
            "elapsed_s": 0.0,
            "calls_used": 0,
        }
        # Checked before anything is created: what is written must read back.
        manifest, _ = parse_manifest(json.loads(_dumps(manifest)))
        cls.check_root(root)
        _make_dirs(root)
        digest = hashlib.sha256(prompt.encode("utf-8", "surrogatepass")).hexdigest()[:8]
        path = _new_folder(root, f"{now:%Y%m%d-%H%M%S}-{digest}")
        store = cls(path, _lock(path), manifest, plan)
        try:
            _make_dir(path / _CACHE)
            _write_json(path / _MANIFEST, manifest)
        except BaseException:
            store.close()
            raise
        return store

    @classmethod
    def resume(cls, root: Path, run_id: str) -> RunStore:
        """The run `run_id` under `root` with its prompt, plan, opts, saved totals, checkpoint,
        contract and scenarios (SPEC R22)."""
        path = cls.resolve_run(root, run_id)
        if not path.is_dir():
            raise RunStoreError(
                f"no run {run_id} in {root}; check the id, and XDG_STATE_HOME if the run was "
                "started with another one"
            )
        cls.check_root(root)
        lock = _lock(path)  # before anything is read
        try:
            manifest, plan = _load(path / _MANIFEST, parse_manifest)
            store = cls(path, lock, manifest, plan)
            store._search_start = _load_optional(path / _CHECKPOINT, parse_checkpoint)
            store._contract = _load_optional(path / _CONTRACT, parse_contract)
            store._scenarios = _load_optional(path / _SCENARIOS, parse_scenarios)
        except BaseException:
            os.close(lock)
            raise
        return store

    @staticmethod
    def check_root(root: Path) -> None:
        """Raise RunStoreError unless runs can be kept under `root`: neither it nor a parent
        holds a `.git` (also behind a symlink), and it or its nearest existing parent is a
        writable folder (SPEC R23). Reads only, so `--dry` can use it."""
        root = Path(os.path.abspath(root))
        for path in (root, root.resolve()):
            for folder in (path, *path.parents):
                if os.path.lexists(folder / ".git"):
                    raise RunStoreError(
                        f"the state folder {root} is inside the git repository {folder}; set "
                        "XDG_STATE_HOME to a folder outside any git repository"
                    )
        existing = next(p for p in (root, *root.parents) if os.path.lexists(p))
        if not (existing.is_dir() and os.access(existing, os.W_OK | os.X_OK)):
            raise RunStoreError(
                f"the state folder {root} cannot be created or written ({existing} is not a "
                "writable folder); set XDG_STATE_HOME to a writable folder"
            )

    @staticmethod
    def resolve_run(root: Path, run_id: str) -> Path:
        """The folder of `run_id` under `root`. Only a run id is accepted, never a path, and the
        folder may be neither a symlink nor outside `root` (ADR-007)."""
        if not re.fullmatch(RUN_ID_PATTERN, run_id, flags=re.ASCII):
            raise RunStoreError(
                f"{run_id!r} is not a run id (like 20261004-132507-1a2b3c4d); pass the id the "
                "run printed, not a path"
            )
        path = root / run_id
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RunStoreError(f"the run folder {path} is a symlink; remove it by hand")
        return path

    @staticmethod
    def clean(root: Path, run_id: str | None = None) -> tuple[int, int]:
        """Remove one run or all runs under `root`: (removed, skipped because a live run holds
        its lock). Only real folders named by a run id are removed; a symlink is never
        followed, and a folder is removed while its lock is held (SPEC R23)."""
        if run_id is not None:
            paths = [RunStore.resolve_run(root, run_id)]
        else:
            names = os.listdir(root) if root.is_dir() else []
            paths = [root / n for n in names if re.fullmatch(RUN_ID_PATTERN, n, flags=re.ASCII)]
        removed = skipped = 0
        for path in paths:
            if path.is_symlink() or not path.is_dir():
                continue
            try:
                lock = _lock(path)
            except RunStoreError:
                skipped += 1
                continue
            try:
                shutil.rmtree(path)
            finally:
                os.close(lock)
            removed += 1
        return removed, skipped

    def close(self) -> None:
        """Release the run folder (its lock); calling it again does nothing."""
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    @property
    def run_id(self) -> str:
        return self._path.name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def prompt(self) -> str:
        return self._manifest["prompt"]

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def opts(self) -> dict[str, object]:
        return copy.deepcopy(self._manifest["opts"])

    @property
    def calls_used(self) -> int:
        return self._manifest["calls_used"]

    @property
    def elapsed_s(self) -> float:
        return self._manifest["elapsed_s"]

    def save_progress(self, calls_used: int, elapsed_s: float) -> None:
        """The paid calls and monotonic seconds used so far, saved after every paid call so a
        resume continues the same budget and clock (SPEC R17, R22)."""
        manifest = {**self._manifest, "calls_used": calls_used, "elapsed_s": elapsed_s}
        self._manifest, _ = self._save(_MANIFEST, manifest, parse_manifest)

    def search_start_or_record(self, used: int, elapsed: float) -> tuple[int, float]:
        """The call count and clock reading when the search first began: recorded by the first
        call and returned unchanged ever after, also on resume (ADR-004)."""
        if self._search_start is None:
            doc = {"search_start": {"used": used, "elapsed": elapsed}}
            self._search_start = self._save(_CHECKPOINT, doc, parse_checkpoint)
        return self._search_start

    def save_contract(self, contract: Contract) -> None:
        doc = {"contract": dataclasses.asdict(contract)}
        self._contract = self._save(_CONTRACT, doc, parse_contract)

    def contract(self) -> Contract | None:
        return self._contract

    def save_scenarios(self, scenarios: Sequence[Scenario]) -> None:
        doc = {"scenarios": [dataclasses.asdict(s) for s in scenarios]}
        self._scenarios = self._save(_SCENARIOS, doc, parse_scenarios)

    def scenarios(self) -> list[Scenario] | None:
        return None if self._scenarios is None else list(self._scenarios)

    def cache_get(self, call: Call) -> CacheEntry | None:
        """The stored reply or tombstone of `call`, or None for a miss. An entry that does not
        parse, or whose key fields do not hash to its name, is a miss and is deleted."""
        key = cache_key(call)
        path = self._path / _CACHE / f"{key}.json"
        try:
            raw = _read_bytes(path)
        except FileNotFoundError:
            return None
        except OSError:  # a symlink, never followed, or another unreadable entry: damaged
            raw = b""
        try:
            doc = json.loads(raw)
        except (ValueError, RecursionError):
            doc = None
        _refuse_newer(path, doc)
        entry = parse_cache_entry(doc, key)
        if entry is None:
            path.unlink(missing_ok=True)
        return entry

    def cache_put(self, call: Call, reply: Reply) -> None:
        """Store a successful reply with the duration of the live call (ADR-004)."""
        fields = {"text": reply.text, "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out}
        self._write_cache(call, outcome="ok", reply=fields, duration_s=reply.duration_s)

    def record_failure(self, call: Call, error: str, duration_s: float) -> None:
        """A tombstone for a call that failed all its attempts inside the search: `CachedBackend`
        replays it as `CallFailed`, so a resume decides as the original run did (SPEC R22)."""
        self._write_cache(call, outcome="failed", error=error, duration_s=duration_s)

    def log_call(
        self,
        role: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cached: bool,
        duration_s: float,
    ) -> None:
        """Append one JSON line to `calls.jsonl`, closing first a line a crash cut short."""
        entry = {"role": role, "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out}
        entry |= {"cached": cached, "duration_s": duration_s, "schema_version": SCHEMA_VERSION}
        line = _dumps(entry).encode() + b"\n"
        fd = _open_private(self._path / "calls.jsonl", os.O_RDWR | os.O_APPEND)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                line = b"\n" + line
            while line:
                line = line[os.write(fd, line) :]
        finally:
            os.close(fd)

    def open_log(self) -> TextIO:
        """`gepa.log`, opened for appending; the caller closes it."""
        fd = _open_private(self._path / "gepa.log", os.O_WRONLY | os.O_APPEND)
        return os.fdopen(fd, "a", encoding="utf-8", errors="backslashreplace")

    def cwd(self) -> Path:
        """The empty working folder of the child process (ARCHITECTURE section 9)."""
        path = self._path / _CWD
        if not path.is_dir():
            _make_dir(path)
        return path

    def _save[T](self, name: str, doc: dict[str, Any], parse: Callable[[Any], T]) -> T:
        """Write a state file that is known to read back, and return what it reads back as."""
        doc = json.loads(_dumps({**doc, "schema_version": SCHEMA_VERSION}))
        parsed = parse(doc)
        _write_json(self._path / name, doc)
        return parsed

    def _write_cache(self, call: Call, **fields: object) -> None:
        folder = self._path / _CACHE
        if not folder.is_dir():
            _make_dir(folder)
        doc = {"schema_version": SCHEMA_VERSION, "key_fields": dataclasses.asdict(call), **fields}
        _write_json(folder / f"{cache_key(call)}.json", doc)


def _load[T](path: Path, parse: Callable[[Any], T]) -> T:
    """Parse one state file; RunStoreError naming it when it is missing, damaged or newer."""
    try:
        doc = json.loads(_read_bytes(path))
        _refuse_newer(path, doc)
        return parse(doc)
    except FileNotFoundError:
        raise _damaged(path, "it is missing") from None
    except (OSError, KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
        raise _damaged(path, error) from error


def _load_optional[T](path: Path, parse: Callable[[Any], T]) -> T | None:
    return _load(path, parse) if os.path.lexists(path) else None


def _damaged(path: Path, why: object) -> RunStoreError:
    return RunStoreError(
        f"{path.name} in the run folder {path.parent} is damaged ({why}); start a new run, or "
        f"remove this one with `autoimprover clean {path.parent.name}`"
    )


def _refuse_newer(path: Path, doc: object) -> None:
    version = newer_version(doc)
    if version is not None:
        raise RunStoreError(
            f"{path.name} in {path.parent} was written by a newer version of autoimprover "
            f"(format {version}); upgrade autoimprover to use this run"
        )


def _read_bytes(path: Path) -> bytes:
    """The file's bytes; a symlink is not followed (OSError ELOOP)."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as file:
        return file.read()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _dumps(doc: object) -> str:
    return json.dumps(doc, sort_keys=True, allow_nan=False)


def _make_dir(path: Path) -> None:
    os.mkdir(path, _DIR_MODE)
    os.chmod(path, _DIR_MODE)  # the umask may have narrowed it


def _make_dirs(path: Path) -> None:
    """Create `path` and its missing parents, each 0700 (Path.mkdir gives parents the umask's)."""
    for folder in reversed([path, *path.parents]):
        if not folder.exists():
            with contextlib.suppress(FileExistsError):  # another run created it meanwhile
                _make_dir(folder)


def _new_folder(root: Path, base: str) -> Path:
    for n in range(1, _MAX_ID_SUFFIX + 1):
        path = root / (base if n == 1 else f"{base}-{n}")
        try:
            _make_dir(path)
        except FileExistsError:
            continue
        return path
    raise RunStoreError(f"more than {_MAX_ID_SUFFIX} runs named {base} in {root}; try again")


def _lock(folder: Path) -> int:
    """A descriptor holding the exclusive flock on `folder`'s run.lock. The kernel drops it when
    the descriptor is closed or the process dies, so there is no stale lock and no pid."""
    try:
        fd = _open_private(folder / _LOCK, os.O_RDWR)
    except OSError as error:
        raise RunStoreError(f"cannot open {folder / _LOCK}: {error}; remove it by hand") from error
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RunStoreError(
            f"run {folder.name} is already running in another process; wait for it to end"
        ) from None
    return fd


def _open_private(path: Path, flags: int) -> int:
    """Open or create a 0600 file without following a symlink at `path`."""
    fd = os.open(path, flags | os.O_CREAT | os.O_NOFOLLOW, _FILE_MODE)
    os.fchmod(fd, _FILE_MODE)
    return fd


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    """Write to a temporary file in the target's own folder (so `os.replace` never crosses a
    filesystem), fsync it, move it into place, then fsync the folder (ADR-007)."""
    data = _dumps(doc).encode()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            os.fchmod(file.fileno(), _FILE_MODE)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    folder = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(folder)
    finally:
        os.close(folder)
