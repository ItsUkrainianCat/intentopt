# ADR-007: The run folder has one owner, versioned JSON, atomic writes and named clocks

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R17, R22, R23, R24

## Context

A run can take 45 minutes and is resumed after an interruption (R22) from files the tool wrote itself. The G3 round 1 vote found that the run folder, the call cache and the call log had no format version, no atomic-write rule and no rule for truncated or foreign files, that the prompt and plan needed for a resume were not stored, and that the 45-minute clock had no name (wall or monotonic) and no behaviour across `--resume` or a laptop suspend.

## Decision

- **Location and permissions.** `$XDG_STATE_HOME/autoimprover/runs/<id>/` (default `~/.local/state/...`), directory mode 0700, files 0600, never inside a git repository (`open_runs_root` walks the parents of the folder looking for `.git`). If the folder cannot be created or written, or lies inside a git repository, the run exits 2 before any paid call and names `XDG_STATE_HOME`. `/improve` points `XDG_STATE_HOME` at a folder under `$TMPDIR`, because the sandboxed shell it runs in cannot write `~/.local/state`. `<id>` is `YYYYMMDD-HHMMSS-<first 8 hex of SHA-256 of the prompt>`, with `-2`, `-3`... on a collision; the time is used for the name only.
- **One owner, one writer.** `runstore.py` is the only module that opens files in the run folder; `CachedBackend` and the runner call its functions. A live run holds `fcntl.flock(LOCK_EX | LOCK_NB)` on `run.lock` until it exits; a second `--resume` of the same id while it is held is refused (exit 2) and `clean` skips a locked folder. The kernel releases the lock when the process dies, so there is no stale lock and no pid check (a pid is only meaningful inside one pid namespace, and each sandboxed shell command gets its own).
- **Run ids are never paths.** `--resume <id>` and `clean <id>` accept only a string that fully matches `RUN_ID_PATTERN` (`types.py`); `resolve_run` joins it to the runs folder, resolves it and requires `is_relative_to(runs)` and that it is not a symlink before any read or `rmtree`. A path, `..` or an absolute name exits 2.
- **Files**, each a JSON document with `"schema_version": 1` (the call log is JSON Lines, one object per line):
  - `manifest.json`: schema version, the prompt, the `Plan`, the flags that affect the result, creation time (UTC, display only), `elapsed_s` (monotonic seconds used so far) and `calls_used` (paid calls so far), both saved after every paid call.
  - `contract.json`, `scenarios.json`.
  - `checkpoint.json`: the stage reached and `search_start` (`Budgeted.used` and clock reading when the search first began), written once and read by `--resume` for the replay-invariant stopper (ADR-004); a corrupt copy refuses `--resume`.
  - `calls.jsonl`: one line per call that reached the backend (role, model, tokens, cached flag, duration); append only.
  - `cache/<sha256>.json`: one file per successful call: the key fields, the reply and `duration_s`.
  - `gepa.log`: GEPA's progress output (ADR-004).
- **Atomic writes.** Every JSON file is written to a temporary name in the same folder (never in the system temp folder, which may be another filesystem: `os.replace` would fail with EXDEV), flushed and `fsync`ed, then moved with `os.replace`. A reader never sees a half-written file made by this tool.
- **Reading.** A `schema_version` newer than the tool knows: refuse with exit 2 ("written by a newer version"). A cache entry that does not parse or whose key fields do not match its name: treated as a miss, deleted and recomputed. A manifest, contract, scenarios or checkpoint that does not parse: `--resume` refuses with exit 2 naming the file. A last partial line in `calls.jsonl` is ignored. Unknown extra files in the folder are ignored and never deleted by `--resume`.
- **Cache key.** SHA-256 of the canonical JSON of all `Call` fields (including `sample`) plus the schema version, so a change to `Call` changes every key rather than silently reusing old replies.
- **Clocks.** Budgets and deadlines use `time.monotonic()`: it does not jump with time-zone, daylight-saving or manual clock changes, and on Linux it does not advance during suspend, so closing the laptop lid does not eat the budget. `elapsed_s` in the manifest is saved after every paid call, so `--resume` continues the same 45 minutes; `calls_used` is saved with it and `Budgeted` starts from it, so a resume continues the same budget (cache hits are free and never reach `Budgeted`, so without the saved count every resume would grant a fresh budget). Wall-clock time (UTC) appears only in names and logs, never in a decision. Tests inject a fake clock.
- **Resume.** `--resume <id>` reads the prompt, plan, `calls_used` and `elapsed_s` from the manifest (a conflicting flag is ignored with a notice), reruns the flow, and the cache replays every paid call. The error output of exits 3, 130 and a crash prints `resume with: [XDG_STATE_HOME=<dir> ]autoimprover --resume <id>` (the prefix when the state folder is not the default, as under `/improve`), never a path to paste.
- **Removal.** `autoimprover clean` removes all runs, `clean <id>` one (validated as above); there is no automatic expiry in v1.

## Consequences

Interrupted runs and corrupt files have a defined outcome. A schema change is a version bump with a decision, not an accident. The `fsync` per file is cheap next to a model call. Cache files are many small files (one per call); a run holds at most a few hundred.

## Alternatives rejected

- **SQLite for the cache:** one more thing that can corrupt, and the files are easy to inspect and delete.
- **Wall-clock deadlines:** break across suspend, resume, time-zone and daylight-saving changes.
- **GEPA's own `run_dir` for resume:** GEPA state is a pickle that is loaded automatically on the next run (ADR-004); loading a file from a folder the user may have edited is the wrong trust boundary.
