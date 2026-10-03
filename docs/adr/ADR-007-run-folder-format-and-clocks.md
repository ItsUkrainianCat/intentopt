# ADR-007: The run folder has one owner, versioned JSON, atomic writes and named clocks

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R17, R22, R23, R24

## Context

A run can take 45 minutes and is resumed after an interruption (R22) from files the tool wrote itself. The G3 round 1 vote found that the run folder, the call cache and the call log had no format version, no atomic-write rule and no rule for truncated or foreign files, that the prompt and plan needed for a resume were not stored, and that the 45-minute clock had no name (wall or monotonic) and no behaviour across `--resume` or a laptop suspend.

## Decision

- **Location and permissions.** `$XDG_STATE_HOME/autoimprover/runs/<id>/` (default `~/.local/state/...`), directory mode 0700, files 0600, never inside a git repository. If the folder cannot be created or written, the run exits 2 before any paid call and names `XDG_STATE_HOME`. `<id>` is `YYYYMMDD-HHMMSS-<first 8 hex of SHA-256 of the prompt>`, with `-2`, `-3`... on a collision; the time is used for the name only.
- **One owner.** `runstore.py` is the only module that opens files in the run folder; `CachedBackend` and the runner call its functions.
- **Files**, each a JSON document with `"schema_version": 1` (the call log is JSON Lines, one object per line):
  - `manifest.json`: schema version, the prompt, the `Plan`, the flags that affect the result, creation time (UTC, display only), `elapsed_s` (monotonic seconds used so far).
  - `contract.json`, `scenarios.json`, `checkpoint.json` (stage reached, split, seed scores).
  - `calls.jsonl`: one line per call that reached the backend (role, model, tokens, cached flag, duration); append only.
  - `cache/<sha256>.json`: one file per successful call: the key fields and the reply.
  - `gepa.log`: GEPA's progress output (ADR-004).
- **Atomic writes.** Every JSON file is written to a temporary name in the same folder, flushed and `fsync`ed, then moved with `os.replace`. A reader never sees a half-written file made by this tool.
- **Reading.** A `schema_version` newer than the tool knows: refuse with exit 2 ("written by a newer version"). A cache entry that does not parse or whose key fields do not match its name: treated as a miss, deleted and recomputed. A manifest, contract, scenarios or checkpoint that does not parse: `--resume` refuses with exit 2 naming the file. A last partial line in `calls.jsonl` is ignored. Unknown extra files in the folder are ignored and never deleted by `--resume`.
- **Cache key.** SHA-256 of the canonical JSON of all `Call` fields (including `sample`) plus the schema version, so a change to `Call` changes every key rather than silently reusing old replies.
- **Clocks.** Budgets and deadlines use `time.monotonic()`: it does not jump with time-zone, daylight-saving or manual clock changes, and on Linux it does not advance during suspend, so closing the laptop lid does not eat the budget. `elapsed_s` in the manifest is saved after every paid call, so `--resume` continues the same 45 minutes. Wall-clock time (UTC) appears only in names and logs, never in a decision. Tests inject a fake clock.
- **Resume.** `--resume <id>` reads the prompt and plan from the manifest (a conflicting flag is ignored with a notice), reruns the flow, and the cache replays every paid call.
- **Removal.** `autoimprover clean` removes all runs, `clean <id>` one; there is no automatic expiry in v1.

## Consequences

Interrupted runs and corrupt files have a defined outcome. A schema change is a version bump with a decision, not an accident. The `fsync` per file is cheap next to a model call. Cache files are many small files (one per call); a run holds at most a few hundred.

## Alternatives rejected

- **SQLite for the cache:** one more thing that can corrupt, and the files are easy to inspect and delete.
- **Wall-clock deadlines:** break across suspend, resume, time-zone and daylight-saving changes.
- **GEPA's own `run_dir` for resume:** GEPA state is a pickle that is loaded automatically on the next run (ADR-004); loading a file from a folder the user may have edited is the wrong trust boundary.
