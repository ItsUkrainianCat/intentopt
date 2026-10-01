# Starting a project with this kit

A build runs in a Claude Code session started **inside the project folder** (`~/projects/<name>`), not from `$HOME`: git worktrees, the ruflo ledger and the sandbox are all tied to the session's folder.

Never run `ruflo init`, the `ruflo-core:init-project` skill or anything through `npx` in a project. This kit replaces them.

## A. New project

1. In `~/projects/`, outside the sandbox (`gh` reads its token from the keyring):
   `gh repo create <name> --private --template ItsUkrainianCat/project-kit --clone`
2. Rename the package: `git mv src/project_name src/<package>`. Replace `project-name`, `project_name`, `PROJECT_NAME`, `PACKAGE` and `ONE_LINE_PURPOSE` in `pyproject.toml`, `CLAUDE.md`, `src/<package>/__init__.py` and `tests/test_package.py`.
3. `uv lock`, then `just check` must be green.
4. Commit and push `main`.
5. Open a terminal, `cd ~/projects/<name> && claude`, and paste the first prompt below.

## B. Existing repository

1. Tag the current state: `git tag v<x>-legacy`.
2. Copy from the kit: `justfile`, `.python-version`, `.claude/agents/`, `.claude/settings.json`, `docs/kit/`, `docs/adr/ADR-000-template.md`; merge `.gitignore`; adapt the `CLAUDE.md` skeleton; bring `pyproject.toml` to the kit's shape (`src/` layout, exact dev pins, tool configuration inside).
3. Code that the build will replace stays where it is, outside `src/` and outside the checks, until gate G5 removes it.
4. `uv lock`, `just check` green, commit, push `main`. Then step A5.

## First prompt for the project session

> Read CLAUDE.md, docs/kit/START.md and docs/kit/BUILD-LOOP.md. Run the G0 self-check and report each line as pass or fail with the evidence. If every line passes, start gate G1 and ask me the specification questions one group at a time.

## G0 self-check

The lead runs these first, in the project session. A failed line is fixed before anything else; do not work around it.

| # | Check | Pass means |
|---|---|---|
| 1 | `pwd` and `git rev-parse --show-toplevel` | both print this project folder, not `$HOME` |
| 2 | `just check`, in the sandbox | exit 0: the `uv` cache is writable from a project session |
| 3 | `git status --short` | empty |
| 4 | `memory_search` (ruflo MCP tool) with the project's topic | results from the global store: home notes or `patterns` |
| 5 | `task_create {type:"feature", description:"<project> G0 start", tags:["<project>","build","G0"]}` and `task_update` to `in_progress`, then `git status --short` | still empty: ledger files are ignored |
| 6 | the session's agent list | contains `architect`, `coder`, `tester`, `reviewer` |
| 7 | `git worktree list` | only the main checkout |
| 8 | `free -m`, and the number of other Claude sessions | at least 3 GB available, swap used under 20 GB, at most 3 other sessions; otherwise say so and wait |
| 9 | swarm and hive, as in "Build start" of `BUILD-LOOP.md`: `swarm_status` / `swarm_init`, `hive-mind_init`, the three seats | `hive-mind_status` lists exactly `rev-spec`, `rev-safety`, `rev-tests` |
| 10 | one throwaway vote: propose `type:"selfcheck:g0"` with `strategy:"quorum", quorumPreset:"majority"`, then vote `false`, `true`, `true` for the three seats | the proposal answers `required 2, totalNodes 3`; the last vote answers `resolved: true, result: "approved"` |
| 11 | the status line | line 2 shows `Swarm ○ 0 active`; line 3 shows the build row with `Build G0`, `hive 3 seats` and the self-check vote |

Record the result of each line in the `task_complete` of the G0 task. If line 9, 10 or 11 fails, run `python3 ~/Documents/ruv-stack-audit/harness/ruflo_protocol_rehearsal.py` and report what it prints before going on.
