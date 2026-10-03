# autoimprover

A command-line prompt optimizer: GEPA search over a prompt, scored by running the candidate on the user's own examples.

## Definition of done

`just check` is green (format, lint, types, tests). Nothing counts as finished before that.

## Layout

- `src/autoimprover/` code · `tests/` tests · `tests/acceptance/` black-box tests written from the spec
- `docs/SPEC.md` what the tool must do, approved by the user
- `docs/ARCHITECTURE.md` how: modules, interfaces, the work-package table
- `docs/adr/` decisions that are hard to reverse
- `docs/kit/` the process this repository is built with (`START.md`, `BUILD-LOOP.md`)
- `legacy/autoimprover/` the 0.1.0 script (tag `v0.1.0-legacy`), kept for reference; outside the checks; gate G5 removes it

## How work is done here

- Follow `docs/kit/BUILD-LOOP.md` (full mode): gates G0 to G7, each with a command that proves it; swarm and hive records, eight-seat supermajority votes (6 of 8, a reproduced high or medium blocks whatever the tally) at G3 and G6, shadow model routing, gate workers, lessons at G7.
- Agents: `architect`, `coder`, `tester`, `reviewer` (`.claude/agents/`). One writer per git worktree; each owns the files listed for its package in `docs/ARCHITECTURE.md`.
- `pyproject.toml`, `uv.lock`, `justfile`, `.gitignore`, `CLAUDE.md`, `docs/**`, `.claude/**`, `src/autoimprover/types.py`, `tests/conftest.py` and `tests/fakes.py` belong to the lead session (the work-package table in `docs/ARCHITECTURE.md` has the full list).
- Dependencies: exact pins in `pyproject.toml`, hashes in `uv.lock`, always `uv run --frozen`. Adding one is the lead's job and needs an ADR.
- Tests never call a real LLM or the network. The real backend (`claude -p`) is exercised only by the user's `just smoke`.
- Only the lead session uses ruflo, through its MCP tools, and only one lead session works in this folder at a time.
- The README is a `docs` work package for `coder`, merged before the G6 vote; after that vote only `docs/BUILD-LOG.md` and the tag change.

## Commit policy (pre-approved by the user for this repository, 2026-10-01)

- An agent commits on its own worktree branch when `just check` is green there.
- The lead commits on `main` at a green gate and pushes `main` to the private origin.
- Explicit `git add <paths>` only. No `Co-Authored-By` trailer.
- Tags, releases and anything public need the user's explicit yes.

## Deliberate choices

- `gepa==0.1.4` comes from PyPI with its hash in `uv.lock`; the 0.1.0 script used an unpinned git URL.
- GEPA must run with `parallel=False`: its default starts one LLM call per CPU core, and each call here is a `claude` process.
- `--bare` is not used for `claude -p`: it needs an API key and never reads the subscription login.

## Starting points for gate G1 (history: superseded by `docs/SPEC.md` and `docs/ARCHITECTURE.md`; where they differ, those two win, for example no `--eval rubric`, no `doctor` subcommand, no regex matching in v1)

- **What is scored:** the candidate prompt run on the user's examples (JSONL: input, optional expected; exact, contains or regex match; optional judge with criteria). The 0.1.0 rubric only graded the prompt's wording and never ran it; keep it as `--eval rubric`, labelled "style score".
- **Backend:** one interface; v1 ships `fake` (tests) and `claude-cli`. The local model at `127.0.0.1:8080` is the first follow-up, as task model only, and its output is untrusted.
- **Budget:** count every LLM call, not GEPA's metric calls (reflection calls are outside that limit). Superseded by `docs/SPEC.md` R17 (default 100, ceiling 300, 45 minutes, fixed costs reserved up front, one call at a time, plan shown before the run starts).
- **Research ideas in v1** (`~/projects/autogepa/techniques/`): disk cache and hard budget (05), prompt-length report and growth cap (10), a "seed already near the ceiling" check (01), a holdout split from 8 examples up. The ideas that change GEPA's search wait.
- **Containing `claude -p`:** one place builds the command (`--safe-mode --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 --model <m> --system-prompt=<s> --output-format json`; user text on stdin only), with a scrubbed environment. `doctor` and the first call of every run abort unless the session reports no plugins, no MCP servers and no tools.
