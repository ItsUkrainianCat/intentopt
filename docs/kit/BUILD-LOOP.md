# Build loop

Seven gates. A gate is passed only when its command shows it; "looks done" is not a state. The lead is the Claude Code session started in the project folder. Subagents do the writing; ruflo keeps the record and the memory.

## Gates

| Gate | Artifact in the repository | Green means | Reviewed by | Push |
|---|---|---|---|---|
| **G0 Start** | kit files, a trivial test | the self-check in `START.md` passes | lead | yes |
| **G1 Specification** | `docs/SPEC.md`: numbered requirements, each with the command or test that proves it; non-goals; limits and budgets | the file says `Status: Approved` with a date | **user** | no |
| **G2/G3 Architecture** | `docs/ARCHITECTURE.md` (key flows, modules, work-package table, requirement map), ADRs, a skeleton commit with the interface stubs | `just check` green on the skeleton; no file in two packages | `reviewer` briefed to attack it, then the user | yes |
| **G4 Refinement** | one branch per work package, tests first | the lead reruns `just check` in each worktree; `git diff --name-only <base>..<branch>` lists only files the package owns | lead | no |
| **G5 Integration** | the branches merged into `main`, one at a time | `just check` green after every merge; the acceptance tests pass | lead | yes |
| **G6 Review** | findings with `file:line` and what was done about each, in the gate commit message | no open high or medium finding; `git status --short` empty after the reviewer | `reviewer`, given the spec, the ADRs and the diff range, not the lead's opinion | yes |
| **G7 Completion** | README, version, tag | the user runs the live check; worktrees and their branches removed | **user** | tag on the user's yes |

SPARC names: Specification = G1, Pseudocode = the key flows in G2/G3, Architecture = G3, Refinement = G4 to G6, Completion = G7.

## G1: what a good spec contains

- One paragraph: who uses the tool and for what.
- Numbered requirements `R1 … Rn`. Each names the command or test that proves it, including the failure paths (bad input, limit reached).
- Non-goals: what the first version deliberately does not do.
- Limits: time, calls, cost, memory, concurrency.
- Open questions are asked of the user, in small groups, before the status is set to Approved.

## G4: briefing an agent

One brief per agent, under 6 KB. Name files to read; do not paste them.

```
Package: <name from the work-package table>
Start commit: <sha>            (your first command must print this)
You own: <file globs>
Interface to keep: docs/ARCHITECTURE.md, section <n>
Acceptance: <command>
Read first: docs/SPEC.md (R<i>, R<j>), docs/ARCHITECTURE.md, docs/adr/ADR-00<k>
```

Spawn with the Agent tool: `subagent_type` = `coder` or `tester`, a `name`, and nothing else in the brief that states your own conclusions. The agent definitions already set the model and the worktree isolation.

## G5: merging

For each branch, in the dependency order of the work-package table:

1. In its worktree: `just check`.
2. `git diff --name-only <start commit>..<branch>`: every path must be inside the package's globs. `git log --format=%B <start commit>..<branch>` must contain no `Co-Authored-By`.
3. On `main`: `git merge --no-ff <branch>`, then `just check`. If it is red, `git merge --abort` or revert the merge, and send the finding back to the agent.
4. After the last merge: `git worktree remove <path>` for each worktree. Delete the branches after G6.

## What ruflo records (lead only)

Tools are `mcp__plugin_ruflo-core_ruflo__<name>`; load them with ToolSearch. Subagents never call them.

| When | Call | With |
|---|---|---|
| G0, and before each design decision | `memory_search` | the topic, no namespace |
| a gate starts | `task_create` | `type: "feature"`, `description: "<project> G<n> <name>"`, `tags: ["<project>"]` |
| a gate is green | `task_complete` | the `taskId`, `result: {commit, check: "green"}` |
| a package is handed to an agent | `claims_claim` | `issueId: "<project>/<package>"`, `claimant: "agent:<name>:<type>"` |
| a subagent starts | `agent_spawn`, then `agent_update` | `agentType`, `agentId: "<project>-<name>"`, `config: {provider: "ollama"}`, **no `task` argument**; then `status: "busy"` |
| a subagent reports | `agent_terminate`, `claims_release` | the same ids |
| G7 | `memory_store` | one or two lessons, `namespace: "patterns"`; never `claude-memories` |

These calls block nothing. Worktrees, the ownership diff and `just check` are the enforcement. The `agent_spawn` record is what makes the status line's `Swarm` cell show `◉ n active` while agents work.

## Limits on this machine

- 2 agents at once on the house line, 3 on the phone hotspot. The lead does no heavy work while they run.
- Before spawning: at most 3 other Claude sessions, at least 3 GB RAM available, swap used under 20 GB.
- One heavy build at a time on the machine. Tests run under `timeout 300`.
- Agents never run containers, `npx`, the `ruflo` CLI or `claude`. If the lead needs a container: one, with `--memory=6g --memory-swap=8g --pids-limit=2048 --cpus=4`, a unique name, no `--restart`.
- Count API errors in the first round; add a third agent only if there were none.

## After the build

Write down what this checklist got wrong and fix it in `~/projects/project-kit`. A step that had to be improvised belongs here before the next project starts.
