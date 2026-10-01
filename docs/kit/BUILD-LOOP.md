# Build loop (full mode)

Seven gates. A gate is passed only when its command shows it; "looks done" is not a state. The lead is the Claude Code session started in the project folder. Subagents do the writing. ruflo keeps the swarm and hive records, counts the votes, logs the routing picks and stores the lessons.

Every ruflo call below was proven against a throwaway project by `~/Documents/ruv-stack-audit/harness/ruflo_protocol_rehearsal.py`. Rerun that script after a ruflo upgrade, before the next build.

## Gates

| Gate | Artifact in the repository | Green means | Decided by | Push |
|---|---|---|---|---|
| **G0 Start** | kit files, a trivial test | the self-check in `START.md` passes | lead | yes |
| **G1 Specification** | `docs/SPEC.md`: numbered requirements, each with the command or test that proves it; non-goals; limits | the file says `Status: Approved` with a date | **user** | no |
| **G2/G3 Architecture** | `docs/ARCHITECTURE.md` (key flows, modules, work-package table, requirement map), ADRs, a skeleton commit with the interface stubs | `just check` green on the skeleton; no file in two packages; **vote approved** | three reviewers vote, then the user | yes |
| **G4 Refinement** | one branch per work package, tests first; the `docs` package (README) is one of them | lead reruns `just check` in each worktree; `git diff --name-only <base>..<branch>` lists only files the package owns | lead | no |
| **G5 Integration** | the branches merged into `main`, one at a time | `just check` green after every merge; the acceptance tests pass | lead | yes |
| **G6 Release vote** | findings with `file:line` and what was done about each | no open high finding; **vote approved** | three reviewers vote | yes |
| **G7 Completion** | `docs/BUILD-LOG.md`, version, tag | the user runs the live check; worktrees and their branches removed | **user** | tag on the user's yes |

SPARC names: Specification = G1, Pseudocode = the key flows in G2/G3, Architecture = G3, Refinement = G4 to G6, Completion = G7.

After the G6 vote only `docs/BUILD-LOG.md` and the tag may change: the vote is bound to a commit.

## What ruflo does in a build (lead only)

Tools are `mcp__plugin_ruflo-core_ruflo__<name>`; load them with ToolSearch. Subagents never call them. Use these MCP tools only, never the `ruflo` CLI, for these records, and run one lead session per project: the routing state is kept per process and a second writer overwrites it.

### Build start

1. Recall: `memory_search {query}` with no namespace, and `neural_predict {input: "<topic>"}` for this project's own lessons.
2. Swarm: `swarm_status`; only if it says `no_swarm`: `swarm_init {topology:"hierarchical", maxAgents:3, strategy:"specialized"}`.
3. Hive: `hive-mind_init {topology:"star", queenId:"lead"}` returns `hiveToken` (calling it again returns the same token, so do that after a context compaction). `hive-mind_status` must list no workers; remove leftovers with `hive-mind_leave {agentId, hiveToken}`. Then `hive-mind_join {agentId, role:"specialist", hiveToken}` for the seats `rev-spec`, `rev-safety`, `rev-tests`.
4. Trajectory, one per build: `hooks_intelligence_trajectory-start {task:"<project> build <n>", agent:"lead"}`; keep the id with `hive-mind_memory {action:"set", key:"build.trajectory", value:{id}}`.
5. Each gate is a task: `task_create {type:"feature", description:"<project> G<n> <name>", tags:["<project>","build","G<n>"]}`, then `task_update {taskId, status:"in_progress"}`; when green `task_complete {taskId, result:{commit, check:"green"}}` and one `hooks_intelligence_trajectory-step {trajectoryId, action:"G<n> <name>", result:"<one line>", quality:<0 to 1, your judgement, never omitted>}`.

### Every subagent

Before spawning:
- `claims_claim {issueId:"<project>:<package>", claimant:"agent:<name>:<type>"}`. A refusal means that package already has an owner: do not spawn. Ids may contain only letters, digits, `_ - . :`.
- `agent_spawn {agentType, agentId:"<project>-<name>", model:"opus", config:{provider:"ollama"}}` with **no `task` argument**, then `agent_update {agentId, status:"busy"}`. The id must start with a letter and contain only letters, digits, `_` and `-`.
- Shadow routing: `hooks_model-route {task:"<label>"}` and note the returned `model`. The label is one short line from a fixed vocabulary, for example "Implement <package> with tests", "Write acceptance tests", "Review architecture", "Review release". The agent still runs on Opus.

When it reports:
- Rerun the gate command yourself. Then `hooks_model-outcome {task:"<the same label>", model:"opus", outcome:"success"|"failure"}`: success means green and accepted without you taking the work over.
- `agent_terminate {agentId}`, `claims_release {issueId, claimant}` with the identical claimant string.

### Votes (G3 architecture, G6 release)

1. `hive-mind_status` must list exactly the three seats. A fourth worker raises the threshold to 3 and the proposal can never resolve.
2. `hive-mind_consensus {action:"propose", type:"arch:<short sha of ARCHITECTURE.md's commit>"` or `"release:<short sha of HEAD>", value:{...}, voterId:"lead", strategy:"quorum", quorumPreset:"majority"}`. Check `required === 2` and `totalNodes === 3` in the answer.
3. Three `reviewer` agents, each given one lens, the spec, the ADRs and the commit, and nothing of your own opinion or of the other reviewers. Run two, then one.

| Seat | G3 lens | G6 lens |
|---|---|---|
| `rev-spec` | every requirement is covered by a module and a test | the code does what the spec says, including failure paths |
| `rev-safety` | simplicity and risk: what can be removed, what is hard to reverse | safety and limits: inputs reaching a shell or a path, secrets in output, limits that can be bypassed (the "audit" worker) |
| `rev-tests` | testability and file ownership: interfaces fixed, no file in two packages (the "map" worker) | tests that prove nothing, missing failure paths, requirements without a test (the "testgaps" worker); this seat runs `just check` |

4. Each reviewer ends with one verdict block (format in `.claude/agents/reviewer.md`). Check four things: the JSON parses; `subject` and `head` match the proposal; every `file:line` exists; `verdict` is `reject` exactly when a high or medium finding is present. On a failed check ask the same agent to emit the block again once; if it fails again, rerun the seat. No abstentions.
5. Store the three blocks unedited: `hive-mind_memory {action:"set", key:"vote.<proposalId>", value:{...}}`.
6. Cast the votes one call at a time, rejections first: `hive-mind_consensus {action:"vote", proposalId, voterId:"<seat>", vote:true|false, hiveToken}`. `vote` must be a JSON boolean. Stop when the answer says `resolved: true`; with two equal votes the third is not needed and would be refused.
7. Rule: **a high finding always blocks**, whatever the tally. Otherwise the result decides: `approved` passes the gate and the dissenting reasons go into the build log; `rejected` means fix the findings and open a new proposal whose reviewers see only the fix diff. At most one re-vote per gate; after that the user decides.

### Gate workers

The jobs ruflo's background workers are named for run here as agents at the gate where their result is used, not on a timer:

| Worker | When | Who |
|---|---|---|
| map | G3 | the `rev-tests` seat (file ownership against the skeleton) |
| audit | G6 | the `rev-safety` seat |
| testgaps | G6 | the `rev-tests` seat |
| document | G4 | a `docs` work package for `coder` (README and usage), merged before G6 |
| consolidate | G7 | the lead stores the lessons (below) |
| optimize, benchmark | G5 | only when the spec has a performance requirement; then one `reviewer` with that lens |

### Build end (G7)

- `hooks_intelligence_trajectory-end {trajectoryId, success, feedback:"<one line>"}`.
- One or two lessons with `memory_store {key:"<project>-<slug>", namespace:"patterns", value}`: these are found from every project. Never write to `claude-memories`.
- The same lessons with `neural_train {modelType:"lessons", modelId:"<project>-<build n>", data:[...]}`: this project's own store, read by `neural_predict` at the next start. It is storage and nearest-neighbour lookup; ignore the `accuracy` it reports.
- `docs/BUILD-LOG.md`, committed: for each vote the three verdict blocks and the tally; the routing table (label, router's pick, "Opus ran", passed or failed); each agent's token count where the result reports it; agent runs in total against the 7 of the plain loop.
- Only after that commit: `hive-mind_shutdown {}`. It wipes the hive's memory and keeps the vote history.

What these records are: the votes, the claims and the recall change what happens next. The swarm, registry, tasks, routing picks and trajectory are records; the status line reads them, and the routing log is evidence for deciding later whether the router may choose for real. A routing pick says what the router would have chosen, not that a cheaper model would have passed.

## Briefing an agent (G4)

One brief per agent, under 6 KB. Name files to read; do not paste them.

```
Package: <name from the work-package table>
Start commit: <sha>            (your first command must print this)
You own: <file globs>
Interface to keep: docs/ARCHITECTURE.md, section <n>
Acceptance: <command>
Read first: docs/SPEC.md (R<i>, R<j>), docs/ARCHITECTURE.md, docs/adr/ADR-00<k>
```

Spawn with the Agent tool: `subagent_type` = `coder` or `tester`, a `name`, and nothing in the brief that states your own conclusions. The agent definitions set the model and the worktree isolation.

## Merging (G5)

For each branch, in the dependency order of the work-package table:

1. In its worktree: `just check`.
2. `git diff --name-only <start commit>..<branch>`: every path must be inside the package's globs. `git log --format=%B <start commit>..<branch>` must contain no `Co-Authored-By`.
3. On `main`: `git merge --no-ff <branch>`, then `just check`. If it is red, `git merge --abort` or revert the merge, and send the finding back to the agent.
4. After the last merge: `git worktree remove <path>` for each worktree. Delete the branches after G6.

## Never in a build

`ruflo hive-mind spawn --claude` (it starts Claude with permission checks off), the daemon and `hooks_worker-*`, the `nested-queen` agents, `coordination_*`, `daa_*`, `workflow_run`, `autopilot_*`, `hooks_build-agents`, `neural_compress`, `ruflo neural optimize`, and any `*_stats` or `*_status` health number in a report: these either execute nothing, are unsafe, or return invented values.

## Limits on this machine

- 2 agents at once on the house line, 3 on the phone hotspot. The lead does no heavy work while they run.
- Before spawning: at most 3 other Claude sessions, at least 3 GB RAM available, swap used under 20 GB.
- One heavy build at a time on the machine. Tests run under `timeout 300`.
- Agents never run containers, `npx`, the `ruflo` CLI or `claude`.
- Expect about 12 agent runs for a three-package build (the plain loop needs 7); one re-vote adds about 2.

## After the build

Write down what this checklist got wrong and fix it in `~/projects/project-kit`. A step that had to be improvised belongs here before the next project starts.
