# Build loop (full mode)

Seven gates. A gate is passed only when its command shows it; "looks done" is not a state. The lead is the Claude Code session started in the project folder. Subagents do the writing. ruflo keeps the swarm and hive records, counts the votes, logs the routing picks and stores the lessons.

Every ruflo call below was proven against a throwaway project by `~/Documents/ruv-stack-audit/harness/ruflo_protocol_rehearsal.py`. Rerun that script after a ruflo upgrade, before the next build.

## Gates

| Gate | Artifact in the repository | Green means | Decided by | Push |
|---|---|---|---|---|
| **G0 Start** | kit files, a trivial test | the self-check in `START.md` passes | lead | yes |
| **G1 Specification** | `docs/SPEC.md`: numbered requirements, each with the command or test that proves it; non-goals; limits | the file says `Status: Approved` with a date | **user** | no |
| **G2/G3 Architecture** | `docs/ARCHITECTURE.md` (key flows, modules, work-package table, requirement map), ADRs, a skeleton commit with the interface stubs | `just check` green on the skeleton; no file in two packages; **vote approved** (6 of 8, no open blocker) | eight reviewers vote, then the user | yes |
| **G4 Refinement** | one branch per work package, tests first; the `docs` package (README) is one of them | lead reruns `just check` in each worktree; `git diff --name-only <base>..<branch>` lists only files the package owns | lead | no |
| **G5 Integration** | the branches merged into `main`, one at a time | `just check` green after every merge; the acceptance tests pass | lead | yes |
| **G6 Release vote** | findings with `file:line` and what was done about each | no open blocker; **vote approved** (6 of 8) | eight reviewers vote | yes |
| **G7 Completion** | `docs/BUILD-LOG.md`, version, tag | the user runs the live check; worktrees and their branches removed | **user** | tag on the user's yes |

SPARC names: Specification = G1, Pseudocode = the key flows in G2/G3, Architecture = G3, Refinement = G4 to G6, Completion = G7.

After the G6 vote only `docs/BUILD-LOG.md` and the tag may change: the vote is bound to a commit.

## What ruflo does in a build (lead only)

Tools are `mcp__plugin_ruflo-core_ruflo__<name>`; load them with ToolSearch. Subagents never call them. Use these MCP tools only, never the `ruflo` CLI, for these records, and run one lead session per project: the routing state is kept per process and a second writer overwrites it.

### Build start

1. Recall: `memory_search {query}` with no namespace, and `neural_predict {input: "<topic>"}` for this project's own lessons. For the reviewers: `memory_search {namespace:"patterns", threshold:-1, query:"review finding classes <topic>"}`; the `review-class-*` hits become the "known failure classes" checklist appended to every reviewer brief (earlier builds' lessons, not your view of this design). `memory_retrieve {namespace:"patterns", key:"seat-calib-<seat>"}` per seat gives the running calibration (absent in the first builds).
2. Swarm: `swarm_status`; only if it says `no_swarm`: `swarm_init {topology:"hierarchical", maxAgents:3, strategy:"specialized"}`.
3. Hive: `hive-mind_init {topology:"star", queenId:"lead"}` returns `hiveToken` (calling it again returns the same token, so do that after a context compaction). `hive-mind_status` must list no workers; remove leftovers with `hive-mind_leave {agentId, hiveToken}`. Then `hive-mind_join {agentId, role:"specialist", hiveToken}` for the eight seats `rev-spec`, `rev-safety`, `rev-tests`, `rev-break`, `rev-state`, `rev-user`, `rev-docs`, `rev-diff`.
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

Eight seats, quorum `supermajority`: ruflo requires `floor(2n/3)+1` = **6 of 8**. A seat rejects exactly when it holds a high or medium finding, so two rejecting seats cannot block and three do. Proven by the rehearsal (stage 4), including the traps: 0–3 stays pending, 3–3 resolves rejected, votes after resolution are refused, the stored tally is truncated at resolution, a ninth worker raises the threshold to 7.

1. `hive-mind_status` must list exactly the eight seats. `required` is recomputed from the live worker count on every call: no join or leave while a proposal is pending.
2. `hive-mind_consensus {action:"propose", type:"arch:<short sha of the skeleton commit>"` or `"release:<short sha of HEAD>", value:{round:<n>, previous:<proposalId or null>}, voterId:"lead", strategy:"quorum", quorumPreset:"supermajority"}`. Check `required === 6` and `totalNodes === 8` in the answer. Create `.claude/votes/<proposalId>/` (gitignored) for the reports.
3. Eight fresh `reviewer` agents, two at a time (three on the hotspot), one heavy seat with one light one per wave: W1 `rev-spec` + `rev-break`, W2 `rev-safety` + `rev-state`, W3 `rev-tests` + `rev-docs`, W4 `rev-user` + `rev-diff`. Per seat: `claims_claim {issueId:"<project>:<gate>-r<n>-<seat>", claimant:"agent:<seat>-r<n>:reviewer"}`, `agent_spawn {agentId:"<project>-<seat>-r<n>", …}`, `hooks_model-route {task:"Review architecture"}` or `"Review release"`. The brief (template below) carries the lens verbatim from this table, the subject, the proposalId, the report path and the known-failure-classes checklist, and nothing of your opinion or of the other seats. `rev-break` and `rev-user` probe or run **before** reading ARCHITECTURE and the ADRs; the others read first.

| Seat | G3 lens | G6 lens | May run |
|---|---|---|---|
| `rev-spec` | every requirement is covered by a module and a test that can fail | the code does what the spec says, including failure paths | read, git |
| `rev-safety` | simplicity and risk: what can be removed, what is hard to reverse; side effects outside the tool's own files | safety and limits by reading the flow: inputs reaching a shell or a path, secrets in output, limits that can be bypassed, reads or writes outside the tool's own directories (the "audit" worker) | read, git |
| `rev-tests` | testability and file ownership: interfaces fixed, no file in two packages (the "map" worker) | tests that prove nothing, missing failure paths, requirements without a test (the "testgaps" worker) | `just check`, `just test <path>`, pytester probes |
| `rev-break` | for every limit, deadline and guard in SPEC and the ADRs: the single enforcement point in the skeleton and a path around it; probe the test guards and boundary scans | break every guard, limit, deadline, size cap, retry and concurrency rule with a probe; no finding without a run | probes in the scratch directory |
| `rev-state` | every persisted format has a schema version, one owner, an atomic write and a defined reading of partial or foreign content; every clock named (monotonic vs wall, time zone) | corrupt, partial or foreign input to each format; clock skew, DST, re-run idempotency, rollback state; host assumptions (paths, units, env) checked read-only against this machine | format probes; `ls`, `systemctl --user show` and `list-*` |
| `rev-user` | every user-visible state has an exit code, a stream and a notice; nothing reads as success without being one | run the CLI offline on each failure path with a scratch `HOME` and `XDG_*`; output, exit code, `--json`, stderr against the spec; misleading or stale messages | CLI runs in a scratch HOME |
| `rev-docs` | SPEC vs ARCHITECTURE vs ADRs vs skeleton vs CLAUDE.md: every claim in one is true of the others; requirement map complete | every "done" or "fixed" claim in BUILD-LOG, ADR status and README is true at HEAD; departures from the ADRs | read, git |
| `rev-diff` | the diff since the approved-SPEC commit: every hunk required by SPEC or ARCHITECTURE, nothing extra, ownership respected | round 1: the diff since the G3 commit; round ≥ 2: `<previous head>..<head>`: what changed that no finding asked for, what the fix broke, interface changes without an ADR | git only |

4. A seat's final message starts with its verdict block (format in `.claude/agents/reviewer.md`), and its report at `.claude/votes/<proposalId>/<seat>.md` starts with the same block. Check: the block is first; the JSON parses; `seat`, `subject`, `head` and `proposal` match; every `file:line` exists at `head`; `verdict` is `reject` exactly when a high or medium finding is present; the report exists and its finding ids equal the block's; every high and medium has a `repro`. Then reproduce: run each `repro.cmd` at `head` in a clean tree (or a temporary worktree, removed afterwards) and log the result; for `kind:"read"` confirm the location and log "confirmed by reading". On a failed check ask the same agent once for the block again; if it fails again, rerun the seat with the same voterId. No abstentions.
5. Store the eight blocks unedited: `hive-mind_memory {action:"set", key:"vote.<proposalId>.<seat>", value:<block>}`, and append them to `docs/BUILD-LOG.md`, the durable copy (`hive-mind_shutdown` wipes hive memory).
6. Cast all eight votes, one call at a time, rejections first: `hive-mind_consensus {action:"vote", proposalId, voterId:"<seat>", vote:true|false, hiveToken}`. `vote` must be a JSON boolean: ruflo counts the string `"false"` as an approval, so after every rejection check that `votesAgainst` went up by one. The proposal resolves on its own: at the sixth approval, at 3–3, or when all eight are cast; votes after `resolved: true` are refused, which is expected: note them. The tally for the record is the eight blocks; ruflo's history entry shows the truncated count.
7. Decision, layered on the tally: **O1** a high finding always blocks; if you cannot reproduce it, the user decides before the gate can pass. **O2** a medium blocks until fixed when you reproduced it, or when two seats found it independently (same file and function or section, or the same requirement number; both ids logged). **O3** a medium from one seat that you could not reproduce, or that has no runnable or readable evidence, is `disputed`: it never blocks alone, is never edited in the block, and goes to the user at sign-off with your attempt. The gate passes when the tally is `approved` and no O1 or O2 blocker is open. Build-log line per vote: `tally 6–2 (ruflo: approved) · blocks: approve 6 / reject 2 · overrides: O2 rev-break F1 reproduced → FAILED`.
8. `rejected`, or an open blocker: fix everything in one commit (a test for every medium or high fix, `just check` green, a finding → fix table in the build log), then a new proposal on the new sha. Round ≥ 2 briefs add the fix range `<previous head>..<head>` and the path of the seat's **own** previous block, in this order: (1) the diff as new code through your lens, (2) each prior finding → `prior_status`, (3) the full subject; never the other seats' findings or the lead's fix table (only `rev-docs` reads the build log, by lens). All eight seats run every round. At most three proposals per gate; then the user decides from the open-items table. A high still open after round 3 is retired only by a SPEC amendment or a Deliberate-choices line the user approves, followed by one user-granted extra round. Lens texts are frozen for the gate.
9. Calibration, per seat per proposal, kept by you (ruflo cannot weight voters): `hive-mind_memory {action:"set", key:"calib.<proposalId>.<seat>", value:{round, verdict, n_high, n_medium, n_low, findings:[{id, severity, corroborated_by:[seats], reproduced:"yes"|"no"|"na", outcome:"fixed"|"disputed"|"accepted"|"false", unique}], block_first_try, rerun, truncated}}`. The derived table (precision_hm = (fixed + accepted + corroborated) / (high + medium), unique_true, false_high, overlap) goes to the build log and to `patterns` at G7. After three builds: retire a seat (merge its lens into the one it overlaps most) when `unique_true` was 0 in all three and `overlap` ≥ 50 %; redesign a lens when `false_high` ≥ 1 per build or `precision_hm` < 0.5; fix a brief when block reruns ≥ ⅓ of runs. `rev-spec`, `rev-safety` and `rev-tests` are gate workers: redesigned, never retired. A retirement is a kit commit citing the table; with seven seats the propose answer must say `required === 5`.

### Gate workers

The jobs ruflo's background workers are named for run here as agents at the gate where their result is used, not on a timer:

| Worker | When | Who |
|---|---|---|
| map | G3 | the `rev-tests` seat (file ownership against the skeleton) |
| audit | G6 | the `rev-safety` seat |
| testgaps | G6 | the `rev-tests` seat |
| document | G4 | a `docs` work package for `coder` (README and usage), merged before G6 |
| consolidate | G7 | the lead stores the lessons (below) |
| optimize, benchmark | G5 | only when the spec has a performance requirement; then one `reviewer` with that lens, which never joins the hive (it would change `required`) |

### Build end (G7)

- `hooks_intelligence_trajectory-end {trajectoryId, success, feedback:"<one line>"}`.
- One or two lessons with `memory_store {key:"<project>-<slug>", namespace:"patterns", value}`: these are found from every project. Never write to `claude-memories`.
- One `memory_store` in `patterns` per confirmed finding class of this build, `key:"review-class-<slug>"` (for example `review-class-urllib-timeout-per-operation`), value: the class, how it was found and how it was fixed; these feed the reviewers' checklist at the next build start.
- Seat calibration: `memory_store {namespace:"patterns", key:"seat-calib-<seat>-<project>-<build n>", value:<the derived row>}` for each seat, and update the running `seat-calib-<seat>` summary.
- The same lessons with `neural_train {modelType:"lessons", modelId:"<project>-<build n>", data:[...]}`: this project's own store, read by `neural_predict` at the next start. It is storage and nearest-neighbour lookup; ignore the `accuracy` it reports.
- `docs/BUILD-LOG.md`, committed: for each vote the eight verdict blocks, the 8-block tally next to ruflo's, the rule in force (6 of 8), the overrides with each reproduction result and the disputed list; the seat-calibration table (eight rows); the routing table (label, router's pick, "Opus ran", passed or failed); each agent's token count where the result reports it; agent runs in total against the 7 of the plain loop.
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

## Briefing a reviewer (G3, G6)

One brief per seat, under 6 KB, the lens copied verbatim from the table. Round 1:

```
Seat: <seat>            Subject: arch:<sha> | release:<sha>       Head: <full sha>
Proposal: <proposalId>  Round: 1   Report: .claude/votes/<proposalId>/<seat>.md
Lens: <the G3 or G6 cell of the table, verbatim>
May run: <the "May run" cell>
Read: docs/SPEC.md, docs/adr/*, docs/ARCHITECTURE.md, <the commit>     (rev-break, rev-user: probe or run first, read after)
Range (rev-diff only): <approved-SPEC commit>..<sha>
Known failure classes from earlier builds: <the review-class-* hits, one line each>
```

Round ≥ 2 adds, in this order: `Fix range: <previous head>..<head>`, `Your previous block: .claude/votes/<previous proposalId>/<seat>.md`, and the instruction "first the diff as new code through your lens, then each of your prior findings (fixed, open or regressed), then the full subject". Never the other seats' findings or the lead's fix table.

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
- Expect about 22 agent runs for a three-package build: 6 outside the votes and 8 per vote at G3 and G6 (the plain loop needs 7); each extra round adds 8. A vote is four waves of two seats, about 30 to 40 minutes; no coder runs during a vote.

## After the build

Write down what this checklist got wrong and fix it in `~/projects/project-kit`. A step that had to be improvised belongs here before the next project starts.
