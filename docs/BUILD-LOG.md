# BUILD-LOG

Loop state for the self-paced `/loop` build (started 2026-10-03 13:23 in a session inside this folder; the earlier `/loop 1m` job was cancelled). Each tick reads this, does the next unchecked step, and updates it.

- [x] G0 kit adopted (earlier)
- [x] G1 SPEC approved 2026-10-03
- [x] G2 ARCHITECTURE draft 1 written 2026-10-03
- [x] ADR-001..004 written (Proposed) 2026-10-03 11:50
- [x] Read the GEPA paper pp.1-12 (11:55). Revised SPEC (R4, R10, R14a, R15, R15a, R17 default 100/300/45 min), ADR-002, ADR-003, ARCHITECTURE. Findings: validation dominates rollouts; Pareto beats best-only/beam; merge mixed so off; weak-to-strong transfer so confirm on the target model; shorter prompts correlate with better scores. Not yet read: appendix C (reflection meta-prompt) and D.1 (merge) and E.4 (optimizer configs): read them before writing the reflection template (WP5).
- [x] Audit round 1 (lead self-review, 12:05). Fixed: (1) BIG GAP: most user prompts are one-off tasks with no varying input, so scenarios are now "situations" for kind=task (R5, R10a); (2) R7 contradicted R8; (3) split sizes contradicted the holdout minimum (formula in R15); (4) noise estimate was a single difference (now 2x, printed); (5) judge could be talked into passes (verbatim-quote rule R10b); (6) fixed costs were not reserved (R17); (7) default target model made confirmation meaningless (now sonnet); (8) no failure handling or exit codes (R2, R24); (9) run data privacy (R23); (10) acceptance measure overstated significance. Still open: ADR for kind=task design (ADR-005), fresh-reader audit by a reviewer agent at the G3 vote.
- [x] ADR-005 (task vs template scenarios) and ADR-006 (own reflection prompt) written 12:12
- [x] Read paper appendix C (meta-prompt) and E.4 (minibatch 3, merge max 5, Df=train / Dpareto=val). Key finding: the paper's meta-prompt tells the rewriter to copy "niche domain facts" from examples into the prompt; with synthetic scenarios that would inject invented content, hence ADR-006. Not read: D.1 (merge algorithm, only needed if `--merge` ships) and appendix L (example optimised prompts, useful as style references for the reflection template).
- [x] WP0 draft 12:20: `src/autoimprover/types.py` (Call, Reply, Backend, Check, Contract, Scenario, Models, Plan, Outcome, errors, constants) + `tests/test_types.py`; `just check` green (11 tests). Uncommitted until the G3 vote passes (skeleton commit follows the vote).
- [x] 12:30 Kit sync: repo copy of the kit was stale (3-seat 2-of-3 votes); copied `docs/kit/*`, `.claude/agents/*` from `~/projects/project-kit` (commit 20c86d0) and fixed CLAUDE.md. The vote is EIGHT seats, 6 of 8 (supermajority), two agents at a time in 4 waves, bound to a commit.
- [x] 12:31 Skeleton commit fa47dff (SPEC, ARCHITECTURE, ADR-001..006, types.py + tests, `just check` green). The vote subject is this sha; the vote comes AFTER the skeleton commit (earlier note was wrong).
- [x] USER STEP done 2026-10-03: `git push origin main` pushed fa47dff..fbccc3b to the private origin (the sandbox has no GitHub credentials, so pushes come from the user's terminal).
- [ ] G3 vote needs: ruflo build-start records (swarm, hive with 8 seats, trajectory, task for G3), 4 waves x 2 reviewer agents, RAM >= 3 GB free and swap < 20 GB at each wave (11:56: 3.0 GB free, swap 18.7 GiB: borderline). Prepare the 8 briefs (< 6 KB each) before spawning.
- [x] 13:30 ARCHITECTURE: full CLI flag list with R-numbers, `clean` subcommand (R23, was missing), ADR-001..006 list, "Open points for the G3 vote" (WP6 proof now includes R23). No separate `docs/adr` index file: the ARCHITECTURE list is the index. ARCHITECTURE changed after fa47dff, so the G3 vote subject is the next commit, not fa47dff. `just check` green (11 tests).
- [x] 16:15 `--facts-from-examples` was in ADR-006 but not in the approved SPEC. The lead removed it from v1 (ARCHITECTURE CLI row and open points, ADR-006 decision text now says "follow-up, needs a SPEC amendment") so the documents match the SPEC the user approved. REVERSIBLE: if the user wants the flag, add an R-number to the SPEC and restore it. To be shown to the user at G3 sign-off. This changes the vote subject again (new sha, see below).
- [ ] Add skeleton stubs for the other modules only when their package starts (avoid empty files)
- [ ] Wait for the user's real-call output (see USER STEP above) before WP1; meanwhile prepare the skeleton plan (WP0 file list, `types.py` draft) so G3 can happen as soon as RAM allows reviewers
- [ ] USER STEP (12:00): real `claude -p --safe-mode ...` call. From the lead's sandbox it returns `"Not logged in"` (the sandbox hides the login file; not retried outside the sandbox on purpose). Flags are accepted by claude 2.1.287 and the JSON result has `result`, `is_error`, `terminal_reason`, `usage`, `total_cost_usd`, `modelUsage`. Still unknown: does `--safe-mode` itself still use the subscription login, and does the JSON report plugins/MCP/tools (needed for the R18 self-check)? The user runs both commands below in their own terminal and pastes the output:
  1. `echo "Reply with exactly: OK" | claude -p --safe-mode --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 --model claude-haiku-4-5-20251001 --system-prompt "You are a test." --output-format json`
  2. the same command with `--output-format stream-json --verbose` (look for the `init` message: plugins, mcp_servers, tools)
  WP1 (backend) waits for this; design, WP2, WP3 do not.
- [ ] G3 vote (8 reviewers, 6 of 8) then WP0 skeleton commit
- [ ] G4 WP1..WP7
- [ ] G5 merge, remove legacy/
- [ ] G6 release vote, user `just smoke`
- [ ] G7 tag (needs user yes)

## RESUME CHECKLIST (written 12:00, loop paused)

The `/loop 1m` job was cancelled at 12:00 because every remaining step is blocked outside the lead's reach:
1. **Wrong session location.** This build was driven from a session started in `$HOME`. The kit requires a session started inside `~/projects/optimizer` (ruflo records and the vote live in that folder's state). Start `cd ~/projects/optimizer && claude`, then `/loop 1m <same prompt>`.
2. **Machine headroom for the vote** (8 reviewers, 2 at a time): >= 3 GB RAM available, swap < 20 GB, <= 3 other sessions. At 11:57: 2.4 GB available, swap 20.2 GiB, load 6. Close heavy apps or other sessions first.
3. **User steps pending:** `git push origin main` (no credentials in the sandbox); the two `claude -p` outputs (see USER STEP above).
4. **Next lead actions in the project session, in order:** build-start records (recall, swarm, hive with 8 seats, trajectory, G3 task); write 8 briefs; propose `arch:fa47dff...`; four waves; decide per O1-O3; fix findings; second proposal if needed; then user sign-off; then WP1..WP7.

Blockers at 13:30 (every remaining item needs one of these):
- Machine headroom: 13:30 2.8 GB available, swap 21.6 GiB, load 24; 16:12 2.2 GB available, swap 20.1 GiB, load 2.9 (need 3 GB and under 20 GiB). Blocks the G3 vote and every agent.
- USER: the two `claude -p` outputs (blocks WP1 only). (Push done; `--facts-from-examples` dropped by the lead 16:15, reversible.)
- Not mine: untracked `.bashrc`, `.gitconfig` and similar dotfiles appear in this folder's `git status` (sandbox artefacts); never `git add` them.

Notes:
- Resources at 11:47: RAM available ~2 GB, swap ~18.7 GiB used; agent spawn rule needs 3 GB free, so the lead works alone until that clears. Re-check before spawning.
- `gh` is unauthenticated inside the sandbox; pushes to the private origin may need the user's terminal.
