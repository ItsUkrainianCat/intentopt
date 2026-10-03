# BUILD-LOG

Loop state for the `/loop 1m` build (cron job 64ac1690, session-only). Each tick reads this, does the next unchecked step, and updates it.

- [x] G0 kit adopted (earlier)
- [x] G1 SPEC approved 2026-10-03
- [x] G2 ARCHITECTURE draft 1 written 2026-10-03
- [x] ADR-001..004 written (Proposed) 2026-10-03 11:50
- [x] Read the GEPA paper pp.1-12 (11:55). Revised SPEC (R4, R10, R14a, R15, R15a, R17 default 100/300/45 min), ADR-002, ADR-003, ARCHITECTURE. Findings: validation dominates rollouts; Pareto beats best-only/beam; merge mixed so off; weak-to-strong transfer so confirm on the target model; shorter prompts correlate with better scores. Not yet read: appendix C (reflection meta-prompt) and D.1 (merge) and E.4 (optimizer configs): read them before writing the reflection template (WP5).
- [x] Audit round 1 (lead self-review, 12:05). Fixed: (1) BIG GAP: most user prompts are one-off tasks with no varying input, so scenarios are now "situations" for kind=task (R5, R10a); (2) R7 contradicted R8; (3) split sizes contradicted the holdout minimum (formula in R15); (4) noise estimate was a single difference (now 2x, printed); (5) judge could be talked into passes (verbatim-quote rule R10b); (6) fixed costs were not reserved (R17); (7) default target model made confirmation meaningless (now sonnet); (8) no failure handling or exit codes (R2, R24); (9) run data privacy (R23); (10) acceptance measure overstated significance. Still open: ADR for kind=task design (ADR-005), fresh-reader audit by a reviewer agent at the G3 vote.
- [x] ADR-005 (task vs template scenarios) and ADR-006 (own reflection prompt) written 12:12
- [x] Read paper appendix C (meta-prompt) and E.4 (minibatch 3, merge max 5, Df=train / Dpareto=val). Key finding: the paper's meta-prompt tells the rewriter to copy "niche domain facts" from examples into the prompt; with synthetic scenarios that would inject invented content, hence ADR-006. Not read: D.1 (merge algorithm, only needed if `--merge` ships) and appendix L (example optimised prompts, useful as style references for the reflection template).
- [x] WP0 draft 12:20: `src/autoimprover/types.py` (Call, Reply, Backend, Check, Contract, Scenario, Models, Plan, Outcome, errors, constants) + `tests/test_types.py`; `just check` green (11 tests). Uncommitted until the G3 vote passes (skeleton commit follows the vote).
- [ ] Add `--facts-from-examples`, `--kind`, `--target-model`, `--merge`, `--force-low-budget`, `--trust-search` to the CLI list in ARCHITECTURE; add ADR-005/006 to the G3 pack; add `docs/adr` index
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

Notes:
- Resources at 11:47: RAM available ~2 GB, swap ~18.7 GiB used; agent spawn rule needs 3 GB free, so the lead works alone until that clears. Re-check before spawning.
- `gh` is unauthenticated inside the sandbox; pushes to the private origin may need the user's terminal.
