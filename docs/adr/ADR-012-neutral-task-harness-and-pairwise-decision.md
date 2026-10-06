# ADR-012: Task runs get a neutral system prompt; the pick uses pairwise preference

- **Status:** Accepted 2026-10-06 (from the first real `just bench`, user-run: 5 vague prompts, 0 improved, 5 ties, naive baseline 0 wins 5 losses, median 27 s)
- **Requirement(s):** R10a, R25, R26

## Context

The judge's reasons in the bench folder showed three things. (1) For prompts that sound like agent work (photos on a drive, files, an export job) the task model's answers were attempts to run commands ("only runs directory exploration commands", "repeated tool calls in an empty directory"): a task call without a system prompt gets Claude Code's default agent persona from `claude -p`. Every score was computed on contaminated answers. (2) For vague prompts the original scores 1.00 on the checks derived from itself, so no rewrite can show a gain of 0.1: the absolute metric has no headroom. (3) Naive rewrites lose because the answers get "overbuilt, too long for a casual question, with invented assumptions": the pairwise judge measures something real that the absolute checks do not.

## Decision

- `TASK_SYSTEM` (types.py): a fixed neutral system prompt, applied by `claude_cli` to every task call that has no system prompt of its own, so all paths (deep, fast, checked, bench) get the same harness. The report keeps saying that tool use is not exercised.
- The fast, checked and deep picks use pairwise preference between answers (both orders, agreement or tie; original vs original gives the noise) without an absolute floor in the fast and checked picks (the contract check is the meaning floor; the absolute judge calls would add K+2 calls to a wave that must fit the clock); the judge's reasons feed the reflective generation. The bench's pairwise judge is the same component (`bench_judge`), so the bench measures what the pipeline decides on, plus a blind re-check on fresh scenarios.
- A self-calibrating planner (observed per-call seconds from earlier runs) is the next item: under a loaded machine two of five 30 s runs were cut by the deadline.

## Consequences

Scores from before this change are not comparable (different harness). The pairwise stage adds judge calls (2 per rewrite plus 2 for the noise pair, one wave) and replaces the per-prompt absolute judge calls of those tiers. The deep tier keeps GEPA's absolute metric for its search but confirms with pairwise preference.
