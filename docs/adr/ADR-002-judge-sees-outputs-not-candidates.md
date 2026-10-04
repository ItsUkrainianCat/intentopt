# ADR-002: The judge sees outputs and a checklist, never the candidate prompt

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R10, R14, R16

## Context

Release 0.1.0 asked one model to grade the wording of the prompt on six traits and never ran the prompt, so the search rewarded prompts that sounded good to a judge (longer, more agent-flavoured). A judge that reads the candidate text can still be flattered by it, and an LLM judge that shares a model with the task shares its blind spots.

## Decision

1. The task model runs the candidate on a scenario. 2. Programmatic checks run first (format, length, required and forbidden strings from the contract). 3. A binary checklist, derived once from the contract and the scenario's `criteria`, is judged by a model different from the task model and from the target model (SPEC R14, so the run that decides the result is never graded by the model that produced it). The judge receives the scenario input, the output and the checklist, and never the candidate text. 4. Score = checks passed / checks total; failed checks and a short output excerpt are the ASI returned to GEPA, and per-group sub-scores go in `side_info["scores"]` so GEPA tracks them as separate objectives. 5. The judge is called once per candidate per batch of scenarios (one call holds all outputs and checklists), which keeps the call cost near one task call per scenario; the batch is capped at `JUDGE_BATCH_MAX` (6) scenarios, which is also the holdout cap, so every scoring pass is one judge call and the judge prompt stays short. With `--examples`, a scenario's `expected` adds one judged content check ("agrees with the reference answer in substance") and each `criteria` string one judged check. This needs GEPA's `batch_evaluator` hook (the per-pair `evaluator` would call the judge once per scenario); on that path `oa.log()` capture is not available, so the ASI (failed checks, short output excerpts) is returned inside each `side_info`.

## Consequences

The score measures behaviour, not style. Cost is one task call per scenario plus one judge call per batch, plus programmatic checks, which are free, so cheap checks should carry as much weight as the contract allows. One exception, the R6 contract check at the end: its judge call sees the candidate (as the "output" next to the original, ADR-008) because the question is fidelity, not scoring. It can only veto a candidate, never raise a score or pick the winner, so flattering it gains nothing the holdout would not undo. The judge can still be wrong on a check; the holdout and the noise threshold (R3, R12) contain that.

## Alternatives rejected

- **Wording rubric (0.1.0):** rewards style, not results.
- **Pairwise judge against the seed on every evaluation:** doubles cost and gives GEPA no per-check feedback; kept for the final acceptance measure (SPEC section 5) only.
- **One scalar 0-10 judge score:** noisy; binary checks are more stable and give actionable feedback.
