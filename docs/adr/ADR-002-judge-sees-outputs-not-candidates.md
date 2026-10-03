# ADR-002: The judge sees outputs and a checklist, never the candidate prompt

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R10, R14, R16

## Context

Release 0.1.0 asked one model to grade the wording of the prompt on six traits and never ran the prompt, so the search rewarded prompts that sounded good to a judge (longer, more agent-flavoured). A judge that reads the candidate text can still be flattered by it, and an LLM judge that shares a model with the task shares its blind spots.

## Decision

1. The task model runs the candidate on a scenario. 2. Programmatic checks run first (format, length, required and forbidden strings from the contract). 3. A binary checklist, derived once from the contract and the scenario's `criteria`, is judged by a model different from the task model. The judge receives the scenario input, the output and the checklist, and never the candidate text. 4. Score = checks passed / checks total; failed checks and a short output excerpt are the ASI returned to GEPA, and per-group sub-scores go in `side_info["scores"]` so GEPA tracks them as separate objectives. 5. The judge is called once per candidate per batch of scenarios (one call holds all outputs and checklists), which halves the call cost; the batch size is capped so the judge prompt stays short.

## Consequences

The score measures behaviour, not style. Cost is two calls per evaluation (task plus judge) plus programmatic checks, which are free, so cheap checks should carry as much weight as the contract allows. The judge can still be wrong on a check; the holdout and the noise threshold (R3, R12) contain that.

## Alternatives rejected

- **Wording rubric (0.1.0):** rewards style, not results.
- **Pairwise judge against the seed on every evaluation:** doubles cost and gives GEPA no per-check feedback; kept for the final acceptance measure (SPEC section 5) only.
- **One scalar 0-10 judge score:** noisy; binary checks are more stable and give actionable feedback.
