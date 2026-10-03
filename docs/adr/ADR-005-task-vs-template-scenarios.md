# ADR-005: One-off task prompts are tested on synthesised situations, reusable templates on inputs

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R5, R10a, R11, R15

## Context

GEPA learns from a set of task instances: the candidate instruction is run on many inputs and the failures teach the rewriter (paper Algorithm 1). Reusable prompts (system prompts, skills, slash commands) fit that directly. Most prompts the user types into Claude Code are one-off requests with no varying input, so there is nothing to generalise over, and running the same request many times only measures sampling noise.

## Decision

The intake call classifies the prompt as `template` or `task` (the user may override with `--kind`). For `template`, scenarios are inputs and the candidate is the system prompt. For `task`, scenarios are short *situations* (plausible contexts the request could arrive in: different project sizes, constraints, missing details) and the task model receives situation plus candidate as one user message. Situations may not contradict the contract and may not add requirements; the synthesis prompt says so and the contract check (R6) runs on the final candidate only, so situations are labelled as test data in the report. For `task` prompts the checks measure whether the answer meets the contract (covers every stated requirement, respects constraints, asks no unnecessary questions, keeps the format).

## Consequences

The same GEPA machinery serves both kinds and the held-out split keeps its meaning. A `task` prompt that depends on tools or files cannot be exercised single-turn without tools (R18); the report states this. Situation quality depends on the synthesis call, so the report lists them and `--examples` can replace them.

## Alternatives rejected

- **Repeat the same request N times:** measures noise, gives GEPA no diversity to reflect on.
- **Let the model use tools in a sandbox:** large attack surface and cost, breaks R18's lockdown.
- **Refuse `task` prompts:** excludes the user's main case.
