# ADR-004: Every model call goes through one Backend seam that counts and caches

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R17, R18, R19, R20, R22

## Context

GEPA's `max_metric_calls` does not count reflection calls, and each call here is a `claude -p` process on the user's Max plan. Calls must be countable, capped, resumable and safe, and tests must never reach a real model.

## Decision

A `Backend` protocol has one method, `complete(Call) -> Reply`. `ClaudeCliBackend` is the only code that builds the `claude -p` command (flags and scrubbed environment as in R18) and aborts on its first call if the session reports plugins, MCP servers or tools. Two wrappers sit on top: `CachedBackend` (disk cache keyed by the full call, stored in the run folder) and `BudgetedBackend` (counts every call by role, enforces call and wall-clock limits, raises `BudgetExhausted`, which the runner turns into "return the best so far"). `FakeBackend` serves scripted replies for all tests. Prompts and outputs are passed on stdin or as single arguments, never through a shell.

## Consequences

One place to audit for safety and cost; resume (R22) falls out of the cache; tests are deterministic. The single-turn, no-tools setting means the task model cannot exercise tool use, so prompts that depend on tools are scored on what the model *says* it would do; the report notes this limit.

## Alternatives rejected

- **Count only GEPA metric calls:** misses reflection, intake, synthesis and judge calls.
- **Anthropic API with a key:** violates the no-API-key rule for this machine.
- **`--bare`:** needs an API key and never reads the subscription login.
