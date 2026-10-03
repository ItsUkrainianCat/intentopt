# ADR-001: We use `gepa` directly and do not add DSPy in v1

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R15, Limits (section 4)

## Context

The artifact to improve is one plain-text prompt. `gepa==0.1.4` exposes `optimize_anything`, which takes a string candidate, an evaluator returning `(score, side_info)`, a `dataset` and a `valset`. DSPy 3.4.0 offers `dspy.GEPA`, which optimises the instructions of a DSPy *program*; it needs a `dspy.LM` for every model and a program wrapper around a bare prompt. Every added dependency pulls a large tree (litellm and more) into a tool whose checks must be reproducible.

## Decision

`gepa` is the only runtime dependency, imported only in `runner.py`. DSPy is not imported anywhere in v1. A later comparison benchmark may add it under a new ADR.

## Consequences

Smaller install and audit surface; evaluator and reflection are plain callables over our own `Backend`. We lose DSPy's ready-made metrics and `auto` budget presets, so the budget logic is ours (R17). If the user later wants to improve DSPy programs, a separate adapter and ADR are needed.

## Alternatives rejected

- **`dspy.GEPA` with a custom `BaseLM` over `claude -p`:** extra dependency tree and an adapter for no gain on a single prompt.
- **MIPROv2 or SIMBA from DSPy:** the paper reports GEPA ahead of MIPROv2 with far fewer rollouts, and both need the same DSPy wrapper.
