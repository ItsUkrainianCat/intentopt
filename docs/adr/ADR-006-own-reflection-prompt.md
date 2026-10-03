# ADR-006: We supply our own reflection prompt instead of the paper's or the library's

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R6, R8, R9, R16

## Context

The paper's meta-prompt (Appendix C) shows the rewriter the current instruction, a minibatch of inputs, outputs and feedback, and asks for a new instruction that includes "niche and domain specific factual information" found in the examples, because that information may not be available to the assistant later. That is right when examples are real data. Here the scenarios are mostly synthetic, so copying their facts into the prompt would add invented content the user never wrote, which is exactly the deviation the tool must avoid. The library default (`optimize_anything_reflection_prompt_template`) is a different, generic template.

## Decision

`ReflectionConfig.reflection_prompt_template` is our own, built per strictness level (R8). It keeps the paper's structure (current instruction, minibatch with outputs and failed checks, request for one new instruction in a code block, minibatch size 3 as in the paper's runs) and adds: the intent contract with "keep verbatim" items; "do not add facts, names, numbers or requirements that appear only in the examples"; "prefer deleting or tightening over adding"; "preserve the author's voice, language, structure and every literal (R9)"; "change only what a failed check points to"; the length cap in tokens. There is no switch in v1 that restores the paper's fact copying, even for user-supplied `--examples`: `docs/SPEC.md` has no requirement for it. A `--facts-from-examples` flag is a candidate follow-up (needs a SPEC amendment and its own test that copied facts come from user data only).

## Consequences

Less overfitting to synthetic scenarios and smaller diffs; some gains the paper's wording would find (injecting domain facts) are deliberately left out for synthetic runs. The template is a tested artifact: unit tests assert the contract items and rules appear in the rendered reflection call. GEPA renders its template with plain string replacement, so a candidate or contract text that contains a placeholder such as `<side_info>` would have feedback spliced into it: the runner escapes the placeholders in candidate and contract text before rendering. The template also asks the model for two or three plain-language lines on what it changed and why; the runner keeps them per candidate lineage and they are the "what changed and why" lines of the report (R2), so no extra model call is needed. An empty reply or one without a code block is rejected by the reflection wrapper (ADR-004) instead of becoming a candidate.

## Alternatives rejected

- **Paper's meta-prompt unchanged:** leaks synthetic facts into the answer.
- **Library default template:** not designed for fidelity to a user-written prompt.
- **Post-filter only (rely on R6):** wastes budget on candidates that are rejected at the end.
