# ADR-003: Scenarios are split three ways and the holdout decides the answer

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R3, R11, R12, R13, R15

## Context

GEPA uses `dataset` for reflection minibatches and `valset` for accepting candidates and keeping the Pareto frontier, so both are seen by the search and can be overfit. With synthetic scenarios and a noisy judge, a small gain on `valset` is as likely luck as skill.

## Decision

Scenarios are split with a fixed seed into `dataset`, `valset` and a holdout by the formulas of SPEC R15 (n >= 8: holdout `max(3, round(0.35 n))`, valset `clamp(round(0.25 n), 2, 4)`, dataset the rest and never below 3; n=8 gives 3/2/3). The search never sees the holdout. The original is scored twice on the holdout, on the target model, as two independent runs (the second has `Call.sample = 1`, so it is not served from the call cache). A candidate is returned only if it beats the original on the holdout, also on the target model, by more than `max(0.05, 2 x |difference between the two runs|)` (SPEC R12), after passing the contract, length and literal checks; otherwise the original is returned unchanged. Below 8 scenarios there is no holdout and the original is returned unless the user passes `--trust-search`, in which case the result is marked unverified.

Paper evidence (arXiv:2507.19457, Obs. 1, 3, 5, 6): most of GEPA's rollouts go to scoring on the Pareto set, so `valset` stays at 2 to 4 scenarios; Pareto selection beats best-only and beam search, so it stays on, with `frontier_type="hybrid"` over scenario x check group to keep the frontier useful with few scenarios; merge is off by default because its effect was mixed (helped GPT-4.1 Mini, hurt Qwen3-8B); prompts optimised on a weaker model transferred to a stronger one, so the final holdout comparison runs on the model the user will use (R14a).

## Consequences

Honest "no reliable improvement" outcomes, which is the safe failure for a tool that rewrites what the user wrote. Fewer scenarios feed the search, so users with few examples get less improvement; they can add `--examples`. The holdout costs extra calls, counted in the budget.

## Alternatives rejected

- **Use `valset` as the final check:** it already steered the search.
- **Always return the search winner:** hides noise and overfitting.
- **Cross-validation:** multiplies cost beyond the 100-call default (SPEC R17).
