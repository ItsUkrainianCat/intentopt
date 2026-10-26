# ADR-014: The deep tier decides by pairwise preference when the examples carry no reference

- **Status:** Accepted 2026-10-26 (the user chose this design, "pairwise replaces absolute", over a stricter "both must pass"; not yet validated on live calls: the Claude Code weekly limit was reached the same day)
- **Requirement(s):** R3, R12, R14a, R17, R25

## Context

ADR-012 moved the fast and checked picks to a blind pairwise preference and said the deep tier "keeps GEPA's absolute metric for its search but confirms with pairwise preference". The code never did the second half: `runner.confirmed` took the first finalist whose absolute holdout score beat the original's by `max(0.05, 2 x noise)`. The SPEC said "the deep tier keeps its absolute metric". The reason for ADR-012 applies as much to deep: for a prompt whose generated checks the original already passes, an absolute score has no headroom, and it cannot see an answer that is overbuilt, too long or built on invented assumptions.

## Decision

- **Without references** (not every example carries `expected` or `criteria`; synthesised scenarios never do), the finalists are decided by the pairwise judge of the fast tiers (`fast_pairwise`, `bench_judge`), through `deep_pairwise.py`. The search still uses GEPA's absolute valset metric.
- **Run, not scored.** Each finalist is run once on the holdout on the target model (R14a). The judge sees the original prompt as the request and the two answers anonymously, in both orders, one call per order for all scenarios, never a candidate prompt (R10). A scenario counts only when both orders agree.
- **Noise.** The original's two holdout runs (R12) are compared with each other the same way; the scenarios with an agreed winner are the noise.
- **Win.** The first finalist (best valset score first, after the free gates and the contract check) that wins more scenarios than it loses, by more than the noise, is returned, `verified`. Otherwise the original is returned: `no_reliable_improvement`.
- **Failures.** A pairwise call that fails all its attempts ends the run with a BackendError (R24); there is no wave to fit in, so it is never dropped as a tie (unlike stage C of the fast tiers).
- **Reference mode is unchanged:** every example with a reference keeps the absolute rule of R3 and R12 (ADR-012's exception, ADR-013); so does the 0.95 "already strong" check (R13), which is read from the original's two scored holdout runs.
- **Reserve.** `final` becomes `FINALISTS x holdout` task calls, `FINALISTS` contract checks, `2 x FINALISTS` pairwise calls and 2 for the noise (it was `FINALISTS x (holdout + judge calls)` plus the checks). For 12 scenarios that is 23, not 18; the smallest budget that affords 4 iterations is 91, not 86; 40 examples need `--budget 105`, above the default ceiling of 100.
- **Report.** `score_before` and `score_after` of a result decided this way are the shares of the holdout scenarios the original and the rewrite won, `noise` the share with an agreed winner between the original's two runs and `margin` the lead minus the noise, as in the fast tiers; the reason holds `deep_pairwise.MARK` and the report words them as a preference, never a score.

## Consequences

- A deep run can now return an improvement for a prompt whose generated checks the original already passes, and can no longer return one that only scored higher on checks. Nothing in this ADR has been measured on a live model; the 34/48 result (references) is not affected.
- The deep default budget affords fewer iterations for the same plan (4 instead of 5 at 100 calls and 12 scenarios).
- A holdout of 3 to 6 scenarios gives a coarse preference and a coarse noise, as it did for the absolute rule; the report prints both.
- Results JSON has no new field; a consumer that reads `score_before` of a deep result must read `reason` to know which rule decided it (the Claude Code mod still labels them as scores: open).

## Alternatives rejected

- **Both rules must pass:** stricter, but then the absolute rule's lack of headroom still blocks every vague prompt, which is what ADR-012 measured.
- **Score the finalist as well and report it:** spends `FINALISTS x judge calls` for a number that decides nothing.
