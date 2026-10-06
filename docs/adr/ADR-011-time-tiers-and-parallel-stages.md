# ADR-011: Time is the knob; the default run is a 30-second parallel pipeline

- **Status:** Accepted 2026-10-05 (the user asked for a default of 30 seconds, at most about a minute, with the time settable by a tag; plan approved the same day)
- **Requirement(s):** R25, R17, R14a, R22

## Context

The first live `/improve` showed the research design (GEPA search, 100 sequential calls, up to 45 minutes) is the wrong default for interactive use. Each call is one `claude -p` process (1.5 to 10 s, more with thinking), calls ran one at a time (R17), and GEPA needs 5 to 7 rounds of about 13 sequential calls. About three sequential steps fit into 30 seconds.

## Decision

- `--time` is the primary knob (default 30 s). It selects a tier: quick (15 to 24 s), fast (25 to 59 s), checked (1 to 9 min), deep (10 min and up, the existing GEPA search with this clock). `--deep` is `--time 20m`.
- Fast and quick are pipelines of **stages whose calls run in parallel** (a thread pool, `--workers`, default 4): intake, synthesis and K rewrites together; then all task runs together; then all judge calls and contract checks together; then free gates and the pick. No iterative search.
- Calls ask `claude --effort low` (a new `Call.effort` field, part of the cache key); models per role are chosen for speed (task Haiku; intake and rewrites Sonnet; the judge by the timing probe, default Sonnet with `--effort low` unless measured otherwise). R14 still holds (judge is neither task nor target model).
- **Evidence level is stated, not hidden**: fast and quick results are `verified: false` ("fast check": picked and scored on the same few scenarios, no noise measured); checked and deep are holdout-verified as before.
- Thread safety: `BudgetedBackend` counters, `ResilientBackend` failure counts, `RunStore.save_progress` and `log_call` get locks; `Evaluator` takes `workers`. Determinism: cache keys are content hashes, so a resume replays finished calls whatever the thread order; results never depend on completion order (stages gather by index).
- The hard clock is the existing `BudgetedBackend` deadline; the fast stage runner checks the remaining time before each optional call and skips what cannot finish; if no rewrite has passed every gate by the deadline the original is returned.

## Customisation (user request, 2026-10-05)

Models and effort are user settings, not constants: the existing `--task-model`, `--judge-model`, `--reflect-model`, `--target-model`, and new `--effort` (all roles) with `--task-effort`, `--judge-effort`, `--reflect-effort` per role; values `low|medium|high|xhigh|max|default`. Tier defaults apply only where no flag is given (fast tiers: effort low everywhere, models task Haiku / reflect Sonnet / judge Opus; deep: effort `default`, today's models). Effort is applied in one place, a thin `EffortBackend` wrapper above the cache that sets `Call.effort` from the role when the call has none (so the cache key includes it and no module that builds calls changes); `Models` defaults become tier-dependent (`default_models(tier)`). The mod forwards every one of these flags.

## Amendment 2026-10-06: rewrites must be able to change meaning-bearing structure

The first live runs returned rewrites that only fixed grammar ("a", "?"), and one user screenshot confirmed it. Causes: candidates are written blind (before any output is seen), `conservative` strictness plus "prefer deleting" leaves only trimming, and no strategy makes an implicit request explicit. Decisions: fast tiers default to `balanced`; strategies `clarify`, `structure`, `tighten` (+ `specify` from K=4); near-identity rewrites are dropped; the original is run twice to measure noise (WP10b); from 45 s a second generation reflects on the first one's failed checks and outputs (K2 rewrites, GEPA's reflective mutation), which needs about 18 s more (reflect 9 s, run 6 s, judge 6 s, estimated with the 3.4 s overhead) and is therefore not part of the 30 s default.

## Amendment 2026-10-06 (third live run)

A `--time 45s` run on a vague prompt ("so im building a prompt improver app. i think it should have multiple features. and be customizable") rewrote it well (`clarify`: it added the request for features, customisation and design) but the contract check vetoed the rewrite with `no-new-goal`, because intake had recorded the goal as "without making a specific request"; the run returned the original. Decision: the intake goal is the request the prompt clearly implies (R5), and `no-new-goal` allows making a stated or clearly implied request explicit (R6); an unrelated task, topic, fact or requirement still fails it. Same run: the six Haiku task runs took 10 to 13 s each (600-token answers), so stage B took 12.8 s instead of 6.3 s and the first generation used 33 s of 45; decision: scoring runs in the fast tiers carry a fixed "at most 120 words" suffix for every candidate alike (R25), the planner assumes 150 output tokens per task call.

## Measured facts (the user's timing probe, 2026-10-05, claude 2.1.287, about 350 output tokens, wall clock per call)

| model | effort low | default effort | output tokens (low / default) |
|---|---|---|---|
| haiku | 7.8 s | 7.4 s | 372 / 386 |
| sonnet | 5.7 s | 10.3 s | 298 / 1190 |
| opus | 7.5 s | 12.7 s | 359 / 1137 |

- Output speed is about 70 to 85 tokens per second on all three models; **a call costs about 2.3 s fixed (process start, login, request) plus output tokens / 75 s**. The model's size does not change latency; the number of output tokens does.
- `--effort low` removes the thinking tokens of Sonnet and Opus and halves their time (Haiku does not think by default).
- Four parallel Haiku calls took 8.5 s against 7.8 s for one: parallelism is near-perfect at 4. Available memory fell by about 135 MB during the four calls (the 244 MB binary is shared), so 6 to 8 workers are affordable.

## Decisions from the facts

1. Every fast-tier call asks `--effort low`. Models are chosen for quality, not speed (rewrites and intake Sonnet, judge Opus per R14, task Haiku, target model only in the checked tier).
2. **Latency is spent on output tokens, so every fast-tier reply is kept short**: the judge replies with pass/fail and a short quote (at most 8 words), the rewrites are as long as the prompt, the contract is compact, task runs are NOT capped: probe 6 (2026-10-05) showed that `CLAUDE_CODE_MAX_OUTPUT_TOKENS` does not truncate, it turns an over-long reply into an error (`is_error: true`, 240 tokens written, no time saved), so it must not be used. The planner assumes about 200 output tokens per task call (about 5 s); a task call that cannot finish before its stage deadline is abandoned (the per-call timeout is the stage's remaining time) and that scenario simply does not count for that prompt. Truncating a reply by streaming and killing the child at N tokens is possible (`--include-partial-messages`) but not worth its complexity now.
3. (Revised 2026-10-05, ADR-002 kept.) Stage C is one wave of K+2 calls: one scenario-judge call per prompt (output-only: the judge never sees the candidate, ADR-002) and ONE batched contract-check call for all surviving rewrites (the veto-only exception of ADR-002; the R6 format with one scenario `contract-k` per rewrite, up to 6 per call). Scenario synthesis in the fast tiers is a kind-agnostic call that needs no contract, so it runs beside the intake and the rewrites in stage A (rewrites carry the ADR-006 rules, the free literal list and the length cap, not the contract; the contract check afterwards is what enforces it).
4. A pure function `fast_plan(time_s, workers)` derives K (rewrites), M (scenarios), the output cap and the estimated seconds from the model above (2.4 s + tokens / 70 s per call, waves = ceil(calls / workers)); `--dry` prints it; default workers 6 everywhere.
5. Estimated critical path at the default: stage A about 9 s (intake is the longest call), stage B about 6 to 11 s, stage C about 8 to 10 s: 25 to 30 s. This is tight: the deadline degrades gracefully (fewer scenarios or rewrites, never an unvetted rewrite).

## Consequences

An interactive run takes about 20 to 30 s instead of up to 45 minutes, at the price of weaker evidence in the default tier (labelled); the research mode stays for those who want proof. More concurrent `claude` processes use more memory (default 4 workers; the user's machine had about 1.6 GB free) and subscription rate. The deep tier needs no change. `Call.effort` changes every cache key once: run folders from earlier versions replay as cache misses, never as errors.

## Alternatives rejected

- **Tune GEPA down:** at 30 s it would run one iteration, which is not a search.
- **One rewrite, no scoring, as the only mode:** cheap but no evidence at all; kept only as the `quick` tier.
- **Warm worker processes (stream-json input):** saves spawn time but one process per model and system prompt; later if the probe shows spawn cost dominates.
