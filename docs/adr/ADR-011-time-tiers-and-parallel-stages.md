# ADR-011: Time is the knob; the default run is a 30-second parallel pipeline

- **Status:** Accepted 2026-10-05 (the user asked for a default of 30 seconds, at most about a minute, with the time settable by a tag; plan approved the same day)
- **Requirement(s):** R25, R17, R14a, R22

## Context

The first live `/improve` showed the research design (GEPA search, 100 sequential calls, up to 45 minutes) is the wrong default for interactive use. Each call is one `claude -p` process (1.5 to 10 s, more with thinking), calls ran one at a time (R17), and GEPA needs 5 to 7 rounds of about 13 sequential calls. About three sequential steps fit into 30 seconds.

## Decision

- `--time` is the primary knob (default 30 s). It selects a tier: quick (15 to 24 s), fast (25 to 59 s), checked (1 to 4 min), deep (5 min and up, the existing GEPA search with this clock). `--deep` is `--time 20m`.
- Fast and quick are pipelines of **stages whose calls run in parallel** (a thread pool, `--workers`, default 4): intake, synthesis and K rewrites together; then all task runs together; then all judge calls and contract checks together; then free gates and the pick. No iterative search.
- Calls ask `claude --effort low` (a new `Call.effort` field, part of the cache key); models per role are chosen for speed (task Haiku; intake and rewrites Sonnet; the judge by the timing probe, default Sonnet with `--effort low` unless measured otherwise). R14 still holds (judge is neither task nor target model).
- **Evidence level is stated, not hidden**: fast and quick results are `verified: false` ("fast check": picked and scored on the same few scenarios, no noise measured); checked and deep are holdout-verified as before.
- Thread safety: `BudgetedBackend` counters, `ResilientBackend` failure counts, `RunStore.save_progress` and `log_call` get locks; `Evaluator` takes `workers`. Determinism: cache keys are content hashes, so a resume replays finished calls whatever the thread order; results never depend on completion order (stages gather by index).
- The hard clock is the existing `BudgetedBackend` deadline; the fast stage runner checks the remaining time before each optional call and skips what cannot finish; if no rewrite has passed every gate by the deadline the original is returned.

## Consequences

An interactive run takes about 20 to 30 s instead of up to 45 minutes, at the price of weaker evidence in the default tier (labelled); the research mode stays for those who want proof. More concurrent `claude` processes use more memory (default 4 workers; the user's machine had about 1.6 GB free) and subscription rate. The deep tier needs no change. `Call.effort` changes every cache key once: run folders from earlier versions replay as cache misses, never as errors.

## Alternatives rejected

- **Tune GEPA down:** at 30 s it would run one iteration, which is not a search.
- **One rewrite, no scoring, as the only mode:** cheap but no evidence at all; kept only as the `quick` tier.
- **Warm worker processes (stream-json input):** saves spawn time but one process per model and system prompt; later if the probe shows spawn cost dominates.
