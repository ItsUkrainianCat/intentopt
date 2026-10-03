# ADR-004: Every model call goes through one Backend seam that counts, caches, retries and stops

- **Status:** Proposed (revised 2026-10-03 after G3 round 1)
- **Date:** 2026-10-03
- **Requirement(s):** R17, R18, R19, R20, R22, R24

## Context

GEPA's `max_metric_calls` does not count reflection calls, and each call here is a `claude -p` process on the user's Max plan. Calls must be countable, capped, resumable and safe, and tests must never reach a real model. Probes in the G3 round 1 vote showed what GEPA 0.1.4 does with exceptions raised inside the search: with `raise_on_exception=True` (the default) a `BudgetExhausted` escapes `optimize_anything` and the result is lost; with `False` the error is scored 0.0 and the search keeps going, so a dead backend burns the whole budget and exit 3 never happens.

## Decision

1. **Protocol.** `Backend.complete(Call) -> Reply`. `Call` carries `role, model, user, system, json_schema, sample`; all of them are part of the cache key. `sample` exists so the second seed run (R12) is a different call.
2. **Stack**, outermost first: `Cached(Resilient(Budgeted(ClaudeCli | FakeBackend)))`.
   - `ClaudeCli` is the only code that builds the `claude -p` command (R18) and the scrubbed environment. User text goes on stdin only. The system prompt is passed as `--system-prompt=<text>` (one argument, so a leading `-` cannot be read as an option) and is limited to `SYSTEM_PROMPT_MAX_BYTES` (100,000, under the 131,072-byte limit of one argument); the runner rejects a longer candidate before running it. The call has a timeout of `min(300 s, clock left)`. The first call of a run aborts with exit 4 if the session reports plugins, MCP servers or tools.
   - `Budgeted` counts every call that reaches it, holds the current `limit` and the monotonic clock, and raises `BudgetExhausted`. It is the single enforcement point. The limit starts at `budget - final_costs` (the final steps' share of the R17 reserve): intake, synthesis and the seed runs are paid from it as they happen and the search gets what is left. `raise_limit(budget)` opens the final steps' share once the search has ended. `--dry` shows the same split up front.
   - `Resilient` retries a failed call twice (each attempt goes through `Budgeted` and counts) and raises `BackendError` after three consecutive calls that failed all their retries (R24).
   - `Cached` is keyed by the SHA-256 of all `Call` fields, stores only successful replies, one atomic file per call (ADR-007); a hit does not touch the budget.
3. **Stopping GEPA.** The runner wraps the evaluator and the reflection callable. The wrappers catch `BudgetExhausted` and `BackendError`, record the cause in a `RunState` (`abort`: budget, clock or backend) and return neutral results, so GEPA never sees these exceptions. `RunState.completed` lists only candidates that finished a full evaluation. A stopper in GEPA's `stop_callbacks` ends the search when `abort` is set, when the calls left in the search share are fewer than one iteration costs, or when 75 % of the wall clock is used. `raise_on_exception` stays True so genuine bugs still surface.
4. **After the search.** `abort == backend` becomes exit 3 with the run folder kept. A budget or clock stop continues with the finalists drawn from `RunState.completed`, run from the reserve through the same gates as any other result (R3, R6, R7, R9); if the clock runs out before a finalist is confirmed, the original is returned.
5. **GEPA output.** A custom logger writes GEPA's progress to the run folder and stdout is redirected there while GEPA runs, so stdout holds only the result (R2). `run_dir` is None: no pickled GEPA state is created or loaded.
6. `FakeBackend` serves scripted replies for all tests. Prompts and outputs are never passed through a shell.

## Consequences

One place to audit for safety and cost; the reserve is protected because the search can only spend its own share; resume (R22) falls out of the cache because GEPA with the same seed and the same cached replies retraces its steps for free; a failing backend ends the run in three failed calls instead of burning the budget. The single-turn, no-tools setting means the task model cannot exercise tool use, so prompts that depend on tools are scored on what the model *says* it would do; the report notes this limit. The adapter (wrappers, `RunState`, stopper) is non-trivial code in `runner.py` and needs its own tests against the pinned GEPA.

## Alternatives rejected

- **Count only GEPA metric calls:** misses reflection, intake, synthesis and judge calls.
- **Let `BudgetExhausted` propagate out of GEPA:** loses the result (probe, round 1).
- **`raise_on_exception=False`:** scores errors as 0.0 and keeps searching (probe, round 1).
- **Anthropic API with a key:** violates the no-API-key rule for this machine.
- **`--bare`:** needs an API key and never reads the subscription login.
- **Budget counted above the cache:** would charge cache hits that cost nothing.
