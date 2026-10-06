# ARCHITECTURE: autoimprover 0.2

Status: Draft 5 (2026-10-04). Draft 2 answered G3 round 1 (rejected 0-6), Draft 3 two pre-flight reviews, Draft 4 G3 round 2 (rejected 0-6), this draft two more pre-flight reviews. Implements `docs/SPEC.md`. Needs the G3 vote (round 3, the last the kit allows before the user decides). Every fact has one home: numbers in `types.py` and `docs/SPEC.md`, decisions in `docs/adr/`.

## 1. Key flows

### improve (R3, R4, R11-R15, R17, R22, R24)

```
main(argv, backend=None, now=None):
  opts = parse(argv)                                              # exit 2 on bad usage; models resolved to full ids; judge vs task and target checked (R14)
  if opts.resume: store = RunStore.resume(id)                     # takes run.lock first, then reads manifest, checkpoint, contract.json, scenarios.json (R22)
  plan = Plan(models, budget, strictness, ...)
  scenarios = read(--examples) or None                            # free; a bad line exits 2; a prompt containing "<curr_param>" or "<side_info>" exits 2 (ADR-006)
  n = len(scenarios) if given else SYNTH_COUNT (12)
  if n >= 8: h, v = split_sizes(n)                                # R15: holdout min(6, max(3, half_up(0.35 n))), valset clamp(half_up(0.25 n), 2, 4)
  else:      h, v = 0, n                                          # R11, --trust-search only: no holdout; every scenario is both dataset and valset
  m = min(MINIBATCH_SIZE, size of dataset); jc(k) = ceil(k / JUDGE_BATCH_MAX)   # judge calls for a pass over k scenarios (1 for k <= 6; 2 for n = 7)
  pre   = 1 + (1 if synthesising) + (2 * (h + jc(h)) if h else 0) + (v + jc(v))   # intake, synthesis, two seed runs (h task calls + the judge calls each), GEPA's seed valset pass
  final = (3 * (h + jc(h)) if h else 0) + 3                           # up to 3 finalist runs on the target model + up to 3 contract checks
  iter_cost = 1 + 2 * (m + 1) + (v + jc(v))                       # an accepted child: reflect, parent minibatch, child minibatch, valset pass; a rejected one costs 1 + 2 * (m + 1)
  search_calls = budget - pre - final                             # a rejected reflection reply costs one more call (GEPA retries it): a planning estimate, not a limit
  iterations = search_calls // iter_cost                          # worst case; the R4 floor uses it; --dry also prints the best case
  refusal = first of: runs folder not usable (read-only check) | search_calls < 0 | (iterations < 4 and not --force-low-budget)
  if --dry: print plan, search_calls, both iteration counts, final time share, "a real run would refuse: <refusal>"; exit 0   # R4: zero calls, writes nothing
  if refusal: exit 2                                              # R4, R17, R23
  if n < 8 and not --trust-search: return Unchanged("no holdout") # R11, decided before any paid call
  store  = store or RunStore.open_or_create(plan, prompt)         # exit 2: not writable, inside a git repository, newer version; holds run.lock (flock) for the run
  clock  = Clock(now, elapsed=store.elapsed_s)                    # monotonic; one object shared by ClaudeCli, Budgeted and the runner
  budgeted = Budgeted(raw, limit=budget - final, used=store.calls_used, clock=clock, deadline=SEARCH_CLOCK_SHARE * wall_clock)
  backend  = Cached(Resilient(budgeted))
  try:                                                            # CallFailed outside the GEPA adapter ends the run: BackendError, exit 3, folder kept (R24)
    contract  = store.contract() or extract_contract(backend, prompt)       # R5; kind from --kind or the guess; a resumed run reads the saved one
    scenarios = store.scenarios() or scenarios or synthesize(backend, prompt, contract)    # R11; the run folder's copy wins on resume
    train, val, holdout = split(scenarios, seed)                  # R15 (n < 8: train = val = scenarios, holdout empty)
    if h:
      base = [runner.score_holdout(prompt, holdout, models.target, sample=i) for i in (0, 1)]   # R12, R14a: both on the target model; judge calls carry the same sample
      noise = abs(base[0] - base[1]); threshold = max(0.05, 2 * noise)                         # R12
      if mean(base) >= 0.95: return Unchanged("already strong")   # R13
    state = RunState(search_start=store.search_start_or_record(budgeted.used, clock.elapsed))   # ADR-004: the replay-invariant meter starts here
    result = gepa.optimize_anything(prompt, batch_evaluator=adapter(evaluator, state), dataset=train, valset=val,
               config=GEPAConfig(engine=EngineConfig(parallel=False, run_dir=None, ...),
                                 reflection=ReflectionConfig(reflection_lm=reflection_wrapper(state), ...),
                                 tracking=TrackingConfig(logger=RunLogger(store)),
                                 stop_callbacks=[state.stopper()]))     # stdout redirected into gepa.log meanwhile
    state.raise_if_aborted()                                      # backend -> exit 3, lockdown -> exit 4, bug -> exit 1; run folder kept
    budgeted.raise_limit(budget, deadline=wall_clock)             # the final steps' share is spendable now
    free_ok = [c for c in state.completed minus the seed if length_ok(c) and literals_preserved(c)]     # R7, R9: free gates first
    finalists = top 3 of free_ok by valset score, each contract-checked (R6; at most 3 judged checks)
    if not h:                                                     # --trust-search: only a finalist that beats the seed on the valset, verified=False (R11)
      best = first finalist with valset score > the seed's; return Improved(best, verified=False, ...) or Unchanged("no candidate beat the seed")
    for cand in finalists (best first):
      if runner.score_holdout(cand, holdout, models.target) > mean(base) + threshold:
        return Improved(cand, verified=True, stop=state.stop, changes=lineage_notes(cand))     # R3, R2
    return Unchanged("no reliable improvement", stop=state.stop)
  except BudgetExhausted:                                         # in the final steps
    return Unchanged("ran out of budget or time before a candidate was confirmed", stop=...)   # never an unvetted answer (R17)
  except CallFailed as e: raise BackendError(str(e))
```

### Execution and judging (R10, R10a, R10b)

`template`: system = candidate, user = scenario input. `task`: user = situation + candidate, no system prompt. The task model's outputs for a batch of at most `JUDGE_BATCH_MAX` scenarios go to one judge call with the checklist; the judge never sees the candidate when it scores. The one exception is the R6 contract check at the end, where the judge compares the candidate with the original; it can only veto a candidate, never raise a score (ADR-002). Replies and the quote rule: ADR-008. An omitted check is `unknown` and is left out of the score (more than 30 % unknown scores 0, R24).

### Stop and failure inside GEPA (R17, R18, R24; ADR-004)

GEPA swallows every exception raised in the reflection callable and, with `raise_on_exception=True`, re-raises evaluator exceptions out of `optimize_anything`; after a reflection exception it retries that task once more (`reflective_mutation.py`), so a rejected reply costs a second paid call (probes, G3). So the runner wraps both and keeps backend exceptions away from GEPA's own handling. The wrappers catch `Exception` only: `KeyboardInterrupt` passes through to `cli.main`, which exits 130.

- **Evaluator adapter** (batch): per pair, a `CallFailed` gives a neutral `(0.0, {"incomplete": ...})`, the candidate is not added to `state.completed`, and the failure is recorded as a tombstone (below). A `CallFailed` while scoring the seed candidate (the original) is different: the seed's score steers every acceptance and the `--trust-search` comparison, so it is recorded as `abort = backend` (exit 3), never as a 0. A `BudgetExhausted` (the hard limit or the deadline fired inside the search) is a **stop**, not an abort: it is recorded as `state.stop` (`clock` if the deadline raised it, else `budget`), every later pair returns neutral without a call, the search ends at the next stopper check, and the finalists come from `state.completed`. `BackendError`, `SessionNotLockedDown` or any other exception is recorded in `state.abort` and also makes later pairs neutral.
- **Reflection wrapper**: calls the model with `sample = state.next_reflect_index()` (a running index, so a stalled search cannot replay one cached proposal for free, and a replayed run asks the same calls). The reply must hold the new instruction between the delimiter lines `INSTRUCTION_BEGIN` and `INSTRUCTION_END` (ADR-008; a fence cannot be the delimiter because the instruction may contain code blocks, R9); the wrapper cuts it out and hands GEPA the instruction wrapped in one outer fence, because GEPA keeps everything from the first to the last fence. A reply with no delimiters or an empty instruction, an instruction containing a placeholder token, and a `CallFailed` (also recorded as a tombstone) all raise `SkipProposal`: GEPA retries once, then skips the iteration, and the run goes on. `BackendError`, `SessionNotLockedDown` and anything else are recorded in `state.abort` and re-raised (GEPA swallows them); a `BudgetExhausted` is recorded as a stop as above; once `state.abort` or `state.stop` is set the wrapper raises without calling.
- **Meter** (`SearchMeter`): counts each **distinct** call (by cache key) once, the first time the search issues it, live or served from the cache, retries excluded, and sums that first occurrence's duration (live: measured; hit or tombstone: stored in the cache entry). A repeat of an identical call inside one run costs nothing, live or cached, so it is not charged. Replaying a run issues the same distinct calls in the same order, so the meter advances exactly as the original did.
- **Stopper** (`stop_callbacks`): True when `state.abort` or `state.stop` is set; when `calls_left < iter_cost`, where `calls_left = (budget - final) - search_start.used - meter.calls`; or when `search_start.elapsed + meter.seconds + meter.seconds_per_call * iter_cost >= SEARCH_CLOCK_SHARE * wall_clock` (a one-iteration look-ahead, so the hard deadline rarely cuts the last iteration midway). It reads nothing but the meter, `state` and the constants, never the saved call total or the elapsed time directly.
- **Stop cause** for the report (`StopCause`, GEPA 0.1.4 never ends a search by itself, probe): the calls condition or a hard-limit `BudgetExhausted` -> `budget`, the normal ending; the clock condition or a deadline `BudgetExhausted` -> `clock`. Only `clock` prints the "cut short" notice. A run that did not search (already strong, no holdout) has `stop = None`.
- After GEPA returns, `state.raise_if_aborted()` turns `backend`, `lockdown` and unexpected causes into exits 3, 4 and 1; a stop is not an exit.
- The hard limits are separate and unchanged: `Budgeted` stops live spending at its limit and deadline whatever the meter says.

### Failures count per role and model (R24)

`ResilientBackend` retries a failed call twice (each attempt counts). After that the call is a `CallFailed`, and it counts toward `MAX_CONSECUTIVE_FAILURES` **per (role, model id)**: three consecutive failed calls of the same role to the same model end the run (`BackendError`). Another role's or another model's success does not reset it, so a judge or reflection model that is down while the task model still answers, or a reflection that always fails while the judge (the same Opus model by default) succeeds, ends the run instead of burning the budget (probes, G3 pre-flight and round 3).

A reply that is valid for the JSON schema but fails `Check`, `Scenario` or `Contract` validation (for example a judged check with `"arg": ""`) is retried as `Call(..., sample = sample + 1)`: a new cache key, so the cache cannot serve the bad reply back, on this run or on `--resume`. Each such reply counts as a failed attempt; after the retries it is a `CallFailed`. The validating callers (WP2, WP3, WP4) own this loop. An `"arg": ""` on a judged check is read as null before validation.

### Resume (R22)

`--resume <id>` takes `run.lock`, then loads the manifest (prompt, plan, flags, `calls_used`, `elapsed_s`), `checkpoint.json` (`search_start`), `contract.json` and `scenarios.json` (a resumed run uses the saved contract and scenarios, not a recomputed contract or a possibly edited `--examples` file), reruns the flow above, and the cache replays every paid call, successes and tombstones alike. `Budgeted` and `Clock` start from the saved totals, so live spending continues the same budget and the same 45 minutes (replayed hits never reach `Budgeted`). The stopper reads only the meter, so a run cut in its last iteration or in the final steps replays the whole search to the same stop point and reaches the same finalists. GEPA's call order is deterministic for a fixed seed (probed at every cut point for n = 8, 12, 40), and reflection calls carry their running index, so the replay asks exactly the calls that were recorded.

**Tombstones.** A call that failed all its retries *inside the search* is recorded by the wrapper that swallowed it, `store.record_failure(call, error, duration_s)`, and `CachedBackend` replays it as a `CallFailed` without a live call, so the replay decides as the original did and the saved budget total cannot refuse a retry that the original never made (probe, round 3). Not recorded, so a resume tries them again: the call whose failure raised `BackendError` (the third consecutive one) and every failure outside the search (intake, synthesis, seed runs, finalist runs, contract checks). The consecutive-failure counter lives in `Resilient`, below the cache, so replayed tombstones do not advance it; after a resume the run may tolerate up to two more failures than an uninterrupted one would, which is accepted.

## 2. Modules (`src/autoimprover/`)

| Module | Responsibility and public interface | Spec |
|---|---|---|
| `types.py` | dataclasses, protocols, constants shared by all: `Backend`, `BatchEvaluator`, `Call`, `Reply`, `Contract`, `Check`, `Scenario`, `Models`, `Plan`, `Outcome` (with `changes` and the search-model scores), the reply schemas `INTAKE_SCHEMA`, `SYNTH_SCHEMA`, `JUDGE_SCHEMA`, exceptions (`CallError`, `CallFailed`, `BackendError`, `BudgetExhausted`, `SessionNotLockedDown`), exit codes, `canonical_model`, `default_models` | all |
| `claude_cli.py` | `ClaudeCliBackend(clock, ...)`: the only builder of the `claude -p` command, scrubbed environment, per-call timeout, lockdown check on every call from the `init` line of the stream (ADR-009, section 9); depends on `Clock` from `backend.py` | R17, R18, R19 |
| `backend.py` | `Clock`; `BudgetedBackend(raw, limit, used, clock, deadline)` with `raise_limit`, `ResilientBackend(inner)` (failure count per model id), `CachedBackend(inner, store)` (entries store the call's `duration_s`; tombstones replay as `CallFailed`; `Reply` carries the duration for the search meter); the layers of section 6 (everything except the `claude -p` process) | R17, R19, R20, R24 |
| `runformat.py` | the pure JSON formats of the run folder: field specs, validators, parse functions for manifest, plan, checkpoint, contract, scenarios and cache entries, the cache key hash; no file I/O, imported only by `runstore.py` | R22, R23 |
| `runstore.py` | `RunStore`: `open_or_create` and `resume(id)` (both take `run.lock` first; `resume` then reads manifest, checkpoint, contract and scenarios), `save_progress(calls_used, elapsed_s)`, `search_start_or_record(used, elapsed)`, `cache_get/put`, `log_call`, `record_failure(call, error, duration_s)` (tombstones; the cache key is `runformat.cache_key`, re-exported by `runstore`), `save_contract` / `contract()`, `save_scenarios` / `scenarios()`, `open_log()` (for `gepa.log`), `cwd()` (the empty working folder of the child process), `clean(id=None)`, `resolve_run(id)`, `check_root()` (read-only usability test for `--dry`); the only module that touches the run folder | R22, R23 |
| `contract.py` | `extract_contract(backend, model, prompt, kind=None) -> Contract` (`model` is the reflection model, ADR-008); `Violation(check_id: str, text: str)`; `check(backend, judge_model, contract, original, candidate) -> list[Violation]` (the programmatic checks plus one judged call, ADR-008); `literals(prompt) -> tuple[str, ...]`, `literals_preserved(original, candidate) -> bool` | R5, R6, R9 |
| `scenarios.py` | `read_examples(path: Path) -> list[Scenario]` (a bad line raises `ValueError("line N: ...")`, which `cli` turns into exit 2); `synthesize(backend, model, prompt, contract) -> list[Scenario]` (`model` is the reflection model); `split_sizes(n) -> tuple[int, int, int]` (holdout, valset, dataset; n >= 8); `split(scenarios, seed) -> Split`, `Split(train, val, holdout)` being tuples of `Scenario` (n < 8: `train == val == all`, `holdout == ()`) | R11, R15 |
| `evaluator.py` | `Evaluator(backend, contract, task_model, judge_model, sample=0)` implementing `BatchEvaluator` (the runner builds one with `models.task` for the search and one with `models.target` and `sample` 0 or 1 for the seed and finalist runs): task call per scenario, programmatic checks, one judge call per at most `JUDGE_BATCH_MAX` scenarios, quote rule; a `CallFailed` on a task call or the judge call returns an `incomplete` entry instead of raising (`BudgetExhausted`, `BackendError`, `SessionNotLockedDown` propagate); `side_info["scores"]` per group and the ASI in other keys | R10, R10a, R10b, R16, R24 |
| `search.py` | the GEPA seam, the only importer of `gepa`: `SearchMeter` (distinct calls once, seconds), `RunState` (`abort`, `stop`, `completed`, reflection index, `raise_if_aborted`), the batch evaluator adapter, the reflection wrapper (delimiter parsing, `SkipProposal`), the stopper, the stdout redirect and `RunLogger`, `run_search(...) -> SearchResult` (candidates with valset scores and lineage notes) | R12, R15, R15a, R16, R17, R22, R24 |
| `runner.py` | `improve(prompt, plan, backend, store, ...) -> Outcome`; `fixed_costs` (its `iterations` and `iterations_best` fields are the estimate), `refusal`; reflection prompt per strictness (ADR-006); `score_holdout`; gates (contract, length cap, literals); finalist selection, `--trust-search`, the final steps and the `Outcome` with its report fields | R3, R4, R6, R7, R8, R11-R14a, R17, R22 |
| `report.py` | `render(outcome, ...)`: word diff, summary lines, JSON, error object; the only writer to stdout | R2 |
| `parallel.py` | `parallel_map(fn, items, workers) -> list`: results in input order, inline for `workers <= 1`, pending items cancelled on the first error (lowest input index raised), Ctrl-C re-raised at once; the only place threads are made | R25 |
| `efforts.py` | `EffortBackend(inner, efforts)`: sets `Call.effort` by role when unset; sits above the cache so the effort is part of the key | R25 |
| `contract_text.py` | the fixed texts of the intake call and the contract check (`INTAKE_SYSTEM`, `CONTRACT_SYSTEM`, `CONTRACT_MANY_SYSTEM`, `NO_NEW_GOAL`); no logic | R5, R6 |
| `fastplan.py` | tiers (`tier_for`), the latency model (3.4 s per call + output tokens / 70 s, task tokens `task_tokens(prompt_tokens)` = 150 up to 50 prompt tokens, +3 per token, at most 600, waves = ceil(calls / workers)) and `fast_plan(time_s, workers, prompt_tokens, have_examples) -> FastPlan` (rewrites K, scenarios M, holdout, generations, stages, estimate), `misfit`, `shrink` | R25 |
| `fast_prompts.py` | rewrite strategies (clarify, structure, tighten, specify) and their calls, the kind-agnostic synthesis call, the reflection call of generation 2, `parse_rewrite`, `parse_synth`, `FAST_TASK_SUFFIX`, `FastEvaluator` (the Evaluator with the 120-word suffix) | R25 |
| `delimiters.py` | `BEGIN_MARKS`, `END_MARKS` and `span(lines)`: the one reading of a rewrite reply's delimiter lines (`<<<INSTRUCTION`, bare or spaced `<<<`), used by `fast_prompts.parse_rewrite` and the deep parser of `search.py`; imports only `types` | ADR-008 |
| `fast_calibrate.py` | the in-run latency model: `Timed` (records each distinct call's seconds, output tokens and planned tokens), `fit` (Theil-Sen medians clamped to 0.5x-1.5x), `planned_tokens`, `token_ratio` (`RATIO` 1 to 4), `Timed.model()` and `grow` (extra pick scenarios after stage A within 0.85 of the clock) | R25, ADR-011 |
| `reference_score.py`, `reference_text.py` | the reference-scored decision (SPEC R25, WP21): the win rule over summed scores and the noise, the batched agreement judge call and its schema, the fast and checked labels | R25 |
| `fast_reference.py` | stages C, C2 and E of a fast or checked run with references: the judge calls, `Judged.score`, the reflection evidence | R25 |
| `bench_hidden.py` | the bench's scoring of an item's hidden examples (`eval_from`): both prompts run on each, one or two batched reference judge calls, wins, ties, losses and pass rates | R26 |
| `report_text.py` | `REASON_LINES` and the fast meanings, split from `report.py` | R2 |
| `fast.py`, `fast_stages.py` | `improve_fast(prompt, plan, fplan, ...) -> Outcome`: stage A (intake, synthesis, rewrites in one wave), free gates, B (the original twice and the rewrites on the scenarios), C (one output-only judge call per prompt and one batched contract check), D (pick: gain above max(0.1, 2 x noise) and more wins than losses), R/B2/C2 (reflective second generation from 45 s), E (held-out confirmation in the checked tier); a stage that does not fit the clock is shrunk, a cut returns the best complete result | R25 |
| `fast_pairwise.py`, `pairwise_text.py` | the pairwise preference pick of the fast and checked tiers (ADR-012; `Rewrite`, `Judged`, `Win`, `won` and `best_ungated`, the ranking of `--ungated` of SPEC R25): per rewrite two output-only batched judge calls (both orders), a verdict counts only when both orders agree, the original against itself gives the noise, a rewrite wins iff wins - losses > noise; `pairwise_text.py` holds the judge's text and schema without imports (bench_judge imports gepa) | R25, R26 |
| `bench.py`, `bench_judge.py`, `bench_report.py`, `cli_bench.py` | `autoimprover bench` (takes a run's model, effort, workers and strictness flags through `cli_options`): the prompt set loader, each prompt through the ordinary pipeline, the blind pairwise judge on fresh scenarios in both orders (agreement or tie, `PAIRWISE_SCHEMA`), the naive baseline, the summary and its `--dry` plan; Ctrl-C exits 130 with `summary.json` | R26 |
| `cli_options.py`, `cli_input.py`, `cli_plan.py`, `cli_fast.py` | flag parsing and defaults per tier, prompt/examples input, the plan views (`PlanView`, `FastView`, `DRY_KEYS`), the fast flow helpers (saved plan, progress tee to stderr) | R1, R2, R4, R25 |
| `cli.py` | `main(argv, *, backend=None, now=None) -> int` (`backend` replaces the raw model layer, `now` the monotonic clock) and the subcommand `clean`. Flags: `--dry`, `--force-low-budget`, `--json`, `--file`, `--examples`, `--kind template\|task`, `--budget`, `--strictness`, `--allow-growth`, `--task-model`, `--judge-model`, `--reflect-model`, `--target-model`, `--merge`, `--trust-search`, `--resume <id>` | R1, R2, R4, R14, R22, R23 |
| `.claude-plugin/`, `hooks/` (repo root) | the Claude Code mod: `plugin.json`, `marketplace.json`, `hooks.json`, `register.js` (every `$` call), pure modules `args.js`, `argv.js`, `report.js`, `stream.js`, `pane.js`; `/improve` and `/optimize` run the CLI as a child process (ADR-010) | R19, R21 |

Dependencies point one way: `cli -> runner -> search -> {evaluator, backend}`, `runner -> {contract, scenarios}`, `{evaluator, contract, scenarios} -> backend -> runstore -> types`, and `cli -> claude_cli -> backend` (the CLI builds the real raw layer; tests inject their own). `gepa` is imported only in `search.py`; nothing else touches GEPA types or `oa.log` (not available on the batch path).

Interfaces fixed at the skeleton commit are in `types.py` (read it, not a copy here) and the `cli.main` signature. Signatures of the other modules above are fixed by this table; an owner may add private helpers but changes a public signature only through the lead.

## 3. Work packages

Every file has exactly one owner. The lead owns `pyproject.toml`, `uv.lock`, `justfile` (including the `smoke` recipe added before G6), `.gitignore`, `.python-version`, `CLAUDE.md`, `docs/**`, `.claude/**`, `src/autoimprover/__init__.py`, `src/autoimprover/types.py`, `tests/conftest.py`, `tests/fakes.py`, `tests/test_types.py`, `tests/test_guards.py`, `tests/test_fakes.py`, `tests/test_package.py`.

| WP | Owner (agent) | Exclusive files | Needs | Proof |
|---|---|---|---|---|
| WP1a backend layers + runstore | `coder` | `src/autoimprover/backend.py`, `runformat.py`, `runstore.py`, `tests/test_backend.py`, `tests/test_backend_layers.py`, `tests/test_runstore.py`, `tests/test_runstore_files.py` | skeleton | R17, R19, R22, R23, R24 |
| WP1b claude cli | `coder` | `src/autoimprover/claude_cli.py`, `tests/test_claude_cli.py` | WP1a (`Clock`), the user's two real-call outputs | R17 (timeout), R18, R19 |
| WP2 contract | `coder` | `contract.py`, `tests/test_contract.py`, `tests/test_contract_literals.py`, `tests/test_contract_check.py` | skeleton | R5, R6, R9 |
| WP3 scenarios | `coder` | `scenarios.py`, `tests/test_scenarios.py` (and `tests/test_scenarios_synth.py` once the synthesis tests move there) | skeleton | R11, R15 |
| WP4 evaluator | `coder` | `evaluator.py`, `tests/test_evaluator.py`, `tests/test_evaluator_judge.py` | WP2, WP3 | R10, R10a, R10b, R16, R24 |
| WP5a search | `coder` | `src/autoimprover/search.py`, `tests/test_search.py`, `tests/test_search_resume.py`; one change to `backend.py` and `tests/test_backend_layers.py` (WP1a is closed): `CachedBackend` gains `record_failures`, a switch that makes it store a tombstone for every `CallFailed` it lets through while on | WP1a-WP4 (merged) | R15a, R16, R17, R22, R24 |
| WP5b runner | `coder` | `runner.py`, `tests/test_runner.py`, `tests/test_runner_flow.py`, `tests/test_runner_prompts.py` | WP5a | R3, R4, R6, R7, R8, R11-R14a, R17, R22 |
| WP6 report + cli | `coder` | `report.py`, `cli.py`, `tests/test_report.py`, `tests/test_cli.py` | WP5 | R1, R2, R4, R14, R22, R23 |
| WP7 docs | `coder` | `README.md` | WP6 | R19, R21 |
| WP9 mod | `coder` | `.claude-plugin/**`, `hooks/**`, `tests/mod/*.test.ts` (run by `claude plugin test`), `tests/test_mod_package.py`, the README section on the mod | WP6, WP7 | R19, R21 |
| WP8 acceptance | `tester` | `tests/acceptance/**` (including its own `conftest.py`) | skeleton only; written from the SPEC and ADR-008 without reading `src/` | the A proofs; every test that asserts a candidate is NOT returned has a twin on `happy_backend` asserting that one IS, so it cannot pass for the wrong reason |

WP1a, WP2, WP3 and WP8 can run in parallel (at most 2 writers at once, each in its own worktree). pytest runs with `--import-mode=importlib` and `pythonpath = ["tests"]`, so `tests/test_cli.py` and `tests/acceptance/test_cli.py` coexist and helpers come from `from fakes import ...`.

## 4. Skeleton (committed before any package starts)

`types.py` (all shared types and constants), `cli.py` as a stub with the final `main` signature (WP6 replaces its body), `tests/conftest.py` (the R20 guards), `tests/fakes.py` (`ScriptedBackend` with `duration_s` and `FakeClock`, `by_role`, `failing`, the reply builders `intake_reply`, `synth_reply`, `judge_reply`, `reflection_reply` and the complete `happy_backend` with its `MARKER`), the tests of these, `pyproject.toml` with `[project.scripts] autoimprover = "autoimprover.cli:main"` and the pytest options. The other modules do not exist yet: each owner creates its file from the table in section 2, so no empty stub is shared between packages.

## 5. Requirement map

| Req | Module (WP) | Proof, and an input that makes it fail |
|---|---|---|
| R1 input | `cli.py` (6) | T: 20,001 chars, empty, NUL byte -> exit 2; CRLF and CR are normalised to LF before anything else sees the prompt |
| R2 output, exit codes | `report.py`, `cli.py` (6) | A: each row of section 8 |
| R3 unchanged unless reliable | `runner.py` (5b) | A with fake: candidate gain below threshold -> original, exit 0 |
| R4 dry run, low budget | `cli.py` (6), `runner.fixed_costs`, `runner.refusal` (5) | A: `--dry` makes zero calls, writes nothing and exits 0 even for budget 30 or an unusable state folder; T: budget 66 with 12 scenarios is refused (worst case 2 iterations); T: n = 8, 12, 40 give 6, 5, 4 worst-case iterations at budget 100, and n < 8 (no holdout) has its own fixed costs |
| R5 contract, kind | `contract.py` (2) | T: kind guessed, `--kind` overrides |
| R6 contract gate | `contract.py` (2), gate in `runner.py` (5b) | A: planted violation never returned |
| R7 length cap | `runner.py` (5b) | T: 1.26x candidate rejected at conservative |
| R8 strictness | `runner.py` (5b) | T: the three reflection templates differ and carry the cap |
| R9 literals | `contract.py` (2) | T: hand-written seeded property test (no new dependency) |
| R10 batched scoring, groups | `evaluator.py` (4) | T: one judge call per batch, `side_info["scores"]` per group |
| R10a execution by kind | `evaluator.py` (4) | T: template vs task call shape |
| R10b quote rule | `evaluator.py` (4) | T: a pass with an empty, blank or absent quote counts as a failed check |
| R11 scenarios, holdout rules | `scenarios.py` (3), `runner.py` (5b) | T: n=7 gives no holdout and no paid call without `--trust-search`; with it dataset = valset = all scenarios, and a finalist not above the seed on the valset is never returned; a failed call while scoring the seed is exit 3, not a 0 |
| R12 noise, `Call.sample` | `types.py` (skeleton), `runner.py` (5b) | T: seed run 2 is a live call; threshold uses `2 x \|diff\|` |
| R13 already strong | `runner.py` (5b) | A: seed 0.95 stops with no search |
| R14 judge != task, != target | `types.py` (skeleton), flags in `cli.py` (6) | T: aliases resolved; `--target-model opus` picks the fallback judge |
| R14a target confirmation | `runner.py` (5b) | A: winner on search model, loser on target is not returned |
| R15 split, search wiring | `scenarios.py` (3), `search.py` (5a) | T: n = 8, 10, 12, 30, 40 sizes; `GEPAConfig` builds on 0.1.4 |
| R15a small valset, strict improvement | `search.py` (5a) | T: calls counted on `fake` |
| R16 ASI to reflection | `evaluator.py` (4), template in `runner.py` (5b) | T: failed checks and excerpts in the reflection prompt |
| R17 budget, reserve, clock, timeout | `backend.py` (1a), `claude_cli.py` (1b: timeout), `search.py` (5a: stopper, meter), `runner.py` (5b: reserve) | T, A: limit, deadline, stop keeps candidates, resume keeps the count; T with `FakeClock`: the stopper's one-iteration look-ahead keeps the search inside its clock share; T: an identical repeated call is charged once |
| R18 command, lockdown, argv rules | `claude_cli.py` (1b) | T: argv has no user text; every call checks lockdown (so the first live call of a process too); S |
| R19 untrusted data | `backend.py` (1), `types.py` (skeleton), the mod (9: the prompt goes through a file and an argument list, never through the model) | T hostile strings; `tests/mod/flow.test.ts`, `test_mod_package.py` |
| R20 no real model or network | `tests/conftest.py` (skeleton) | T `test_guards.py` |
| R21 `/improve` mod | `.claude-plugin/**`, `hooks/**` (9) | T `claude plugin test`, `claude plugin validate`, drift test; S |
| R22 resume | `runstore.py` (1a), `search.py` (5a), `runner.py` (5b), `cli.py` (6) | A: a run cut at each stage (mid-search, last iteration, final steps) resumes with zero repeated successful paid calls, the same budget and the same finalists; T: a call that failed before the cut is tried again live |
| R23 run folder, clean | `runstore.py` (1), `cli.py` (6) | T: 0700, not in git, `clean <id>` rejects a path |
| R24 failures, retries, exit 3 | `backend.py` (1a), `evaluator.py` (4), `search.py` (5a), `runner.py` (5b) | A with `failing()`: the first failed call outside the search (intake) -> exit 3 with the run folder; T: three consecutive failed calls to one model inside the search -> exit 3 even while another model succeeds; one failed reflection call only skips an iteration |

## 6. Enforcement points

The single place that enforces each limit, and why nothing goes around it:

| Rule | Enforced in | Why it cannot be bypassed |
|---|---|---|
| Call limit (R17) | `BudgetedBackend.complete` | every live call passes it; cache hits sit above it and cost nothing, and a resume starts it from the saved total |
| Search stop, replay-invariant (R17, R22) | `search.RunState.stopper` over `SearchMeter` (issued calls and seconds, hits counted like live calls) | the stopper reads nothing else, so a replay decides exactly as the original did |
| Reserve (R17) | `BudgetedBackend.limit`, set to `budget - final`, raised only by `runner` after the search | the search has no handle on the limit |
| Wall clock, deadline (R17) | `Clock` (shared) read by `BudgetedBackend` and `ClaudeCliBackend` | one object, monotonic |
| Per-call timeout (R17) | `claude_cli._run`: `min(300 s, deadline() - clock.elapsed())` | the only place a process starts |
| Retries, consecutive failures per model (R24) | `ResilientBackend.complete` | sits directly above `Budgeted`, so every attempt counts; the counter is keyed by model id |
| Cache key, hits free | `CachedBackend._key` over all `Call` fields | `Call` is frozen; a field test pins the list |
| Judge != task, target (R14) | `Models.__post_init__` after `canonical_model` | no `Models` exists without it |
| argv, stdin, system-prompt size (R18) | `Call.__post_init__` and `claude_cli._argv` | user text is only ever put on stdin, except a template candidate, which is the system prompt and rides in `--system-prompt=<text>` (the R18 test asserts exactly that) |
| Lockdown (R18) | `ClaudeCliBackend` on every call (ADR-009); `state.raise_if_aborted` re-raises | covers a resumed run whose first live call is a reflection |
| Prompt size, NUL (R1) | `cli.read_prompt` | the only reader of the prompt |
| Length cap (R7) | `runner.length_ok` | called by the gate that every answer passes |
| Contract and literals (R6, R9) | `contract.check`, called only from `_Run.finalists` in `runner.py` | the answer is built only from the finalists that pass the free gates and the contract check |
| Holdout hidden (R15) | `scenarios.split` returns a `Split`; `main` unpacks it and only `runner.score_holdout` reads the holdout | GEPA gets `train` and `val` lists only |
| Run folder writes, atomicity (R22, R23) | `runstore._atomic_write` | no other module opens run-folder files |
| One live run per folder (R22) | `RunStore.open_or_create`: `fcntl.flock(LOCK_EX \| LOCK_NB)` on `run.lock`, held until exit; `clean` skips a locked folder | the kernel drops the lock when the process dies, so no stale lock and no pid check (pids are namespace-local in sandboxed shells) |
| Run id, no path (R23) | `runstore.resolve_run` (pattern `RUN_ID_PATTERN` and `is_relative_to`) | `--resume` and `clean` call it |
| State folder writable, not in git (R23) | `RunStore.check_root` (read-only, used by `--dry`) and the same checks inside `open_or_create` and `resume` | called before any paid call |
| stdout only the result (R2) | `report.Emitter` writes, `report.render` builds; `cli.main` redirects GEPA output into `store.open_log()` | one writer |
| No real model, network, process, home (R20) | `tests/conftest.py` `_ruv_guards`, shadowing checked at collection | tripwire for tests written in good faith |

## 7. Formats

Run folder files, owner `runstore.py`, schema version 1, atomic writes, how partial or foreign content is read, clocks and concurrency: ADR-007. Summary: JSON with `schema_version`; written to a temporary name in the same folder and moved with `os.replace`; a corrupt cache entry is a miss, a corrupt manifest refuses `--resume` (exit 2); times are UTC for display, durations monotonic; one run folder has one writer (a second `--resume` of the same id while one is live is refused through an `flock` on `run.lock`). `checkpoint.json` holds only `search_start` (the `Budgeted.used` and clock reading when the search first began), written once and read by `--resume`; cache entries hold the reply, the key fields and `duration_s`; the files and the child's `cwd/` folder are listed in ADR-007. The tool reads user files only through `read_examples` (JSONL: `input`, optional `expected`, optional `criteria`; one bad line exits 2 naming the line).

## 8. User-visible states

| State | Exit | stdout | stderr |
|---|---|---|---|
| improved (holdout-verified) | 0 | the improved prompt (`--json`: one object) | report; a notice if the search was cut short by the clock |
| unchanged: no reliable improvement, already strong, no holdout, no candidate beat the seed (`--trust-search`), ran out of budget or time before confirming | 0 | the original prompt (`--json`: one object) | report with the reason |
| `--trust-search` result (only a finalist that beats the seed on the valset) | 0 | the prompt | report and a loud notice "not verified on a holdout" (`verified: false`) |
| `--dry` (also when a real run would refuse, for any reason including an unusable state folder) | 0 | the plan (`--json`: one object) | nothing |
| internal error (a bug) | 1 | nothing (`--json`: error object) | `error: internal error: <type>`, `run folder: <path>` and the resume line when a run folder exists |
| bad input or usage; judge equals task or target; fixed costs above the budget; fewer than 4 iterations without `--force-low-budget`; state folder not writable or inside a git repository; newer or corrupt run folder; run id not an id | 2 | nothing (`--json`: error object) | `error: <what and the flag or folder to change>` |
| backend failure (R24): three consecutive failed calls, or any failed call outside the search (intake, synthesis, seed runs, finalists, contract checks) | 3 | nothing (`--json`: error object) | `error: ...`, `run folder: <path>`, `resume with: [XDG_STATE_HOME='<dir>' ]autoimprover --resume <id>` (the prefix, shell-quoted, appears when the state folder is not the default) |
| session not locked down (R18) | 4 | nothing (`--json`: error object) | `error: ...` and `run folder: <path>` when one exists |
| interrupted (Ctrl-C) | 130 | nothing (`--json`: error object) | `error: interrupted`, and when a run folder exists `run folder: <path>` and `resume with: ...` |
| `clean` | 0 (also when nothing to remove), 2 on bad usage | nothing (`--json`: `{"status":"cleaned","removed":N,"skipped":N}`) | what was removed and what was skipped because a run holds its lock |

`--json` always writes exactly one object to stdout: success has `status`, `prompt`, `verified`, `stop`, `changes`, `reason`, `reason_code`, `diff` (word level), `contract`, `score_before`, `score_after`, `search_score_before`, `search_score_after`, `noise`, `margin`, `length_ratio`, `calls_used`, `run_dir`; errors have `{"status":"error","code":N,"error":"...","run_dir":"..."}`. Notices stay on stderr in both modes. No row reads as success without being one: only rows with exit 0 print a prompt on stdout, and an unverified or cut-short result says so on stderr and in `--json`.

## 9. External calls

One kind of child process, built in one place (`claude_cli._argv`):

`claude -p --safe-mode --settings '{"outputStyle":"default"}' --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 --model <full id> --output-format stream-json --verbose [--system-prompt=<text>] [--json-schema <schema>]`; the `init` line must report no tools, MCP servers, skills, slash commands or extra agents and the default output style (`plugins` is informational), the final `result` line carries the text (`structured_output` for a schema call): ADR-009, real output in `tests/fixtures/claude_cli/`

- stdin: the user text of the call (prompt, scenario, outputs). A system prompt, when the call has one (a template candidate, or the fixed instruction of a role), is the one argument `--system-prompt=<text>`: at most `SYSTEM_PROMPT_MAX_BYTES`, no NUL. Template candidates therefore appear in `/proc/<pid>/cmdline` for the length of a call (readable by local users); the run folder is 0700 but this exposure is accepted, and a file option is used instead if the real-call check shows the CLI has one.
- environment: scrubbed allowlist (`PATH`, `HOME`, `LANG`, `LC_*`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TMPDIR`, and when set the network settings the sandboxed shell of `/improve` needs: `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, `NO_PROXY` in both cases, `SSL_CERT_FILE`), no `ANTHROPIC_*`, no tokens; the allowlist is one constant in `claude_cli.py`.
- working directory: the empty folder `store.cwd()` = `<run>/cwd` (so no project files are read).
- timeout: `min(CALL_TIMEOUT_S, clock.remaining(deadline))`, where `deadline` is the current one (the search deadline until `raise_limit`, the full wall clock after); the process is killed by its exact pid; reply cap 1 MiB, larger is a `CallError`.
- retries: `ResilientBackend`, 2, each counted.
- how `/improve` starts the tool (R21, ADR-010): the mod writes the prompt to a file in a private 0700 folder, spawns `uv run --frozen --project <plugin root> autoimprover --json --file <file> --target-model <session model>` by argument list with `$.process.spawn` (a process a mod starts runs outside Claude Code's sandbox, with the user's login), streams stderr into the status line, reads the one JSON object of R2 from stdout, shows it in a pane, deletes the prompt file in every path, and puts the improved prompt into the prompt box only when the user presses "Use it". Cancel ends the stream, which kills the child by its exact pid.

No other child process, no network call, no `shell=True`.

## 10. Left out on purpose

DSPy and any second optimiser (ADR-001); GEPA merge by default (R15); the local model as task model (first follow-up); `--facts-from-examples` (ADR-006); tool use in candidate runs (single turn, no tools, R18); parallel calls; a hosted service; automatic expiry of run folders; a file option for the system prompt until the real-call check shows one exists; regular-expression checks (R19); a pinned property-testing library (R9 uses a hand-written seeded test; pyright runs from the host and is not pinned by `uv.lock`).

## 11. Open questions (provisional choices)

- (answered 2026-10-05, ADR-009) `--safe-mode` uses the subscription login; the init line of the stream reports tools, MCP servers, skills, commands, agents and the output style; plugins are listed but inactive. Still open: the init line of a schema call (probe 5).
- Judge fallback when the target is the default judge: Sonnet 5.5 (R14).
- The background-run mechanics of `/improve` (timeout, output retrieval, sandbox network) are checked only by `just smoke`; provisional: as in section 9.
- `--merge` is accepted and handed to GEPA, but with a valset under GEPA's overlap floor of 5 (ours is at most 4) it never merges (probe, G3 pre-flight), so at default sizes it changes nothing; kept because SPEC R15 lists it, and a candidate for removal if the user agrees.
- Model aliases are pinned in `types.py` (`MODEL_ALIASES`); the claude CLI may resolve `opus` to a newer id later. Provisional: the user passes full ids when they care.

## ADRs (all Proposed until the G3 vote; index: `docs/adr/`)

ADR-001 gepa only, no DSPy in v1. ADR-002 judge sees outputs, not candidates. ADR-003 three-way split and holdout rule. ADR-004 one backend seam, budget, stop and failure handling. ADR-005 one-off tasks are tested on synthesised situations, templates on inputs. ADR-006 our own reflection prompt. ADR-007 run folder format, atomic writes, clocks, lock and resume. ADR-008 wire formats of every model message (intake, synthesis, task, judge, contract check, reflection).
