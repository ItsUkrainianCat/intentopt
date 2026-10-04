# ARCHITECTURE: autoimprover 0.2

Status: Draft 4 (2026-10-03). Draft 2 answered G3 round 1 (rejected 0-6), Draft 3 two pre-flight reviews, this draft G3 round 2 (rejected 0-6). Implements `docs/SPEC.md`. Needs the G3 vote (round 3, the last the kit allows before the user decides). Every fact has one home: numbers in `types.py` and `docs/SPEC.md`, decisions in `docs/adr/`.

## 1. Key flows

### improve (R3, R4, R11-R15, R17, R22, R24)

```
main(argv, backend=None):
  opts = parse(argv)                                              # exit 2 on bad usage; models resolved to full ids; judge vs task and target checked (R14)
  if opts.resume: manifest = RunStore.load(id); opts = manifest.opts   # prompt and plan come from the run (R22)
  plan = Plan(models, budget, strictness, ...)
  scenarios = read(--examples) or None                            # free; a bad line exits 2
  n = len(scenarios) if given else 12
  h, v = split_sizes(n)                                           # R15: holdout min(6, max(3, half_up(0.35 n))), valset clamp(half_up(0.25 n), 2, 4)
  pre   = 1 + (1 if synthesising) + 2 * (h + 1) + (v + 1)         # intake, synthesis, two seed runs (h task calls + 1 judge each), GEPA's seed valset pass
  final = 3 * (h + 1) + 3                                         # up to 3 finalists on the target model + up to 3 contract checks
  iter_cost = 1 + 2 * (MINIBATCH_SIZE + 1) + (v + 1)              # worst case, an accepted child: reflect, parent minibatch, child minibatch, valset pass
  search_calls = budget - pre - final                             # a rejected child costs 1 + 2 * (MINIBATCH_SIZE + 1); --dry prints both counts
  iterations = search_calls // iter_cost                          # worst case; the R4 floor uses it
  refusal = first of: runs folder not usable (read-only check) | search_calls < 0 | (iterations < 4 and not --force-low-budget)
  if --dry: print plan, search_calls, both iteration counts, final time share, "a real run would refuse: <refusal>"; exit 0   # R4: zero calls, writes nothing
  if refusal: exit 2                                              # R4, R17, R23
  if n < 8 and not --trust-search: return Unchanged("no holdout") # R11, decided before any paid call
  store  = RunStore.open_or_create(plan, prompt)                  # exit 2: not writable, inside a git repository, newer version; holds run.lock (flock) for the run
  clock  = Clock(elapsed=store.elapsed_s)                         # monotonic; one object shared by ClaudeCli, Budgeted and the runner
  budgeted = Budgeted(raw, limit=budget - final, used=store.calls_used, clock=clock, deadline=SEARCH_CLOCK_SHARE * wall_clock)
  backend  = Cached(Resilient(budgeted))
  try:                                                            # CallFailed outside the GEPA adapter ends the run: BackendError, exit 3, folder kept (R24)
    contract  = extract_contract(backend, prompt)                 # R5; kind from --kind or the guess
    scenarios = scenarios or synthesize(backend, prompt, contract)    # R11
    train, val, holdout = split(scenarios, seed)                  # R15
    base = [score_set(prompt, holdout, models.target, sample=i) for i in (0, 1)]   # R12, R14a: both on the target model; judge calls carry the same sample
    noise = abs(base[0] - base[1]); threshold = max(0.05, 2 * noise)               # R12
    if mean(base) >= 0.95: return Unchanged("already strong")     # R13
    state = RunState(search_start=store.search_start_or_record(budgeted.used, clock.elapsed))   # ADR-004: the replay-invariant meter starts here
    result = gepa.optimize_anything(prompt, batch_evaluator=adapter(evaluator, state), dataset=train, valset=val,
               config=GEPAConfig(engine=EngineConfig(parallel=False, run_dir=None, ...),
                                 reflection=ReflectionConfig(reflection_lm=reflection_wrapper(state), ...),
                                 tracking=TrackingConfig(logger=RunLogger(store)),
                                 stop_callbacks=[state.stopper()]))     # stdout redirected into gepa.log meanwhile
    state.raise_if_aborted()                                      # backend -> exit 3, lockdown -> exit 4, bug -> exit 1; run folder kept
    budgeted.raise_limit(budget, deadline=wall_clock)             # the final steps' share is spendable now
    finalists = top 3 of state.completed minus the seed, by val score, that pass contract, length cap and literals   # R6, R7, R9
    if n < 8: finalists = [c for c in finalists if val(c) > val(seed)]; return best as Improved(verified=False) or Unchanged   # --trust-search (R11)
    for cand in finalists (best first):
      if holdout_score(cand, models.target) > mean(base) + threshold:
        return Improved(cand, verified=True, stop=state.stop, changes=lineage_notes(cand))     # R3, R2
    return Unchanged("no reliable improvement", stop=state.stop)
  except BudgetExhausted:                                         # in the final steps
    return Unchanged("ran out of budget or time before a candidate was confirmed", stop=...)   # never an unvetted answer (R17)
  except CallFailed as e: raise BackendError(str(e))
  # --trust-search (n < 8) has no holdout, so the seed runs and the 0.95 check are skipped and the search starts after the split
```

### Execution and judging (R10, R10a, R10b)

`template`: system = candidate, user = scenario input. `task`: user = situation + candidate, no system prompt. The task model's outputs for a batch of at most `JUDGE_BATCH_MAX` scenarios go to one judge call with the checklist; the judge never sees the candidate. It returns `{check: pass, quote}`; a pass whose quote is not in the output counts as a fail, an omitted check is `unknown` and is left out of the score (more than 30 % unknown scores 0, R24). Wire formats of every message: ADR-008.

### Stop and failure inside GEPA (R17, R18, R24; ADR-004)

GEPA swallows every exception raised in the reflection callable (`reflective_mutation.py`) and, with `raise_on_exception=True`, re-raises evaluator exceptions out of `optimize_anything` (probes, G3). So the runner wraps both and keeps backend exceptions away from GEPA's own handling:

- **Evaluator adapter** (batch): per pair, a `CallFailed` gives a neutral `(0.0, {"incomplete": ...})` and the candidate is not added to `state.completed`; `BudgetExhausted`, `BackendError`, `SessionNotLockedDown` or any other exception is recorded in `state.abort` and every later pair returns neutral without a call.
- **Reflection wrapper**: calls the model with `sample = state.next_reflect_index()` (a running index, so a stalled search cannot replay one cached proposal for free, and a replayed run asks the same calls). A reply that is empty or has no code block, and a `CallFailed`, both raise `SkipProposal`: GEPA skips that iteration and nothing is recorded. `BackendError` (the third consecutive failed call), `BudgetExhausted`, `SessionNotLockedDown` and anything else are recorded in `state.abort` and re-raised (GEPA swallows them); once `state.abort` is set the wrapper raises without calling.
- **Stopper** (`stop_callbacks`): True when `state.abort` is set, when `search_calls_at_start - meter.issued_calls < iter_cost`, or when `meter.issued_seconds` has reached the search share of the clock. The **meter** counts every call the search issues, a cache hit exactly like a live call (retries excluded), and sums each call's duration (live: measured; hit: the duration stored in its cache entry). Replaying a run therefore advances the meter exactly as the original did, and the stopper decides the same at every iteration; it never reads the saved call total or the elapsed time directly.
- After GEPA returns, `state.raise_if_aborted()` turns `backend`, `lockdown` and unexpected causes into exits 3, 4 and 1. A `budget` or `clock` cause is not a failure: the finalists come from `state.completed`.
- The hard limits are separate and unchanged: `Budgeted` stops live spending at its limit and deadline whatever the meter says.

### Resume (R22)

`--resume <id>` loads the manifest (prompt, plan, flags, `calls_used`, `elapsed_s`) and `checkpoint.json` (`search_start`: the `Budgeted.used` and clock reading when the search first began), reruns the flow above, and the cache replays every paid call. `Budgeted` and `Clock` start from the saved totals, so live spending continues the same budget and the same 45 minutes (replayed hits never reach `Budgeted`). The stopper reads only the meter, so a run cut in its last iteration or in the final steps replays the whole search to the same stop point, reaches the same finalists and continues. GEPA's call order is deterministic for a fixed seed (probed), and reflection calls carry their running index, so the replay asks exactly the calls that were cached.

## 2. Modules (`src/autoimprover/`)

| Module | Responsibility and public interface | Spec |
|---|---|---|
| `types.py` | dataclasses, protocols, constants shared by all: `Backend`, `BatchEvaluator`, `Call`, `Reply`, `Contract`, `Check`, `Scenario`, `Models`, `Plan`, `Outcome` (with `changes` and the search-model scores), the reply schemas `INTAKE_SCHEMA`, `SYNTH_SCHEMA`, `JUDGE_SCHEMA`, exceptions (`CallError`, `CallFailed`, `BackendError`, `BudgetExhausted`, `SessionNotLockedDown`), exit codes, `canonical_model`, `default_models` | all |
| `backend.py` | `Clock`; `ClaudeCliBackend(clock)`, `BudgetedBackend(raw, limit, used, clock, deadline)` with `raise_limit`, `ResilientBackend(inner)`, `CachedBackend(inner, store)` (entries store the call's `duration_s`; `Reply` gains it for the search meter); the layers of section 6 | R17-R20, R24 |
| `runstore.py` | `RunStore`: `open_or_create` (takes `run.lock`), `load(id)`, `save_progress(calls_used, elapsed_s)`, `search_start_or_record(used, elapsed)`, `cache_get/put`, `log_call`, `clean(id=None)`, `resolve_run(id)`, `check_root()` (read-only usability test for `--dry`); the only module that touches the run folder | R22, R23 |
| `contract.py` | `extract_contract(backend, prompt, kind) -> Contract`; `check(contract, candidate) -> list[Violation]` (programmatic part plus one judged call); `literals(prompt)`, `literals_preserved(original, candidate)` | R5, R6, R9 |
| `scenarios.py` | `read_examples(path) -> list[Scenario]`; `synthesize(backend, prompt, contract) -> list[Scenario]`; `split_sizes(n)`; `split(scenarios, seed) -> Split(train, val, holdout)` | R11, R15 |
| `evaluator.py` | `Evaluator(backend, models, contract)` implementing `BatchEvaluator`: task call per scenario, programmatic checks, one judge call per batch, quote rule, `side_info["scores"]` per group and the ASI in other keys | R10, R10a, R10b, R16, R24 |
| `runner.py` | `improve(prompt, plan, backend, store, ...) -> Outcome`; `RunState` with the replay-invariant `SearchMeter`; adapter and reflection wrapper; stopper; reflection prompt per strictness; `iterations_afforded`, `fixed_costs`; lineage notes for the report; gates (contract, length cap, literals); the only importer of `gepa` | R3, R4, R7, R8, R12-R17, R24 |
| `report.py` | `render(outcome, ...)`: word diff, summary lines, JSON, error object; the only writer to stdout | R2 |
| `cli.py` | `main(argv, *, backend=None) -> int` and the subcommand `clean`. Flags: `--dry`, `--force-low-budget`, `--json`, `--file`, `--examples`, `--kind template\|task`, `--budget`, `--strictness`, `--allow-growth`, `--task-model`, `--judge-model`, `--reflect-model`, `--target-model`, `--merge`, `--trust-search`, `--resume <id>` | R1, R2, R4, R14, R22, R23 |
| `commands/improve.md`, `commands/optimize.md` (repo root) | Claude Code slash command text (`/improve`, alias `/optimize`) | R19, R21 |

Dependencies point one way: `cli -> runner -> {evaluator, contract, scenarios} -> backend -> runstore -> types`. `gepa` is imported only in `runner.py`; nothing else touches GEPA types or `oa.log` (not available on the batch path).

Interfaces fixed at the skeleton commit are in `types.py` (read it, not a copy here) and the `cli.main` signature. Signatures of the other modules above are fixed by this table; an owner may add private helpers but changes a public signature only through the lead.

## 3. Work packages

Every file has exactly one owner. The lead owns `pyproject.toml`, `uv.lock`, `justfile` (including the `smoke` recipe added before G6), `.gitignore`, `.python-version`, `CLAUDE.md`, `AGENTS.md`, `docs/**`, `.claude/**`, `legacy/**` (removed at G5), `src/autoimprover/__init__.py`, `src/autoimprover/types.py`, `tests/conftest.py`, `tests/fakes.py`, `tests/test_types.py`, `tests/test_guards.py`, `tests/test_fakes.py`, `tests/test_package.py`.

| WP | Owner (agent) | Exclusive files | Needs | Proof |
|---|---|---|---|---|
| WP1 backend + runstore | `coder` | `src/autoimprover/backend.py`, `runstore.py`, `tests/test_backend.py`, `tests/test_runstore.py` | skeleton, the user's real-call output | R17, R18, R19, R22, R23, R24 |
| WP2 contract | `coder` | `contract.py`, `tests/test_contract.py` | skeleton | R5, R6, R9 |
| WP3 scenarios | `coder` | `scenarios.py`, `tests/test_scenarios.py` | skeleton | R11, R15 |
| WP4 evaluator | `coder` | `evaluator.py`, `tests/test_evaluator.py` | WP1-WP3 | R10, R10a, R10b, R16, R24 |
| WP5 runner | `coder` | `runner.py`, `tests/test_runner.py` | WP1-WP4 | R3, R4, R6, R7, R8, R11-R17, R22, R24 |
| WP6 report + cli | `coder` | `report.py`, `cli.py`, `tests/test_report.py`, `tests/test_cli.py` | WP5 | R1, R2, R4, R14, R22, R23 |
| WP7 docs | `coder` | `README.md`, `commands/improve.md`, `commands/optimize.md`, `tests/test_improve_command.py` | WP6 | R19, R21 |
| WP8 acceptance | `tester` | `tests/acceptance/**` (including its own `conftest.py`) | skeleton only; written from the SPEC and ADR-008 without reading `src/` | the A proofs; every test that asserts a candidate is NOT returned has a twin on `happy_backend` asserting that one IS, so it cannot pass for the wrong reason |

WP1 to WP3 and WP8 can run in parallel (at most 2 writers at once, each in its own worktree). pytest runs with `--import-mode=importlib` and `pythonpath = ["tests"]`, so `tests/test_cli.py` and `tests/acceptance/test_cli.py` coexist and helpers come from `from fakes import ...`.

## 4. Skeleton (committed before any package starts)

`types.py` (all shared types and constants), `cli.py` as a stub with the final `main` signature (WP6 replaces its body), `tests/conftest.py` (the R20 guards), `tests/fakes.py` (`ScriptedBackend`, `by_role`, `failing`, the reply builders `intake_reply`, `synth_reply`, `judge_reply`, `reflection_reply` and the complete `happy_backend` with its `MARKER`), the tests of these, `pyproject.toml` with `[project.scripts] autoimprover = "autoimprover.cli:main"` and the pytest options. The other modules do not exist yet: each owner creates its file from the table in section 2, so no empty stub is shared between packages.

## 5. Requirement map

| Req | Module (WP) | Proof, and an input that makes it fail |
|---|---|---|
| R1 input | `cli.py` (6) | T: 20,001 chars, empty, NUL byte -> exit 2 |
| R2 output, exit codes | `report.py`, `cli.py` (6) | A: each row of section 8 |
| R3 unchanged unless reliable | `runner.py` (5) | A with fake: candidate gain below threshold -> original, exit 0 |
| R4 dry run, low budget | `cli.py` (6), `runner.fixed_costs`, `runner.iterations_afforded` (5) | A: `--dry` makes zero calls, writes nothing and exits 0 even for budget 30 or an unusable state folder; T: budget 66 with 12 scenarios is refused (worst case 2 iterations) |
| R5 contract, kind | `contract.py` (2) | T: kind guessed, `--kind` overrides |
| R6 contract gate | `contract.py` (2), gate in `runner.py` (5) | A: planted violation never returned |
| R7 length cap | `runner.py` (5) | T: 1.26x candidate rejected at conservative |
| R8 strictness | `runner.py` (5) | T: the three reflection templates differ and carry the cap |
| R9 literals | `contract.py` (2) | T: hand-written seeded property test (no new dependency) |
| R10 batched scoring, groups | `evaluator.py` (4) | T: one judge call per batch, `side_info["scores"]` per group |
| R10a execution by kind | `evaluator.py` (4) | T: template vs task call shape |
| R10b quote rule | `evaluator.py` (4) | T: a pass with a quote not in the output scores 0 |
| R11 scenarios, holdout rules | `scenarios.py` (3), `runner.py` (5) | T: n=7 gives no holdout and no paid call without `--trust-search`; with it, a finalist below the seed on the valset is never returned |
| R12 noise, `Call.sample` | `types.py` (skeleton), `runner.py` (5) | T: seed run 2 is a live call; threshold uses `2 x \|diff\|` |
| R13 already strong | `runner.py` (5) | A: seed 0.95 stops with no search |
| R14 judge != task, != target | `types.py` (skeleton), flags in `cli.py` (6) | T: aliases resolved; `--target-model opus` picks the fallback judge |
| R14a target confirmation | `runner.py` (5) | A: winner on search model, loser on target is not returned |
| R15 split, search wiring | `scenarios.py` (3), `runner.py` (5) | T: n = 8, 10, 12, 30, 40 sizes; `GEPAConfig` builds on 0.1.4 |
| R15a small valset, strict improvement | `runner.py` (5) | T: calls counted on `fake` |
| R16 ASI to reflection | `evaluator.py` (4), template in `runner.py` (5) | T: failed checks and excerpts in the reflection prompt |
| R17 budget, reserve, clock, timeout | `backend.py` (1), `runner.py` (5) | T, A: limit, deadline, stop keeps candidates, resume keeps the count |
| R18 command, lockdown, argv rules | `backend.py` (1) | T: argv has no user text; first live call checks lockdown; S |
| R19 untrusted data | `backend.py` (1), `types.py` (skeleton), `commands/improve.md` (7) | T hostile strings; `test_improve_command.py` |
| R20 no real model or network | `tests/conftest.py` (skeleton) | T `test_guards.py` |
| R21 slash command | `commands/*.md` (7) | T text test; S |
| R22 resume | `runstore.py` (1), `runner.py` (5), `cli.py` (6) | A: a run cut at each stage (mid-search, last iteration, final steps) resumes with zero repeated paid calls, the same budget and the same finalists |
| R23 run folder, clean | `runstore.py` (1), `cli.py` (6) | T: 0700, not in git, `clean <id>` rejects a path |
| R24 failures, retries, exit 3 | `backend.py` (1), `evaluator.py` (4), `runner.py` (5) | A with `failing()`: the first failed call outside the search (intake) -> exit 3 with the run folder; T: three consecutive failed calls inside the search -> exit 3; one failed reflection call only skips an iteration |

## 6. Enforcement points

The single place that enforces each limit, and why nothing goes around it:

| Rule | Enforced in | Why it cannot be bypassed |
|---|---|---|
| Call limit (R17) | `BudgetedBackend.complete` | every live call passes it; cache hits sit above it and cost nothing, and a resume starts it from the saved total |
| Search stop, replay-invariant (R17, R22) | `RunState.stopper` over `SearchMeter` (issued calls and seconds, hits counted like live calls) | the stopper reads nothing else, so a replay decides exactly as the original did |
| Reserve (R17) | `BudgetedBackend.limit`, set to `budget - final`, raised only by `runner` after the search | the search has no handle on the limit |
| Wall clock, deadline (R17) | `Clock` (shared) read by `BudgetedBackend` and `ClaudeCliBackend` | one object, monotonic |
| Per-call timeout (R17) | `ClaudeCliBackend._run`: `min(300 s, clock.remaining())` | the only place a process starts |
| Retries, consecutive failures (R24) | `ResilientBackend.complete` | sits directly above `Budgeted`, so every attempt counts |
| Cache key, hits free | `CachedBackend._key` over all `Call` fields | `Call` is frozen; a field test pins the list |
| Judge != task, target (R14) | `Models.__post_init__` after `canonical_model` | no `Models` exists without it |
| argv, stdin, system-prompt size (R18) | `Call.__post_init__` and `ClaudeCliBackend._argv` | user text is only ever put on stdin |
| Lockdown (R18) | `ClaudeCliBackend` on the first live call of the process; `state.raise_if_aborted` re-raises | covers a resumed run whose first live call is a reflection |
| Prompt size, NUL (R1) | `cli.read_prompt` | the only reader of the prompt |
| Length cap (R7) | `runner.within_length` | called by the gate that every answer passes |
| Contract and literals (R6, R9) | `contract.check`, called only from `runner.gate` | the answer is built only from `gate` output |
| Holdout hidden (R15) | `scenarios.split` returns a `Split`; only `runner.score_holdout` reads `.holdout` | GEPA gets `train` and `val` lists only |
| Run folder writes, atomicity (R22, R23) | `runstore._atomic_write` | no other module opens run-folder files |
| One live run per folder (R22) | `RunStore.open_or_create`: `fcntl.flock(LOCK_EX \| LOCK_NB)` on `run.lock`, held until exit; `clean` skips a locked folder | the kernel drops the lock when the process dies, so no stale lock and no pid check (pids are namespace-local in sandboxed shells) |
| Run id, no path (R23) | `runstore.resolve_run` (pattern `RUN_ID_PATTERN` and `is_relative_to`) | `--resume` and `clean` call it |
| State folder writable, not in git (R23) | `runstore.open_runs_root` | called before any paid call |
| stdout only the result (R2) | `report.emit`; `cli.main` redirects GEPA output into `gepa.log` | one writer |
| No real model, network, process, home (R20) | `tests/conftest.py` `_ruv_guards`, shadowing checked at collection | tripwire for tests written in good faith |

## 7. Formats

Run folder files, owner `runstore.py`, schema version 1, atomic writes, how partial or foreign content is read, clocks and concurrency: ADR-007. Summary: JSON with `schema_version`; written to a temporary name in the same folder and moved with `os.replace`; a corrupt cache entry is a miss, a corrupt manifest refuses `--resume` (exit 2); times are UTC for display, durations monotonic; one run folder has one writer (a second `--resume` of the same id while one is live is refused through an `flock` on `run.lock`). `checkpoint.json` holds the stage and `search_start` (the `Budgeted.used` and clock reading when the search first began), which `--resume` reads; cache entries hold the reply, the key fields and `duration_s`. The tool reads user files only through `read_examples` (JSONL: `input`, optional `expected`, optional `criteria`; one bad line exits 2 naming the line).

## 8. User-visible states

| State | Exit | stdout | stderr |
|---|---|---|---|
| improved (holdout-verified) | 0 | the improved prompt (`--json`: one object) | report; a notice if the search was cut short by budget or clock |
| unchanged: no reliable improvement, already strong, no holdout, no candidate beat the seed (`--trust-search`), ran out of budget or time before confirming | 0 | the original prompt (`--json`: one object) | report with the reason |
| `--trust-search` result (only a finalist that beats the seed on the valset) | 0 | the prompt | report and a loud notice "not verified on a holdout" (`verified: false`) |
| `--dry` (also when a real run would refuse, for any reason including an unusable state folder) | 0 | the plan (`--json`: one object) | nothing |
| internal error (a bug) | 1 | nothing (`--json`: error object) | `error: internal error: <type>` and `run folder: <path>` |
| bad input or usage; judge equals task or target; fixed costs above the budget; fewer than 4 iterations without `--force-low-budget`; state folder not writable or inside a git repository; newer or corrupt run folder; run id not an id | 2 | nothing (`--json`: error object) | `error: <what and the flag or folder to change>` |
| backend failure (R24): three consecutive failed calls, or any failed call outside the search (intake, synthesis, seed runs, finalists, contract checks) | 3 | nothing (`--json`: error object) | `error: ...`, `run folder: <path>`, `resume with: [XDG_STATE_HOME=<dir> ]autoimprover --resume <id>` (the prefix appears when the state folder is not the default) |
| session not locked down (R18) | 4 | nothing (`--json`: error object) | `error: ...` |
| interrupted (Ctrl-C) | 130 | nothing (`--json`: error object) | `error: interrupted`, `run folder: <path>`, `resume with: ...` |
| `clean` | 0 (also when nothing to remove), 2 on bad usage | nothing (`--json`: `{"status":"cleaned","removed":N}`) | what was removed |

`--json` always writes exactly one object to stdout: success has `status`, `prompt`, `verified`, `stop`, scores, `noise`, `margin`, `length_ratio`, `calls_used`, `run_dir`; errors have `{"status":"error","code":N,"error":"...","run_dir":"..."}`. Notices stay on stderr in both modes. No row reads as success without being one: only rows with exit 0 print a prompt on stdout, and an unverified or cut-short result says so on stderr and in `--json`.

## 9. External calls

One kind of child process, built in one place (`ClaudeCliBackend._argv`):

`claude -p --safe-mode --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 --model <full id> --output-format json [--system-prompt=<text>] [--json-schema <schema>]`

- stdin: the user text of the call (prompt, scenario, outputs). A system prompt, when the call has one, is the one argument `--system-prompt=<text>`: at most `SYSTEM_PROMPT_MAX_BYTES`, no NUL. Template candidates therefore appear in `/proc/<pid>/cmdline` for the length of a call (readable by local users); the run folder is 0700 but this exposure is accepted, and a file option is used instead if the real-call check shows the CLI has one.
- environment: scrubbed allowlist (`PATH`, `HOME`, `LANG`, `LC_*`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TMPDIR`, and when set the network settings the sandboxed shell of `/improve` needs: `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, `NO_PROXY` in both cases, `SSL_CERT_FILE`), no `ANTHROPIC_*`, no tokens; the exact list waits for the user's real-call output.
- working directory: an empty folder `<run>/cwd` (so no project files are read).
- timeout: `min(CALL_TIMEOUT_S, clock.remaining())`, where `remaining()` is measured to the current deadline (the search deadline until `raise_limit`, the full wall clock after); the process is killed by its exact pid; reply cap 1 MiB, larger is a `CallError`.
- retries: `ResilientBackend`, 2, each counted.
- how `/improve` starts the tool (R21): `commands/improve.md` tells Claude to write the prompt to a 0600 file under `$TMPDIR`, run `autoimprover --file <path>` in a **background** Bash call with an explicit timeout above the wall clock (the foreground limit is 10 minutes, the default background limit 30; a run can take 45), read the output when it ends, and delete the prompt file. The run folder and the `XDG_STATE_HOME` it used are printed, so `--resume` and `clean` can be run later with the same variable.

No other child process, no network call, no `shell=True`.

## 10. Left out on purpose

DSPy and any second optimiser (ADR-001); GEPA merge by default (R15); the local model as task model (first follow-up); `--facts-from-examples` (ADR-006); tool use in candidate runs (single turn, no tools, R18); parallel calls; a hosted service; automatic expiry of run folders; a file option for the system prompt until the real-call check shows one exists; regular-expression checks (R19); a pinned property-testing library (R9 uses a hand-written seeded test; pyright runs from the host and is not pinned by `uv.lock`).

## 11. Open questions (provisional choices)

- Does `--safe-mode` use the subscription login, and does the JSON reply report plugins, MCP servers and tools (R18 self-check)? Provisional: check on the first live call from the reply envelope; wait for the user's real-call output before WP1.
- Judge fallback when the target is the default judge: Sonnet 5.5 (R14).
- The background-run mechanics of `/improve` (timeout, output retrieval, sandbox network) are checked only by `just smoke`; provisional: as in section 9.
- Model aliases are pinned in `types.py` (`MODEL_ALIASES`); the claude CLI may resolve `opus` to a newer id later. Provisional: the user passes full ids when they care.

## ADRs (all Proposed until the G3 vote; index: `docs/adr/`)

ADR-001 gepa only, no DSPy in v1. ADR-002 judge sees outputs, not candidates. ADR-003 three-way split and holdout rule. ADR-004 one backend seam, budget, stop and failure handling. ADR-005 one-off tasks are tested on synthesised situations, templates on inputs. ADR-006 our own reflection prompt. ADR-007 run folder format, atomic writes, clocks, lock and resume. ADR-008 wire formats of every model message (intake, synthesis, task, judge, contract check, reflection).
