# ARCHITECTURE: autoimprover 0.2

Status: Draft 3 (2026-10-03). Draft 2 answered G3 round 1 (rejected 0-6); this draft also answers two non-voting pre-flight reviews. Implements `docs/SPEC.md`. Needs the G3 vote (round 2). Every fact has one home: numbers in `types.py` and `docs/SPEC.md`, decisions in `docs/adr/`.

## 1. Key flows

### improve (R3, R4, R11-R15, R17, R22, R24)

```
main(argv, backend=None):
  opts = parse(argv)                                              # exit 2 on bad usage (R1, R14: models resolved to full ids)
  if opts.resume: manifest = RunStore.load(id); opts = manifest.opts   # plan and prompt come from the run (R22)
  plan = Plan(models, budget, strictness, ...)
  scenarios = read(--examples) or None                            # free
  n = len(scenarios) if given else SYNTH_COUNT (12)
  h, v = split_sizes(n)                                           # R15: holdout min(6, max(3, half_up(0.35 n))), valset clamp(half_up(0.25 n), 2, 4)
  pre   = (1 + 1[if synthesising]) + 2 * (h + 1)                  # intake, synthesis, two seed runs (h task calls + 1 judge each)
  final = 3 * (h + 1) + 3                                         # up to 3 finalists on the target model + up to 3 contract checks
  iter_cost = 1 + (MINIBATCH_SIZE + 1) + (v + 1)                  # reflect + minibatch + valset pass of an accepted child
  search_calls = budget - pre - final
  iterations = search_calls // iter_cost
  if --dry: print plan, search_calls, iterations, final time share, and "a real run would refuse: <reason>" if so; exit 0   # R4: zero calls
  if search_calls < 0: exit 2                                     # R17: the fixed costs alone exceed the budget
  if iterations < 4 and not --force-low-budget: exit 2            # R4
  if n < 8 and not --trust-search: return Unchanged("no holdout") # R11, decided before any paid call
  store = RunStore.open_or_create(plan, prompt)                   # exit 2: not writable, inside a git repository, newer version (ADR-007)
  clock = Clock(elapsed=store.elapsed_s)                          # monotonic; one object shared by ClaudeCli, Budgeted and the runner
  budgeted = Budgeted(raw, limit=budget - final, used=store.calls_used, clock=clock, deadline=SEARCH_CLOCK_SHARE * wall_clock)
  backend  = Cached(Resilient(budgeted))
  contract  = extract_contract(backend, prompt)                   # R5; kind from --kind or the guess
  scenarios = scenarios or synthesize(backend, prompt, contract)  # R11
  train, val, holdout = split(scenarios, seed)                    # R15
  base = [score_set(prompt, holdout, models.target, sample=i) for i in (0, 1)]   # R12, R14a: both on the target model; judge calls carry the same sample
  #   a CallFailed here is not a score: it becomes BackendError -> exit 3 (R24)
  noise = abs(base[0] - base[1]); threshold = max(0.05, 2 * noise)               # R12
  if mean(base) >= 0.95: return Unchanged("already strong")       # R13
  state = RunState()
  result = gepa.optimize_anything(prompt, batch_evaluator=adapter(evaluator, state), dataset=train, valset=val,
             config=GEPAConfig(engine=EngineConfig(parallel=False, run_dir=None, ...),
                               reflection=ReflectionConfig(reflection_lm=reflection_wrapper(state), ...),
                               tracking=TrackingConfig(logger=RunLogger(store)),
                               stop_callbacks=[state.stopper(budgeted)]))   # stdout redirected into gepa.log meanwhile
  state.raise_if_aborted()                                        # backend -> exit 3, lockdown -> exit 4, bug -> exit 1; run folder kept
  budgeted.raise_limit(budget, deadline=wall_clock)               # the final steps' share is spendable now
  try:
    finalists = top 3 of state.completed minus the seed, by val score, that pass contract, length cap and literals   # R6, R7, R9
    for cand in finalists (best first):
      if holdout_score(cand, models.target) > mean(base) + threshold:
        return Improved(cand, verified=True, stop=state.stop)     # R3
    return Unchanged("no reliable improvement", stop=state.stop)
  except BudgetExhausted:
    return Unchanged("ran out of budget or time before a candidate was confirmed", stop=...)   # never an unvetted answer (R17)
  # --trust-search with n < 8: no seed runs; the best finalist that passes the gates, verified=False (R11)
```

### Execution and judging (R10, R10a, R10b)

`template`: system = candidate, user = scenario input. `task`: user = situation + candidate, no system prompt. The task model's outputs for a batch of at most `JUDGE_BATCH_MAX` scenarios go to one judge call with the checklist; the judge never sees the candidate. It returns `{check: pass, quote}`; a pass whose quote is not in the output counts as a fail, an omitted check is `unknown` and is left out of the score (more than 30 % unknown scores 0, R24).

### Stop and failure inside GEPA (R17, R18, R24; ADR-004)

GEPA swallows every exception raised in the reflection callable (`reflective_mutation.py`) and, with `raise_on_exception=True`, re-raises evaluator exceptions out of `optimize_anything` (probes, G3). So neither wrapper lets a backend exception reach GEPA's own handling:

- **Evaluator adapter** (batch): per pair, `CallFailed` gives a neutral `(0.0, {"incomplete": ...})` and the candidate is not added to `state.completed`; `BudgetExhausted`, `BackendError`, `SessionNotLockedDown` or any other exception is recorded in `state.abort` and every later pair returns neutral without a call.
- **Reflection wrapper**: calls the model with `sample = state.next_reflect_index()` (a running index, so a stalled search cannot replay cached proposals for free, and a resumed run still replays identically); rejects an empty reply or one without a code block (raises `SkipProposal`, GEPA skips the iteration); on any other exception it records the cause in `state.abort` and re-raises, which GEPA swallows; once `state.abort` is set it raises without calling.
- **Stopper** (`stop_callbacks`): True when `state.abort` is set, when `budgeted.limit - budgeted.used < iter_cost`, or when the search deadline has passed.
- After GEPA returns, `state.raise_if_aborted()` turns `backend`, `lockdown` and unexpected causes into exits 3, 4 and 1. A `budget` or `clock` cause is not a failure: the finalists come from `state.completed`.

### Resume (R22)

`--resume <id>` loads the manifest (prompt, plan, flags), reruns the flow above, and the cache replays every paid call. The manifest carries `calls_used` and `elapsed_s`, saved after every paid call, and `Budgeted` and `Clock` start from them, so a resume continues the same budget and the same 45 minutes. GEPA's call order is deterministic for a fixed seed (probed), so replay retraces the same steps.

## 2. Modules (`src/autoimprover/`)

| Module | Responsibility and public interface | Spec |
|---|---|---|
| `types.py` | dataclasses, protocols, constants shared by all: `Backend`, `BatchEvaluator`, `Call`, `Reply`, `Contract`, `Check`, `Scenario`, `Models`, `Plan`, `Outcome`, exceptions (`CallError`, `CallFailed`, `BackendError`, `BudgetExhausted`, `SessionNotLockedDown`), exit codes, `canonical_model`, `default_models` | all |
| `backend.py` | `Clock`; `ClaudeCliBackend(clock)`, `BudgetedBackend(raw, limit, used, clock, deadline)` with `raise_limit`, `ResilientBackend(inner)`, `CachedBackend(inner, store)`; the layers of section 6 | R17-R20, R24 |
| `runstore.py` | `RunStore`: `open_or_create`, `load(id)`, `save_progress(calls_used, elapsed_s)`, `cache_get/put`, `log_call`, `checkpoint`, `clean(id=None)`, `resolve_run(id)`; the only module that touches the run folder | R22, R23 |
| `contract.py` | `extract_contract(backend, prompt, kind) -> Contract`; `check(contract, candidate) -> list[Violation]` (programmatic part plus one judged call); `literals(prompt)`, `literals_preserved(original, candidate)` | R5, R6, R9 |
| `scenarios.py` | `read_examples(path) -> list[Scenario]`; `synthesize(backend, prompt, contract) -> list[Scenario]`; `split_sizes(n)`; `split(scenarios, seed) -> Split(train, val, holdout)` | R11, R15 |
| `evaluator.py` | `Evaluator(backend, models, contract)` implementing `BatchEvaluator`: task call per scenario, programmatic checks, one judge call per batch, quote rule, `side_info["scores"]` per group and the ASI in other keys | R10, R10a, R10b, R16, R24 |
| `runner.py` | `improve(prompt, plan, backend, store, ...) -> Outcome`; `RunState`; adapter and reflection wrapper; stopper; reflection prompt per strictness; `iterations_afforded`; gates (contract, length cap, literals); the only importer of `gepa` | R3, R4, R7, R8, R12-R17, R24 |
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
| WP8 acceptance | `tester` | `tests/acceptance/**` (including its own `conftest.py`) | skeleton only; written from the SPEC without reading `src/` | the A proofs |

WP1 to WP3 and WP8 can run in parallel (at most 2 writers at once, each in its own worktree). pytest runs with `--import-mode=importlib` and `pythonpath = ["tests"]`, so `tests/test_cli.py` and `tests/acceptance/test_cli.py` coexist and helpers come from `from fakes import ...`.

## 4. Skeleton (committed before any package starts)

`types.py` (all shared types and constants), `cli.py` as a stub with the final `main` signature (WP6 replaces its body), `tests/conftest.py` (the R20 guards), `tests/fakes.py` (`ScriptedBackend`, `by_role`, `failing`), the tests of these, `pyproject.toml` with `[project.scripts] autoimprover = "autoimprover.cli:main"` and the pytest options. The other modules do not exist yet: each owner creates its file from the table in section 2, so no empty stub is shared between packages.

## 5. Requirement map

| Req | Module (WP) | Proof, and an input that makes it fail |
|---|---|---|
| R1 input | `cli.py` (6) | T: 20,001 chars, empty, NUL byte -> exit 2 |
| R2 output, exit codes | `report.py`, `cli.py` (6) | A: each row of section 8 |
| R3 unchanged unless reliable | `runner.py` (5) | A with fake: candidate gain below threshold -> original, exit 0 |
| R4 dry run, low budget | `cli.py` (6), `runner.iterations_afforded` (5) | A: `--dry` makes zero calls; budget 30 refuses |
| R5 contract, kind | `contract.py` (2) | T: kind guessed, `--kind` overrides |
| R6 contract gate | `contract.py` (2), gate in `runner.py` (5) | A: planted violation never returned |
| R7 length cap | `runner.py` (5) | T: 1.26x candidate rejected at conservative |
| R8 strictness | `runner.py` (5) | T: the three reflection templates differ and carry the cap |
| R9 literals | `contract.py` (2) | T: hand-written seeded property test (no new dependency) |
| R10 batched scoring, groups | `evaluator.py` (4) | T: one judge call per batch, `side_info["scores"]` per group |
| R10a execution by kind | `evaluator.py` (4) | T: template vs task call shape |
| R10b quote rule | `evaluator.py` (4) | T: a pass with a quote not in the output scores 0 |
| R11 scenarios, holdout rules | `scenarios.py` (3), `runner.py` (5) | T: n=7 gives no holdout and no paid call without `--trust-search` |
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
| R22 resume | `runstore.py` (1), `runner.py` (5), `cli.py` (6) | A: interrupted run resumes with zero repeated paid calls and the same budget |
| R23 run folder, clean | `runstore.py` (1), `cli.py` (6) | T: 0700, not in git, `clean <id>` rejects a path |
| R24 failures, retries, exit 3 | `backend.py` (1), `evaluator.py` (4), `runner.py` (5) | A with `failing()`: three failed calls -> exit 3, folder kept |

## 6. Enforcement points

The single place that enforces each limit, and why nothing goes around it:

| Rule | Enforced in | Why it cannot be bypassed |
|---|---|---|
| Call limit (R17) | `BudgetedBackend.complete` | every live call passes it; cache hits sit above it and cost nothing |
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
| Run id, no path (R23) | `runstore.resolve_run` (pattern `RUN_ID_PATTERN` and `is_relative_to`) | `--resume` and `clean` call it |
| State folder writable, not in git (R23) | `runstore.open_runs_root` | called before any paid call |
| stdout only the result (R2) | `report.emit`; `cli.main` redirects GEPA output into `gepa.log` | one writer |
| No real model, network, process, home (R20) | `tests/conftest.py` `_ruv_guards`, shadowing checked at collection | tripwire for tests written in good faith |

## 7. Formats

Run folder files, owner `runstore.py`, schema version 1, atomic writes, how partial or foreign content is read, clocks and concurrency: ADR-007. Summary: JSON with `schema_version`; written to a temporary name in the same folder and moved with `os.replace`; a corrupt cache entry is a miss, a corrupt manifest refuses `--resume` (exit 2); times are UTC for display, durations monotonic; one run folder has one writer (a second `--resume` of the same id while one is live is refused through a lock file). The tool reads user files only through `read_examples` (JSONL: `input`, optional `expected`, optional `criteria`; one bad line exits 2 naming the line).

## 8. User-visible states

| State | Exit | stdout | stderr |
|---|---|---|---|
| improved (holdout-verified) | 0 | the improved prompt (`--json`: one object) | report; a notice if the search was cut short by budget or clock |
| unchanged: no reliable improvement, already strong, no holdout, ran out of budget or time before confirming | 0 | the original prompt (`--json`: one object) | report with the reason |
| `--trust-search` result | 0 | the prompt | report and a loud notice "not verified on a holdout" (`verified: false`) |
| `--dry` (also when a real run would refuse) | 0 | the plan (`--json`: one object) | nothing |
| internal error (a bug) | 1 | nothing (`--json`: error object) | `error: internal error: <type>` and `run folder: <path>` |
| bad input or usage; judge equals task or target; fixed costs above the budget; fewer than 4 iterations without `--force-low-budget`; state folder not writable or inside a git repository; newer or corrupt run folder; run id not an id | 2 | nothing (`--json`: error object) | `error: <what and the flag or folder to change>` |
| backend failure (R24) | 3 | nothing (`--json`: error object) | `error: ...`, `run folder: <path>`, `resume with: autoimprover --resume <id>` |
| session not locked down (R18) | 4 | nothing (`--json`: error object) | `error: ...` |
| interrupted (Ctrl-C) | 130 | nothing (`--json`: error object) | `error: interrupted`, `run folder: <path>`, `resume with: ...` |
| `clean` | 0 (also when nothing to remove), 2 on bad usage | nothing (`--json`: `{"status":"cleaned","removed":N}`) | what was removed |

`--json` always writes exactly one object to stdout: success has `status`, `prompt`, `verified`, `stop`, scores, `noise`, `margin`, `length_ratio`, `calls_used`, `run_dir`; errors have `{"status":"error","code":N,"error":"...","run_dir":"..."}`. Notices stay on stderr in both modes. No row reads as success without being one: only rows with exit 0 print a prompt on stdout, and an unverified or cut-short result says so on stderr and in `--json`.

## 9. External calls

One kind of child process, built in one place (`ClaudeCliBackend._argv`):

`claude -p --safe-mode --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 --model <full id> --output-format json [--system-prompt=<text>] [--json-schema <schema>]`

- stdin: the user text of the call (prompt, scenario, outputs). A system prompt, when the call has one, is the one argument `--system-prompt=<text>`: at most `SYSTEM_PROMPT_MAX_BYTES`, no NUL. Template candidates therefore appear in `/proc/<pid>/cmdline` for the length of a call (readable by local users); the run folder is 0700 but this exposure is accepted, and a file option is used instead if the real-call check shows the CLI has one.
- environment: scrubbed allowlist (`PATH`, `HOME`, `LANG`, `LC_*`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TMPDIR`), no `ANTHROPIC_*`, no tokens; the exact list waits for the user's real-call output.
- working directory: an empty folder `<run>/cwd` (so no project files are read).
- timeout: `min(CALL_TIMEOUT_S, clock.remaining())`; the process is killed by its exact pid; reply cap 1 MiB, larger is a `CallError`.
- retries: `ResilientBackend`, 2, each counted.

No other child process, no network call, no `shell=True`.

## 10. Left out on purpose

DSPy and any second optimiser (ADR-001); GEPA merge by default (R15); the local model as task model (first follow-up); `--facts-from-examples` (ADR-006); tool use in candidate runs (single turn, no tools, R18); parallel calls; a hosted service; automatic expiry of run folders; a file option for the system prompt until the real-call check shows one exists; regular-expression checks (R19); a pinned property-testing library (R9 uses a hand-written seeded test; pyright runs from the host and is not pinned by `uv.lock`).

## 11. Open questions (provisional choices)

- Does `--safe-mode` use the subscription login, and does the JSON reply report plugins, MCP servers and tools (R18 self-check)? Provisional: check on the first live call from the reply envelope; wait for the user's real-call output before WP1.
- Judge fallback when the target is the default judge: Sonnet 5.5 (R14).
- Model aliases are pinned in `types.py` (`MODEL_ALIASES`); the claude CLI may resolve `opus` to a newer id later. Provisional: the user passes full ids when they care.

## ADRs (all Proposed until the G3 vote; index: `docs/adr/`)

ADR-001 gepa only, no DSPy in v1. ADR-002 judge sees outputs, not candidates. ADR-003 three-way split and holdout rule. ADR-004 one backend seam, budget, stop and failure handling. ADR-005 one-off tasks are tested on synthesised situations, templates on inputs. ADR-006 our own reflection prompt. ADR-007 run folder format, atomic writes, clocks and resume.
