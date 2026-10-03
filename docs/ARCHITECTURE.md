# ARCHITECTURE: autoimprover 0.2

Status: Draft 2 (2026-10-03), revised after G3 round 1 (rejected 0-6, see `docs/BUILD-LOG.md`). Implements `docs/SPEC.md`. Needs the G3 vote.

## Flow (pseudocode)

```
improve(prompt, opts):
  plan = Plan(models, budget, strictness)                          # R14: judge != task, judge != target
  pre, final = pre_search_costs(plan), final_costs(plan)           # R17: intake+synthesis+2 seed runs | up to 3 finalists + 3 contract checks
  search_calls = budget - pre - final                              # R17: computed BEFORE any call
  if search_calls < 0: exit 2                                      # R17: the fixed costs alone exceed the budget
  if iterations_afforded(search_calls) < 4 and not force_low_budget: exit 2   # R4
  if --dry: print plan; exit 0                                     # R4: zero calls
  store = RunStore.open_or_create(plan, prompt)                    # R23, R22; exit 2 if not writable (ADR-007)
  budgeted = Budgeted(raw, limit=budget - final, clock=Monotonic(store.elapsed_s))   # pre-search costs are paid from this limit as they happen
  backend  = Cached(Resilient(budgeted))                           # ADR-004: hits are free, every attempt counts
  contract = extract_contract(backend, prompt)                     # R5; the first call checks the lockdown (exit 4, R18)
  scenarios = load(examples) or synthesize(backend, prompt, contract)   # R11: inputs (template) or situations (task)
  train, val, holdout = split(scenarios, seed)                     # R15: sizes by formula
  if no holdout and not --trust-search: return Unchanged("no holdout")    # R11
  # seed scores on the TARGET model, two independent runs: Call.sample 0 and 1 (R12, R14a)
  base = [score_set(prompt, holdout, target, sample=i) for i in (0, 1)]   # a failed run is retried; still failing -> exit 3, never a 0 (R24)
  noise = abs(base[0] - base[1]); threshold = max(0.05, 2 * noise)        # R12
  if mean(base) >= 0.95: return Unchanged("already strong")        # R13
  state = RunState()
  result = gepa.optimize_anything(seed=prompt, batch_evaluator=adapter(Evaluator, state), dataset=train, valset=val,
             config=GEPAConfig(parallel=False, run_dir=None, tracking=RunLogger(store), stop_callbacks=[state.stopper(...)], ...))
  if state.abort == "backend": exit 3, run folder kept            # R24
  budgeted.raise_limit(budget)                                     # the final steps' share is spendable now
  finalists = top 3 of state.completed by val score that pass contract, length cap and literal checks   # R6, R7, R9
  for cand in finalists (best first):
      if state.out_of_time(): break                                # never an unvetted answer (R17)
      if holdout_score(cand, target) > mean(base) + threshold: return Improved(cand, verified=True, stop=state.cause)   # R3
  return Unchanged("no reliable improvement", stop=state.cause)    # R3
  # --trust-search without a holdout: best finalist that passes the gates, verified=False (R11)
```

Candidate execution: `template` -> system=candidate, user=input; `task` -> user=situation+candidate (R10a). The judge returns `{check: pass, quote}`; a pass whose quote is not in the output is a fail (R10b).

## Layers of the backend (ADR-004)

`Cached(Resilient(Budgeted(ClaudeCli | Scripted)))`, outermost first.

- **ClaudeCli**: the only builder of the `claude -p` command (R18), scrubbed environment, per-call timeout `min(300 s, clock left)`. User text (prompt, scenario, output) goes on stdin only. The system prompt is one argument `--system-prompt=<text>` (cannot be read as an option) and at most `SYSTEM_PROMPT_MAX_BYTES`; a longer candidate fails the length gate before it is run. The first call of a run checks the lockdown.
- **Budgeted**: counts every call that reaches it, holds the current `limit` (`budget - final_costs` from the start, so intake, synthesis, seed runs and search share what is left of it; the full budget after `raise_limit`), keeps the monotonic clock (elapsed seconds carried over from the run folder), raises `BudgetExhausted`. The single enforcement point of R17.
- **Resilient**: retries a failed call twice (each attempt passes through `Budgeted`, so it counts), and raises `BackendError` after three consecutive calls that failed all their retries (R24).
- **Cached**: key = SHA-256 of all `Call` fields including `sample`; stores successful replies only, one atomic file per call (ADR-007). A hit costs nothing and does not touch the budget.

## Stop and failure inside GEPA (ADR-004)

GEPA never sees a backend exception. The adapter around the evaluator and the reflection callable catches `BudgetExhausted` and `BackendError`, records the cause in `RunState` (`abort` = budget | clock | backend), and returns neutral results. `RunState.completed` holds only candidates that finished a full evaluation, so a half-scored candidate can never be returned. The stopper handed to GEPA's `stop_callbacks` returns True when `abort` is set, when fewer calls remain in the search share than one iteration costs, or when 75 % of the clock is used (`SEARCH_CLOCK_SHARE`). `raise_on_exception` stays True so real bugs still surface. GEPA's own progress lines go to `<run>/gepa.log` through a custom logger, and the runner redirects stdout into the same file while GEPA runs, so stdout carries only the result (R2). `run_dir` is None: GEPA keeps no pickled state; resume (R22) reruns the flow and the cache replays every paid call, so GEPA retraces its steps for free (same seed, same replies).

## Modules (`src/autoimprover/`)

| Module | Responsibility | Spec |
|---|---|---|
| `types.py` | dataclasses, Protocols, constants shared by all: `Backend`, `BatchEvaluator`, `Call`, `Contract`, `Scenario`, `Models`, `Plan`, `Outcome`, exit codes | all |
| `backend.py` | `ClaudeCliBackend`, `BudgetedBackend`, `ResilientBackend`, `CachedBackend`, `FakeBackend` (the shipped fake: scripted replies; tests inject a backend through `cli.main(argv, backend=...)` and use `tests/fakes.py`) | R17-R20, R24 |
| `runstore.py` | the run folder: manifest, contract, scenarios, call log, cache, checkpoint; atomic writes; resume; `clean`; writability check (ADR-007) | R22, R23 |
| `contract.py` | extract contract; `check(contract, candidate)` programmatic part plus judged part; literal preservation (placeholders, code, URLs, paths, quotes) | R5, R6, R9 |
| `scenarios.py` | read JSONL; synthesize; three-way split with a fixed seed | R11, R15 |
| `evaluator.py` | run the task model on a batch of scenarios; programmatic checks; batched checklist judge (one call per candidate per batch); returns `(score, side_info)` per scenario with `side_info["scores"]` per check group and the ASI in other keys | R10, R10a, R10b, R16 |
| `runner.py` | GEPA wiring (the only importer of `gepa`: `batch_evaluator`, `hybrid` frontier, merge off, small valset), `RunState` and stopper, strictness templates, length cap, noise, holdout decision on the target model, iteration estimate for `--dry`, early stop | R3, R4, R7, R8, R12-R15a, R17, R24 |
| `report.py` | word diff, summary, JSON | R2 |
| `cli.py` | `main(argv, *, backend=None) -> int` (fixed signature; a stub is in the skeleton); subcommand `clean`. Flags: `--dry`, `--force-low-budget`, `--json`, `--examples`, `--kind template\|task`, `--budget`, `--strictness`, `--allow-growth`, `--task-model`, `--judge-model`, `--reflect-model`, `--target-model`, `--merge`, `--trust-search`, `--resume`, `--file` | R1, R2, R4, R14, R22, R23 |
| `commands/improve.md` (repo root) | Claude Code slash command text (`/improve`, `/optimize`) | R19, R21 |

Dependencies point one way: `cli -> runner -> {evaluator, contract, scenarios} -> backend -> runstore -> types`. `gepa` is imported only in `runner.py`; nothing else touches GEPA types or `oa.log` (it is not available on the batch path).

## Interfaces (fixed at the skeleton commit; the source of truth is `types.py`)

```python
class Backend(Protocol):
    def complete(self, call: Call) -> Reply: ...
@dataclass(frozen=True) class Call: role: Role; model: str; user: str; system: str = ""; json_schema: str | None = None; sample: int = 0
@dataclass(frozen=True) class Reply: text: str; cached: bool = False; tokens_in: int = 0; tokens_out: int = 0
@dataclass(frozen=True) class Contract: goal: str; kind: Kind; keep: tuple[str,...]; constraints: tuple[str,...]; output_format: str; language: str; tone: str; checks: tuple[Check,...]
@dataclass(frozen=True) class Scenario: id: str; input: str; expected: str | None; criteria: tuple[str,...]
class BatchEvaluator(Protocol):
    def __call__(self, candidate: str, scenarios: Sequence[Scenario]) -> list[tuple[float, dict[str, Any]]]: ...
def main(argv: Sequence[str] | None = None, *, backend: Backend | None = None) -> int: ...   # cli.py
```

## Outcomes, streams and exit codes (R2)

| State | Exit | stdout | stderr |
|---|---|---|---|
| improved (holdout-verified) | 0 | the improved prompt (`--json`: one object) | report, notice if the search was cut short by budget or clock |
| unchanged: no reliable improvement, already strong, no holdout, ran out of time before confirming | 0 | the original prompt | report with the reason |
| `--trust-search` result | 0 | the prompt | report, loud notice "not verified on a holdout" (`verified: false` in `--json`) |
| `--dry` | 0 | the plan | nothing |
| bad input or usage; judge equals task or target; reserve above budget or fewer than 4 iterations without `--force-low-budget`; state folder not writable; run folder from a newer version | 2 | nothing (`--json`: error object) | `error: <what and the flag or folder to change>` |
| backend failure (R24) | 3 | nothing (`--json`: error object) | `error: ...` and `run folder: <path>` (resume with `--resume`) |
| session not locked down (R18) | 4 | nothing (`--json`: error object) | `error: ...` |
| interrupted (Ctrl-C) | 130 | nothing | `run folder: <path>` |
| `clean` | 0 (also when nothing to remove), 2 on bad usage | nothing | what was removed |

`--json` always writes exactly one object to stdout: success has `status`, `prompt`, `verified`, `stop`, scores, `noise`, `margin`, `length_ratio`, `calls_used`, `run_dir`; errors have `{"status":"error","code":N,"error":"...","run_dir":"..."}`. Notices stay on stderr in both modes.

## Requirement map

| Req | Where (module, work package) | Proof |
|---|---|---|
| R1 input | `cli.py` WP6 | T `test_cli.py` |
| R2 output, exit codes | `report.py`, `cli.py` WP6 | A |
| R3 unchanged unless reliable | `runner.py` WP5 | A with fake |
| R4 dry run, low budget | `cli.py` WP6, `iterations_afforded` in `runner.py` WP5 | A |
| R5 contract, kind | `contract.py` WP2 | T |
| R6 contract gate | `contract.py` WP2, gate in `runner.py` WP5 | A with planted violations |
| R7 length cap | `runner.py` WP5 | T |
| R8 strictness | `runner.py` WP5 | T on templates, S |
| R9 literals | `contract.py` WP2 | T property |
| R10 batched scoring, groups | `evaluator.py` WP4 | T with fake |
| R10a execution by kind | `evaluator.py` WP4 | T |
| R10b quote rule | `evaluator.py` WP4 | T hostile outputs |
| R11 scenarios, holdout rules | `scenarios.py` WP3, `runner.py` WP5 | T |
| R12 noise, `Call.sample` | `types.py` WP0, `runner.py` WP5 | T (`test_types.py`, `test_runner.py`) |
| R13 already strong | `runner.py` WP5 | A |
| R14 judge != task, != target | `types.py` WP0, flags in `cli.py` WP6 | T (`test_types.py`, `test_cli.py`) |
| R14a target confirmation | `runner.py` WP5 | A with fake |
| R15 split, search wiring | `scenarios.py` WP3, `runner.py` WP5 | T, A |
| R15a small valset, strict improvement | `runner.py` WP5 | T counts calls |
| R16 ASI to reflection | `evaluator.py` WP4, template in `runner.py` WP5 | T |
| R17 budget, reserve, clock, timeout | `backend.py` WP1 (`Budgeted`), `runner.py` WP5 (reserve, stopper) | T, A |
| R18 command, lockdown, argv rules | `backend.py` WP1 | T, S |
| R19 untrusted data | `backend.py` WP1 (stdin, argument rules), `types.py` WP0 (no regex rule), `commands/improve.md` WP7 (prompt by file) | T hostile strings (`test_backend.py`, `test_improve_command.py`), A |
| R20 no real model or network | `tests/conftest.py` WP0 | T `test_guards.py` |
| R21 slash command | `commands/improve.md` WP7 | T text test, S |
| R22 resume | `runstore.py` WP1, `runner.py` WP5, `--resume` in `cli.py` WP6 | A |
| R23 run folder, clean | `runstore.py` WP1, `clean` in `cli.py` WP6 | T, A |
| R24 failures, retries, exit 3 | `backend.py` WP1 (`Resilient`), `evaluator.py` WP4 (unknown checks), `runner.py` WP5 (exit 3 path) | A with failing fake |

## Work packages

Every file has exactly one owner. The lead owns `pyproject.toml`, `uv.lock`, `justfile`, `.gitignore`, `CLAUDE.md`, `docs/**`, `.claude/**`.

| WP | Owner files | Needs | Proof |
|---|---|---|---|
| WP0 skeleton (lead, done) | `src/autoimprover/__init__.py`, `types.py`, `cli.py` (stub only), `tests/conftest.py`, `tests/fakes.py`, `tests/test_types.py`, `tests/test_guards.py`, `tests/test_fakes.py`, `tests/test_package.py` | none | `just check` green |
| WP1 backend + runstore | `backend.py`, `runstore.py`, `tests/test_backend.py`, `tests/test_runstore.py` | WP0, the user's real-call output | R17, R18, R19, R22, R23, R24 |
| WP2 contract | `contract.py`, `tests/test_contract.py` | WP0 | R5, R6, R9 |
| WP3 scenarios | `scenarios.py`, `tests/test_scenarios.py` | WP0 | R11, R15 |
| WP4 evaluator | `evaluator.py`, `tests/test_evaluator.py` | WP1-WP3 interfaces | R10, R10a, R10b, R16, R24 |
| WP5 runner | `runner.py`, `tests/test_runner.py` | WP1-WP4 | R3, R4, R7, R8, R11-R15a, R17, R24 |
| WP6 report + cli | `report.py`, `cli.py` (takes over the stub), `tests/test_report.py`, `tests/test_cli.py` | WP5 | R1, R2, R4, R14, R22, R23 |
| WP7 docs | `README.md`, `commands/improve.md`, `tests/test_improve_command.py` | WP6 | R19, R21 |
| WP8 acceptance (`tester`) | `tests/acceptance/**` (including `tests/acceptance/conftest.py`) | WP0 only, written from the SPEC without reading `src/` | the A proofs |

WP1 to WP3 and WP8 can run in parallel (at most 2 writers at once, each in its own worktree). pytest runs with `--import-mode=importlib` and `pythonpath = ["tests"]`, so `tests/test_cli.py` and `tests/acceptance/test_cli.py` coexist and helpers come from `from fakes import ...`.

## Risks and answers

- **Judge reward-hacking**: the judge never sees the candidate text, only output plus checklist (R10); holdout is hidden from the search (R15); the judge is never the task or the target model (R14).
- **Overfit to synthetic scenarios**: holdout plus noise threshold (R3, R12); `--examples` lets the user supply real ones.
- **Cost and runaway**: one budget across all roles, a single enforcement point, reserve computed first, cache hits free, per-call timeout (R17, R24).
- **Intent drift**: contract check and literal check gate the answer; strictness caps edits (R6-R9); an answer never skips them, even after a budget or clock stop.
- **GEPA API drift** (0.1.4 pinned): single import seam in `runner.py`, one contract test that imports and builds a `GEPAConfig` with `batch_evaluator`, a stopper and a logger.
- **Prompt injection through the user's prompt or model output**: both are data, never code, shell or paths (R19); `/improve` passes the prompt by file.

## ADRs (all Proposed until the G3 vote; index: `docs/adr/`)

ADR-001 gepa only, no DSPy in v1. ADR-002 judge sees outputs, not candidates. ADR-003 three-way split and holdout rule. ADR-004 one backend seam, budget, stop and failure handling. ADR-005 one-off tasks are tested on synthesised situations, templates on inputs. ADR-006 our own reflection prompt. ADR-007 run folder format, atomic writes and clocks.

## Open points for the G3 vote

- `--facts-from-examples` was removed from v1 by the lead (ADR-006: a follow-up that needs a SPEC amendment). The user can reverse this.
- SPEC amendments made after round 1 (R2, R12, R14, R14a, R17, R18, R19, R21, R23, R24, Q4) are listed at the top of `docs/SPEC.md`; only the R14 judge change was put to the user and answered.
