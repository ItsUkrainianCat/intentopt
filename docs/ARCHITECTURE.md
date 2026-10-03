# ARCHITECTURE: autoimprover 0.2

Status: Draft 1 (2026-10-03). Implements `docs/SPEC.md` (approved 2026-10-03). Needs the G3 vote.

## Flow (pseudocode)

```
improve(prompt, opts):
  plan = Plan(models, budget, strictness)                       # R4: --dry stops here
  backend = Budgeted(Cached(ClaudeCli | Fake), budget)          # R17, R18: every call counted
  contract = extract_contract(backend, prompt)                  # R5: also kind = template | task
  reserve = fixed_costs(plan); search_budget = budget - reserve # R17: refuse if reserve > budget
  scenarios = load(examples) or synthesize(backend, prompt, contract)   # R11: inputs (template) or situations (task)
  train, val, holdout = split(scenarios, seed)                  # R15: sizes by formula
  # candidate execution: template -> system=candidate, user=input; task -> user=situation+candidate (R10a)
  # judge returns {check: pass, quote}; a pass whose quote is not in the output is a fail (R10b)
  base = score_set(prompt, holdout) twice -> noise              # R12
  if mean(base) >= 0.95: return Unchanged("already strong")     # R13
  result = gepa.optimize_anything(seed=prompt, dataset=train, valset=val,
             evaluator=Evaluator(task, judge, contract), config=GEPAConfig(parallel=False, ...))
  for cand in pareto_candidates(result) ordered by val score:   # best first
      if not passes(contract, cand) or too_long(cand) or literals_changed(cand): continue
      if holdout(cand) > holdout(prompt) + max(0.05, noise): return Improved(cand)
  return Unchanged("no reliable improvement")                   # R3
```

## Modules (`src/autoimprover/`)

| Module | Responsibility | Spec |
|---|---|---|
| `types.py` | dataclasses and Protocols shared by all: `Backend`, `Call`, `Contract`, `Scenario`, `Plan`, `Outcome` | all |
| `backend.py` | `ClaudeCliBackend` (the one command builder, scrubbed env, session self-check), `FakeBackend` (scripted replies), `CachedBackend` (disk), `BudgetedBackend` (count, wall clock, `BudgetExhausted`) | R17-R20 |
| `runstore.py` | run folder: contract, scenarios, call log, cache, checkpoints; resume | R5, R17, R22 |
| `contract.py` | extract contract; `check(contract, candidate)` programmatic part plus judged part; literal-preservation (placeholders, code, URLs, paths, quotes) | R5, R6, R9 |
| `scenarios.py` | read JSONL; synthesize; split 3 ways with fixed seed | R11, R15 |
| `evaluator.py` | run task model on a scenario; programmatic checks; batched checklist judge (one call per candidate per batch); returns `(score, side_info)` with `side_info["scores"]` per check group; `oa.log` for ASI | R10, R14, R16 |
| `runner.py` | GEPA wiring (`hybrid` frontier, merge off, small valset), strictness templates, length cap, noise, holdout decision on the target model, iteration estimate for `--dry`, early stop | R3, R4, R7, R8, R12, R13, R14a, R15, R15a |
| `report.py` | word diff, summary, JSON | R2 |
| `cli.py` | arguments, `--dry`, exit codes, `--examples`, `--budget`, `--strictness`, `--allow-growth`, `--trust-search`, `--resume` | R1, R4, R22 |
| `commands/improve.md` | Claude Code slash command text (`/improve`, `/optimize`) | R21 |

Dependencies point one way: `cli -> runner -> {evaluator, contract, scenarios} -> backend -> runstore -> types`.
`gepa` is imported only in `runner.py` (one seam; everything else is testable without it).

## Interfaces (fixed at the skeleton commit)

```python
class Backend(Protocol):
    def complete(self, call: Call) -> Reply: ...        # Call: role, model, system, user, schema?
@dataclass(frozen=True) class Call: role: Literal["intake","synth","task","judge","reflect"]; model: str; system: str; user: str
@dataclass(frozen=True) class Reply: text: str; cached: bool; tokens_in: int; tokens_out: int
@dataclass(frozen=True) class Contract: goal: str; keep: tuple[str,...]; constraints: tuple[str,...]; output_format: str; language: str; tone: str; checks: tuple[Check,...]
@dataclass(frozen=True) class Scenario: id: str; input: str; expected: str|None; criteria: tuple[str,...]
def evaluate(candidate: str, s: Scenario) -> tuple[float, dict]: ...
```

## Work packages

| WP | Owner files | Needs | Proof |
|---|---|---|---|
| WP0 skeleton (lead) | `types.py`, empty modules, `tests/conftest.py` | none | `just check` green |
| WP1 backend | `backend.py`, `runstore.py`, `tests/test_backend.py`, `tests/test_runstore.py` | WP0 | R17, R18, R20 |
| WP2 contract | `contract.py`, `tests/test_contract.py` | WP0 | R5, R6, R9 |
| WP3 scenarios | `scenarios.py`, `tests/test_scenarios.py` | WP0 | R11, R15 |
| WP4 evaluator | `evaluator.py`, `tests/test_evaluator.py` | WP1-WP3 interfaces | R10, R14, R16 |
| WP5 runner | `runner.py`, `tests/test_runner.py`, `tests/acceptance/test_run.py` | WP1-WP4 | R3, R7, R8, R12, R13, R15 |
| WP6 report+cli | `report.py`, `cli.py`, `tests/test_cli.py`, `tests/acceptance/test_cli.py` | WP5 | R1, R2, R4, R22 |
| WP7 docs | `README.md`, `commands/improve.md` | WP6 | R21 |

WP1 to WP3 can run in parallel (max 2 writers at once, each in its own worktree). The lead owns `pyproject.toml`, `uv.lock`, `justfile`, `docs/**`, `.claude/**`.

## Risks and answers

- **Judge reward-hacking**: the judge never sees the candidate text, only output plus checklist (R10); holdout is hidden from the search (R15).
- **Overfit to synthetic scenarios**: holdout plus noise threshold (R3, R12); `--examples` lets the user supply real ones.
- **Cost**: one budget across all roles, cache, cheap task model (R14, R17).
- **Intent drift**: contract check and literal check gate the answer; strictness caps edits (R6-R9).
- **GEPA API drift** (0.1.4 pinned): single import seam in `runner.py`, one contract test that imports and builds a `GEPAConfig`.
- **Prompt injection through the user's prompt or model output**: both are data, never code or paths (R19).

## ADRs to write at G3
ADR-001 gepa only, no DSPy in v1. ADR-002 judge sees outputs, not candidates. ADR-003 three-way split and holdout rule. ADR-004 one backend seam and budget counting.
