# SPEC: autoimprover 0.2

Status: Approved 2026-10-03 (user: "set the loop!" in reply to the Draft 1 request, with the defaults in section 6 accepted unless changed later).

## 1. Purpose

Take one prompt the user wrote, return one more effective prompt that keeps what the user meant.
Search method: GEPA (Agrawal et al., arXiv:2507.19457), reflective prompt evolution with a Pareto
frontier, run through `gepa==0.1.4` `optimize_anything`. Entry points: the `autoimprover` command and
the Claude Code slash command `/improve` (alias `/optimize`).

"Effective" means the prompt's *outputs* score higher on a fixed check set. Wording quality alone
is never the score (that was the 0.1.0 defect).

## 2. Requirements

Each has a proof: a test (T), an acceptance test (A, black-box, written from this spec) or the
user's `just smoke` (S).

### Input and output
- R1. Input is a prompt from an argument, a file or stdin (UTF-8, 1 to 20,000 chars; otherwise exit 2 with a message). Proof: T.
- R2. Output is exactly one improved prompt plus a report: scores before/after on the holdout, calls used, a word-level diff, and 3 to 6 lines saying what changed and why. Without `--json` the improved prompt alone goes to stdout (so it can be piped) and the report to stderr; `--json` writes one JSON object with both to stdout. Exit codes: 0 done (including "no reliable improvement"), 2 bad input or usage, 3 backend failure (R24), 4 session not locked down (R18). Proof: A.
- R3. If no candidate beats the original on the holdout by more than the measured noise, the original is returned unchanged, the report says "no reliable improvement", and the exit code is 0. Proof: A with the `fake` backend.
- R4. A `--dry` run prints the plan (models, call budget, scenario count, **estimated number of GEPA iterations the budget affords**) and makes zero model calls. A real run whose budget affords fewer than 4 iterations refuses to start unless `--force-low-budget` is given. Proof: A.

### Fidelity: not deviating in ways the user would not like
- R5. Before the search, one call extracts an **intent contract** from the prompt (goal, hard constraints, facts and names to keep, required output format, language, tone) and classifies its **kind**: `template` (a reusable instruction applied to varying inputs: system prompts, skills, slash commands) or `task` (a one-off request, the usual case for prompts typed into Claude Code). `--kind` overrides the guess. The contract is stored in the run folder and shown in the report. Proof: T.
- R6. Every candidate that becomes the answer passes the contract check: all "keep" items present or equivalent, no new goals added, same language and output format. A failing candidate is never returned. Proof: A with planted violations.
- R7. Length cap: the result is at most the strictness cap of R8 times the original's tokens (floor: the original plus 40 tokens, so very short prompts can still gain a sentence); `--allow-growth` removes the cap. The report states the ratio. Proof: T.
- R8. Strictness levels `conservative` (default: minimal edits, structure kept), `balanced`, `bold`. The level sets the reflection instructions and the cap in R7 (1.25x / 1.5x / 2.5x). Proof: T on the prompt templates, S on 5 prompts.
- R9. Placeholders, code blocks, URLs, file paths and quoted strings in the original appear unchanged in the result. Proof: T (property test).

### Scoring
- R10. Candidates are scored by running them. A task model executes the candidate on each scenario; the output is checked by (a) programmatic checks (format, length, required and forbidden strings from the contract), then (b) a binary checklist derived from the contract and judged by a different model that sees the output and the checklist but not the candidate text. Score = share of checks passed. The judge is called **once per candidate per batch of scenarios** (all outputs and checklists in one call), not once per scenario. The evaluator also returns `side_info["scores"]` with one entry per check group (for example `format`, `constraints`, `content`), which GEPA tracks as separate objectives on the frontier. Proof: T with `fake`.
- R10a. How a candidate is executed (one task call). `template`: the candidate is the system prompt and the scenario input is the user message. `task`: the scenario is a short synthesised *situation* (context the request could arrive in, for example different project details or constraints, never invented facts that contradict the contract) and the task model receives the situation followed by the candidate as the user message, with no system prompt. Single-turn and no tools (R18): for `task` prompts the output is the model's answer or plan, and the report says that tool use is not exercised. Proof: T.
- R10b. The judge returns JSON (`--json-schema`) with, for every check, `pass` and a **verbatim quote from the output** as evidence. A pass whose quote is not found in the output (after whitespace normalisation) counts as a fail; a check the judge omits counts as `unknown` (R24). Output text is wrapped in delimiters and the judge is told it is data. This keeps a candidate from talking the judge into passing everything. Proof: T with hostile outputs.
- R11. Scenarios come from `--examples file.jsonl` (fields `input`, optional `expected`, optional `criteria`) or, when absent, from one synthesis call (12 varied inputs or situations including edge cases). With 8 or more scenarios a holdout of at least 3 is kept (split rules in R15) with a fixed seed; fewer than 8 give no holdout, so R3, R12 and R13 cannot be checked and the report says so and returns the original unless `--trust-search` is given. Proof: T.
- R12. Noise: the original is scored twice on the holdout (two independent runs); the win threshold in R3 is `max(0.05, 2 x |difference between the two runs|)`. With a holdout of 3 to 5 scenarios this is a coarse estimate, so the report prints it and the margin by which the winner cleared it. Proof: T.
- R13. If the original already scores at least 0.95 on the holdout, the run stops early and says "already strong, relative to the generated checks" (high scores can also mean weak checks; the report lists the checks so the user can judge). Proof: A.
- R14. The judge model is never the task model. Proof: T (config validation).
- R14a. Target-model confirmation. The search may use a cheaper task model (default Haiku 4.5; the paper's Obs. 6 shows prompts optimised on a weaker model transfer to a stronger one). The final holdout comparison in R3 is run on the model the user will actually use (`--target-model`, default `sonnet`; the `/improve` command passes the session's model; when it differs from the task model, the report shows both scores). A candidate that wins on the search model and loses or ties on the target model is not returned. Proof: A with `fake`.

### Search
- R15. Search uses `gepa.optimize_anything` with `parallel=False`. Scenarios are split three ways: `dataset` (reflection minibatches), `valset` (acceptance and the Pareto frontier) and the holdout, which the search never sees and which only R3, R12 and R13 use. Split sizes for n scenarios (n >= 8): holdout `max(3, round(0.35 n))`, valset `clamp(round(0.25 n), 2, 4)`, dataset the rest (never below 3); so n=8 gives 3/2/3 and n=12 gives 4/3/5 (holdout/valset/dataset). Below 8 see R11. `frontier_type="hybrid"` is used (scenario x check-group), so a small `valset` still gives a useful frontier. `MergeConfig` is off by default: the paper (Obs. 5, Tables 1-2) found merge helped GPT-4.1 Mini but hurt Qwen3-8B, and its timing is an open question; it may be enabled by `--merge` and is not part of the acceptance measure. Pareto-based selection stays on (paper Table 3: +12.4 points vs +6.1 for best-candidate and +5.1 for beam search). Proof: T on the adapter wiring, A.
- R15a. Validation dominates cost in GEPA (paper Obs. 1: most rollouts go to scoring candidates on the Pareto set), so `valset` is kept small (2 to 4 scenarios) and a candidate is scored on it only after it improves on its minibatch (the library's `strict_improvement` acceptance). Proof: T counts calls on `fake`.
- R16. Reflection receives the contract, the strictness level and per-scenario failed checks with short output excerpts (ASI). Proof: T.

### Budget and safety
- R17. Every `claude -p` call (task, judge, reflection, intake, synthesis) counts toward one budget: default 100, ceiling 300, 45 minutes wall clock (the paper's runs used 400 to 7,000 rollouts, so this tool works in a very low-budget regime where the first few reflective updates carry most of the gain; the report says so). The fixed costs are reserved up front and shown by `--dry`: intake, synthesis, two seed runs on the holdout, up to 3 finalists on the target model, up to 3 contract checks. The search gets the rest and stops cleanly when it is used up, then runs the final steps with the reserve. If the reserve alone exceeds the budget the run refuses to start. Identical calls are cached on disk. Proof: A.
- R18. All model calls go through one function that builds the command `claude -p --safe-mode --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 --model M --output-format json` plus `--system-prompt S` where the call has a system prompt (R10a) and `--json-schema` on intake, synthesis and judge calls, with a scrubbed environment; the first call of every run aborts (exit 4) if the session reports plugins, MCP servers or tools. No API key is read or needed. Details wait for the user's real-call check (BUILD-LOG). Proof: T, S.
- R19. Prompts and model outputs are untrusted data: they are passed as stdin or arguments, never evaluated, never used to build shell commands or file paths. Proof: T with hostile strings.
- R20. Tests never call a real model or the network. Proof: `just check`.
- R23. Run data (prompt, contract, scenarios, outputs, call cache) is stored under `$XDG_STATE_HOME/autoimprover/runs/<id>/` with mode 0700, never inside a git repository, and `autoimprover clean` removes it. The prompt is the user's own data, so the report names the folder. Proof: T.
- R24. Failure handling. A call that errors or times out is retried up to 2 times (each retry counts toward the budget). A judge check that stays `unknown` is excluded from the score; a candidate with more than 30 % unknown checks scores 0 with the reason in its ASI. Three consecutive backend failures end the run with exit 3 and the partial run folder kept for `--resume` (R22). A budget stop is not a failure. Proof: A with a failing `fake`.

### Claude Code integration
- R21. `/improve <prompt>` runs the command with `--dry` first when the prompt is longer than 2,000 chars, otherwise runs it directly, shows the report, and then waits for the user's yes before using the improved prompt as the task. `/optimize` is an alias. Proof: S.
- R22. A run can be resumed from its run folder after an interruption without repeating paid calls. Proof: A.

## 3. Non-goals (v1)
Training model weights; multi-prompt pipelines or DSPy programs; the local model at 127.0.0.1:8080 as task model (first follow-up); a hosted service; any network call except `claude -p`.

## 4. Limits
Python 3.12+; one dependency (`gepa==0.1.4`) plus dev tools; one run at a time; Max-plan usage is the only cost.

## 5. Acceptance measure (S)
On 8 to 10 of the user's real prompts, the improved prompt is preferred over the original in a blind A/B by the user in at least 70 % of the cases where the tool returned a change (a go/no-go signal, not a statistical proof: 7 of 10 is not significant against chance), with zero contract violations found by the user, a median length ratio at most 1.25 in `conservative`, and the "no reliable improvement" outcome appearing for prompts the user already considers fine. A pairwise Claude judge is reported alongside but does not decide.

## 6. Open questions for the user
Q1 target prompts (own Claude Code prompts first?), Q2 examples of good output available?, Q3 default budget 100 calls (raised from 60 after reading the paper: 60 buys about 3 iterations), Q4 models (task Haiku 4.5, judge Sonnet 5.5, reflection Opus 5.5), Q5 wait for yes before using the result (R21).
