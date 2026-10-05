# autoimprover

A command-line prompt optimizer. You give it one prompt; it returns one prompt that should work
better and keeps what you meant, or your original when it cannot show an improvement.

"Better" is measured by running the prompt, not by grading its wording: a task model runs each
candidate on test scenarios (your examples, or ones it writes), and a separate judge model checks
the outputs. The search is GEPA (reflective prompt evolution with a Pareto frontier,
arXiv:2507.19457, `gepa==0.1.4`). A rewrite is returned only after it beat the original by more
than the measured noise on held-out scenarios the search never saw, on the model you will use it
with. With fewer than 8 scenarios there is nothing to hold out, and the original is returned unless
you pass `--trust-search`, which marks the result as not verified.

## Status

Version 0.2.0.dev0, not released. Built and tested end to end with a scripted fake model (no test
calls a real model or the network): input checks, contract, scenarios, scoring, GEPA search, final
checks, report, run folders and `--resume`. The real model backend,
`src/autoimprover/claude_cli.py` (the one place that starts `claude -p`, SPEC R18), is built and
tested with a fake `claude` script that prints the format of real `claude -p` output
(`tests/fixtures/claude_cli/`, ADR-009). It is not verified against live runs: the live check is
the user's `just smoke` or a first `/improve`; until it passes, nothing here claims that a live
run, the `/improve` command or the acceptance measure of SPEC section 5 works.

## Install and run

Needs Python 3.12 or newer and `uv`; live runs will also need the `claude` CLI logged in to your
subscription (no API key is read). In the checkout:

```
uv run --frozen autoimprover "Summarise the meeting notes for the team in five bullet points."
uv run --frozen autoimprover --file prompt.txt --examples examples.jsonl
cat prompt.txt | uv run --frozen autoimprover --json
```

The first `uv run --frozen` creates `.venv` from `uv.lock`. From another folder, add
`--project ~/projects/optimizer` after `--frozen`. The prompt comes from one quoted argument, from
`--file`, or from piped stdin (never from a terminal, so the tool does not wait for typing). It
must be UTF-8, 1 to 20,000 characters, without a NUL character and without GEPA's template tokens
`<curr_param>` and `<side_info>`; CRLF line endings become LF.

## Example: the plan of a run

`--dry` prints what a run would do and spend. It makes no model call, writes nothing and exits 0
once the arguments parse, so it works in this build:

```
$ uv run --frozen autoimprover --dry --file prompt.txt
dry run: no model call made, nothing written
models: task claude-haiku-4-5-20251001, judge claude-opus-5-5, reflection claude-opus-5-5, target claude-sonnet-5-5
strictness: conservative, length cap 1.25x the original's tokens (at least the original plus 40)
budget: 100 calls (ceiling 300); fixed costs: 16 before the search, 18 after it, 66 left for the search
scenarios: 12, synthesised by one call; holdout 4, valset 3, dataset 5
iterations: about 5 GEPA iterations (worst case, 13 calls each) to 7 (best case); an estimate: the clock may end the search sooner (live calls take 5 to 45 s)
clock: 45 min; the search may use 33 min 45 s, the final steps keep 11 min 15 s
```

With `--budget 30` it adds `a real run would refuse: the fixed costs (16 calls before the search
and 18 after it) exceed the budget of 30; use --budget 86 or more` and still exits 0. With
`--target-model opus` the judge becomes `claude-sonnet-5-5` (never the task or target model).

## How a run works

1. One call extracts an intent contract from the prompt: goal, things to keep, constraints, output
   format, language, tone, a list of checks, and the kind (`template`: a reusable instruction such
   as a system prompt; `task`: a one-off request). `--kind` overrides the guess.
2. Scenarios come from `--examples`, or one call writes 12. From 8 scenarios up they are split
   into dataset, valset and a holdout of 3 to 6 that the search never sees.
3. The original is scored twice on the holdout, on the target model, in two independent runs; the
   difference is the noise, and a result must beat the original by more than
   `max(0.05, 2 x noise)`. An original that already scores 0.95 or more ends the run there.
4. GEPA searches on the cheaper task model, one call at a time. A candidate is scored by running it
   (`template`: as the system prompt, the scenario as the user message; `task`: the scenario as a
   situation, then the candidate, no system prompt; single turn, no tools) and by checks on the
   output: programmatic ones and a judge that sees the outputs, never the candidate.
5. Up to 3 finalists pass free gates first (length cap, placeholders, code blocks, URLs, paths and
   quoted strings kept unchanged), then a contract check. The first one that beats the original on
   the holdout, on the target model, is returned; otherwise the original is.

## Flags

| Flag | Meaning |
|---|---|
| `--help` | show the help (also `-h`) |
| `--file PATH` | read the prompt from this UTF-8 file |
| `--examples PATH` | test scenarios, JSON Lines (below); without it one call writes 12 |
| `--kind KIND` | `template` or `task`; default: guessed by the contract call |
| `--budget N` | model calls, 1 to 300; default 100 |
| `--strictness LEVEL` | `conservative` (default), `balanced` or `bold`: length cap 1.25x, 1.5x or 2.5x the original's tokens (at least the original plus 40) |
| `--allow-growth` | no length cap |
| `--task-model MODEL` | runs candidates during the search; default `claude-haiku-4-5-20251001` |
| `--judge-model MODEL` | checks the outputs; default `claude-opus-5-5`, or `claude-sonnet-5-5` when the target is Opus |
| `--reflect-model MODEL` | proposes rewrites, writes the contract and scenarios; default `claude-opus-5-5` |
| `--target-model MODEL` | the model you will use the prompt with; the final comparison runs on it; default `claude-sonnet-5-5` |
| `--merge` | let GEPA merge candidates (with this tool's valset of at most 4 it does not merge; see ARCHITECTURE section 11) |
| `--trust-search` | below 8 scenarios, return a rewrite that beat the original on the search's own valset, marked not verified |
| `--force-low-budget` | run although the budget affords fewer than 4 search iterations |
| `--dry` | print the plan; no model call, nothing written |
| `--json` | print one JSON object on stdout |
| `--resume ID` | continue the run with this id |

Model names may be full ids or the aliases `haiku`, `sonnet`, `opus`. A judge equal to the task or
target model is refused (exit 2, naming the flag). The 45-minute wall clock has no flag.

## Examples file

JSON Lines, UTF-8 (a leading BOM is accepted), one object per non-blank line:

```
{"input": "Notes: budget approved, launch moved to May.", "expected": "- Budget approved\n- Launch moved to May"}
{"input": "Notes: nothing decided.", "criteria": ["says that nothing was decided"]}
{"input": "Notes: hire two engineers."}
```

- `input` (required, not only whitespace): for a `template` prompt the user message it receives;
  for a `task` prompt the situation the request arrives in.
- `expected` (optional, string or null): adds one judged check, "the output agrees with the
  reference answer in substance". `criteria` (optional, non-blank strings): one judged check each.

Other keys are ignored; there are no exact-match or regular-expression checks. A bad line is exit 2
naming the line. The file is read once and copied into the run folder.

## Output

Without `--json` the returned prompt alone goes to stdout, so it can be piped, and the report to
stderr: result and reason, whether it is verified, holdout scores on the target model, the noise
and the margin, search scores on the task model, length ratio, calls used, why the search ended
(a clock stop is also a notice), what changed and why, a word diff, the intent contract with its
checks, and the run folder. With `--json` stdout carries exactly one JSON object and notices stay
on stderr. GEPA's own progress goes to `gepa.log` in the run folder, never to stdout.

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | done, also when the original is returned ("no reliable improvement" and the other unchanged reasons) |
| 1 | internal error (a bug); stderr names the run folder and the resume line when there is one |
| 2 | bad input or usage, or a refusal before any paid call (budget, judge model, state folder) |
| 3 | backend failure: a failed call outside the search, or three failed calls in a row of one kind to one model |
| 4 | the `claude` session was not locked down (it reported tools, MCP servers, skills, slash commands, extra agents or a non-default output style) |
| 130 | interrupted (Ctrl-C) |

Every non-zero exit writes `error: <what, and the flag or folder to change>` to stderr and nothing
to stdout (with `--json`, only the error object). Exits 3 and 130 also print `run folder: <path>`
and `resume with: autoimprover --resume <id>` (with an `XDG_STATE_HOME=` prefix when the state
folder is not the default).

## JSON output

The keys of the object on stdout, in order:

- finished run: `status`, `prompt`, `verified`, `stop`, `changes`, `reason`, `reason_code`, `diff`, `contract`, `score_before`, `score_after`, `search_score_before`, `search_score_after`, `noise`, `margin`, `length_ratio`, `calls_used`, `run_dir`
- dry run: `status`, `plan`, `scenarios`, `synthesised`, `holdout`, `valset`, `dataset`, `calls_before_search`, `calls_after_search`, `search_calls`, `iteration_cost`, `iterations`, `iterations_best`, `search_clock_s`, `final_clock_s`, `refusal`, `keeps_original`
- dry run plan: `models`, `strictness`, `budget`, `wall_clock_s`, `allow_growth`, `merge`, `seed`, `tier`, `workers`, `efforts`
- error: `status`, `code`, `error`, `run_dir`
- clean: `status`, `removed`, `skipped`

`status` is `improved` or `unchanged` for a finished run, else `dry`, `error`, `cleaned` or `help`.
`reason_code` is one of `improved`, `no_reliable_improvement`, `already_strong`, `no_holdout`,
`no_candidate_beat_seed`, `unconfirmed_out_of_budget`; `stop` is `budget`, `clock` or null (no
search ran). Scores are shares of checks passed, from 0 to 1, or null when not measured.

## Budget and clock

Every model call counts toward one budget (intake, synthesis, task, judge, reflection), and so does
each retry (a failed call is retried twice); a call answered from the run's disk cache is free.
Fixed costs are reserved before the first call: the contract call, the synthesis call when there
are no examples, two seed runs on the holdout and GEPA's scoring of the original on the valset;
after the search up to 3 finalist runs on the target model and up to 3 contract checks. The search
gets the rest. A run refuses to start (exit 2) when the fixed costs exceed the budget, or when the
budget affords fewer than 4 iterations in the worst case and `--force-low-budget` is not given.
The wall clock is 45 minutes of monotonic time (a suspended laptop does not count), carried across
`--resume`; the search may use 75 % of it. If calls or time run out before a finalist is confirmed,
the original is returned (`unconfirmed_out_of_budget`). A budget of 100 is very low for GEPA (the
paper's runs used 400 to 7,000 rollouts); the first few rewrites carry most of the gain.

## Run folders, `--resume` and `clean`

Each run keeps its data in `$XDG_STATE_HOME/autoimprover/runs/<id>/` (by default
`~/.local/state/autoimprover/runs/<id>/`), folders mode 0700, files 0600. The id is the UTC start
time and 8 hex digits of the prompt's SHA-256, like `20261004-132507-1a2b3c4d`. The folder holds
`manifest.json` (prompt, plan, flags, calls and seconds used), `contract.json`, `scenarios.json`,
`checkpoint.json`, `calls.jsonl`, `cache/` (the model replies), `gepa.log`, `run.lock` and an empty
`cwd/`. A state folder that is not writable or lies inside a git repository is exit 2.

`autoimprover --resume <id>` continues an interrupted run with its saved prompt, plan, flags,
scenarios, call count and clock (never a fresh budget or clock); answered calls are replayed from
the cache for free. Other flags are ignored with a notice (except `--json`; `--dry` is exit 2). A
second `--resume` of a run that is still going is refused. `autoimprover clean <id>` removes one
run folder, `autoimprover clean` all of them, skipping runs still going; both exit 0, also when
there is nothing to remove. Both accept only a run id, never a path. Nothing expires on its own.

## Trust model: what is stored and what is trusted

- Your prompt, your examples and the model replies are your data. A run keeps them in its run
  folder until you run `clean`; the report names the folder.
- Prompts, examples and model outputs are treated as data: never evaluated, never used to build a
  shell command or a file path. Checks written by a model are limited to `contains`,
  `not_contains`, `max_chars` and `min_chars`, never regular expressions.
- The judge never sees the candidate when it scores, only the outputs and the checklist, and a
  pass counts only with a verbatim quote found in the output. The final contract check sees the
  candidate and can only reject it. The judge is never the task or the target model.
- Text a model wrote is stripped of terminal escape sequences and control characters when printed.
- Every model call is `claude -p --safe-mode --settings '{"outputStyle":"default"}' --tools ""
  --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 ...` (SPEC
  R18, ADR-009), with user text on stdin, a scrubbed environment and a 300 s timeout per call. A
  call ends the run with exit 4 when its session reports tools, MCP servers, skills, slash
  commands, agents beyond the four built-in ones, or an output style other than the default; the
  installed plugins it lists are not active under `--safe-mode` and do not count. Built and tested
  with a fake `claude`; not yet verified against live runs (see "Status").

## Use it inside Claude Code

This repository is also a Claude Code plugin whose mod (`hooks/register.js`, ADR-010) adds
`/improve` and its alias `/optimize`. It needs Claude Code 2.1.287 or later (the first with mods),
`uv` on the PATH Claude Code runs with, and `claude` logged in to your subscription. Install it
from the repository (private: adding the marketplace needs access to it), then run
`/reload-plugins` in a running session:

```
claude plugin marketplace add ItsUkrainianCat/optimizer
claude plugin install autoimprover@optimizer
```

For one session from a checkout instead: `claude --plugin-dir <path to the checkout>`.

| Command | What it does |
|---|---|
| `/improve <prompt>` | improve the prompt (it may span lines); CLI flags such as `--budget 60` or `--strictness balanced` go before it, and a prompt that starts with `--` goes after a lone `--` |
| `/improve --file PATH` | the prompt from a file; a relative path is read from the session's folder |
| `/improve --dry <prompt>` | the plan only: no model call, nothing written |
| `/improve --resume [ID]` | continue the last run, or the run with that id |
| `/improve clean [ID]` | remove the last run's folder, or that run's |
| `/improve cancel` | stop the run in progress; it stays resumable |

`/optimize` takes the same. One run at a time per session; `cancel` works while Claude is busy.

What the mod does, and nothing more:

- It writes a typed prompt to `prompt.txt` in a folder that `mktemp -d` makes under `$TMPDIR`
  (else `/tmp`) with mode 0700, then sets the file to mode 0600 with `chmod` (Claude Code's file
  write takes no mode; until the `chmod`, the 0700 folder keeps other users out), and starts
  `uv run --frozen --project <plugin folder> autoimprover --json --file <that file>
  --target-model <the session's model>` plus your flags, as an argument list with no shell. The
  folder is removed when the child ends, also when it fails or is cancelled. The prompt never
  goes through the model.
- A prompt longer than 2,000 characters (a `--file` larger than 2,000 bytes) gets a `--dry` run
  first; when the plan says a real run would refuse, the mod stops there with that reason.
- The tool's virtual environment is `$XDG_CACHE_HOME/autoimprover/venv` (by default
  `~/.cache/autoimprover/venv`, through `UV_PROJECT_ENVIRONMENT`), never the plugin's folder.
- The child's last line on stderr is the status line. When the run ends with an improved
  prompt, that prompt appears in the prompt box by itself, as a draft: edit it if you want and
  press Enter to send it. The mod sends nothing. If the box already holds a draft you typed, the
  draft is kept and the pane says so. A result that is not verified on a holdout
  (`--trust-search`) goes in too, and the toast and the pane say "NOT verified". A kept original,
  a failure, a cancel or a plan (`--dry`) puts nothing in the box.
- The run's JSON object becomes a pane: the result and reason, what happened to the prompt box,
  the scores, the margin over the noise, the length ratio, what changed and the improved prompt.
  "Use it" puts the improved prompt into the box again, replacing what is there ("Use it (not
  verified)" for a `--trust-search` result), "Copy" copies it, "Close" closes the pane. Where
  nothing can draw a pane (`claude -p "/improve ..."`), the command waits for the run, prints the
  report as its text (including whether the prompt went into the box) and exits with the tool's
  exit code (1 for a failure of the mod itself, 2 when `uv` is missing).
- It keeps the last run id in the plugin's store for `--resume` and `clean`. Run data lives in the
  default state folder, `~/.local/state/autoimprover/runs/<id>/` (see "Run folders").

Trust surface: `claude plugin validate --strict` lists what the module hooks and calls. It hooks
`session.start` (to register the two commands), `command.run` (its own commands) and `ui.render`
(its own pane only); never `prompt.submit`, so it does not see your other prompts. It calls
`process.run` (`uv --version`, `mktemp`, `chmod`, `rm`, the plan, `clean`), `process.spawn` (the
run), `fs.write` (the prompt file), `fs.stat` (the size of a `--file`), `env.get` (`HOME`,
`XDG_CACHE_HOME`, `TMPDIR`), `session.model`, `session.cwd`, `session.surfaces`, `store.*`,
`prompt.read` (to keep a draft you typed), `prompt.fill`, `command.register` and `ui.*`; no
model, MCP or network call.
`tests/test_mod_package.py` pins these lists. A mod runs inside Claude Code with your permissions,
and what it starts runs outside Claude Code's sandbox with your normal login, which is how the
tool's `claude -p` calls find it; those calls run with `--safe-mode`, so this mod is off in them.

Not verified until your live run: the mod is type-checked against the declarations of Claude Code
2.1.287 and its tests (`tests/mod/*.test.ts`, run by `claude plugin test`) use a stubbed engine.
Whether it loads in a live session, finds `uv`, gets a model id `--target-model` accepts from
the session, kills the child on `cancel`, puts the prompt into the box, and how the pane looks,
are checked only by running it.

## Limits and non-goals

- Single turn, no tools: for a `task` prompt the output scored is the model's answer or plan.
- The holdout has 3 to 6 scenarios, so the noise estimate is coarse; high scores can also mean
  weak checks, so the report lists them. Model aliases are pinned to the ids above.
- One run per run folder; runs with different ids may run at the same time.
- Not in v1 (SPEC section 3): training model weights, multi-prompt pipelines or DSPy programs, the
  local model as task model, a hosted service, any network call except `claude -p`.

## Development

Requirements: `docs/SPEC.md`; design: `docs/ARCHITECTURE.md`, `docs/adr/`. `just check` (format,
lint, types, tests) is the definition of done; tests use the fakes in `tests/fakes.py` and never a
real model or the network. The 0.1.0 script is the git tag `v0.1.0-legacy`.
