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

Time tiers (SPEC R25, ADR-011): `--time` picks how a run works, and the default is the 30-second
**fast** tier, not the GEPA search. quick (15 to 24 s) checks one rewrite against the intent
contract; fast (25 to 59 s) runs a few rewrites, and the original twice, on a few scenarios in
parallel stages and returns the best rewrite only when it beats the original by more than
max(0.1, twice the difference of the original's two runs), labelled "fast check: scored on the
same few scenarios it was picked on, noise measured from two runs of the original on those
scenarios, not verified on held-out scenarios". Its rewrites may make an implied request explicit
and organise what you wrote (strategies clarify, structure, tighten, specify), and from 45 s, when
the time allows, a second generation reflects on the first one's outputs and failed checks (the
reflective step of GEPA) and is scored the same way; checked (1 to 9 minutes) adds a held-out check
on the target model. In the fast and checked tiers every scoring run, of the original and of every
rewrite, ends with the same request to answer in at most 120 words, so the runs stay short; it belongs to
the measurement, never to a returned prompt. Deep (10
minutes and up, `--deep` is `--time 20m`) is the GEPA search that "How a run works" and "Budget
and clock" below describe. Only checked and deep results are verified. The timings are estimates
from one timing probe; live runs are not yet measured.

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
tier: fast (--time 30 s), 6 calls at a time
models: task claude-haiku-4-5-20251001, judge claude-opus-5-5, reflection claude-sonnet-5-5, target claude-sonnet-5-5
effort: task low, judge low, reflection low
strictness: balanced, length cap 1.5x the original's tokens (at least the original plus 40)
rewrites: 2; scenarios: 2, synthesised by one call (2 to pick on, 0 held out)
stages:
  A: intake, synthesis and rewrites: 4 calls, about 8.8 s
  B: task runs: 8 calls, about 11.1 s
  C: judge and contract checks: 5 calls, about 5.5 s
  D: free gates and pick: 0 calls, about 0.0 s
estimate: 17 calls in about 25 s of 30 s; budget: 51 calls (ceiling 300)
evidence: a fast check: scored on the scenarios it is picked on, noise measured from two runs of the original, not verified on held-out scenarios
```

The stage times come from the latency model of ADR-011 (about 3.4 s per call, the start-up
measured with the trims of ADR-009, plus its output tokens at 70 per second, one slowest call per
wave of `--workers` calls; a task run is assumed to write 150 tokens, as it asks for at most 120
words). With `--workers 1` it adds `a real run would refuse: the fast plan
needs about 82 s, more than --time 30 s; give a longer --time, or more --workers (now 1)` and
still exits 0. With `--deep` it prints the plan of the GEPA
search instead: its budget, fixed costs, split, estimated iterations and clock share; with
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
| `--time DURATION` | the run's wall clock and its tier, a whole number with `s`, `m` or `h`: quick from `15s`, fast from `25s`, checked from `1m`, deep from `10m`; default `30s`, at most `3h`; below `15s` exit 2 |
| `--deep` | the GEPA search: the same as `--time 20m` (not together with `--time`) |
| `--workers N` | calls a stage runs at a time, 1 to 16; default 6 |
| `--budget N` | model calls, 1 to 300; default: three times the plan's estimate in the quick, fast and checked tiers, and in deep one call per 12 s of `--time`, from 20 to 100 |
| `--strictness LEVEL` | `conservative`, `balanced` or `bold`: length cap 1.25x, 1.5x or 2.5x the original's tokens (at least the original plus 40); default `balanced` in the quick, fast and checked tiers, `conservative` in deep |
| `--allow-growth` | no length cap |
| `--task-model MODEL` | runs the prompts on the scenarios; default `claude-haiku-4-5-20251001` |
| `--judge-model MODEL` | checks the outputs; default `claude-opus-5-5`, or `claude-sonnet-5-5` when the target is Opus |
| `--reflect-model MODEL` | writes the rewrites, the contract and the scenarios; default `claude-sonnet-5-5`, in deep `claude-opus-5-5` |
| `--target-model MODEL` | the model you will use the prompt with; the held-out comparison (checked, deep) runs on it; default `claude-sonnet-5-5` |
| `--effort LEVEL` | `claude --effort` of every role: `low`, `medium`, `high`, `xhigh`, `max` or `default` (the model's own); default `low`, in deep `default` |
| `--task-effort LEVEL` | the task role's effort, over `--effort` |
| `--judge-effort LEVEL` | the judge role's effort, over `--effort` |
| `--reflect-effort LEVEL` | the effort of the contract, scenario and rewrite calls, over `--effort` |
| `--merge` | deep only: let GEPA merge candidates (with this tool's valset of at most 4 it does not merge; see ARCHITECTURE section 11) |
| `--trust-search` | deep only: below 8 scenarios, return a rewrite that beat the original on the search's own valset, marked not verified |
| `--force-low-budget` | deep only: run although the budget affords fewer than 4 search iterations |
| `--dry` | print the plan; no model call, nothing written |
| `--json` | print one JSON object on stdout |
| `--resume ID` | continue the run with this id |

Model names may be full ids or the aliases `haiku`, `sonnet`, `opus`. A judge equal to the task or
target model is refused (exit 2, naming the flag). A flag given always wins over the tier's
default; `--merge`, `--trust-search` and `--force-low-budget` with a tier other than deep are
exit 2. A quick, fast or checked plan whose estimate is longer than `--time` is refused (exit 2;
`--dry` says so).

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
| 2 | bad input or usage, or a refusal before any paid call (budget, judge model, state folder, a plan longer than `--time`) |
| 3 | backend failure: a failed call outside the search, or three failed calls in a row of one kind to one model |
| 4 | the `claude` session was not locked down (it reported tools, MCP servers, skills, slash commands, extra agents or a non-default output style) |
| 130 | interrupted: Ctrl-C, or SIGTERM (what `/improve cancel` sends); no call starts after it and the running `claude` children are killed |

Every non-zero exit writes `error: <what, and the flag or folder to change>` to stderr and nothing
to stdout (with `--json`, only the error object). Exits 3 and 130 also print `run folder: <path>`
and `resume with: autoimprover --resume <id>` (with an `XDG_STATE_HOME=` prefix when the state
folder is not the default).

## JSON output

The keys of the object on stdout, in order:

- finished run: `status`, `prompt`, `verified`, `stop`, `changes`, `reason`, `reason_code`, `diff`, `contract`, `score_before`, `score_after`, `search_score_before`, `search_score_after`, `noise`, `margin`, `length_ratio`, `calls_used`, `run_dir`, `mode`, `elapsed_s`, `meaning`, `verified_text`, `margin_text`
- dry run: `status`, `plan`, `scenarios`, `synthesised`, `holdout`, `valset`, `dataset`, `calls_before_search`, `calls_after_search`, `search_calls`, `iteration_cost`, `iterations`, `iterations_best`, `search_clock_s`, `final_clock_s`, `refusal`, `keeps_original`, `tier`, `workers`, `efforts`, `rewrites`, `stages`, `est_calls`, `est_seconds`
- dry run plan: `models`, `strictness`, `budget`, `wall_clock_s`, `allow_growth`, `merge`, `seed`, `tier`, `workers`, `efforts`
- error: `status`, `code`, `error`, `run_dir`
- clean: `status`, `removed`, `skipped`

`status` is `improved` or `unchanged` for a finished run, else `dry`, `error`, `cleaned` or `help`.
`reason_code` is one of `improved`, `no_reliable_improvement`, `already_strong`, `no_holdout`,
`no_candidate_beat_seed`, `unconfirmed_out_of_budget`; `stop` is `budget`, `clock` or null (no
search ran, or every stage ran as planned). `mode` is the tier, `quick`, `fast`, `checked` or
`deep`; `verified` is true only for a checked or deep result confirmed on held-out scenarios. In a
quick or fast result `score_before` and `score_after` are taken on the scenarios the rewrite was
picked on, not held out; in a checked result the `search_score_*` keys are. In a fast result
`noise` is the difference between the original's two runs on those scenarios, and a rewrite had
to gain more than max(0.1, 2 x noise). Scores are shares of checks passed, from 0 to 1, or null
when not measured. `elapsed_s` is the run's clock in seconds; `meaning`, `verified_text` and
`margin_text` are the report's lines of those names without their labels, null when it prints
none. A dry run's keys are the same in every
tier, null where they do not apply: the search's numbers (`valset` to `final_clock_s`) in the
quick, fast and checked tiers, `rewrites`, `stages`, `est_calls` and `est_seconds` in deep; each
stage is `name`, `calls` and `seconds`.

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

## Measuring the tool (bench)

`autoimprover bench` (SPEC R26) shows whether the tool's rewrites are better, which a single run
cannot. It runs every prompt of a set through the ordinary pipeline (the tier of `--time`, default
`30s`), one prompt after the other, then compares each returned rewrite with its original: both run
with the plain call on the target model (no 120-word suffix) on 4 fresh scenarios, and the judge
model sees the original request and two anonymous answers per scenario, in both orders. A scenario
counts only when both orders pick the same side, so a judge that prefers a position gives ties,
never wins; a rewrite wins when it wins more scenarios than it loses. A prompt returned unchanged
is a tie and costs no comparison. `--baseline naive` also compares a one-call "Improve this
prompt." rewrite (reflection model, low effort) with the original, on the same scenarios.

```
uv run --frozen autoimprover bench --dry                 # the plan: calls and time, no call made
uv run --frozen autoimprover bench --baseline naive --json > bench.json
```

Flags: `--prompts FILE` (JSON Lines: `id`, `prompt`, optional `kind`, optional `examples`; default
the 20 varied prompts of `bench/prompts.jsonl`), `--time`, `--limit N` (1 to 25; a set of more
needs it), `--baseline naive|none`, `--judge-model`, `--seed S` (0 to 999, for the bench's own
calls), `--json`, `--dry`, and `bench` comes first. It spends subscription calls: at `30s` the
plan is about 17 calls per run plus 17 per comparison (13 more with the naive baseline), so the
whole set is at most 680 calls in about 21 minutes. The summary (text, or one JSON object with
`--json`) gives the improved rate, wins, ties and losses with the win rate among improved prompts,
the baseline's, median and 90th percentile seconds per run, total calls and a row per prompt: ids,
codes and numbers, never a prompt. `contract_violations` is `null`: the pipeline never returns a
rewrite its contract check vetoed, and the pairwise judge sees answers, not prompts, so the bench
has no count of its own yet. Ctrl-C prints the summary of the prompts already measured. The
bench exits 0 whatever the result (it measures, it does not gate); 2 for bad usage or a plan that
would refuse, 3 when every run failed. Each prompt's run folder, with the bench's own calls in its
cache, is under `$XDG_STATE_HOME/autoimprover/bench/<id>/<prompt id>/`; `clean` does not remove
these, so delete `bench/<id>/` by hand when done.

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
