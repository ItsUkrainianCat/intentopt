# CLI reference

The project is presented as **IntentOpt**; the Python package, executable and local state directory remain `autoimprover` for compatibility. Run commands from the repository root.

A command-line prompt optimizer. You give it one prompt; it returns one prompt that should work
better and keeps what you meant, or your original when it cannot show an improvement. "Better" is
measured by running the prompt, not by grading its wording: a task model runs the original and
each rewrite on test scenarios (your examples, or ones it writes), and a separate judge model
checks the outputs without seeing candidate rewrite text. A pairwise judge also sees the
original request; a reference judge sees the example and its checks.

How much a result proves depends on the time you give it (`--time`, default 30 s). The default
**fast** run returns a rewrite only when it beat the original by more than the noise of two runs of
the original, on the same few scenarios it was picked on: a fast check, not verified on held-out
scenarios. A **checked** run (1 to 9 minutes) also confirms the winner on held-out scenarios on the
model you will use it with, and only then calls it verified. A **deep** run (`--deep`) is a GEPA
search (reflective prompt evolution with a Pareto frontier, arXiv:2507.19457, `gepa==0.1.4`) whose
result must pass the configured held-out check before being marked verified.

## Status

Experimental development build. See [research evidence and limitations](RESEARCH.md) for the
current evidence record, including reported live benchmarks. This reference contains historical
timing estimates; `--dry` and `--help` describe the plan for your checkout. Model IDs are backend
configuration, not a guarantee of provider availability.

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

## Choosing `--time`

`--time` is the run's hard wall clock and picks its tier (SPEC R25, ADR-011). Calls and seconds
below are historical pairwise-mode estimates for a short prompt on the default 6 workers;
reference mode can use more examples and reflection rounds. A longer prompt or fewer workers
shrink the plan. Inspect `--dry` for the current input and configuration.

| `--time` | Tier | What runs | Calls (estimate) | Verified |
|---|---|---|---|---|
| 15 to 24 s | quick | the intent contract and one `clarify` rewrite side by side, then the contract check; no scenario is scored | 3 at 15 s (about 13 s) | no |
| 25 to 59 s | fast | the stages below on 2 to 4 scenarios with up to 3 rewrites; from 45 s, when it fits, a second round of up to 2 rewrites | 17 at 30 s (about 25 s), 25 at 45 s (34 s), 35 at 59 s (48 s) | no |
| 1 to 9 min | checked | the fast stages with up to 6 rewrites (2 at 1 min), then the winner and the original on 2 to 4 held-out scenarios on the target model | 33 at 1 min (50 s), 58 at 90 s (75 s), 80 at 5 min | yes |
| 10 min and up | deep | the GEPA search ("The deep tier" below), one call at a time; `--deep` is `--time 20m` | its budget: 100 at 20 min | yes |

How to choose: the default 30 s for a short run whose result you read before using it;
`--time 45s` for the second round, which rewrites from what the first round's answers got wrong;
`--time 1m` or more when you want the result checked on scenarios it was not picked on, on your
model; `--deep` for the search. Below 15 s a run is refused (exit 2). The deep tier's default
budget is one call per 12 s of `--time` (20 to 100 calls); with 12 synthesised scenarios that
affords at least 4 search iterations only from `--time 18m`, so a shorter deep run refuses unless
you give `--budget 86` or more, or `--force-low-budget` (`--dry` says so).

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
rewrites: 1; scenarios: 3, synthesised by one call (3 to pick on, 0 held out)
stages:
  A: intake, synthesis and rewrite: 3 calls, about 8.8 s
  B: task runs: 9 calls, about 11.1 s
  C: pairwise judge and contract checks: 5 calls, about 5.1 s
  D: free gates and pick: 0 calls, about 0.0 s
estimate: 17 calls in about 25 s of 30 s; budget: 51 calls (ceiling 300)
evidence: a fast check: preferred over the original by a pairwise judge on the scenarios it is picked on, noise measured by comparing the original with itself, not verified on held-out scenarios
```

The stage times come from the latency model of ADR-011 (about 3.4 s per call, the start-up
measured with the trims of ADR-009, plus its output tokens at 70 per second, one slowest call per
wave of `--workers` calls; a task run is assumed to write 150 tokens, as it asks for at most 120
words, and a pairwise judge call 40 per scenario, a winner and one short reason). With `--workers 1` it adds `a real run would refuse: the fast plan
needs about 82 s, more than --time 30 s; give a longer --time, or more --workers (now 1)` and
still exits 0. With `--deep` it prints the plan of the GEPA
search instead: its budget, fixed costs, split, estimated iterations and clock share; with
`--target-model opus` the judge becomes `claude-sonnet-5-5` (never the task or target model).

## How a run works

The following describes pairwise mode; see "Reference examples and learned rules" for the
reference-mode differences. The fast and checked tiers are stages whose calls run side by side (`--workers`, default 6):

1. **A**, one wave: one call extracts the intent contract (goal, things to keep, constraints,
   output format, language, tone, checks, and the kind: `template`, a reusable instruction such as
   a system prompt, or `task`, a one-off request; `--kind` overrides the guess). When the prompt
   only states a situation or an intention, the goal is the request it clearly implies. Beside it,
   one call writes the scenarios (or your `--examples` are used), and K calls write rewrites, each
   with its own strategy: `clarify` (state the request the prompt only implies, in your words),
   `structure` (organise your content into role, context, task and expected output), `tighten`
   (remove redundancy), and from K=4 `specify` (make audience, format and constraints explicit
   where they are implied). A rewrite is dropped at once if it changes no meaning word (only case,
   punctuation, whitespace or articles), is longer than the length cap, or loses a placeholder,
   code block, URL, path or quoted string of the original.
2. **B**: the original runs twice and every rewrite once on each scenario, on the task model
   (`template`: the prompt is the system prompt and the scenario the user message; `task`: the
   scenario as a situation, then the prompt, under a fixed neutral system prompt, a plain assistant
   with no tools, so the model does not act as Claude Code's agent; single turn, no tools). Every one
   of these scoring runs ends with the same request to answer in at most 120 words, which keeps
   the runs short; it belongs to the measurement and never to a returned prompt.
3. **C**, one wave: for each rewrite, two judge calls compare its answers with those of the
   original's first run on all the scenarios at once, once with the original's answers first and
   once second; the judge sees your original prompt as the request and two anonymous answers per
   scenario, never a rewrite, and says which is better, or a tie, with one short reason. Two more
   calls compare the original's two runs the same way, and one judge call checks every rewrite
   against the contract and can only veto.
4. A scenario counts for a side only when both orders pick it, so a judge that prefers a position
   makes ties, never wins. The noise is the number of scenarios where the original's two runs had
   a winner. A rewrite wins only when it kept the contract and won more scenarios than it lost, by
   more than the noise.
5. From 45 s, when the time allows, a second round: the reflection model reads the best one or two
   rewrites that kept the contract (or the original), with the judge's reasons for the scenarios
   each lost or tied, and writes up to 2 new rewrites, which pass the same gates and are compared
   with the same answers of the original.
6. **D**: the winner with the largest lead is picked (a tie goes to the shorter). In the fast tier
   it is returned, labelled a fast check; with no winner the original is.
7. **E** (checked only): the winner and the original run once each on 2 to 4 held-out scenarios on the
   target model, with the same 120-word request; the winner is returned, verified, only when it
   beats the original there by more than 0.05 (no noise is measured on the held-out scenarios).

Before each stage the time and calls left are compared with its estimate; a stage that does not
fit shrinks (scenarios first) or is skipped, and a rewrite that has not passed every gate when time
runs out is never returned: the original is (`unconfirmed_out_of_budget`). The quick tier writes
the contract and one `clarify` rewrite side by side (no scenarios), runs the contract check, and
returns the rewrite when it passes and the free gates.

**What a fast check means, and what it does not.** It means the rewrite kept your intent (as the
contract check judged it), passed the free gates, and on 2 to 4 scenarios, the same ones it was
picked on, a judge preferred its answers to your original's in both orders on more scenarios than
it lost, by more than your original's two runs differ. It does not claim that it is
better on other inputs, on the model you will use it with (the runs are on the task model), or for
answers longer than 120 words; the noise comes from two runs only. Read it before you use it; a
checked run tests it on held-out scenarios.

**The deep tier** (10 minutes and up):

1. The intent contract as above; scenarios from `--examples`, or one call writes 12. From 8
   scenarios up they are split into dataset, valset and a holdout of 3 to 6 that the search never
   sees; below 8 the original is kept unless `--trust-search`.
2. The original is scored twice on the holdout, on the target model; a result must beat it by more
   than `max(0.05, 2 x noise)`. An original that already scores 0.95 or more ends the run there.
3. GEPA searches on the cheaper task model, one call at a time, scoring each candidate by running
   it and judging the outputs as above (without the 120-word request).
4. Up to 3 finalists pass the free gates, then the contract check. The first one that beats the
   original on the holdout, on the target model, is returned; otherwise the original is.

## Reference examples and learned rules

When every supplied example has `expected` or `criteria`, fast and checked runs score agreement
with those references instead of pairwise preference. The original is run twice to estimate
noise. A rewrite must improve more examples than it worsens, improve at least one, and exceed
the noise margin while passing the intent and literal-preservation gates.

In this mode the `induce` rewrite learns rules and output format from pick examples. The contract
check also sees those examples; held-out examples do not enter rewrite generation. Reflection
can run up to three rounds when time and call budget allow, using failed pick examples. Checked
confirmation runs the original twice and the candidate once on the held-out examples on the
target model, with reference scoring and a noise margin. The 120-word scoring suffix still applies.
Reference-mode length and planning rules differ from the pairwise estimates below; inspect `--dry`.

For an external evaluation split, a benchmark item can set `eval_from`: the optimizer receives
only the examples before that index, and the benchmark grades on the hidden remainder. The
repository includes `bench/examples-set.jsonl`, with three synthetic policy tasks, 14 examples
available to the optimizer and 16 hidden per task. The optimizer's internal holdout and this
external benchmark split are different levels of evaluation.

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
| `--ungated` | a measuring aid for the fast and checked tiers: returns the best-ranked candidate without a shown win; results are not verified |
| `--dry` | print the plan; no model call, nothing written |
| `--json` | print one JSON object on stdout |
| `--resume ID` | continue the run with this id |

Model names may be full ids or the aliases `haiku`, `sonnet`, `opus`. A judge equal to the task or
target model is refused (exit 2, naming the flag). A flag given always wins over the tier's
default; `--merge`, `--trust-search` and `--force-low-budget` with a tier other than deep are
exit 2. A quick, fast or checked plan whose estimate is longer than `--time` is refused (exit 2;
`--dry` says so).

Models and effort: the task model runs the scoring runs, the judge checks outputs and the contract,
the reflection model writes the contract, the scenarios and the rewrites, and the target model (the
one you will use the prompt with) runs the held-out comparison of the checked and deep tiers. The
quick, fast and checked tiers ask every role for `--effort low`, which the timing probe found
halves Sonnet's and Opus's time; deep leaves each model its own default. Effort is part of a
call's cache key.

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

When every example has `expected` or `criteria`, the fast and checked tiers decide by agreement with them (the reference-scored label), not by a blind preference. Other keys are ignored; there are no exact-match or regular-expression checks. A bad line is exit 2
naming the line. The file is read once and copied into the run folder. In reference mode, short categorical reference labels are ordered round robin before the
pick/held-out split when there are 2 to 12 labels; otherwise file order is retained. Only pick
examples reach rewrite generation. The checked tier keeps the original without a call when no
held-out examples are left.

## Output

Without `--json` the returned prompt alone goes to stdout, so it can be piped, and the report to
stderr: result and reason, what the result means and whether it is verified, the tier and the
seconds it took, the scores labelled by where they were taken (on the scenarios a rewrite was
picked on, on held-out scenarios, or in the search), the noise and the margin, length ratio, calls
used, whether a stage or the search was cut short (a clock stop is also a notice), what changed
and why, a word diff, the intent contract with its checks, and the run folder. The stages' progress
goes to stderr line by line while the run goes. With `--json` stdout carries exactly one JSON
object and notices stay on stderr. GEPA's own progress goes to `gepa.log` in the run folder, never
to stdout.

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
folder is not the default). A cancelled run keeps its folder: `--resume` continues it, and the
calls it had answered are replayed from its cache for free.

## JSON output

The keys of the object on stdout, in order:

- finished run: `status`, `prompt`, `verified`, `stop`, `changes`, `reason`, `reason_code`, `diff`, `contract`, `score_before`, `score_after`, `search_score_before`, `search_score_after`, `noise`, `margin`, `length_ratio`, `calls_used`, `run_dir`, `mode`, `elapsed_s`, `meaning`, `verified_text`, `margin_text`
- dry run: `status`, `plan`, `scenarios`, `synthesised`, `holdout`, `valset`, `dataset`, `calls_before_search`, `calls_after_search`, `search_calls`, `iteration_cost`, `iterations`, `iterations_best`, `search_clock_s`, `final_clock_s`, `refusal`, `keeps_original`, `tier`, `workers`, `efforts`, `rewrites`, `stages`, `est_calls`, `est_seconds`, `ungated`
- dry run plan: `models`, `strictness`, `budget`, `wall_clock_s`, `allow_growth`, `merge`, `seed`, `tier`, `workers`, `efforts`
- error: `status`, `code`, `error`, `run_dir`
- clean: `status`, `removed`, `skipped`

`status` is `improved` or `unchanged` for a finished run, else `dry`, `error`, `cleaned` or `help`.
`reason_code` is one of `improved`, `no_reliable_improvement`, `already_strong`, `no_holdout`,
`no_candidate_beat_seed`, `unconfirmed_out_of_budget`; `stop` is `budget`, `clock` or null (no
search ran, or every stage ran as planned). `mode` is the tier, `quick`, `fast`, `checked` or
`deep`; `verified` is true only for a checked or deep result confirmed on held-out scenarios. In a
quick or fast result `score_before` and `score_after` are taken on the scenarios the rewrite was
picked on, not held out; in a checked result the `search_score_*` keys are. In pairwise mode they are
preference shares: the shares of those scenarios the judge gave to the original and
to the rewrite. In reference mode they are the shares of reference checks passed. In a fast result `noise` is the share of those scenarios where the original's two
runs had a winner, and a rewrite had to lead by more than it. Elsewhere scores are shares of
checks passed, from 0 to 1. Any of them is null when not measured. `elapsed_s` is the run's clock in seconds; `meaning`, `verified_text` and
`margin_text` are the report's lines of those names without their labels, null when it prints
none. A dry run's keys are the same in every
tier, null where they do not apply: the search's numbers (`valset` to `final_clock_s`) in the
quick, fast and checked tiers, `rewrites`, `stages`, `est_calls` and `est_seconds` in deep; each
stage is `name`, `calls` and `seconds`.

## Budget and clock

Every model call counts toward one budget (contract, synthesis, task, judge, reflection), and so
does each retry (a failed call is retried twice); a call answered from the run's disk cache is
free. `--time` is monotonic time (a suspended laptop does not count), carried across `--resume`;
each call's timeout is 300 s or the time left, whichever is shorter, and no call starts after the
deadline. In the quick, fast and checked tiers the whole `--time` is the clock and the default
budget is three times the plan's estimate, room for retries; when either runs out, what has not
passed every gate is dropped and the original is returned (`unconfirmed_out_of_budget`).

In the deep tier fixed costs are reserved before the first call: the contract call, the synthesis
call when there are no examples, two seed runs on the holdout and GEPA's scoring of the original
on the valset; after the search up to 3 finalist runs on the target model and up to 3 contract
checks. The search gets the rest and 75 % of the clock. A deep run refuses to start (exit 2) when
the fixed costs exceed the budget, or when the budget affords fewer than 4 iterations in the worst
case and `--force-low-budget` is not given. If calls or time run out before a finalist is
confirmed, the original is returned. A budget of 100 is very low for GEPA (the paper's runs used
400 to 7,000 rollouts); the first few rewrites carry most of the gain.

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
- The judge never sees a candidate prompt when it compares or scores: in the fast and checked
  tiers' pairwise mode it sees your original prompt as the request and two anonymous answers;
  reference mode supplies scenario inputs, outputs and reference checks. When it scores
  outputs against the checklist, a pass counts only with a verbatim quote found in the output. The
  final contract check sees the candidate and can only reject it. The judge is never the task or
  the target model.
- Text a model wrote is stripped of terminal escape sequences and control characters when printed.
- Every model call is `claude -p --safe-mode --settings '{"outputStyle":"default"}' --tools ""
  --strict-mcp-config --disable-slash-commands --no-session-persistence --max-turns 1 ...` (SPEC
  R18, ADR-009), with user text on stdin, a scrubbed environment and a 300 s timeout per call. A
  call ends the run with exit 4 when its session reports tools, MCP servers, skills, slash
  commands, agents beyond the four built-in ones, or an output style other than the default; the
  installed plugins it lists are not active under `--safe-mode` and do not count. The check runs
  on every call; it refused a live call once and has passed live since the fix (see "Status").

## Use it inside Claude Code

This repository is also a Claude Code plugin whose mod (`hooks/register.js`, ADR-010) adds
`/improve` and its alias `/optimize`. It needs Claude Code 2.1.287 or later (the first with mods),
`uv` on the PATH Claude Code runs with, and `claude` logged in to your subscription. Install it
from the repository, then run
`/reload-plugins` in a running session:

```
claude plugin marketplace add ItsUkrainianCat/intentopt
claude plugin install autoimprover@intentopt
```

For one session from a checkout instead: `claude --plugin-dir <path to the checkout>`.

| Command | What it does |
|---|---|
| `/improve <prompt>` | improve the prompt (it may span lines); CLI flags such as `--time 45s` or `--strictness conservative` go before it, and a prompt that starts with `--` goes after a lone `--` |
| `/improve --file PATH` | the prompt from a file; a relative path is read from the session's folder |
| `/improve --dry <prompt>` | the plan only: no model call, nothing written |
| `/improve --resume [ID]` | continue the last run, or the run with that id |
| `/improve clean [ID]` | remove the last run's folder, or that run's |
| `/improve cancel` | stop the run in progress; it stays resumable |

`/optimize` takes the same. One run at a time per session; `cancel` works while Claude is busy.
Without `--time` a run is the 30-second fast tier.

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
  draft is kept and the pane says so. A result that is not verified on held-out scenarios (a quick
  or fast check, or `--trust-search`) goes in too, and the toast and the pane say "NOT verified"
  ("fast check" for the quick and fast tiers). A kept original, a failure, a cancel or a plan
  (`--dry`) puts nothing in the box.
- The run's JSON object becomes a pane: the result and reason, what happened to the prompt box,
  the scores, the margin over the noise, the length ratio, what changed and the improved prompt.
  "Use it" puts the improved prompt into the box again, replacing what is there ("Use it (not
  verified)" for a result that is not verified), "Copy" copies it, "Close" closes the pane. Where
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

Verified live on 2026-10-05: the mod loads, `/improve --dry` prints the plan, a real `/improve`
starts the tool, and a failed run shows its error pane with the resume line. Not verified until
your next live run: the mod is type-checked against the declarations of Claude Code 2.1.287 and its
tests (`tests/mod/*.test.ts`, run by `claude plugin test`) use a stubbed engine. Whether it gets a
model id `--target-model` accepts from every session, kills the child on `cancel`, puts an
improved prompt into the box, and how the result pane looks, are checked only by running it.

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
needs it), `--baseline naive|none`, the run's own model, effort, `--workers` and `--strictness` flags
(`--task-model`, `--judge-model`, `--reflect-model`, `--target-model`, `--effort` and the
per-role efforts: a weak `--target-model claude-haiku-4-5-20251001` leaves more room for a
rewrite to help), `--ungated` (a measuring aid for the fast and checked tiers: returns the
best-ranked candidate without a shown win; results are not verified), `--seed S` (0 to 999, for
the bench's own calls), `--json`, `--dry`, and
`bench` comes first. It spends subscription calls: at `30s` the
plan is about 17 calls per run plus 17 per comparison (13 more with the naive baseline), so the
whole set is at most 680 calls in about 21 minutes. The summary (text, or one JSON object with
`--json`) gives the improved rate, wins, ties and losses with the win rate among improved prompts,
the baseline's, median and 90th percentile seconds per run, total calls and a row per prompt: ids,
codes and numbers, never a prompt. `contract_violations` is `null`: the pipeline never returns a
rewrite its contract check vetoed, and the pairwise judge sees answers, not prompts, so the bench
has no count of its own yet. The bench exits 0 whatever the result (it measures, it does not
gate); 2 for bad usage or a plan that would refuse, 3 when every run failed, 130 on Ctrl-C: then
stdout stays empty (with `--json` it holds the error object), the partial summary of the prompts
already measured follows the `error:` line on stderr, and its JSON object is saved as
`bench/<id>/summary.json` (`"interrupted": true`). Each prompt's run folder, with the bench's own
calls in its cache, is under `$XDG_STATE_HOME/autoimprover/bench/<id>/<prompt id>/`; `clean` does
not remove these, so delete `bench/<id>/` by hand when done. Reported live benchmark runs and their limitations are recorded in [RESEARCH.md](RESEARCH.md).

## Limits and non-goals

- Single turn, no tools: for a `task` prompt the output scored is the model's answer or plan.
- The evidence is thin by design in the short tiers: a fast check is judged on 2 to 4 scenarios,
  the same ones it picks on, with a noise estimate from two runs; the checked tier uses a small held-out set; reference mode scores two original runs there
  for noise, while pairwise mode has no held-out noise estimate; the deep holdout has 3 to 6 scenarios, so
  its noise estimate is coarse. High scores can also mean weak checks, so the report lists them.
- Scoring runs in the fast and checked tiers ask for answers of at most 120 words, so a prompt
  whose value shows only in long answers is measured on short ones.
- Model aliases are pinned to the ids above. One run per run folder; runs with different ids may
  run at the same time.
- Not in v1 (SPEC section 3): training model weights, multi-prompt pipelines or DSPy programs, the
  local model as task model, a hosted service, any network call except `claude -p`.

## Development

Requirements: [SPEC.md](SPEC.md); design: [ARCHITECTURE.md](ARCHITECTURE.md), [ADRs](adr/). `just check` (format,
lint, types, tests) is the definition of done; tests use the fakes in `tests/fakes.py` and never a
real model or the network. `just bench` runs the bench; `just smoke` is a live check on one
prompt (both spend subscription calls). The 0.1.0 script is the git tag `v0.1.0-legacy`.
