# IntentOpt

**Intent-preserving prompt optimization, evaluated against examples.**

IntentOpt is an experimental Python CLI for improving a single prompt while preserving its
constraints, literals and intended task. It generates candidate rewrites, runs them against
scenarios, and returns the original when the configured checks cannot support a change.

With reference examples, it can infer missing rules and score agreement with expected outputs.
Without references, it uses paired output comparisons by a separate judge model. Longer runs
can check a candidate on held-out examples. The `verified` label means that the configured
held-out check passed; it is not a guarantee of generalization or safety.

The project display name is **IntentOpt**. The package and executable remain `autoimprover` for
compatibility. This is a development build, not a stable release or a claim of state-of-the-art
performance.

## Try it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From this checkout:

```sh
# Inspect a plan without making model calls or writing a run folder.
uv run --frozen autoimprover --dry "Summarize meeting notes in five bullet points."

# Inspect the supplied reference-task benchmark without running it.
uv run --frozen autoimprover bench --prompts bench/examples-set.jsonl --time 2m --dry
```

Live runs require the `claude` CLI authenticated to an account with access to the selected
models. They consume subscription usage. Inspect the plan and choose a budget before running:

```sh
uv run --frozen autoimprover --file prompt.txt --examples examples.jsonl --time 2m
```

Prompts, examples and model replies are saved in local run folders and sent to the configured
model backend during a live run. Use public or appropriately authorized data. See the
[CLI reference](docs/USAGE.md) for options, model configuration, storage and cleanup.

## Research and evidence

The research question is whether example-guided rewriting and conservative acceptance gates
can improve task performance without changing user intent. The implementation includes
reference scoring, noise checks, held-out evaluation, categorical example balancing,
reflection rounds, a GEPA deep-search tier, and a resumable call cache.

The repository contains a scripted-model test suite and three synthetic policy benchmark tasks.
The build log reports a small live experiment with 34/48 hidden examples passed by returned
prompts versus 23/48 by originals. Those numbers describe one historical run on three tasks;
raw live-run artifacts are not published here, and subsequent changes have not been validated
by that experiment. They are a pilot observation, not an independently reproducible result.
See [research evidence, protocol and limitations](docs/RESEARCH.md).

## Support the work

Useful support includes independent evaluations, additional task datasets, reviews of failure
cases, and funding for repeated model evaluations and artifact curation. The
[research roadmap](docs/ROADMAP.md) gives concrete milestones and completion criteria.
Use repository issues to discuss a scoped contribution or potential sponsorship; no funding
program or delivery commitment is implied.

## Explore and contribute

- [CLI reference and Claude Code integration](docs/USAGE.md)
- [Research methodology and evidence](docs/RESEARCH.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)
- [Specification](docs/SPEC.md), [architecture](docs/ARCHITECTURE.md), and [design decisions](docs/adr/)
- [Development record](docs/BUILD-LOG.md) — historical notes, not a substitute for published artifacts

Run `just check` for formatting, lint, type checks and the offline test suite. `just smoke` and
`just bench` make live model calls and are separate from that check.
