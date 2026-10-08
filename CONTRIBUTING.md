# Contributing to IntentOpt

IntentOpt is an experimental research tool. Contributions should make its behavior easier to
inspect, its evaluation more reliable, or its limitations clearer. The Python package and
executable currently remain `autoimprover`.

## Before changing behavior

Read [the specification](docs/SPEC.md), [architecture](docs/ARCHITECTURE.md), and relevant
[design decisions](docs/adr/). Open an issue for a substantial change with the problem,
proposed behavior and evidence that would show it works. Small corrections can go directly
to a pull request.

## Local checks

Use Python 3.12+, `uv` and `just`. The `types` recipe also needs `pyright` on PATH.

```sh
uv sync --frozen
just check
```

`just check` runs formatting, lint, type checks and scripted-model tests. Tests use fixtures
and fakes, not live model calls. Keep that property: a normal test run must not consume account
usage or require credentials. For a behavior change, add a focused regression test at the
relevant boundary. Avoid model-specific assumptions in deterministic tests.

`just smoke` and `just bench` are separate live checks that consume subscription usage. Use
`--dry` to inspect a plan first. Contributors are not expected to pay for live evaluations
merely to submit a patch. Describe any missing live validation in the pull request.

## Pull requests

Explain the concrete problem and resulting behavior, list the relevant checks and disclose
remaining limits. Update docs when a public flag, report field, model role, acceptance gate or
storage format changes. Preserve CLI and run-folder compatibility unless a migration is
explicitly designed and documented.

Keep prompts, examples and model outputs as data: never execute them or interpolate them into
shell commands. Do not weaken budgets, deadlines or intent checks to obtain a better benchmark
number. Never add credentials, private prompts, local run caches or account/session files.

## Datasets and reported results

Share only data you have permission to publish and explain provenance, intended task,
reference construction and license. Review examples for private information. Keep hidden
benchmark examples out of generation and distinguish the external benchmark split from the
optimizer's internal holdout.

Follow [the research protocol](docs/RESEARCH.md) when reporting live results. Include the exact
commit, commands, dataset version, models and budgets; distinguish reproducible released
artifacts from personal observations. Report regressions and unchanged outcomes alongside
wins. A test passing with fake replies is implementation evidence, not a performance result.
