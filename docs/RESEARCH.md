# Research notes

## Question and scope

Can example-guided prompt rewriting improve a task's measured performance while preserving
its intended goal, constraints and literals? IntentOpt explores this question in a single-prompt,
single-turn setting. It does not train model weights or establish a general prompt-optimization
benchmark result.

There are two different measurement settings:

- **Reference-guided:** every example supplies an expected answer or explicit criteria. Pick
  examples can inform rewrite generation; a judge scores outputs against those references.
- **Preference-guided:** outputs are compared anonymously against the original, in both answer
  orders. Agreement across orders reduces position bias; it does not remove judge-model bias.

The deep tier integrates GEPA through the pinned `gepa==0.1.4` dependency. Integration with an
existing search method is not a claim to have invented it. Design decisions are recorded in
[the ADRs](adr/); behavior and intended acceptance measures are in [SPEC.md](SPEC.md).

Related method: Agrawal et al., [GEPA: Reflective Prompt Evolution Can Outperform
Reinforcement Learning](https://arxiv.org/abs/2507.19457). The results of that paper
belong to its authors and do not establish performance for this implementation.

## Evidence inventory

| Evidence | Available in this repository | What it supports |
|---|---|---|
| Python source and pinned dependency lock | `src/autoimprover/`, `pyproject.toml`, `uv.lock` | Inspecting and running the implementation |
| Scripted-model and process-fixture tests | `tests/` | Deterministic checks of gates, budgets, failures, resume and report behavior; not real-model quality |
| Synthetic policy tasks | `bench/examples-set.jsonl`, `bench/examples/` | Repeating the benchmark procedure with configured live models |
| Design and development record | [BUILD-LOG.md](BUILD-LOG.md), [SPEC.md](SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md) | Historical implementation decisions and author-reported observations |
| Raw artifacts for the historical live runs below | Not published in this repository | Independent verification of the reported scores is currently unavailable |

A runnable dataset and procedure are not the same as reproducing a historical score. Model
behavior and availability can change, and tests with scripted replies cannot establish live
quality. The test count in older build-log entries belongs to those commits, not necessarily
the current checkout.

## Historical pilot observations

These figures are transcribed from the development record, not recomputed from public raw
outputs. They should be cited as author-reported pilot observations. Date labels follow the
build log, which is not a formal experiment registry.

| Recorded experiment | Reported observation | Limits |
|---|---|---|
| Ungated preference benchmark, 20 prompts, 90 s (WP21 context) | Best candidates: 4 wins, 9 ties, 7 losses; naive baseline: 1 win, 4 ties, 15 losses | Weak and mixed preference signal; no demonstrated general advantage |
| After WP22, 2026-10-07 00:30, three reference tasks with 10 hidden examples each | Originals 21/30; returned prompts 24/30. Refund improved 4 to 7; the other two prompts were retained | One run; small sample; original scores varied across runs |
| After WP24, 2026-10-07 15:30, three tasks with 16 hidden examples each | Originals 23/48; returned prompts 34/48; naive baseline 28/48, computed from the local cache in the historical account | One run; three tasks; unpublished cache; later WP25 split changes are not evaluated by these figures |

For the WP24 observation, reported original / tool / naive hidden passes were priority
8 / 15 / 12, refund 8 / 12 / 4, and escalation 7 / 7 / 12, each out of 16. The tool retained the
original escalation prompt. The reported median optimizer duration was 77.8 seconds. These
numbers cannot establish statistical significance, cross-domain performance or a durable
advantage over the naive baseline. The two reference experiments used different datasets and
must not be combined into a trend or pooled estimate.

WP25 subsequently balanced categorical pick examples by label. The historical WP24 experiment
therefore does not validate the current split behavior. An updated, repeated evaluation remains
an open milestone.

## Evaluation protocol for a publishable result

1. Record the repository commit, dependency lock, dataset hashes, command, time and call budgets,
   role model IDs, efforts, worker count, baseline, date and backend version. A seed documents
   the benchmark's sampling configuration; it does not guarantee deterministic provider outputs.
2. Fix the dataset and analysis plan before testing. For a benchmark item with `eval_from`, the
   optimizer receives only the examples before that index. The remaining examples are hidden
   from generation and used by the external benchmark. Do not adapt rules after inspecting
   hidden outputs and then reuse the same split as evidence of generalization.
3. Keep the optimizer's internal pick/held-out split separate from the external benchmark
   split. Only pick examples enter induction and reflection. Check split leakage explicitly,
   including duplicated and near-duplicated examples.
4. Compare the original, the gated optimizer and a one-call rewrite baseline on the same hidden
   inputs and target model configuration. Treat an unchanged returned prompt as a distinct
   outcome. In preference mode, judge both answer orders; in reference mode, report agreement
   with references and retain the failed checks.
5. Repeat across runs and more independently designed tasks. Predeclare repetition count and
   primary metric. Report per-task results and variation across runs; account for task and
   example dependence when estimating uncertainty. Do not treat all examples as independent
   evidence for broad generalization.
6. Report regressions, unchanged rate, contract violations from independent review, calls,
   elapsed time, failures and interruptions as well as aggregate gains. Audit a sample of
   judge decisions with human review; judge agreement is a measurement, not ground truth.
7. Publish a privacy-reviewed artifact bundle: manifest, sanitized outputs and judge decisions,
   metric calculation and exact commands. Exclude credentials and private user data; disclose
   any redactions that prevent recomputation. A useful result must allow another person to
   recompute the reported metric from the released artifacts.

Planning commands, which make no model calls:

```sh
uv run --frozen autoimprover bench --prompts bench/examples-set.jsonl --time 2m --baseline naive --dry
uv run --frozen autoimprover bench --prompts bench/prompts.jsonl --time 90s --baseline naive --dry
```

Removing `--dry` runs the live experiment and consumes subscription usage. Use
[USAGE.md](USAGE.md) for result files, flags and local cleanup. Raw run directories contain
prompts, examples and model replies; review them before sharing.

## Limitations and interpretation

- Reference labels and criteria can be wrong or incomplete. Synthesized scenarios can miss the
  task's difficult cases, and induction can infer incorrect boundary rules.
- The current synthetic policy suite covers three tasks. It is insufficient evidence for
  performance across writing, coding, science or other task families.
- Short-tier scoring appends a 120-word answer request. This can change performance for tasks
  whose value depends on longer output. The suffix is never part of the returned prompt.
- Noise estimates based on two original runs are coarse. Multiple candidates and reflection
  rounds increase selection pressure; passing a small internal holdout is not a statistical
  guarantee or an externally audited result.
- The contract check is model-mediated and can miss changes in meaning. Literal preservation
  is narrower than semantic preservation. Independent human review is still needed.
- A distinct judge model reduces direct role overlap but does not establish independence from
  shared model-family biases. Pairwise preference and reference agreement measure different
  things and should not be reported as interchangeable scores.
- `--ungated`, quick runs and fast runs do not provide held-out verification. The `verified`
  flag records a configured check, not a guarantee about unseen real-world use.
- Live checked, deep, resume and integration coverage is narrower than scripted test coverage.
  Historical acceptance targets in SPEC section 5 are targets, not achieved public results.

See [ROADMAP.md](ROADMAP.md) for the work required to strengthen the evidence.
