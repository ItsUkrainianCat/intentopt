# Research roadmap

These are proposed milestones, not funded commitments or promised dates. Each is complete only
when its evidence can be inspected. The immediate priority is reproducible evaluation rather
than a stronger performance claim.

| Milestone | Work | Completion evidence |
|---|---|---|
| Reproduce the current reference pipeline | Repeat the three policy tasks after categorical balancing; record exact model and run configuration | Public sanitized run artifacts, recomputable scores, per-task outcomes and run-to-run variation |
| Expand evaluation coverage | Add independently designed tasks and boundary cases, review references and duplicate leakage | Versioned dataset, documented split policy, reference review and leakage audit |
| Compare methods fairly | Original, one-call rewrite and gated optimizer under declared budgets; ablate induction, balancing and reflection | Predeclared analysis, repeated runs, per-task comparisons, failures and cost/time tradeoffs |
| Audit intent preservation | Human review of changed prompts and failed contract checks | Review rubric, anonymized decisions, measured violation rate and documented disagreements |
| Test live operational behavior | Checked and deep tiers, cancellation/resume and Claude Code integration | Sanitized smoke-test records with environment, commands and failures |
| Improve backend portability | Assess another backend through the existing backend seam | Design decision, compatibility tests and a limited live evaluation without hiding backend differences |
| Prepare a stable release | Resolve documented gaps, verify install paths and compatibility, align public documentation | Passing required checks, release notes, reproducible install instructions and explicitly scoped support |

## Where support helps

Independent evaluators can challenge the hidden-task protocol and judge behavior. Dataset
contributors can provide permissively shareable tasks with reviewed references. Engineering
contributors can improve failure handling, evidence exports and backend compatibility.

Financial support could fund repeated model evaluations, human review and artifact preparation.
A sponsorship discussion should define a scope, budget and deliverable before work starts;
there is no implied sponsor program, guaranteed result or paid endorsement. Sponsorship should
not determine which results are published.

Open an issue to propose a task or scoped support. Include what evidence would convince you
that it is complete. See [CONTRIBUTING.md](../CONTRIBUTING.md) and
[RESEARCH.md](RESEARCH.md) for contribution and reporting standards.
