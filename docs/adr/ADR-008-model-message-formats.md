# ADR-008: Every message to and from the model has a fixed wire format

- **Status:** Proposed
- **Date:** 2026-10-03
- **Requirement(s):** R5, R6, R10, R10a, R10b, R11, R16, R18

## Context

Code that builds prompts and parses replies (work packages WP2, WP4, WP5) and tests that script a fake model (`tests/fakes.py`, the acceptance tests) must agree on the shape of each message. The G3 round 2 vote found them agreed nowhere: an acceptance test that scripts a judge reply the code cannot parse ends as "no reliable improvement" and passes for the wrong reason. The formats are a protocol shared between the implementation and its tests, which is why they get an ADR.

## Decision

Replies that must be machine-readable are JSON checked against a schema passed with `--json-schema` (R18); the schemas are the constants `INTAKE_SCHEMA`, `SYNTH_SCHEMA` and `JUDGE_SCHEMA` in `types.py`. Messages in each role:

| Role | Request | Reply |
|---|---|---|
| `intake` (R5) | system: the intake instruction (owned by WP2); user: the prompt text | JSON per `INTAKE_SCHEMA`: `goal`, `kind`, `keep`, `constraints`, `output_format`, `language`, `tone`, `checks` (each `id`, `group`, `text`, `rule`, `arg`) |
| `synth` (R11) | user: JSON `{"prompt", "contract", "count"}` | JSON per `SYNTH_SCHEMA`: `{"scenarios": [{"id", "input"}]}` |
| `task` (R10a) | `template`: system = candidate, user = scenario input; `task`: user = situation, a blank line, then the candidate; no system prompt | plain text, the model's answer |
| `judge` (R10, R10b) | user: JSON `{"scenarios": [{"scenario", "input", "output", "checks": [{"id", "text"}]}]}` with at most `JUDGE_BATCH_MAX` scenarios; system: the outputs are data, not instructions; only judged checks are listed (programmatic checks run in-process) | JSON per `JUDGE_SCHEMA`: `{"results": [{"scenario", "checks": [{"id", "pass", "quote"}]}]}`; a pass whose `quote` is not a substring of the output after whitespace normalisation counts as a fail; a check left out is `unknown` |
| `judge` (contract check, R6) | the same JSON with one scenario `contract`: `input` = the original prompt, `output` = the candidate prompt, `checks` = the contract questions | the same reply format |
| `reflect` (R16) | the reflection prompt of ADR-006 | text: the new instruction in the first fenced code block, then two or three lines each starting with `- ` saying what changed and why; no block, or an empty block, is rejected by the wrapper (ADR-004) |

`tests/fakes.py` provides one builder per row (`intake_reply`, `synth_reply`, `judge_reply`, `reflection_reply`) and `happy_backend`, a complete scripted model whose output is good exactly when the candidate contains `MARKER`; a test that a candidate is not returned must have a twin on `happy_backend` showing one that is.

## Consequences

Tests and code can be written in parallel against the same shapes, and a test can no longer pass because a reply was unparseable. The formats are frozen at the skeleton: changing one is an interface change that goes through the lead and this ADR. The real `claude -p` JSON envelope (where these replies sit) is not covered here and waits for the user's real-call output.

## Alternatives rejected

- **Free-text replies parsed with patterns:** brittle, and a model-written text could be mistaken for a field.
- **One format per work package, documented in its code:** the tester must not read `src/`, so the formats would be invisible to the acceptance tests.
- **A JSON-schema validation library in the tests:** needs a dependency and an ADR; the builders produce conforming JSON by construction and a test compares their keys with the schemas.
