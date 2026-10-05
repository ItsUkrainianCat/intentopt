# ADR-009: `claude -p` is called with stream-json and checked on every call

- **Status:** Accepted 2026-10-05 (the facts come from the user's real calls, claude 2.1.287; SPEC R18 amended, the user confirms the amendment)
- **Requirement(s):** R18, R17 (timeout), R19

## Context

The user ran the call of R18 in their own terminal (BUILD-LOG, USER STEP 12:00, answered 2026-10-05). Observed:

1. The subscription login works with `--safe-mode` (`apiKeySource: "none"`).
2. `--output-format json` returns one object (`result`, `is_error`, `subtype`, `terminal_reason`, `usage`, `total_cost_usd`, `modelUsage`, `duration_ms`) with nothing about plugins, MCP servers or tools.
3. `--output-format stream-json --verbose` prints JSON lines: an `init` line (`tools`, `mcp_servers`, `skills`, `slash_commands`, `agents`, `output_style`, `plugins`, `model`, `cwd`), `assistant` and `thinking_tokens` lines, a `rate_limit_event`, and last the same `result` object. Fixtures: `tests/fixtures/claude_cli/`.
4. Under `--safe-mode` the `plugins` array still lists every installed plugin (about 70 on the user's machine) although none contributes a tool, skill, agent, command or hook. The R18 rule "no plugins" would abort every run.
5. `--safe-mode` does not neutralise the user's `outputStyle` setting: with `--system-prompt "You are a test."` the model still saw the user's output style (749 input tokens, from a plain terminal too). With `--settings '{"outputStyle":"default"}'` it saw 422 tokens: the SDK preamble, the system prompt, the environment block (cwd, platform, OS, model), the user's e-mail address and the date. Those cannot be removed with a flag and are accepted.

## Decision

- Every call uses `--output-format stream-json --verbose` and `--settings '{"outputStyle":"default"}'` in addition to the flags of R18. One code path: the backend reads the `init` line and the last `{"type":"result"}` line and ignores the rest.
- The lockdown check runs on every call (a superset of "the first live call of every process"): it aborts with `SessionNotLockedDown` (exit 4) unless `tools == []` (on a call with a `--json-schema`: `tools` is `[]` or exactly `["StructuredOutput"]`, the internal tool that returns the parsed answer: seen in the user's first real run, 2026-10-05), `mcp_servers == []`, `skills == []`, `slash_commands == []`, `agents` is a subset of the four built-ins (`claude`, `Explore`, `general-purpose`, `Plan`) and `output_style == "default"`. `plugins` is informational and never checked. A missing or malformed `init` line is also `SessionNotLockedDown`.
- The reply text is the `result` field of the final object; `is_error: true`, a `terminal_reason` other than `completed`, a missing final object or a non-JSON line before it are a `CallError` (retried by the Resilient layer, R24). Tokens come from `usage`, `duration_s` from the clock, not from `duration_ms`.
- The working directory of the child is a fresh empty folder under the run folder (so no project file can be read), the environment is scrubbed as in R18.

## Amendment 2026-10-05

The user's first real `/improve` run showed that the `init` line of a schema call lists `tools: ["StructuredOutput"]`; the first version of the check refused it (exit 4 on intake). `StructuredOutput` is allowed in `tools` on calls that carry a schema and nowhere else.

## Consequences

The first call of a process shows whether the session is locked down on that machine, with no extra call. A user whose `claude` setup adds tools in `--safe-mode` gets exit 4 with the field that failed. The user's e-mail address and the date stay in every model context (accepted: the calls go to the same account). If a later `claude` version renames an `init` field, the check fails closed.

## Alternatives rejected

- **Keep `--output-format json` and check once with a probe call:** one more paid call per process and two code paths.
- **Check `plugins == []`:** aborts every run on any machine with installed plugins.
- **Copy the credentials into a clean `CLAUDE_CONFIG_DIR`:** touches the login file; rejected on principle.
