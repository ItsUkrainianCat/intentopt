# ADR-010: `/improve` is a Claude Code mod that runs the CLI

- **Status:** Accepted 2026-10-05 (the user asked for a mod, not a plugin, and approved the plan `whimsical-napping-marshmallow`; SPEC R19 and R21 amended, the user confirms)
- **Requirement(s):** R19, R21, R22, R23

## Context

`commands/improve.md` made Claude run the tool through its sandboxed Bash: `claude -p` returned "Not logged in" there, a foreground call stops after 10 minutes and a background one after 30 unless given more, state had to be pointed at `$TMPDIR`, and the prompt travelled through the model, which had to be told to treat it as data. Claude Code 2.1.287 (installed here) ships mods: plugins whose JavaScript hooks run inside Claude Code and can register commands, start processes outside the sandbox, draw panes and fill the prompt box (docs: code.claude.com/docs/en/plugins/mods).

## Decision

- A mod in this repository's plugin (`.claude-plugin/plugin.json`, `hooks/hooks.json` with `modules`, `hooks/register.js` and small modules) registers `/improve` and `/optimize` in `session.start` and handles them in `command.run`. It is a thin front end: the Python CLI stays the only implementation of the behaviour.
- Subscriptions are limited to `session.start`, `command.run` (for `improve` and `optimize`) and `ui.render` (matcher `{component: Pane, requestId: improve}`, for the mod's own pane); no `prompt.submit`, no `$.model.complete`, no `$.mcp`. `claude plugin validate` lists these hooks and the API calls the mod makes: `command.register`, `env.get` (HOME, TMPDIR, XDG_CACHE_HOME), `fs.stat`, `fs.write`, `process.run`, `process.spawn`, `prompt.fill`, `session.cwd`, `session.model`, `session.surfaces`, `store.get/set/delete`, `ui.close/copy/invalidate/log/open/resolve/status/toast`. `tests/test_mod_package.py` pins the calls found in the source and `just mod-check` runs validate and `claude plugin test`. The plugin root is the repository (the CLI sources must ship with the plugin), so validate always warns that `CLAUDE.md` at the plugin root is not loaded as context; that warning is accepted, `--strict` is used on the marketplace file only.
- The child is started by an argument list; the prompt goes through a 0600 file only; `UV_PROJECT_ENVIRONMENT` points outside the plugin cache; the child's stdout is the one JSON object of R2.
- `commands/improve.md` and `commands/optimize.md` are removed (a mod command with a taken name fails to register).
- Tests: `claude plugin test` (`*.test.ts`) for argument parsing, argv building, report rendering, cancel, missing `uv`, not-logged-in; pytest drift tests (plugin version equals `pyproject.toml`, `hooks.json` module path exists, no hardcoded home path, the install tree holds `src/` and `uv.lock`).

## Amendment 2026-10-05 (user request)

An improved prompt is put into the prompt box automatically (`$.prompt.read` first: a draft the user already typed is never overwritten), so the next step is the user pressing Enter or editing; the pane keeps Use it, Copy and Close.

## Consequences

The login and sandbox problems disappear (the child is outside the sandbox, as any process a mod starts), nothing relays the prompt through the model, progress and review are real UI, and runs can last the full 45 minutes. The cost: a mod runs with the user's full permissions and unsandboxed, so the trust surface is the mod's code (small, listed by `validate`, public in the repository); mods do not draw in `claude -p` or the VS Code chat panel (the command's text is the fallback) and are off under `--safe-mode`, which our backend uses, so the model-calling child never loads the mod. The mods API can change between Claude Code releases; the README names the tested version.

## Alternatives rejected

- **Plain plugin with a markdown command:** sandbox login failure, Bash time limits, prompt through the model.
- **A `prompt.submit` hook that improves every prompt:** a run costs about 100 calls and up to 45 minutes, and it would drift the user's intent unasked.
- **An MCP server:** a 45-minute tool call fights tool timeouts and gives no UI.
