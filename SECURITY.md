# Security

This is experimental software. Its model-generated checks are evidence about a
particular run, not a security guarantee.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature if enabled for this
repository. If it is unavailable, open an issue requesting a private reporting
channel without including exploit details, credentials, or private run data.
The maintainer handles reports on a best-effort basis; no response deadline is
promised. Development currently targets the latest source revision.

## Data handling

Live runs send prompts, examples, and model responses through the logged-in
Claude CLI to its model service. Treat this as external processing. Do not use
confidential data unless you are authorized to send it to that service.

Run folders contain prompts, examples, cached replies, and logs. They use private
filesystem permissions and are rejected inside Git repositories. Keep these
folders out of issues and benchmark uploads; inspect and redact an export first.
Use the CLI's `clean` command for ordinary runs. Benchmark state has separate
cleanup instructions in the usage guide.

The subprocess backend disables tools and MCP integrations, scrubs its
environment, and checks session capabilities. The Claude Code plugin itself
runs with the user's permissions. These controls do not make untrusted model
output infallible or prevent sensitive data in the input from reaching the model.

## Before publishing

Scan the full Git history, all published branches and tags, and the current
files. A clean scanner result means no supported patterns were detected; it
does not prove that every secret or private datum is absent. Review commit
metadata, fixtures, benchmark data, release files, and Actions artifacts too.

If a credential has ever been committed, revoke or rotate it before cleaning
history. Adding a path to `.gitignore` does not remove already tracked content.
