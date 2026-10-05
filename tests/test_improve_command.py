"""The `/improve` slash command and its alias `/optimize`, and the README, read as text (SPEC R19,
R21; R2 for the exit codes). Each R21 element is pinned to a phrase of `commands/improve.md`, the
alias must say the same, and every flag, exit code, JSON key and the `--dry` example the README
shows must match the code and SPEC R2, so the documents cannot drift from what the tool does."""

import json
import re
from pathlib import Path

import pytest

from autoimprover import cli, report
from autoimprover.types import (
    EXIT_BACKEND,
    EXIT_INTERNAL,
    EXIT_INTERRUPTED,
    EXIT_NOT_LOCKED_DOWN,
    EXIT_OK,
    EXIT_USAGE,
    WALL_CLOCK_DEFAULT_S,
    Outcome,
)

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
SPEC = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
# The longest a background Bash call of Claude Code may run (its `timeout` ceiling, in ms).
BACKGROUND_MAX_MS = 7_200_000
TOOL = 'uv run --frozen --project "$HOME/projects/optimizer" autoimprover'
STATE = 'export XDG_STATE_HOME="${TMPDIR:-/tmp}/autoimprover-state"'
# A value for each README flag that takes one; parse() checks only these three.
SAMPLE = {"--kind": "task", "--strictness": "balanced", "--budget": "100"}


def command(name: str) -> tuple[dict[str, str], str]:
    """The frontmatter fields and the body of `commands/<name>.md`."""
    text = (ROOT / "commands" / f"{name}.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{name}.md has no frontmatter"
    head, body = text[4:].split("\n---\n", 1)
    return dict(line.split(": ", 1) for line in head.splitlines()), body


def tool_line(body: str, *, dry: bool) -> str:
    """The one line that starts the tool on the prompt file, with or without `--dry`."""
    lines = [
        line.strip()
        for line in body.splitlines()
        if f'{TOOL} {"--dry " if dry else ""}--file "DIR/prompt.txt"' in line
    ]
    assert len(lines) == 1, f"expected one {'dry' if dry else 'real'} run line, got {lines}"
    return lines[0]


# --- R21: what /improve must do, each element pinned to its phrase -------------------------------

R21_PHRASES = {
    "prompt written with the Write tool": r"Use the Write tool to write the prompt to "
    r"`DIR/prompt\.txt`",
    "prompt never in a shell line": r"Never put the prompt text into a Bash command",
    "private folder under TMPDIR": r'`mktemp -d "\$\{TMPDIR:-/tmp\}/autoimprover-prompt\.'
    r'XXXXXX"`',
    "background call": r"`run_in_background: true`",
    "run folder and state folder printed": r"print the run folder \(the report's `run folder:` "
    r"line\) and the `XDG_STATE_HOME=` line",
    "session model as target": r"MODEL is the exact model id you are running as",
    "dry run above 2,000 chars": r'Run `wc -m < "DIR/prompt\.txt"` with Bash\. If it prints more '
    r"than 2000, run this dry run first",
    "wait for the user's yes": r"Do not act on it until the user answers yes\.",
}


@pytest.mark.parametrize("phrase", R21_PHRASES.values(), ids=R21_PHRASES.keys())
def test_improve_states_each_r21_step(phrase):
    _, body = command("improve")
    prose = " ".join(body.split())  # a phrase may wrap across lines
    assert re.search(phrase, prose), f"commands/improve.md lacks: {phrase}"


def test_the_run_passes_the_file_the_state_folder_and_the_target_model_then_deletes_the_file():
    _, body = command("improve")
    run = tool_line(body, dry=False)
    assert run.startswith(f'{STATE}; echo "XDG_STATE_HOME=$XDG_STATE_HOME"; {TOOL} --file ')
    assert f'{TOOL} --file "DIR/prompt.txt" --target-model MODEL >' in run
    tool_at, delete_at = run.index(TOOL), run.find('; rm -f -- "DIR/prompt.txt";')
    assert delete_at > tool_at, "the prompt file is not deleted after the run in the same call"


def test_the_dry_run_uses_the_same_state_folder_and_target_model():
    _, body = command("improve")
    assert tool_line(body, dry=True) == (
        f'{STATE}; {TOOL} --dry --file "DIR/prompt.txt" --target-model MODEL'
    )


def test_the_background_timeout_is_above_the_wall_clock_and_within_the_ceiling():
    _, body = command("improve")
    found = re.findall(r"`timeout: (\d+)`", body)
    assert len(found) == 1, found
    assert WALL_CLOCK_DEFAULT_S * 1000 < int(found[0]) <= BACKGROUND_MAX_MS


def test_the_prompt_reaches_the_command_once_and_never_a_shell_line():
    """SPEC R19: `$ARGUMENTS` is substituted once, between marker lines, outside every shell line;
    no `!` pre-run shell and no positional `$1` that Claude Code would also substitute."""
    _, body = command("improve")
    assert body.count("$ARGUMENTS") == 1
    assert "\n<<<PROMPT\n$ARGUMENTS\nPROMPT>>>\n" in body
    assert "!`" not in body and not re.search(r"\$\d", body)


@pytest.mark.parametrize("name", ["improve", "optimize"])
def test_the_frontmatter_values_are_plain_yaml_scalars(name):
    """A `: ` or ` #` inside an unquoted value breaks the YAML frontmatter Claude Code reads."""
    fields, _ = command(name)
    assert all(": " not in value and " #" not in value for value in fields.values())


def test_the_frontmatter_describes_the_command_and_allows_only_its_tools():
    fields, _ = command("improve")
    assert fields["description"] and fields["argument-hint"] == "<prompt to improve>"
    tools = [tool.strip() for tool in fields["allowed-tools"].split(",")]
    assert tools == ["Write", "Read", "Bash(mktemp -d:*)", "Bash(wc -m:*)"]


def test_optimize_is_the_same_command_under_another_name():
    improve_fields, improve_body = command("improve")
    optimize_fields, optimize_body = command("optimize")
    assert optimize_body == improve_body
    assert optimize_fields.pop("description").startswith("Alias of /improve - ")
    improve_fields.pop("description")
    assert optimize_fields == improve_fields


# --- the README against the code and SPEC R2 -----------------------------------------------------


def section(title: str) -> str:
    """The README text under `## <title>`, up to the next `## ` heading."""
    start = README.index(f"\n## {title}\n")
    end = README.find("\n## ", start + 1)
    return README[start : None if end == -1 else end]


def readme_flags() -> list[tuple[str, str]]:
    """(flag, metavar or "") for each row of the README's flags table."""
    return re.findall(r"^\| `(--[a-z-]+)(?: ([^`]+))?` \|", section("Flags"), flags=re.M)


def test_every_flag_the_readme_lists_is_parsed_by_the_cli():
    for flag, metavar in readme_flags():
        argv = [flag, SAMPLE.get(flag, "x")] if metavar else [flag]
        assert flag[2:].replace("-", "_") in cli.parse(argv).given, flag


def test_the_readme_lists_every_flag_of_the_help(capsys):
    assert cli.main(["--help"]) == EXIT_OK
    shown = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", capsys.readouterr().out))
    assert {flag for flag, _ in readme_flags()} == shown


def test_the_readme_exit_codes_are_those_of_spec_r2():
    r2 = next(line for line in SPEC.splitlines() if line.startswith("- R2. "))
    listed = r2[r2.index("Exit codes:") : r2.index("Every non-zero exit")]
    spec = {int(code) for code in re.findall(r"(?:: |, )(\d+) [a-z]", listed)}
    readme = {int(code) for code in re.findall(r"^\| (\d+) \|", section("Exit codes"), re.M)}
    codes = {EXIT_OK, EXIT_INTERNAL, EXIT_USAGE, EXIT_BACKEND, EXIT_NOT_LOCKED_DOWN}
    assert readme == spec == codes | {EXIT_INTERRUPTED}


def readme_keys(label: str) -> list[str]:
    line = next(
        line for line in section("JSON output").splitlines() if line.startswith(f"- {label}: ")
    )
    return re.findall(r"`([a-z_]+)`", line.split(": ", 1)[1])


def test_the_readme_json_keys_are_those_the_tool_writes(capsys):
    finished = Outcome(status="unchanged", prompt="p", reason="r", reason_code="no_holdout")
    assert readme_keys("finished run") == list(report.outcome_object(finished, "p", None))
    assert readme_keys("error") == list(report.error_object(EXIT_USAGE, "e", ""))
    assert cli.main(["--dry", "--json", "Summarise the notes."]) == EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    assert readme_keys("dry run") == list(plan)
    assert readme_keys("dry run plan") == list(plan["plan"])
    assert cli.main(["clean", "--json"]) == EXIT_OK
    assert readme_keys("clean") == list(json.loads(capsys.readouterr().out))


def test_the_readme_dry_example_is_the_tool_output(capsys):
    assert cli.main(["--dry", "Summarise the meeting notes."]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("dry run: no model call made, nothing written\n")
    assert f"$ uv run --frozen autoimprover --dry --file prompt.txt\n{out}```" in README
