"""The Claude Code mod that ships `/improve` and `/optimize` (SPEC R19, R21; ADR-010), read as
text: the plugin and marketplace manifests against `pyproject.toml`, the hooks module's trust
surface (the events it hooks, the calls it makes on `$`, the variables it reads, no home path),
the pure modules' tables against the CLI (`FLAGS`, `REASON_CODES`, the run id pattern), and the
README against the code and SPEC R2. No process is started: the mod's behaviour is tested by the
`*.test.ts` files under `claude plugin test`, its static analysis by `claude plugin validate`."""

import json
import re
import tomllib
from pathlib import Path

from autoimprover import cli, report
from autoimprover.types import (
    EXIT_BACKEND,
    EXIT_INTERNAL,
    EXIT_INTERRUPTED,
    EXIT_NOT_LOCKED_DOWN,
    EXIT_OK,
    EXIT_USAGE,
    REASON_CODES,
    RUN_ID_PATTERN,
    Outcome,
)

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
README = (ROOT / "README.md").read_text(encoding="utf-8")
SPEC = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
# The trust surface of ADR-010 as `claude plugin validate` lists it: the events the module hooks
# (a pane is drawn by a `ui.render` hook on its own pane only), the calls it makes on `$`, and the
# environment variables it reads.
EVENTS = ["session.start", "command.run", "ui.render"]
CALLS = {
    "command.register",
    "env.get",
    "fs.stat",
    "fs.write",
    "process.run",
    "process.spawn",
    "prompt.fill",
    "session.cwd",
    "session.model",
    "session.surfaces",
    "store.delete",
    "store.get",
    "store.set",
    "ui.close",
    "ui.copy",
    "ui.invalidate",
    "ui.log",
    "ui.open",
    "ui.resolve",
    "ui.status",
    "ui.toast",
}
ENV_READS = {"HOME", "TMPDIR", "XDG_CACHE_HOME"}
# Flags of the CLI the mod never forwards: its own help, the JSON mode it always sets, and
# `--resume`, which it handles as a subcommand.
NOT_FORWARDED = {"--help", "--json", "--resume"}


def manifest(name: str) -> dict:
    return json.loads((ROOT / ".claude-plugin" / name).read_text(encoding="utf-8"))


def hooks_source(name: str) -> str:
    return (HOOKS / name).read_text(encoding="utf-8")


def js_table(source: str, name: str) -> str:
    """The text of `export const <name> = ...` up to the first line that closes it."""
    match = re.search(rf"^export const {name} = (.*?)^\S", source, flags=re.M | re.S)
    assert match, f"no `export const {name}` in the module"
    return match.group(1)


# --- the manifests -------------------------------------------------------------------------------


def test_the_plugin_manifest_names_the_plugin_and_carries_the_python_version():
    plugin = manifest("plugin.json")
    assert plugin["name"] == "autoimprover"
    assert plugin["version"] == VERSION
    assert plugin["description"]


def test_the_marketplace_lists_this_folder_as_the_plugin_with_the_same_version():
    market = manifest("marketplace.json")
    assert market["name"] and market["owner"]["name"]
    assert [(p["name"], p["source"], p["version"]) for p in market["plugins"]] == [
        ("autoimprover", "./", VERSION)
    ]


def test_the_install_tree_holds_what_uv_run_needs():
    """The plugin is this folder (`source` "./"), and `uv run --frozen --project <root>` needs
    the project, its lock and its code there."""
    for path in ("pyproject.toml", "uv.lock", "src/autoimprover/cli.py", "hooks/hooks.json"):
        assert (ROOT / path).is_file(), path


def test_the_hooks_manifest_loads_the_register_module():
    hooks = json.loads(hooks_source("hooks.json"))
    assert hooks["modules"] == ["./register.js"]
    assert (HOOKS / "register.js").is_file()


def test_the_markdown_commands_are_gone():
    """A mod command whose name a markdown command holds fails to register (ADR-010)."""
    assert not (ROOT / "commands" / "improve.md").exists()
    assert not (ROOT / "commands" / "optimize.md").exists()


# --- the hooks module: trust surface and the static rules of a mod --------------------------------


def test_the_module_hooks_only_its_events():
    events = re.findall(r"\bon\(\s*'([a-z.]+)'", hooks_source("register.js"))
    assert events == EVENTS


def test_the_module_calls_only_the_expected_api():
    calls = set(re.findall(r"\$\.([a-z]+\.[a-zA-Z]+)\(", hooks_source("register.js")))
    assert calls == CALLS
    names = set(re.findall(r"\$\.env\.get\('([A-Z_]+)'\)", hooks_source("register.js")))
    assert names == ENV_READS


def test_the_module_never_submits_prompts_calls_a_model_or_an_mcp_server():
    source = hooks_source("register.js")
    for word in ("prompt.submit", "$.model", "$.mcp", "$.http", "$.agent", "$.tool", "env.set"):
        assert word not in source, word


def test_every_module_is_an_es_module_without_dynamic_imports_or_a_home_path():
    for path in sorted(HOOKS.glob("*.js")):
        source = path.read_text(encoding="utf-8")
        assert "import(" not in source and "require(" not in source, path.name
        assert not re.search(r"/home/|/Users/|projects/optimizer", source), path.name
        imports = re.findall(r"^import\b[^;]*?\bfrom '([^']+)'$", source, flags=re.M)
        assert all(re.fullmatch(r"\./[a-z]+\.js", target) for target in imports), imports
        assert len(imports) == len(re.findall(r"^import\b", source, flags=re.M)), path.name


def test_only_the_register_module_touches_the_engine():
    """Pure modules take no `$` and make no call on it (the mods' static rule)."""
    pure = [p for p in sorted(HOOKS.glob("*.js")) if p.name != "register.js"]
    assert pure, "the pure modules are missing"
    for path in pure:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"(?<![\w$\\])\$(?=\s*[.,)])", source), path.name
        assert "export function register" not in source, path.name


# --- the pure modules against the CLI -------------------------------------------------------------


def test_the_forwarded_flags_are_those_the_cli_parses_with_the_same_arity():
    actions = [a for a in cli._parser()._actions if a.option_strings]
    parsed = {
        next(s for s in a.option_strings if s.startswith("--")): 0 if a.nargs == 0 else 1
        for a in actions
    }
    table = js_table(hooks_source("args.js"), "FLAGS")
    flags = {flag: int(arity) for flag, arity in re.findall(r"\['(--[a-z-]+)', ([01])\]", table)}
    assert flags == {f: n for f, n in parsed.items() if f not in NOT_FORWARDED}


def test_the_reason_codes_of_the_view_are_those_of_the_tool():
    table = js_table(hooks_source("report.js"), "REASON_CODES")
    assert tuple(re.findall(r"'([a-z_]+)'", table)) == REASON_CODES


def test_the_run_id_pattern_is_the_tool_s():
    match = re.search(r"^export const RUN_ID = /\^(.*)\$/$", hooks_source("args.js"), flags=re.M)
    assert match and match.group(1).replace("(?:", "(") == RUN_ID_PATTERN


# --- the README -----------------------------------------------------------------------------------


def section(title: str) -> str:
    """The README text under `## <title>`, up to the next `## ` heading."""
    start = README.index(f"\n## {title}\n")
    end = README.find("\n## ", start + 1)
    return README[start : None if end == -1 else end]


def test_the_readme_says_how_to_use_the_mod_and_names_every_command():
    text = " ".join(section("Use it inside Claude Code").split())
    plugin = manifest("plugin.json")["name"]
    market = manifest("marketplace.json")["name"]
    for phrase in (
        "claude plugin marketplace add ItsUkrainianCat/optimizer",
        f"claude plugin install {plugin}@{market}",
        "--plugin-dir",
        "/reload-plugins",
        "/improve <prompt>",
        "/optimize",
        "/improve --file PATH",
        "/improve --dry",
        "/improve --resume",
        "/improve clean",
        "/improve cancel",
        "2.1.287",
        "claude plugin validate",
    ):
        assert phrase in text, phrase
    assert "commands/improve.md" not in README and "~/.claude/commands" not in README


def readme_flags() -> list[tuple[str, str]]:
    """(flag, metavar or "") for each row of the README's flags table."""
    return re.findall(r"^\| `(--[a-z-]+)(?: ([^`]+))?` \|", section("Flags"), flags=re.M)


# A value for each README flag that takes one; parse() checks only these three.
SAMPLE = {"--kind": "task", "--strictness": "balanced", "--budget": "100"}


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
