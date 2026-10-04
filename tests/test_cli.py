"""The command line before any model call (SPEC R1, R2, R4, R14, R17, R23): reading the prompt,
every usage error and refusal (exit 2, naming the flag or folder to change), `--dry` (the plan,
zero calls, nothing written, exit 0 even when a real run would refuse) and `clean`. Runs through
the real stack are in `test_cli_run.py`."""

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from fakes import FakeClock, ScriptedBackend, happy_backend

from autoimprover import cli
from autoimprover.runstore import RunStore, runs_root
from autoimprover.types import DEFAULT_MODELS, PROMPT_MAX_CHARS, Plan

PROMPT = "Answer the user's request."
RUN_ID = "20261004-120000-abcdef12"


@dataclass
class Run:
    code: int
    out: str
    err: str
    backend: ScriptedBackend

    def obj(self) -> dict:
        """The one JSON object on stdout."""
        lines = self.out.splitlines()
        assert len(lines) == 1, self.out
        return json.loads(lines[0])


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> Run:
    backend = happy_backend(PROMPT)
    code = cli.main(list(argv), backend=backend, now=FakeClock().now)
    out, err = capsys.readouterr()
    return Run(code, out, err, backend)


def state() -> list[str]:
    """What the state folder holds (the guards create it empty for every test)."""
    return sorted(os.listdir(os.environ["XDG_STATE_HOME"]))


def feed(monkeypatch: pytest.MonkeyPatch, data: bytes, tty: bool = False) -> None:
    stream = io.TextIOWrapper(io.BytesIO(data))
    monkeypatch.setattr(stream, "isatty", lambda: tty)
    monkeypatch.setattr("sys.stdin", stream)


# --- reading the prompt (SPEC R1) ----------------------------------------------------------------


def test_the_argument_is_taken_as_given_with_its_line_endings_made_lf():
    assert cli.read_prompt("a\r\nb\rc\n", None, None) == "a\nb\nc\n"


def test_a_file_is_read_as_utf8_with_a_leading_bom_dropped(tmp_path: Path):
    path = tmp_path / "prompt.txt"
    path.write_bytes(b"\xef\xbb\xbfl\xc3\xadne 1\r\nline 2\r\n")
    assert cli.read_prompt(None, str(path), None) == "líne 1\nline 2\n"


def test_stdin_is_read_when_no_argument_or_file_is_given():
    assert cli.read_prompt(None, None, io.BytesIO("héllo\r\n".encode())) == "héllo\n"


def test_an_argument_or_file_wins_over_stdin_which_is_then_not_read(tmp_path: Path):
    unread = io.BytesIO(b"from stdin")
    assert cli.read_prompt("from argv", None, unread) == "from argv" and unread.tell() == 0
    path = tmp_path / "p.txt"
    path.write_text("from file")
    assert cli.read_prompt(None, str(path), unread) == "from file" and unread.tell() == 0


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", "empty"),
        (" \n\t\r\n", "empty"),
        ("a\0b", "NUL"),
        ("use <curr_param> here", "<curr_param>"),
        ("see <side_info>", "<side_info>"),
        ("x" * (PROMPT_MAX_CHARS + 1), "20,000"),
        ("caf\udce9", "UTF-8"),
    ],
)
def test_a_bad_argument_is_refused_naming_the_problem(text: str, problem: str):
    with pytest.raises(cli.UsageError, match=problem) as refused:
        cli.read_prompt(text, None, None)
    assert "prompt argument" in str(refused.value)


def test_the_limit_counts_characters_after_the_line_endings_are_made_lf():
    assert cli.read_prompt("x" * PROMPT_MAX_CHARS, None, None) == "x" * PROMPT_MAX_CHARS
    crlf = "a\r\n" * (PROMPT_MAX_CHARS // 2)
    assert len(cli.read_prompt(crlf, None, None)) == PROMPT_MAX_CHARS


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (b"\xff\xfe bad", "UTF-8"),
        (b"", "empty"),
        (b"a\x00", "NUL"),
        (b"x" * 90_000, "20,000"),
        (b"x" + "€".encode() * 30_000, "20,000"),  # cut mid-character: too long, not bad UTF-8
    ],
)
def test_bad_bytes_in_a_file_or_on_stdin_are_refused_naming_the_source(
    tmp_path: Path, data: bytes, problem: str
):
    path = tmp_path / "prompt.txt"
    path.write_bytes(data)
    with pytest.raises(cli.UsageError, match=problem) as refused:
        cli.read_prompt(None, str(path), None)
    assert "--file" in str(refused.value)
    with pytest.raises(cli.UsageError, match=problem) as refused:
        cli.read_prompt(None, None, io.BytesIO(data))
    assert "stdin" in str(refused.value)


class Endless(io.RawIOBase):
    """A stream that never ends and records how much was asked of it."""

    def __init__(self) -> None:
        self.asked = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        self.asked += len(buffer)
        buffer[:] = b"x" * len(buffer)
        return len(buffer)


def test_a_huge_stdin_is_refused_after_a_bounded_read():
    stream = Endless()
    with pytest.raises(cli.UsageError, match="20,000"):
        cli.read_prompt(None, None, stream)  # type: ignore[arg-type]
    assert stream.asked <= 4 * PROMPT_MAX_CHARS + 8


@pytest.mark.parametrize("name", ["missing.txt", "."])
def test_a_file_that_cannot_be_read_is_refused(tmp_path: Path, name: str):
    with pytest.raises(cli.UsageError, match="--file"):
        cli.read_prompt(None, str(tmp_path / name), None)


def test_no_prompt_at_all_is_refused():
    with pytest.raises(cli.UsageError, match="--file"):
        cli.read_prompt(None, None, None)


def test_main_reads_a_piped_prompt_and_refuses_a_terminal(monkeypatch, capsys):
    feed(monkeypatch, b"Piped prompt.\r\n")
    assert run(capsys, "--dry").code == 0
    feed(monkeypatch, b"never read", tty=True)
    refused = run(capsys, "--dry")
    assert refused.code == 2 and "error: no prompt" in refused.err and refused.out == ""


# --- usage errors and refusals: exit 2, the flag named, nothing written (SPEC R2, R14, R17) -------

REFUSED = [
    (["--budget", "0", PROMPT], "--budget"),
    (["--budget", "301", PROMPT], "--budget"),
    (["--budget", "ten", PROMPT], "--budget"),
    (["--strictness", "wild", PROMPT], "--strictness"),
    (["--kind", "essay", PROMPT], "--kind"),
    (["--judge-model", "sonnet", PROMPT], "--judge-model"),
    (["--judge-model", "claude-haiku-4-5-20251001", PROMPT], "--judge-model"),
    (["--task-model", "opus", PROMPT], "--task-model"),
    (["--target-model", "opus", "--task-model", "sonnet", PROMPT], "--task-model"),
    (["--judge-model", "Sonnet[1m]", "--target-model", "claude-sonnet-5-5", PROMPT], "--judge"),
    (["--task-model", " ", PROMPT], "--task-model"),
    (["--examples", "missing.jsonl", PROMPT], "--examples"),
    (["--file", "missing.txt"], "--file"),
    (["--file", "prompt.txt", PROMPT], "--file"),
    (["--no-such-flag", PROMPT], "--no-such-flag"),
    (["--dry", "--merg", PROMPT], "--merg"),
    ([PROMPT, "a second prompt"], "one quoted argument"),
    (["--resume", f"../{RUN_ID}"], "--resume"),
    (["--resume", "/tmp/x"], "--resume"),
    (["--resume", RUN_ID], "--resume"),
    (["--dry", "--resume", RUN_ID], "--dry"),
    (["Fill in <side_info> here."], "<side_info>"),
    (["clean", "--budget", "5"], "--budget"),
    (["clean", RUN_ID, RUN_ID], "clean"),
    (["clean", "../x"], "clean"),
    (["--budget", "30", PROMPT], "--budget 86"),
    (["--budget", "60", PROMPT], "--force-low-budget"),
]


@pytest.mark.parametrize(("argv", "flag"), REFUSED)
def test_each_refusal_exits_2_naming_the_flag_and_writes_nothing(capsys, argv, flag):
    refused = run(capsys, *argv)
    assert (refused.code, refused.out) == (2, "")
    assert refused.err.startswith("error: ") and flag in refused.err
    assert "run folder" not in refused.err and refused.backend.calls == []
    assert state() == []


@pytest.mark.parametrize(("argv", "flag"), REFUSED)
def test_with_json_each_refusal_is_one_error_object(capsys, argv, flag):
    refused = run(capsys, *argv, "--json")
    assert refused.code == 2 and refused.obj() == {
        "status": "error",
        "code": 2,
        "error": refused.err.splitlines()[0].removeprefix("error: "),
        "run_dir": "",
    }
    assert flag in refused.obj()["error"]


def test_the_examples_files_bad_line_is_named(tmp_path, capsys):
    examples = tmp_path / "ex.jsonl"
    examples.write_text('{"input": "one"}\n{"input": 3}\n')
    refused = run(capsys, "--examples", str(examples), PROMPT)
    assert refused.code == 2 and "--examples" in refused.err and "line 2" in refused.err


def git_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    folder = tmp_path / "repo" / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(folder))
    return folder


def test_a_state_folder_inside_git_is_refused_before_any_call(monkeypatch, tmp_path, capsys):
    folder = git_state(monkeypatch, tmp_path)
    refused = run(capsys, PROMPT)
    assert refused.code == 2 and "XDG_STATE_HOME" in refused.err
    assert refused.backend.calls == [] and not folder.exists()


def test_without_home_or_state_folder_the_run_is_refused(monkeypatch, capsys):
    monkeypatch.delenv("HOME")
    monkeypatch.delenv("XDG_STATE_HOME")
    refused = run(capsys, PROMPT)
    assert refused.code == 2 and "XDG_STATE_HOME" in refused.err


def test_help_goes_to_stdout_and_with_json_as_one_object(capsys):
    shown = run(capsys, "--help")
    assert shown.code == 0 and "--resume" in shown.out and "clean" in shown.out
    assert run(capsys, "--help", "--json").obj()["status"] == "help"


# --- --dry: the plan, zero calls, nothing written, exit 0 (SPEC R4, R14, R17) ---------------------


def test_dry_prints_the_plan_makes_no_call_and_writes_nothing(capsys):
    dry = run(capsys, "--dry", PROMPT)
    assert (dry.code, dry.err, dry.backend.calls) == (0, "", [])
    assert dry.out.startswith("dry run: no model call made, nothing written\n")
    assert "66 left for the search" in dry.out and "would refuse" not in dry.out
    assert state() == []


def test_dry_with_json_is_one_plan_object(capsys):
    plan = run(capsys, "--dry", "--json", "--budget", "120", "--strictness", "bold", PROMPT).obj()
    assert plan["status"] == "dry" and plan["refusal"] is None
    assert (plan["plan"]["budget"], plan["plan"]["strictness"]) == (120, "bold")
    assert plan["plan"]["models"] == {
        "task": DEFAULT_MODELS.task,
        "judge": DEFAULT_MODELS.judge,
        "reflect": DEFAULT_MODELS.reflect,
        "target": DEFAULT_MODELS.target,
    }


def test_dry_exits_0_and_says_why_a_real_run_would_refuse(capsys):
    dry = run(capsys, "--dry", "--json", "--budget", "30", PROMPT)
    assert dry.code == 0 and "exceed the budget of 30" in dry.obj()["refusal"]
    forced = run(capsys, "--dry", "--json", "--budget", "60", "--force-low-budget", PROMPT)
    assert forced.obj()["refusal"] is None


def test_dry_reports_an_unusable_state_folder_and_still_writes_nothing(
    monkeypatch, tmp_path, capsys
):
    folder = git_state(monkeypatch, tmp_path)
    dry = run(capsys, "--dry", PROMPT)
    assert dry.code == 0 and "a real run would refuse: " in dry.out
    assert "XDG_STATE_HOME" in dry.out and not folder.exists()


def test_dry_without_home_reports_it(monkeypatch, capsys):
    monkeypatch.delenv("HOME")
    monkeypatch.delenv("XDG_STATE_HOME")
    dry = run(capsys, "--dry", "--json", PROMPT)
    assert dry.code == 0 and "HOME is not set" in dry.obj()["refusal"]


def test_a_target_that_is_the_default_judge_gets_the_fallback_judge(capsys):
    models = run(capsys, "--dry", "--json", "--target-model", "opus", PROMPT).obj()["plan"]
    assert models["models"]["target"] == "claude-opus-5-5"
    assert models["models"]["judge"] == "claude-sonnet-5-5"


def test_model_aliases_case_and_suffix_resolve_to_full_ids(capsys):
    plan = run(capsys, "--dry", "--json", "--task-model", " Haiku[1m] ", PROMPT).obj()["plan"]
    assert plan["models"]["task"] == "claude-haiku-4-5-20251001"


def examples(tmp_path: Path, n: int) -> str:
    path = tmp_path / f"{n}.jsonl"
    path.write_text("".join(json.dumps({"input": f"example {i}"}) + "\n" for i in range(n)))
    return str(path)


def test_dry_with_few_examples_says_the_original_is_kept_unless_trust_search(tmp_path, capsys):
    few = run(capsys, "--dry", "--json", "--examples", examples(tmp_path, 7), PROMPT).obj()
    assert (few["scenarios"], few["synthesised"], few["holdout"]) == (7, False, 0)
    assert "7 scenarios" in few["keeps_original"] and "--trust-search" in few["keeps_original"]
    trusted = run(
        capsys, "--dry", "--json", "--trust-search", "--examples", examples(tmp_path, 7), PROMPT
    ).obj()
    assert trusted["keeps_original"] is None


def test_dry_with_nine_examples_plans_their_split(tmp_path, capsys):
    plan = run(capsys, "--dry", "--json", "--examples", examples(tmp_path, 9), PROMPT).obj()
    assert (plan["scenarios"], plan["holdout"], plan["valset"], plan["dataset"]) == (9, 3, 2, 4)
    assert plan["calls_before_search"] == 1 + 2 * 4 + 3  # no synthesis call


# --- clean (SPEC R23) ----------------------------------------------------------------------------


def new_run(prompt: str = PROMPT) -> RunStore:
    return RunStore.open_or_create(runs_root(), Plan(models=DEFAULT_MODELS), prompt)


def test_clean_removes_every_run_and_only_runs(tmp_path, capsys):
    first, second = new_run("one"), new_run("two")
    first.close()
    second.close()
    root = runs_root()
    (root / "notes").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (root / RUN_ID).symlink_to(outside)
    cleaned = run(capsys, "clean")
    assert (cleaned.code, cleaned.out) == (0, "")
    assert "removed 2 run folders" in cleaned.err
    assert sorted(os.listdir(root)) == sorted(["notes", RUN_ID])
    assert (outside / "keep.txt").read_text() == "keep"


def test_clean_with_an_id_removes_that_run_only(capsys):
    first, second = new_run("one"), new_run("two")
    first.close()
    second.close()
    cleaned = run(capsys, "clean", first.run_id, "--json")
    assert cleaned.obj() == {"status": "cleaned", "removed": 1, "skipped": 0}
    assert not first.path.exists() and second.path.is_dir()


def test_clean_skips_a_run_that_holds_its_lock(capsys):
    live = new_run()
    try:
        cleaned = run(capsys, "--json", "clean")
        assert cleaned.obj() == {"status": "cleaned", "removed": 0, "skipped": 1}
        assert "still running" in cleaned.err and live.path.is_dir()
    finally:
        live.close()


def test_clean_with_nothing_to_remove_exits_0(capsys):
    assert run(capsys, "clean").code == 0
    assert run(capsys, "clean", RUN_ID).code == 0


def test_clean_refuses_a_symlinked_run_and_leaves_its_target(tmp_path, capsys):
    outside = tmp_path / "outside"
    outside.mkdir()
    runs_root().mkdir(parents=True)
    (runs_root() / RUN_ID).symlink_to(outside)
    refused = run(capsys, "clean", RUN_ID)
    assert refused.code == 2 and "symlink" in refused.err and outside.is_dir()
