"""`autoimprover` through the real stack (SPEC R2, R3, R11, R17, R18, R19, R22, R23, R24): the run
folder in the test's state folder, `Cached(Resilient(Budgeted(raw)))` over a scripted raw layer,
the pinned GEPA and a fake clock. Every exit code of SPEC R2 is reached the way a user would reach
it, and with `--json` stdout always holds exactly one object. A test that shows a candidate is NOT
returned has a twin showing one IS (ARCHITECTURE section 3)."""

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend, failing, happy_backend, reflection_reply

from autoimprover import cli, runner
from autoimprover.backend import BudgetedBackend
from autoimprover.runstore import runs_root
from autoimprover.types import (
    DEFAULT_MODELS,
    SEARCH_CLOCK_SHARE,
    SYNTH_COUNT,
    WALL_CLOCK_DEFAULT_S,
    Call,
    Plan,
    Reply,
    SessionNotLockedDown,
)

PROMPT = "Answer the user's request."
BETTER = f"Answer the user's request {MARKER}."
WORSE = f"{PROMPT} Be brief."


@pytest.fixture(autouse=True)
def _no_disk_flush(monkeypatch: pytest.MonkeyPatch):
    """Writes stay atomic but are not flushed: a run writes hundreds of cache entries, and
    durability is tested with the run store."""
    monkeypatch.setattr(os, "fsync", lambda _fd: None)


@dataclass
class Run:
    code: int
    out: str
    err: str

    def obj(self) -> dict:
        lines = self.out.splitlines()
        assert len(lines) == 1, self.out
        return json.loads(lines[0])


def run(capsys: pytest.CaptureFixture[str], backend, *argv: str) -> Run:
    code = cli.main(list(argv), backend=backend, now=FakeClock().now)
    out, err = capsys.readouterr()
    return Run(code, out, err)


def runs() -> list[Path]:
    root = runs_root()
    return sorted(root.iterdir()) if root.is_dir() else []


def comparable(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if k not in ("calls_used", "run_dir")}


# --- the results of a run (SPEC R2, R3, R23) -----------------------------------------------------


def test_an_improved_prompt_is_one_json_object_with_its_run_folder_kept_private(capsys):
    done = run(capsys, happy_backend(BETTER), "--json", PROMPT)
    obj = done.obj()
    assert done.code == 0 and (obj["status"], obj["prompt"]) == ("improved", BETTER)
    assert (obj["reason_code"], obj["verified"], obj["stop"]) == ("improved", True, "budget")
    assert obj["contract"]["kind"] == "task" and MARKER in obj["diff"]
    assert 0 < obj["calls_used"] <= 100
    [folder] = runs()
    assert obj["run_dir"] == str(folder) and f"run folder: {folder}" in done.err
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    for path in folder.rglob("*"):
        mode = stat.S_IMODE(path.lstat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), path
    assert (folder / "gepa.log").stat().st_size > 0  # GEPA's output, not stdout


def test_without_json_the_improved_prompt_alone_is_on_stdout_and_the_report_on_stderr(capsys):
    done = run(capsys, happy_backend(BETTER), PROMPT)
    assert (done.code, done.out) == (0, BETTER + "\n")
    assert done.err.index("models: task") < done.err.index("result: improved (improved)")
    assert "word diff:" in done.err and "verified: yes" in done.err


def test_its_unchanged_twin_returns_the_original_with_the_reason_code(capsys):
    kept = run(capsys, happy_backend(WORSE), PROMPT)
    assert (kept.code, kept.out) == (0, PROMPT + "\n")
    assert "result: unchanged" in kept.err and "(no_reliable_improvement)" in kept.err
    obj = run(capsys, happy_backend(WORSE), "--json", PROMPT).obj()
    assert (obj["status"], obj["prompt"], obj["reason_code"]) == (
        "unchanged",
        PROMPT,
        "no_reliable_improvement",
    )
    assert obj["diff"] == "" and obj["verified"] is False


def examples(tmp_path: Path, n: int) -> str:
    path = tmp_path / f"{n}.jsonl"
    path.write_text("".join(json.dumps({"input": f"example {i}"}) + "\n" for i in range(n)))
    return str(path)


def test_seven_examples_keep_the_original_without_a_call_or_a_run_folder(tmp_path, capsys):
    backend = happy_backend(BETTER)
    kept = run(capsys, backend, "--json", "--examples", examples(tmp_path, 7), PROMPT)
    obj = kept.obj()
    assert (kept.code, obj["reason_code"], obj["prompt"], obj["run_dir"]) == (
        0,
        "no_holdout",
        PROMPT,
        "",
    )
    assert backend.calls == [] and runs() == []


def test_their_twin_with_trust_search_returns_an_unverified_result(tmp_path, capsys):
    argv = ("--json", "--trust-search", "--examples", examples(tmp_path, 7), PROMPT)
    trusted = run(capsys, happy_backend(BETTER), *argv)
    obj = trusted.obj()
    assert (obj["status"], obj["prompt"], obj["verified"]) == ("improved", BETTER, False)
    assert "NOT VERIFIED" in trusted.err


def test_anything_printed_during_the_run_goes_to_the_log_not_stdout(capsys):
    model = happy_backend(BETTER)

    def chatty(call: Call) -> Reply:
        print("chatter from inside the run")
        return model.complete(call)

    done = run(capsys, ScriptedBackend(lambda call: chatty(call).text), "--json", PROMPT)
    assert done.obj()["status"] == "improved" and "chatter" not in done.err
    [folder] = runs()
    assert "chatter from inside the run" in (folder / "gepa.log").read_text()


# --- every exit code through the real stack (SPEC R2, R18, R24) ----------------------------------


def test_a_backend_that_fails_ends_with_exit_3_the_folder_and_the_resume_line(capsys):
    down = run(capsys, failing("claude exited 1"), PROMPT)
    [folder] = runs()
    assert (down.code, down.out) == (3, "")
    assert down.err.splitlines()[-3:] == [
        "error: " + down.err.splitlines()[-3].removeprefix("error: "),
        f"run folder: {folder}",
        f"resume with: autoimprover --resume {folder.name}",
    ]
    assert "intake" in down.err and "claude exited 1" in down.err


def test_with_json_a_backend_failure_is_one_error_object(capsys):
    down = run(capsys, failing(), "--json", PROMPT)
    [folder] = runs()
    obj = down.obj()
    assert (obj["status"], obj["code"], obj["run_dir"]) == ("error", 3, str(folder))
    assert "backend down" in obj["error"]


def test_a_state_folder_that_is_not_the_default_is_named_in_the_resume_line(
    monkeypatch, tmp_path, capsys
):
    elsewhere = tmp_path / "my state"
    monkeypatch.setenv("XDG_STATE_HOME", str(elsewhere))
    down = run(capsys, failing(), PROMPT)
    [folder] = runs()
    assert folder.is_relative_to(elsewhere)
    assert f"resume with: XDG_STATE_HOME='{elsewhere}' autoimprover --resume {folder.name}" in (
        down.err
    )


def test_a_session_that_is_not_locked_down_ends_with_exit_4(capsys):
    loose = ScriptedBackend(lambda _call: SessionNotLockedDown("the session reports 2 plugins"))
    stopped = run(capsys, loose, PROMPT)
    [folder] = runs()
    assert (stopped.code, stopped.out) == (4, "")
    assert "2 plugins" in stopped.err and f"run folder: {folder}" in stopped.err
    assert "resume with" not in stopped.err
    assert run(capsys, loose, "--json", PROMPT).obj()["code"] == 4


def interrupt(_call: Call) -> str:
    raise KeyboardInterrupt


def test_ctrl_c_ends_with_exit_130_keeps_the_folder_and_names_the_resume_line(capsys):
    stopped = run(capsys, ScriptedBackend(interrupt), PROMPT)
    [folder] = runs()
    assert (stopped.code, stopped.out) == (130, "")
    assert stopped.err.splitlines()[-3:] == [
        "error: interrupted",
        f"run folder: {folder}",
        f"resume with: autoimprover --resume {folder.name}",
    ]
    resumed = run(capsys, happy_backend(BETTER), "--json", "--resume", folder.name)
    assert resumed.obj()["prompt"] == BETTER  # the lock was released


def test_with_json_ctrl_c_is_one_error_object(capsys):
    obj = run(capsys, ScriptedBackend(interrupt), "--json", PROMPT).obj()
    assert (obj["status"], obj["code"], obj["error"]) == ("error", 130, "interrupted")


def test_a_bug_ends_with_exit_1_and_still_one_json_object(capsys):
    buggy = ScriptedBackend(lambda _call: RuntimeError("a bug"))
    crashed = run(capsys, buggy, "--json", PROMPT)
    [folder] = runs()
    obj = crashed.obj()
    assert (crashed.code, obj["code"], obj["run_dir"]) == (1, 1, str(folder))
    assert obj["error"] == "internal error: RuntimeError: a bug"
    assert f"resume with: autoimprover --resume {folder.name}" in crashed.err


def test_a_missing_claude_backend_ends_with_exit_3_before_any_folder(monkeypatch, capsys):
    monkeypatch.setitem(__import__("sys").modules, "autoimprover.claude_cli", None)
    code = cli.main([PROMPT])
    _, err = capsys.readouterr()
    assert code == 3 and "autoimprover.claude_cli" in err and runs() == []


def test_a_claude_backend_that_fails_to_import_something_else_is_a_bug(monkeypatch, capsys):
    def broken(name: str):
        raise ModuleNotFoundError(f"No module named 'helper' (importing {name})", name="helper")

    monkeypatch.setattr(cli.importlib, "import_module", broken)
    code = cli.main([PROMPT])
    _, err = capsys.readouterr()
    assert code == 1 and "internal error: ModuleNotFoundError" in err and runs() == []


# --- the stack a run is given (SPEC R17, R22) ----------------------------------------------------


def test_the_run_gets_the_search_share_and_a_resume_its_saved_count_and_clock(monkeypatch, capsys):
    built: list[tuple] = []

    class Spy(BudgetedBackend):
        def __init__(self, raw, limit, used, clock, deadline, on_call=None):
            built.append((limit, used, deadline, clock.elapsed(), on_call.__name__))
            super().__init__(raw, limit, used, clock, deadline, on_call)

    monkeypatch.setattr(cli, "BudgetedBackend", Spy)
    fake = FakeClock()
    model = happy_backend(BETTER)
    timed = ScriptedBackend(lambda call: model.complete(call).text, duration_s=7.0, clock=fake)
    with pytest.raises(Cut):
        cli.main([PROMPT], backend=Cutting(5, timed), now=fake.now)
    [folder] = runs()
    run(capsys, happy_backend(BETTER), "--resume", folder.name)
    final = runner.fixed_costs(Plan(models=DEFAULT_MODELS), SYNTH_COUNT, True).final
    share = SEARCH_CLOCK_SHARE * WALL_CLOCK_DEFAULT_S
    assert built == [
        (100 - final, 0, share, 0.0, "save_progress"),
        (100 - final, 5, share, 28.0, "save_progress"),
    ]


def test_a_kind_given_on_the_command_line_is_the_contracts_kind(capsys):
    obj = run(capsys, happy_backend(BETTER), "--json", "--kind", "template", PROMPT).obj()
    assert obj["contract"]["kind"] == "template" and obj["status"] == "improved"


def test_a_resumed_run_keeps_its_kind_and_trust_search_and_examples(tmp_path, capsys):
    argv = ["--json", "--kind", "template", "--trust-search", "--examples"]
    with pytest.raises(Cut):
        cli.main([*argv, examples(tmp_path, 7), PROMPT], backend=Cutting(1), now=FakeClock().now)
    [folder] = runs()
    obj = run(capsys, happy_backend(BETTER), "--json", "--resume", folder.name).obj()
    assert (obj["contract"]["kind"], obj["status"], obj["verified"]) == (
        "template",
        "improved",
        False,
    )


# --- resume (SPEC R22) ---------------------------------------------------------------------------


class Cut(BaseException):
    """The process dies in the middle of a live call."""


@dataclass
class Cutting:
    """The happy model, dying at live call `at`; it records every call it was asked."""

    at: int
    model: ScriptedBackend = field(default_factory=lambda: happy_backend(BETTER))
    calls: list[Call] = field(default_factory=list)

    def complete(self, call: Call) -> Reply:
        self.calls.append(call)
        if len(self.calls) == self.at:
            raise Cut
        return self.model.complete(call)


def test_a_run_cut_at_any_point_resumes_to_the_same_outcome_and_continues_the_count(capsys):
    reference_model = happy_backend(BETTER)
    reference = run(capsys, reference_model, "--json", PROMPT).obj()
    total = len(reference_model.calls)
    for at in (1, 2, total // 2, total - 1, total):
        for folder in runs():
            cli.main(["clean", folder.name])
        cut = Cutting(at)
        with pytest.raises(Cut):
            cli.main(["--json", PROMPT], backend=cut, now=FakeClock().now)
        capsys.readouterr()
        [folder] = runs()
        model = happy_backend(BETTER)
        resumed = run(capsys, model, "--json", "--resume", folder.name)
        obj = resumed.obj()
        assert comparable(obj) == comparable(reference), at
        assert obj["calls_used"] == at + len(model.calls), at
        assert obj["run_dir"] == str(folder)


def test_a_finished_run_resumes_to_the_same_outcome_without_a_live_call(capsys):
    first = run(capsys, happy_backend(BETTER), "--json", PROMPT).obj()
    [folder] = runs()
    model = happy_backend(WORSE)  # it would answer differently: only the cache can agree
    again = run(capsys, model, "--json", "--resume", folder.name).obj()
    assert again == first and model.calls == []


def test_a_resumed_run_with_examples_keeps_them_even_when_cut_before_they_were_used(
    tmp_path, capsys
):
    argv = ("--json", "--examples", examples(tmp_path, 9), PROMPT)
    reference = run(capsys, happy_backend(BETTER), *argv).obj()
    cli.main(["clean"])
    with pytest.raises(Cut):
        cli.main(list(argv), backend=Cutting(1), now=FakeClock().now)
    [folder] = runs()
    model = happy_backend(BETTER)
    resumed = run(capsys, model, "--json", "--resume", folder.name).obj()
    assert comparable(resumed) == comparable(reference)
    assert model.count("synth") == 0


@pytest.mark.parametrize("damage", [{"kind": "essay"}, {"trust_search": "yes"}])
def test_a_manifest_with_damaged_saved_flags_refuses_the_resume(capsys, damage):
    with pytest.raises(Cut):
        cli.main([PROMPT], backend=Cutting(1), now=FakeClock().now)
    [folder] = runs()
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["opts"] |= damage
    (folder / "manifest.json").write_text(json.dumps(manifest))
    model = happy_backend(BETTER)
    refused = run(capsys, model, "--resume", folder.name)
    assert (refused.code, refused.out, model.calls) == (2, "", [])
    assert "manifest.json" in refused.err and folder.name in refused.err


def test_resume_shows_the_saved_run_before_it_continues(capsys):
    with pytest.raises(Cut):
        cli.main([PROMPT], backend=Cutting(5), now=FakeClock().now)
    capsys.readouterr()
    [folder] = runs()
    resumed = run(capsys, happy_backend(BETTER), "--resume", folder.name)
    assert f"resuming run {folder.name}: 5 of 100 calls used" in resumed.err
    assert resumed.out == BETTER + "\n"


# --- model-written text never steers the terminal (SPEC R19) -------------------------------------

HOSTILE = f"Answer the user's request {MARKER}.\x1b[2J\x1b]0;owned\x07\x9b31m\u202e\x07"
HOSTILE_WHY = ("tightened\x1b[31m the wording", "kept\x07 the format", "kept \x1b]8;;x\x07it")


def hostile_backend() -> ScriptedBackend:
    model = happy_backend(HOSTILE)

    def script(call: Call) -> str:
        if call.role == "reflect":
            return reflection_reply(HOSTILE, why=HOSTILE_WHY)
        return model.complete(call).text

    return ScriptedBackend(script)


CONTROL = [chr(c) for c in (*range(0, 9), *range(11, 32), *range(127, 160))] + ["\u202e"]


@pytest.mark.parametrize("json_mode", [False, True])
def test_control_characters_in_model_output_never_reach_stdout_or_stderr(capsys, json_mode):
    done = run(capsys, hostile_backend(), *(["--json"] if json_mode else []), PROMPT)
    assert done.code == 0
    for char in CONTROL:
        assert char not in done.out and char not in done.err, repr(char)
    if json_mode:
        assert done.obj()["prompt"] == HOSTILE  # the data itself, escaped by JSON
    else:
        assert done.out == f"Answer the user's request {MARKER}.\n"
        assert "  - tightened the wording" in done.err
