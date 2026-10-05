"""Acceptance tests for SPEC R2: what goes to stdout and stderr, the exit codes and `--json`.

Black-box: each test runs `cli.main` with a scripted raw backend (tests/fakes.py) and reads only the
exit code, stdout, stderr, the run folder and the calls the backend recorded.
"""

import re
import shlex
from pathlib import Path

import pytest
from fakes import MARKER, ScriptedBackend, failing, happy_backend, reflection_reply

from autoimprover.types import REASON_CODES, RUN_ID_PATTERN, Call, SessionNotLockedDown

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."
PLAIN = f"{ORIGINAL} Keep each bullet short."  # a rewrite without MARKER: never better
WHY = ("tightened the wording", "kept the output format", "kept every literal")
# The fields ARCHITECTURE section 8 lists for a successful `--json` object (SPEC R2).
SUCCESS_KEYS = {
    "status",
    "prompt",
    "verified",
    "stop",
    "changes",
    "reason",
    "reason_code",
    "diff",
    "contract",
    "score_before",
    "score_after",
    "search_score_before",
    "search_score_after",
    "noise",
    "margin",
    "length_ratio",
    "calls_used",
    "run_dir",
}
ERROR_KEYS = {"status", "code", "error", "run_dir"}


def _error_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("error: ")]


def _raises(exc: BaseException):
    def script(_call: Call) -> str:
        raise exc

    return script


def _internal_bug() -> ScriptedBackend:
    return ScriptedBackend(_raises(RuntimeError("a bug in a layer")))


def _no_call() -> ScriptedBackend:
    return ScriptedBackend(_raises(RuntimeError("no model call expected")))


def _not_locked_down() -> ScriptedBackend:
    return ScriptedBackend(lambda _call: SessionNotLockedDown("the session reports 2 plugins"))


def _interrupted() -> ScriptedBackend:
    return ScriptedBackend(_raises(KeyboardInterrupt()))


# name -> (raw backend, extra argv, exit code, a run folder exists, a resume line is printed)
EXIT_CASES = {
    "1-internal-error": (_internal_bug, [], 1, True, True),
    "2-usage": (_no_call, ["--budget", "0"], 2, False, False),
    "3-backend-failure": (failing, [], 3, True, True),
    "4-not-locked-down": (_not_locked_down, [], 4, True, False),
    "130-interrupted": (_interrupted, [], 130, True, True),
}


def test_improved_prompt_alone_goes_to_stdout_and_the_report_to_stderr(run_cli):
    """R2, R5, R10a, R17: without --json the improved prompt alone is on stdout (pipeable; GEPA's
    progress output never reaches it), and the report on stderr holds the change lines, the word
    diff, the calls used, the contract, the note that tool use is not exercised and the note on the
    low-budget regime."""
    backend = happy_backend(IMPROVED)
    r = run_cli([ORIGINAL], backend)
    assert r.code == 0, r.err
    assert r.out == IMPROVED + "\n"
    for line in WHY:  # ADR-006: the reflection's three lines are the "what changed and why" lines
        assert line in r.err
    assert MARKER in r.err  # the word-level diff shows the added words
    assert re.search(rf"calls[^\n]*\b{len(backend.calls)}\b", r.err), r.err
    assert "answer the user's request well" in r.err  # R5: the contract is shown in the report
    assert "tool use" in r.err  # R10a: a task prompt's tool use is not exercised, and it says so
    assert "low budget" in r.err.lower() or "low-budget" in r.err.lower()  # R17: says the regime


def test_unchanged_run_prints_the_original_on_stdout_with_exit_0(run_cli):
    """R2, R3: "no reliable improvement" is exit 0 and the original goes to stdout."""
    r = run_cli([ORIGINAL], happy_backend(PLAIN))
    assert r.code == 0, r.err
    assert r.out == ORIGINAL + "\n"
    assert "no reliable improvement" in r.err


def test_json_success_object_carries_every_report_field(run_cli, runs_dir):
    """R2: --json writes one object with the prompt and the report: holdout scores before and
    after, calls used, the word diff, the change lines, verified, the stop cause and a fixed
    reason_code."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--json", ORIGINAL], backend)
    assert r.code == 0, r.err
    obj = r.json()
    assert SUCCESS_KEYS <= obj.keys()
    assert obj["status"] == "improved"
    assert obj["reason_code"] == "improved"
    assert obj["reason"]
    assert obj["prompt"] == IMPROVED
    assert obj["verified"] is True
    assert obj["stop"] == "budget"  # the normal ending: the search used its share of the calls
    assert obj["score_before"] == pytest.approx(0.0)
    assert obj["score_after"] == pytest.approx(1.0)
    assert obj["calls_used"] == len(backend.calls)
    assert list(obj["changes"]) == list(WHY)
    assert MARKER in obj["diff"]
    assert obj["contract"]["goal"] == "answer the user's request well"
    assert obj["length_ratio"] > 1.0
    assert Path(obj["run_dir"]).parent == runs_dir
    assert '"status"' not in r.err  # the object is on stdout only


def test_json_unchanged_object_tells_its_reason_apart(run_cli):
    """R2, R3: the unchanged object carries the original, status unchanged and the fixed
    reason_code for "no reliable improvement"."""
    r = run_cli(["--json", ORIGINAL], happy_backend(PLAIN))
    assert r.code == 0, r.err
    obj = r.json()
    assert SUCCESS_KEYS <= obj.keys()
    assert obj["status"] == "unchanged"
    assert obj["prompt"] == ORIGINAL
    assert obj["reason_code"] == "no_reliable_improvement"
    assert obj["reason_code"] in REASON_CODES
    assert obj["verified"] is False


def test_change_lines_are_capped_at_six(run_cli):
    """R2: the report has 3 to 6 lines saying what changed and why, even when the reflection
    writes more."""
    why = [f"reason number {i}" for i in range(1, 9)]
    inner = happy_backend(IMPROVED)

    def script(call: Call) -> str:
        if call.role == "reflect":
            return reflection_reply(IMPROVED, why=why)
        return inner.complete(call).text

    r = run_cli(["--json", ORIGINAL], ScriptedBackend(script))
    assert r.code == 0, r.err
    changes = r.json()["changes"]
    assert 3 <= len(changes) <= 6
    assert set(changes) <= set(why)


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("case", list(EXIT_CASES))
def test_exit_code_table(run_cli, runs_dir, case, as_json):
    """R2: every non-zero exit (1, 2, 3, 4, 130) writes `error: ...` to stderr and nothing to stdout
    (with --json exactly one error object); exits 1, 3 and 130 also print the run folder and
    `resume with: autoimprover --resume <id>`, exit 4 the run folder; exit 2 (a refusal before any
    paid call) makes no call and leaves no run folder."""
    make, extra, code, has_folder, has_resume = EXIT_CASES[case]
    argv = (["--json"] if as_json else []) + extra + [ORIGINAL]
    r = run_cli(argv, make())
    assert r.code == code, r.err
    assert _error_lines(r.err), r.err
    if as_json:
        obj = r.json()
        assert obj.keys() == ERROR_KEYS
        assert obj["status"] == "error"
        assert obj["code"] == code
        assert obj["error"]
    else:
        assert r.out == ""
    if has_folder:
        folder = r.run_folder()
        assert folder.is_dir()
        assert folder.parent == runs_dir
        assert re.fullmatch(RUN_ID_PATTERN, folder.name)
        if as_json:
            assert Path(r.json()["run_dir"]) == folder
    else:
        assert r.calls == []
        assert not runs_dir.exists() or not any(runs_dir.iterdir())
    if has_resume:
        assert f"resume with: autoimprover --resume {r.run_folder().name}" in r.err


def test_interrupt_is_130_with_error_interrupted(run_cli):
    """R2: Ctrl-C (a KeyboardInterrupt inside a model call) exits 130, `error: interrupted`."""
    r = run_cli([ORIGINAL], _interrupted())
    assert r.code == 130
    assert "error: interrupted" in _error_lines(r.err)[0]


def test_internal_error_names_the_error_type(run_cli):
    """R2: exit 1 says it is an internal error on its `error:` line (ARCHITECTURE section 8)."""
    r = run_cli([ORIGINAL], _internal_bug())
    assert r.code == 1
    assert _error_lines(r.err)[0].startswith("error: internal error")
    assert "RuntimeError" in _error_lines(r.err)[0]


def test_resume_line_carries_a_shell_quoted_state_folder_when_not_the_default(
    run_cli, tmp_path, monkeypatch
):
    """R2, ADR-007: when XDG_STATE_HOME is not the default, the resume line starts with the
    shell-quoted `XDG_STATE_HOME=<dir>` so it can be pasted as is."""
    state = tmp_path / "other state; $(rm -rf x)"
    state.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    r = run_cli([ORIGINAL], failing())
    assert r.code == 3
    line = next(x for x in r.err.splitlines() if x.startswith("resume with: "))
    words = shlex.split(line.removeprefix("resume with: "))
    assert words == [f"XDG_STATE_HOME={state}", "autoimprover", "--resume", r.run_folder().name]
    assert r.run_folder().parent == state / "autoimprover" / "runs"


def test_success_reason_codes_are_from_the_fixed_list(run_cli, examples):
    """R2: each outcome carries one of the fixed reason codes, and they tell the outcomes apart."""
    seen = set()
    for argv, backend in (
        ([ORIGINAL], happy_backend(IMPROVED)),
        ([ORIGINAL], happy_backend(PLAIN)),
        ([ORIGINAL + " " + MARKER], happy_backend(IMPROVED)),
        (["--examples", examples(7), ORIGINAL], happy_backend(IMPROVED)),
    ):
        r = run_cli(["--json", *argv], backend)
        assert r.code == 0, r.err
        seen.add(r.json()["reason_code"])
    assert seen == {"improved", "no_reliable_improvement", "already_strong", "no_holdout"}
    assert seen <= set(REASON_CODES)


USAGE = {  # argv -> the flag the error must name (None: not an error)
    "judge-is-target": (["--judge-model", "sonnet"], "--judge-model"),
    "judge-is-task": (["--judge-model", "haiku"], "--judge-model"),
    "judge-is-target-after-aliases": (
        ["--judge-model", " Claude-Sonnet-5-5[1m] "],
        "--judge-model",
    ),
    "judge-falls-back-for-an-opus-target": (["--target-model", "opus"], None),
    "budget-above-ceiling": (["--budget", "301"], "--budget"),
    "budget-not-a-number": (["--budget", "ten"], "--budget"),
    "unknown-strictness": (["--strictness", "wild"], "--strictness"),
    "unknown-kind": (["--kind", "poem"], "--kind"),
    "unknown-flag": (["--fast"], "--fast"),
}


@pytest.mark.parametrize("case", list(USAGE))
def test_usage_errors_exit_2_and_name_the_flag(run_cli, case):
    """R2, R14, R17: a usage error, including a judge equal to the task or target model after
    aliases, case and a `[1m]` suffix are resolved, exits 2 before any call, prints nothing on
    stdout and names the flag. `--target-model opus` alone is not an error: the judge falls back."""
    extra, flag = USAGE[case]
    backend = happy_backend(IMPROVED)
    r = run_cli(["--dry", *extra, ORIGINAL], backend)
    assert backend.calls == []
    if flag is None:
        assert r.code == 0, r.err
        return
    assert r.code == 2, r.err
    assert r.out == ""
    assert flag in _error_lines(r.err)[0]
