"""Acceptance tests for SPEC R24 (failed calls, retries, exit 3) and the exit-4 lockdown of R18
when it fires inside the search.

A failed call is one whose attempt and 2 retries all raise `CallError` from the raw backend (the
way `claude -p` fails); `cli.main(argv, backend=...)` replaces only that raw layer, so the retry
and failure counting of the real stack run (tests/fakes.py `failing`).
"""

import json

import pytest
from fakes import MARKER, ScriptedBackend, failing, happy_backend, judge_reply

from autoimprover.types import Call, CallError, SessionNotLockedDown

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."
TASK_MODEL, TARGET = "claude-haiku-4-5-20251001", "claude-sonnet-5-5"


def _exit_3_output(r) -> None:
    assert r.code == 3, r.err
    assert r.out == ""
    assert any(line.startswith("error: ") for line in r.err.splitlines())
    folder = r.run_folder()
    assert (folder / "manifest.json").is_file()  # the partial run folder is kept for --resume
    assert f"resume with: autoimprover --resume {folder.name}" in r.err


def test_failed_intake_is_retried_twice_then_ends_the_run_with_exit_3(run_cli):
    """R24: a call that errors is retried up to 2 times (3 attempts, each a paid call); a failed
    call outside the search (intake) ends the run with exit 3 and keeps the run folder."""
    backend = failing()
    r = run_cli([ORIGINAL], backend)
    _exit_3_output(r)
    assert [c.role for c in backend.calls] == ["intake"] * 3
    assert len(set(backend.calls)) == 1


def test_a_retry_that_answers_lets_the_run_go_on(run_cli, override):
    """R24 twin: two failed attempts then an answer is not a failed call; the run goes on, and the
    two failed attempts count toward the budget (R17)."""
    inner = happy_backend(IMPROVED)
    attempts = []

    def intake(call: Call) -> str | Exception:
        attempts.append(call)
        return CallError("flaky") if len(attempts) <= 2 else inner.complete(call).text

    backend = override(inner, intake=intake)
    r = run_cli(["--json", ORIGINAL], backend)
    assert r.code == 0, r.err
    assert [c.role for c in backend.calls[:4]] == ["intake"] * 3 + ["synth"]
    assert r.json()["prompt"] == IMPROVED
    assert r.json()["calls_used"] == len(backend.calls)


def _is_contract_check(call: Call) -> bool:
    return call.role == "judge" and json.loads(call.user)["scenarios"][0]["scenario"] == "contract"


def _is_seed_run(call: Call) -> bool:
    return call.role == "task" and call.model == TARGET and MARKER not in call.user


def _is_seed_scoring(call: Call) -> bool:
    return call.role == "task" and call.model == TASK_MODEL and MARKER not in call.user


def _is_finalist_run(call: Call) -> bool:
    return call.role == "task" and call.model == TARGET and MARKER in call.user


OUTSIDE = {  # where the call fails -> which calls fail
    "synthesis": lambda c: c.role == "synth",
    "seed-run-on-holdout": _is_seed_run,
    "scoring-the-original": _is_seed_scoring,
    "finalist-run": _is_finalist_run,
    "contract-check": _is_contract_check,
}


@pytest.mark.parametrize("where", [*OUTSIDE, "nowhere"])
def test_failed_call_outside_the_search_ends_with_exit_3(run_cli, where):
    """R24: one failed call (all 3 attempts) in a seed run is never scored 0, and neither is one
    while scoring the original inside GEPA; that and a failed synthesis, finalist run or contract
    check end the run with exit 3 (there is no iteration to skip), the run folder kept, and the
    failed call is not asked again. Twin "nowhere": the same run without the failure improves."""
    matches = OUTSIDE.get(where, lambda _c: False)
    inner = happy_backend(IMPROVED)
    first: list[Call] = []

    def script(call: Call) -> str | Exception:
        if matches(call) and not first:
            first.append(call)
        if first and call == first[0]:
            return CallError("model down")
        return inner.complete(call).text

    backend = ScriptedBackend(script)
    r = run_cli([ORIGINAL], backend)
    failed = [c for c in backend.calls if first and c == first[0]]
    if where == "nowhere":
        assert r.code == 0, r.err
        assert r.out == IMPROVED + "\n"
    else:
        _exit_3_output(r)
        assert "no reliable improvement" not in r.err
        assert len(failed) == 3  # tried 3 times, then the run ends instead of scoring it


def _fail_first_reflections(k: int):
    """Reflection replies: the first k distinct reflection calls fail on every attempt."""
    distinct: list[Call] = []
    inner = happy_backend(IMPROVED)

    def reflect(call: Call) -> str | Exception:
        if call not in distinct:
            distinct.append(call)
        if distinct.index(call) < k:
            return CallError("reflection model down")
        return inner.complete(call).text

    return reflect


@pytest.mark.parametrize("failed, code", [(3, 3), (2, 0)], ids=["three-in-a-row", "twin-two"])
def test_three_consecutive_failed_reflections_end_the_run(run_cli, override, failed, code):
    """R24: three consecutive failed calls of the same kind to the same model end the run with exit
    3, although judge calls to that same model (another kind) succeed in between; a failed
    reflection only skips its iteration. Twin: two failed reflections, then the run improves."""
    backend = override(happy_backend(IMPROVED), reflect=_fail_first_reflections(failed))
    r = run_cli([ORIGINAL], backend)
    if code == 3:
        reflect_at = [i for i, c in enumerate(backend.calls) if c.role == "reflect"]
        between = backend.calls[reflect_at[0] : reflect_at[-1]]
        assert any(c.role == "judge" and c.model == "claude-opus-5-5" for c in between)
        _exit_3_output(r)
        assert backend.count("reflect") == 9  # 3 failed calls x 3 attempts, then the run ends
        assert backend.calls[-1].role == "reflect"
    else:
        assert r.code == 0, r.err
        assert r.out == IMPROVED + "\n"


@pytest.mark.parametrize("fail", [True, False], ids=["failed-task-call", "twin"])
def test_failed_search_call_gives_a_neutral_score_and_no_finalist(
    run_cli, override, examples, fail
):
    """R24: inside the search a failed call gives the scenario a neutral score and the candidate
    cannot become a finalist (it is never run on the target model's holdout); the run goes on and
    ends with exit 0. With 8 scenarios the minibatch is the whole dataset, so every evaluation of
    the candidate includes the failing scenario. Twin: no failure, the candidate wins."""
    first: list[Call] = []

    def task(call: Call) -> str | Exception:
        good = MARKER in call.user
        if fail and good and call.model == TASK_MODEL:
            if not first:
                first.append(call)
            if call.user == first[0].user:
                return CallError("task model timed out")
        return "GOOD answer" if good else "BAD answer"

    backend = override(happy_backend(IMPROVED), task=task)
    r = run_cli(["--json", "--examples", examples(8), ORIGINAL], backend)
    assert r.code == 0, r.err
    obj = r.json()
    finalist_runs = [c for c in backend.calls if c.model == TARGET and MARKER in c.user]
    if fail:
        assert sum(1 for c in backend.calls if c == first[0]) == 3  # tried 3 times, never again
        assert finalist_runs == []
        assert obj["prompt"] == ORIGINAL
    else:
        assert finalist_runs
        assert obj["prompt"] == IMPROVED


@pytest.mark.parametrize("unknown, status", [(3, "improved"), (4, "unchanged")])
def test_more_than_30_percent_unknown_checks_score_0(run_cli, override, examples, unknown, status):
    """R24, R10b: a judged check the judge omits is `unknown` and left out of the score; a candidate
    with more than 30 % unknown checks scores 0. With 10 judged checks per scenario (1 contract
    check, 9 criteria), 3 omitted (30 %) still lets the candidate win, 4 (40 %) does not."""
    path = examples(12, criteria=[f"mentions point {i}" for i in range(1, 10)])

    def judge(call: Call) -> str:
        request = json.loads(call.user)
        if request["scenarios"][0]["scenario"] == "contract":
            return judge_reply(call)
        results = []
        for item in request["scenarios"]:
            good = item["output"].startswith("GOOD")
            checks = [
                {"id": c["id"], "pass": good, "quote": item["output"][:20]} for c in item["checks"]
            ]
            assert len(checks) == 10
            results.append(
                {"scenario": item["scenario"], "checks": checks[:-unknown] if good else checks}
            )
        return json.dumps({"results": results})

    r = run_cli(
        ["--json", "--examples", path, ORIGINAL], override(happy_backend(IMPROVED), judge=judge)
    )
    assert r.code == 0, r.err
    assert r.json()["status"] == status


@pytest.mark.parametrize("role", ["reflect", "task", "none"])
def test_lockdown_failure_on_a_resumed_runs_first_live_call_is_exit_4(
    run_cli, cut_at, override, role
):
    """R18, R2: the first live call of every process, a resumed run too, aborts with exit 4 if the
    session is not locked down, however GEPA handles the error (it swallows reflection errors).
    The run is cut (Ctrl-C) just before a call of `role` inside the search; the resumed process's
    first live call is that call and reports plugins. Twin "none": a locked-down session resumes
    to the result."""
    reference = happy_backend(IMPROVED)
    run_cli([ORIGINAL], reference)
    kind = "reflect" if role == "none" else role
    k = next(
        i
        for i, c in enumerate(reference.calls)
        if c.role == kind and (kind != "task" or c.model == TASK_MODEL) and i > 20
    )
    first = run_cli([ORIGINAL], cut_at(happy_backend(IMPROVED), k))
    assert first.code == 130
    lockdown = SessionNotLockedDown("the session reports 1 MCP server")
    if role == "none":
        resumed = happy_backend(IMPROVED)
    else:
        resumed = override(happy_backend(IMPROVED), **{role: lambda _c: lockdown})
    r = run_cli(["--resume", first.resume_id()], resumed)
    assert resumed.calls[0] == reference.calls[k]
    if role == "none":
        assert r.code == 0, r.err
        assert r.out == IMPROVED + "\n"
    else:
        assert r.code == 4, r.err
        assert r.out == ""
        assert any(line.startswith("error: ") for line in r.err.splitlines())
        assert r.run_folder().name == first.resume_id()
        assert len(resumed.calls) == 1
