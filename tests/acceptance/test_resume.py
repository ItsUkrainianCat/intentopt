"""Acceptance tests for SPEC R22 (resume by id without repeating a successful paid call, the same
budget, clock and finalists) and the R17 rule that a resumed run continues the same count and
clock. A run is cut with KeyboardInterrupt, a BaseException raised from inside the k-th model call
(the way Ctrl-C arrives), then resumed in a fresh `cli.main` with a fresh backend and clock.
"""

import json

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend, failing, happy_backend

from autoimprover.types import Call, CallError

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."
TARGET = "claude-sonnet-5-5"
SAME = ("status", "prompt", "reason_code", "verified", "stop", "score_before", "score_after")


def _reflections(calls: list[Call]) -> list[int]:
    return [i for i, c in enumerate(calls) if c.role == "reflect"]


CUTS = {  # where to cut, as an index into the uninterrupted run's calls
    "seed-run": lambda calls: next(i for i, c in enumerate(calls) if c.sample == 1),
    "mid-search": lambda calls: _reflections(calls)[1],
    "last-iteration": lambda calls: _reflections(calls)[-1],
    "final-steps": lambda calls: next(
        i for i, c in enumerate(calls) if c.model == TARGET and MARKER in c.user
    ),
    "last-call": lambda calls: len(calls) - 1,
}


def _cut_and_resume(run_cli, cut_at, make, k, argv=(ORIGINAL,), clock_factory=FakeClock):
    """Run until the k-th call raises KeyboardInterrupt, then resume; return both backends and the
    resumed result."""
    first = cut_at(make(), k)
    r1 = run_cli(list(argv), first, clock_factory())
    assert r1.code == 130, r1.err
    second = make()
    r2 = run_cli(["--resume", r1.resume_id(), "--json"], second, clock_factory())
    return first, second, r2


@pytest.mark.parametrize("cut", list(CUTS))
def test_resumed_run_repeats_no_paid_call_and_reaches_the_same_result(run_cli, cut_at, cut):
    """R22, R17: a run cut at any point (a seed run, mid-search, the last iteration, the final
    steps, the last call) resumes by id to the same finalist and result; no successful paid call
    is made again, the resumed run asks only the calls the original had left (no fresh budget), and
    the call count continues from the saved total."""
    reference = happy_backend(IMPROVED)
    ref = run_cli(["--json", ORIGINAL], reference).json()
    k = CUTS[cut](reference.calls)
    first, second, r2 = _cut_and_resume(run_cli, cut_at, lambda: happy_backend(IMPROVED), k)
    assert r2.code == 0, r2.err
    obj = r2.json()
    assert {f: obj[f] for f in SAME} == {f: ref[f] for f in SAME}
    assert obj["prompt"] == IMPROVED
    assert set(first.calls[:k]).isdisjoint(second.calls)
    assert k + len(second.calls) == len(reference.calls)
    assert ref["calls_used"] <= obj["calls_used"] <= ref["calls_used"] + 1
    assert obj["calls_used"] <= 100


def test_resumed_run_keeps_the_clock(run_cli, cut_at, timed):
    """R22, R17: elapsed time is carried across --resume (monotonic, from the run folder), so a run
    whose search the clock cut short is cut short at the same point after a resume in a new process
    whose clock starts again at 0; a fresh clock would let the search run on."""

    def make(clock: FakeClock):
        return timed(happy_backend(IMPROVED), clock, 60.0)

    ref_clock = FakeClock()
    reference = make(ref_clock)
    ref = run_cli(["--json", ORIGINAL], reference, ref_clock).json()
    assert ref["stop"] == "clock"
    k = _reflections(reference.calls)[0] + 2
    clock1 = FakeClock()
    first = timed(cut_at(happy_backend(IMPROVED), k), clock1, 60.0)
    r1 = run_cli([ORIGINAL], first, clock1)
    assert r1.code == 130
    clock2 = FakeClock()
    second = make(clock2)
    r2 = run_cli(["--resume", r1.resume_id(), "--json"], second, clock2)
    assert r2.code == 0, r2.err
    obj = r2.json()
    assert obj["stop"] == "clock"
    assert obj["prompt"] == ref["prompt"]
    assert k + len(second.calls) == len(reference.calls)


def _fail_nth_reflection(n: int) -> ScriptedBackend:
    """A backend failing every attempt of the n-th distinct reflection call (counting from 0)."""
    inner = happy_backend(IMPROVED)
    distinct: list[Call] = []

    def script(call: Call):
        if call.role == "reflect":
            if call not in distinct:
                distinct.append(call)
            if distinct.index(call) == n:
                return CallError("reflection model down")
        return inner.complete(call).text

    return ScriptedBackend(script)


def test_failed_search_call_is_replayed_as_failed_after_a_resume(run_cli, cut_at):
    """R22: a call that failed inside the search is recorded and replayed as failed, so the resumed
    run decides as the original did: the failed call is not tried again even though the model now
    answers, and the result is the same."""
    reference = _fail_nth_reflection(1)
    ref = run_cli(["--json", ORIGINAL], reference).json()
    failed = [c for c in reference.calls if c.role == "reflect"]
    failed_call = next(c for c in failed if failed.count(c) == 3)
    k = _reflections(reference.calls)[-1]
    first = cut_at(_fail_nth_reflection(1), k)
    r1 = run_cli([ORIGINAL], first)
    assert r1.code == 130
    second = happy_backend(IMPROVED)
    r2 = run_cli(["--resume", r1.resume_id(), "--json"], second)
    assert r2.code == 0, r2.err
    assert failed_call not in second.calls
    assert {f: r2.json()[f] for f in SAME} == {f: ref[f] for f in SAME}
    assert k + len(second.calls) == len(reference.calls)


def test_the_call_whose_failure_ended_the_run_is_tried_again(run_cli, override):
    """R22, R24: after exit 3 from three failed reflections, --resume tries again only the call
    whose failure ended the run; the two earlier failed calls stay failed (replayed, not asked)."""
    inner = happy_backend(IMPROVED)
    first = override(inner, reflect=lambda _c: CallError("reflection model down"))
    r1 = run_cli([ORIGINAL], first)
    assert r1.code == 3, r1.err
    failed = list(dict.fromkeys(c for c in first.calls if c.role == "reflect"))
    assert len(failed) == 3
    second = happy_backend(IMPROVED)
    r2 = run_cli(["--resume", r1.resume_id()], second)
    assert r2.code == 0, r2.err
    assert second.calls[0] == failed[2]
    assert failed[0] not in second.calls and failed[1] not in second.calls
    assert not set(c for c in first.calls if c.role != "reflect") & set(second.calls)


def test_failure_outside_the_search_is_tried_again(run_cli):
    """R22, R24: a run that ended with exit 3 on a failed intake keeps its folder; --resume asks
    the intake again and completes the run."""
    first = failing()
    r1 = run_cli([ORIGINAL], first)
    assert r1.code == 3
    second = happy_backend(IMPROVED)
    r2 = run_cli(["--resume", r1.resume_id()], second)
    assert r2.code == 0, r2.err
    assert second.calls[0] == first.calls[0]
    assert r2.out == IMPROVED + "\n"


def test_resume_reads_prompt_plan_and_scenarios_from_the_run_folder(
    run_cli, cut_at, examples, tmp_path
):
    """R22, ADR-007: a resume reads the prompt, plan and scenarios from the run folder; a new
    prompt, a new --budget and an edited --examples file given with --resume are ignored with a
    notice."""
    path = examples(12)
    reference = happy_backend(IMPROVED)
    ref = run_cli(["--json", "--examples", path, ORIGINAL], reference).json()
    k = _reflections(reference.calls)[1]
    first = cut_at(happy_backend(IMPROVED), k)
    r1 = run_cli(["--examples", path, ORIGINAL], first)
    assert r1.code == 130
    changed = tmp_path / "changed.jsonl"
    changed.write_text(
        "".join(json.dumps({"input": f"edited-scenario-{i}"}) + "\n" for i in range(12))
    )
    second = happy_backend(IMPROVED)
    argv = ["--resume", r1.resume_id(), "--json", "--budget", "300", "--examples", str(changed)]
    r2 = run_cli([*argv, "A different prompt."], second)
    assert r2.code == 0, r2.err
    assert "notice" in r2.err
    assert {f: r2.json()[f] for f in SAME} == {f: ref[f] for f in SAME}
    assert k + len(second.calls) == len(reference.calls)
    assert not [
        c for c in second.calls if "edited-scenario" in c.user or "different prompt" in c.user
    ]


@pytest.mark.parametrize("bad", ["../runs", "/tmp", "runs/20261004-120000-0a1b2c3d", "x"])
def test_resume_accepts_only_a_run_id(run_cli, bad):
    """R23, R2: --resume accepts only a run id, never a path; anything else is exit 2 with an
    `error:` line and no call."""
    r = run_cli(["--resume", bad])
    assert r.code == 2
    assert r.out == ""
    assert r.calls == []
    assert any(line.startswith("error: ") for line in r.err.splitlines())


def test_resume_of_an_unknown_id_is_exit_2(run_cli):
    """R22, R2: a well-formed id with no run folder is refused with exit 2."""
    r = run_cli(["--resume", "20261004-120000-0a1b2c3d"])
    assert r.code == 2
    assert r.calls == []


def _cut_run(run_cli, cut_at):
    reference = happy_backend(IMPROVED)
    run_cli([ORIGINAL], reference)
    k = _reflections(reference.calls)[1]
    first = cut_at(happy_backend(IMPROVED), k)
    r1 = run_cli([ORIGINAL], first)
    assert r1.code == 130
    return r1.run_folder(), first, k, reference


@pytest.mark.parametrize(
    "name, content",
    [("manifest.json", "{not json"), ("checkpoint.json", "{"), ("contract.json", "")],
)
def test_corrupt_run_file_refuses_the_resume(run_cli, cut_at, name, content):
    """ADR-007, R22: a manifest, checkpoint or contract that does not parse refuses --resume with
    exit 2, names the file, and makes no call."""
    folder, *_ = _cut_run(run_cli, cut_at)
    (folder / name).write_text(content)
    second = happy_backend(IMPROVED)
    r = run_cli(["--resume", folder.name], second)
    assert r.code == 2, r.err
    assert second.calls == []
    assert name in next(line for line in r.err.splitlines() if line.startswith("error: "))


def test_run_folder_from_a_newer_version_is_refused(run_cli, cut_at):
    """ADR-007: a schema_version newer than the tool knows refuses --resume with exit 2."""
    folder, *_ = _cut_run(run_cli, cut_at)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["schema_version"] = 99
    (folder / "manifest.json").write_text(json.dumps(manifest))
    second = happy_backend(IMPROVED)
    r = run_cli(["--resume", folder.name], second)
    assert r.code == 2
    assert second.calls == []
    assert "newer" in r.err


def test_corrupt_cache_entry_is_a_miss_and_asked_again(run_cli, cut_at):
    """ADR-007, R22: a cache entry that does not parse is a miss: that one call is asked again and
    the resumed run still reaches the result."""
    folder, first, k, reference = _cut_run(run_cli, cut_at)
    entries = sorted((folder / "cache").glob("*.json"))
    assert entries
    entries[0].write_text("{truncated")
    second = happy_backend(IMPROVED)
    r = run_cli(["--resume", folder.name, "--json"], second)
    assert r.code == 0, r.err
    assert r.json()["prompt"] == IMPROVED
    assert len(set(first.calls[:k]) & set(second.calls)) == 1
    assert k + len(second.calls) == len(reference.calls) + 1
