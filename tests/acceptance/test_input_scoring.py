"""Acceptance tests for how the prompt gets in and how candidates are run and judged, seen from the
calls the scripted backend records: SPEC R1 (input), R5 and R10a (kind and execution shape), R10
and R14 (one blind judge call per batch, never the task or target model), R11 (scenario checks),
R12 (two independent seed runs) and R15 (the three-way split, the hidden holdout, one call at a
time). Message shapes are those of ADR-008.
"""

import io
import json
import re
import sys
import threading

import pytest
from fakes import MARKER, ScriptedBackend, failing, happy_backend

from autoimprover.types import Call

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."
TASK_MODEL, JUDGE, TARGET = "claude-haiku-4-5-20251001", "claude-opus-5-5", "claude-sonnet-5-5"


def _input(call: Call) -> str:
    """The scenario input of a task call (`situation N` from the examples fixture)."""
    match = re.search(r"situation \d+", call.user)
    assert match, call.user
    return match.group(0)


def _feed(monkeypatch, source: str, text: str | bytes, tmp_path) -> list[str]:
    """Hand the prompt to the tool through `source`; return the argv that goes with it."""
    data = text if isinstance(text, bytes) else text.encode("utf-8")
    if source == "file":
        path = tmp_path / "prompt.txt"
        path.write_bytes(data)
        return ["--file", str(path)]
    if source == "stdin":
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))
        return []
    return [data.decode("utf-8")]


@pytest.mark.parametrize("source", ["argument", "file", "stdin"])
@pytest.mark.parametrize(
    "text, seen",
    [
        (ORIGINAL, ORIGINAL),
        ("Line one.\r\nLine two.", "Line one.\nLine two."),
        ("Line one.\rLine two.", "Line one.\nLine two."),
        ("x" * 20_000, "x" * 20_000),
    ],
    ids=["plain", "crlf", "cr", "20000-chars"],
)
def test_prompt_reaches_the_model_from_argument_file_or_stdin(
    run_cli, monkeypatch, tmp_path, source, text, seen
):
    """R1: the prompt comes from an argument, --file or stdin (UTF-8, up to 20,000 chars) with line
    endings normalised to LF; the first model call (intake) receives exactly that text."""
    backend = failing()
    r = run_cli([*_feed(monkeypatch, source, text, tmp_path)], backend)
    assert r.code == 3, r.err  # accepted: the run reached the first paid call, which fails
    assert backend.calls[0].role == "intake"
    assert backend.calls[0].user == seen


BAD_INPUT = {  # name -> (source, prompt text or bytes, words the error must name)
    "empty": ("argument", "", "empty"),
    "nul": ("file", "before\x00after", "NUL"),
    "too-long": ("stdin", "x" * 20_001, "20"),
    "curr-param-token": ("argument", "Use <curr_param> here.", "<curr_param>"),
    "side-info-token": ("file", "Read <side_info> first.", "<side_info>"),
    "not-utf-8": ("file", b"caf\xe9 au lait", "UTF-8"),
}


@pytest.mark.parametrize("case", list(BAD_INPUT))
def test_bad_prompt_is_exit_2_naming_the_problem(run_cli, monkeypatch, tmp_path, case):
    """R1, R2: an empty prompt, a NUL character, more than 20,000 chars, GEPA's template tokens or
    a file that is not UTF-8 exit 2 with a message naming the problem; no call, empty stdout."""
    source, text, names = BAD_INPUT[case]
    backend = happy_backend(IMPROVED)
    r = run_cli([*_feed(monkeypatch, source, text, tmp_path)], backend)
    assert r.code == 2, r.err
    assert r.out == ""
    assert backend.calls == []
    error = next(line for line in r.err.splitlines() if line.startswith("error: "))
    assert names.lower() in error.lower()


def test_missing_prompt_file_is_exit_2_naming_the_flag(run_cli, tmp_path):
    """R1, R2: an unreadable --file is exit 2 and the error names --file."""
    r = run_cli(["--file", str(tmp_path / "missing.txt")])
    assert r.code == 2
    assert "--file" in next(line for line in r.err.splitlines() if line.startswith("error: "))


def _seed_calls(calls: list[Call]) -> list[Call]:
    return [
        c
        for c in calls
        if c.role == "task" and c.model == TARGET and MARKER not in c.system + c.user
    ]


@pytest.mark.parametrize(
    "guess, flag, shape",
    [
        ("task", [], "task"),
        ("template", [], "template"),
        ("task", ["--kind", "template"], "template"),
        ("template", ["--kind", "task"], "task"),
    ],
)
def test_kind_decides_how_a_candidate_is_run(run_cli, guess, flag, shape):
    """R5, R10a: the intake guesses the kind and --kind overrides it. `template`: the candidate is
    the system prompt and the scenario input the user message; `task`: the user message is the
    situation, a blank line, then the candidate, with no system prompt (ADR-008)."""
    backend = happy_backend(IMPROVED, kind=guess)
    r = run_cli(["--json", *flag, ORIGINAL], backend)
    assert r.code == 0, r.err
    assert r.json()["contract"]["kind"] == shape
    seeds = _seed_calls(backend.calls)
    assert seeds
    for call in seeds:
        if shape == "template":
            assert call.system == ORIGINAL
            assert call.user == _input(call)
        else:
            assert call.system == ""
            assert call.user == f"{_input(call)}\n\n{ORIGINAL}"


def test_judge_is_blind_batched_and_never_the_task_or_target_model(run_cli):
    """R10, R14, R12, R18: every judge call goes to a model that is neither the task nor the target
    model, carries a JSON schema, holds at most 6 scenarios and never the candidate text (only the
    R6 contract check compares prompts); each seed run is one judge call over the whole holdout,
    and the second seed run is a new set of calls (sample 1), not a cache replay."""
    backend = happy_backend(IMPROVED)
    r = run_cli([ORIGINAL], backend)
    assert r.code == 0, r.err
    judges = [c for c in backend.calls if c.role == "judge"]
    assert judges
    for call in judges:
        assert call.model not in (TASK_MODEL, TARGET)
        assert call.json_schema
        items = json.loads(call.user)["scenarios"]
        assert 1 <= len(items) <= 6
        if items[0]["scenario"] != "contract":
            assert ORIGINAL not in call.user + call.system
            assert MARKER not in call.user + call.system
    seeds = _seed_calls(backend.calls)
    assert sorted(c.sample for c in seeds) == [0] * 4 + [1] * 4
    assert len(set(seeds)) == 8
    seed_judges = [c for c in judges if c.sample == 1]
    assert len(seed_judges) == 1
    assert {i["input"] for i in json.loads(seed_judges[0].user)["scenarios"]} == {
        _input(c) for c in seeds
    }
    for role in ("intake", "synth"):
        assert all(c.json_schema for c in backend.calls if c.role == role)


def test_examples_expected_and_criteria_become_judged_checks(run_cli, examples):
    """R11: `expected` adds one judged check "agrees with the reference answer in substance" and
    each `criteria` string one judged check; they reach the judge with the scenario."""
    path = examples(12, expected="Three bullets", criteria=["is polite", "names the owner"])
    backend = happy_backend(IMPROVED)
    r = run_cli(["--examples", path, ORIGINAL], backend)
    assert r.code == 0, r.err
    first_seed_judge = next(c for c in backend.calls if c.role == "judge")
    texts = [c["text"] for c in json.loads(first_seed_judge.user)["scenarios"][0]["checks"]]
    assert any("is polite" in t for t in texts)
    assert any("names the owner" in t for t in texts)
    assert any("reference answer" in t for t in texts)
    assert len(texts) == 4  # the contract's one judged check, the reference, two criteria


SPLITS = {8: 3, 10: 4, 12: 4, 40: 6}  # SPEC R15: holdout min(6, max(3, round(0.35 n)))


@pytest.mark.parametrize("n", list(SPLITS))
def test_holdout_size_and_the_search_never_sees_it(run_cli, examples, n):
    """R15, R14a: with n scenarios the holdout has min(6, max(3, round(0.35 n))) of them, both seed
    runs use exactly the holdout on the target model, the search model never sees a holdout
    scenario, and the split is the same on a second run (fixed seed)."""
    holdouts = []
    for _ in range(2):
        backend = happy_backend(IMPROVED)
        r = run_cli(["--examples", examples(n), ORIGINAL], backend)
        assert r.code == 0, r.err
        seeds = _seed_calls(backend.calls)
        holdout = {_input(c) for c in seeds if c.sample == 0}
        assert holdout == {_input(c) for c in seeds if c.sample == 1}
        searched = {_input(c) for c in backend.calls if c.role == "task" and c.model == TASK_MODEL}
        assert len(holdout) == SPLITS[n]
        assert holdout.isdisjoint(searched)
        assert searched
        holdouts.append(holdout)
    assert holdouts[0] == holdouts[1]


def test_below_8_scenarios_every_scenario_is_dataset_and_valset(run_cli, examples):
    """R15, R11: below 8 (with --trust-search) every scenario is used by the search."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--trust-search", "--examples", examples(7), ORIGINAL], backend)
    assert r.code == 0, r.err
    searched = {_input(c) for c in backend.calls if c.role == "task" and c.model == TASK_MODEL}
    assert searched == {f"situation {i}" for i in range(1, 8)}


def test_one_call_at_a_time_on_the_callers_thread(run_cli):
    """R15, R17: GEPA runs with parallel=False, so every model call is made one at a time from the
    caller's thread (each call here would be a `claude -p` process)."""
    inner = happy_backend(IMPROVED)
    threads: set[int] = set()
    busy = []

    def script(call: Call) -> str:
        threads.add(threading.get_ident())
        assert not busy, "two calls at once"
        busy.append(call)
        try:
            return inner.complete(call).text
        finally:
            busy.pop()

    r = run_cli([ORIGINAL], ScriptedBackend(script))
    assert r.code == 0, r.err
    assert threads == {threading.get_ident()}
