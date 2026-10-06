"""The report (SPEC R2, R3, R4, R5, R7, R11, R12, R13, R14a, R17, R23): the human report of every
Outcome, its `--json` object, the plan of `--dry`, the error object, and the one writer to stdout,
which prints at most one result per run and keeps model-written text from steering the terminal
(SPEC R19). What the command line does around them is tested in `test_cli*.py`."""

import dataclasses
import io
import json

import pytest

from autoimprover import cli_plan, report, runner
from autoimprover.types import (
    DEFAULT_MODELS,
    REASON_CODES,
    Check,
    Contract,
    Outcome,
    Plan,
)

ORIGINAL = "Answer the user's request."
BETTER = "Answer the user's request in three short steps."
PLAN = Plan(models=DEFAULT_MODELS)
RUN = "/state/autoimprover/runs/20261004-120000-abcdef12"
CONTRACT = Contract(
    goal="answer the user's request well",
    kind="task",
    keep=("the user's request",),
    constraints=("no tables",),
    output_format="",
    language="en",
    tone="neutral",
    checks=(
        Check(id="c1", group="content", text="answers the request"),
        Check(id="c2", group="format", text="at most 400 characters", rule="max_chars", arg="400"),
    ),
)
IMPROVED = Outcome(
    status="improved",
    prompt=BETTER,
    reason="beats the original on the holdout, on the target model, by more than the noise",
    reason_code="improved",
    verified=True,
    stop="budget",
    changes=("tightened the wording", "kept the output format", "kept every literal"),
    score_before=0.25,
    score_after=1.0,
    search_score_before=0.0,
    search_score_after=1.0,
    noise=0.1,
    margin=0.55,
    length_ratio=1.75,
    calls_used=87,
    run_dir=RUN,
)
UNCHANGED = Outcome(
    status="unchanged",
    prompt=ORIGINAL,
    reason="no reliable improvement: no candidate beat the original on the holdout by more than "
    "the noise",
    reason_code="no_reliable_improvement",
    stop="budget",
    score_before=0.25,
    search_score_before=0.0,
    noise=0.1,
    calls_used=100,
    run_dir=RUN,
)


def outcome_for(code: str) -> Outcome:
    if code == "improved":
        return IMPROVED
    return dataclasses.replace(UNCHANGED, reason_code=code, reason=f"the runner's text for {code}")


# --- the human report (SPEC R2, R3, R5, R7, R12, R13, R14a, R17, R23) -----------------------------


def test_every_reason_code_has_its_own_plain_language_line():
    assert set(report.REASON_LINES) == set(REASON_CODES)
    lines = [report.REASON_LINES[code] for code in REASON_CODES]
    assert all(line.strip() for line in lines) and len(set(lines)) == len(lines)


@pytest.mark.parametrize("code", REASON_CODES)
def test_the_report_of_each_reason_code_carries_its_code_its_reason_and_its_line(code: str):
    outcome = outcome_for(code)
    text = report.render(outcome, ORIGINAL, CONTRACT, PLAN)
    assert f"({code})" in text and outcome.reason in text
    assert report.REASON_LINES[code] in text
    others = [report.REASON_LINES[c] for c in REASON_CODES if c != code]
    assert not any(line in text for line in others)


def test_the_improved_report_snapshot():
    assert report.render(IMPROVED, ORIGINAL, CONTRACT, PLAN) == (
        "result: improved (improved)\n"
        "reason: beats the original on the holdout, on the target model, by more than the noise\n"
        f"meaning: {report.REASON_LINES['improved']}\n"
        "verified: yes, on the holdout, on the target model claude-sonnet-5-5\n"
        "holdout score (target model claude-sonnet-5-5): 0.25 before, 1.00 after\n"
        "noise: 0.10 between the original's two holdout runs; the result cleared the bar of "
        "0.20 by 0.55\n"
        "search score (valset, search model claude-haiku-4-5-20251001): 0.00 before, "
        "1.00 after\n"
        "length: 1.75x the original's tokens\n"
        "calls used: 87 of 100\n"
        "search ended: it used its share of the calls (the normal ending)\n"
        "what changed and why:\n"
        "  - tightened the wording\n"
        "  - kept the output format\n"
        "  - kept every literal\n"
        "word diff: Answer the user's [-request.-] {+request in three short steps.+}\n"
        "intent contract (task):\n"
        "  goal: answer the user's request well\n"
        "  keep: the user's request\n"
        "  constraints: no tables\n"
        "  output format: none required; language: en; tone: neutral\n"
        "  check c1 (content, judged): answers the request\n"
        "  check c2 (format, max_chars 400): at most 400 characters\n"
        "note: a task prompt runs single-turn without tools here, so tool use is not exercised\n"
        "note: 100 calls is a very low budget for GEPA (the paper's runs used 400 to 7,000 "
        "rollouts); the first few reflective updates carry most of the gain\n"
        f"run folder: {RUN} (remove it with: autoimprover clean 20261004-120000-abcdef12)\n"
    )


def test_the_unchanged_report_snapshot():
    assert report.render(UNCHANGED, ORIGINAL, CONTRACT, PLAN) == (
        "result: unchanged, the original prompt is returned (no_reliable_improvement)\n"
        f"reason: {UNCHANGED.reason}\n"
        f"meaning: {report.REASON_LINES['no_reliable_improvement']}\n"
        "holdout score (target model claude-sonnet-5-5): 0.25 for the original\n"
        "noise: 0.10 between the original's two holdout runs; a result had to gain more "
        "than 0.20\n"
        "search score (valset, search model claude-haiku-4-5-20251001): 0.00 for the original\n"
        "calls used: 100 of 100\n"
        "search ended: it used its share of the calls (the normal ending)\n"
        "intent contract (task):\n"
        "  goal: answer the user's request well\n"
        "  keep: the user's request\n"
        "  constraints: no tables\n"
        "  output format: none required; language: en; tone: neutral\n"
        "  check c1 (content, judged): answers the request\n"
        "  check c2 (format, max_chars 400): at most 400 characters\n"
        "note: a task prompt runs single-turn without tools here, so tool use is not exercised\n"
        "note: 100 calls is a very low budget for GEPA (the paper's runs used 400 to 7,000 "
        "rollouts); the first few reflective updates carry most of the gain\n"
        f"run folder: {RUN} (remove it with: autoimprover clean 20261004-120000-abcdef12)\n"
    )


def test_the_bar_is_the_least_gain_when_the_runs_agree():
    agreed = dataclasses.replace(UNCHANGED, noise=0.0)
    text = report.render(agreed, ORIGINAL, CONTRACT, PLAN)
    assert f"a result had to gain more than {runner.MIN_THRESHOLD:.2f}" in text


def test_a_trust_search_result_is_marked_not_verified_and_shows_no_holdout_scores():
    unverified = dataclasses.replace(
        IMPROVED, verified=False, score_before=None, score_after=None, noise=None, margin=None
    )
    text = report.render(unverified, ORIGINAL, CONTRACT, PLAN)
    assert "NOT VERIFIED on a holdout" in text and "--trust-search" in text
    assert "holdout score" not in text and "verified: yes" not in text
    assert "search score (valset" in text
    assert report.REASON_LINES["improved"] not in text and "never saw" not in text
    [meaning] = [line for line in text.splitlines() if line.startswith("meaning: ")]
    assert "validation set" in meaning and "no holdout" in meaning
    assert "no noise was measured" in meaning


def test_only_a_clock_stop_says_the_search_was_cut_short():
    by_clock = report.render(dataclasses.replace(IMPROVED, stop="clock"), ORIGINAL, None, PLAN)
    assert "cut short by the clock" in by_clock
    for stop in ("budget", None):
        text = report.render(dataclasses.replace(UNCHANGED, stop=stop), ORIGINAL, None, PLAN)
        assert "cut short" not in text
    no_search = report.render(dataclasses.replace(UNCHANGED, stop=None), ORIGINAL, None, PLAN)
    assert "search ended: no search ran" in no_search and "low budget" not in no_search


def test_an_already_strong_report_lists_the_checks_so_the_user_can_judge():
    strong = dataclasses.replace(UNCHANGED, reason_code="already_strong", stop=None)
    text = report.render(strong, ORIGINAL, CONTRACT, PLAN)
    assert "check c1 (content, judged): answers the request" in text


def test_a_template_has_no_tool_use_note_and_a_missing_contract_or_folder_is_left_out():
    template = dataclasses.replace(CONTRACT, kind="template")
    assert "tool use" not in report.render(UNCHANGED, ORIGINAL, template, PLAN)
    bare = report.render(dataclasses.replace(UNCHANGED, run_dir=""), ORIGINAL, None, PLAN)
    assert "intent contract" not in bare and "run folder" not in bare


def test_the_search_score_is_only_shown_when_the_search_ran():
    no_search = dataclasses.replace(UNCHANGED, search_score_before=None, stop=None)
    assert "search score" not in report.render(no_search, ORIGINAL, CONTRACT, PLAN)


# --- the --json object (SPEC R2) -----------------------------------------------------------------

JSON_KEYS = {
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
    "mode",
    "elapsed_s",
    "meaning",
    "verified_text",
    "margin_text",
}


def test_the_json_object_has_the_fields_of_the_spec_and_survives_a_round_trip():
    obj = report.outcome_object(IMPROVED, ORIGINAL, CONTRACT)
    assert set(obj) == JSON_KEYS
    assert json.loads(json.dumps(obj)) == obj
    assert obj["contract"] == json.loads(json.dumps(dataclasses.asdict(CONTRACT)))
    assert obj["diff"] == report.word_diff(ORIGINAL, BETTER) and obj["changes"] == list(
        IMPROVED.changes
    )
    assert (obj["status"], obj["reason_code"], obj["verified"]) == ("improved", "improved", True)


def test_an_unchanged_object_has_no_diff_and_a_missing_contract_is_null():
    obj = report.outcome_object(UNCHANGED, ORIGINAL, None)
    assert (obj["diff"], obj["contract"], obj["prompt"]) == ("", None, ORIGINAL)


def test_the_error_object_has_exactly_the_four_fields():
    assert report.error_object(3, "boom", RUN) == {
        "status": "error",
        "code": 3,
        "error": "boom",
        "run_dir": RUN,
    }


# --- model-written text is data (SPEC R19) and the word diff (SPEC R2) ---------------------------

HOSTILE = "a\x1b[2J\x1b]0;owned\x07b\x07c\x9b31md\u202ee\x7ff\rg\x00h\x08i"


def test_terminal_escapes_and_control_characters_are_removed():
    assert report.clean_text(HOSTILE) == "abcdefghi"


def test_newlines_tabs_and_ordinary_unicode_are_kept():
    text = "first line\n\tsecond — ünïcödé 日本語 \u200f"
    assert report.clean_text(text) == text


def test_one_line_also_folds_line_breaks():
    assert report.one_line("a\nb\t c\x1b[1m d") == "a b c d"


@pytest.mark.parametrize(
    ("before", "after", "diff"),
    [
        ("a b c", "a b c", "a b c"),
        ("a b c", "a x c", "a [-b-] {+x+} c"),
        ("a b", "a b c d", "a b {+c d+}"),
        ("a b c", "c", "[-a b-] c"),
        ("one\ntwo", "one  two", "one two"),
    ],
)
def test_the_word_diff_marks_removed_and_added_words(before: str, after: str, diff: str):
    assert report.word_diff(before, after) == diff


def test_a_long_prompt_of_common_words_diffs_only_the_word_that_changed():
    side = " ".join(["the"] * 100)  # difflib's autojunk would treat "the" as junk here
    diff = report.word_diff(f"{side} old {side}", f"{side} new {side}")
    assert diff == f"{side} [-old-] {{+new+}} {side}"


# --- the plan (SPEC R4, R17) ---------------------------------------------------------------------


def view(budget: int = 100, n: int = 12, synthesised: bool = True, **fields) -> cli_plan.PlanView:
    plan = Plan(models=DEFAULT_MODELS, budget=budget)
    costs = runner.fixed_costs(plan, n, synthesising=synthesised)
    return cli_plan.PlanView(plan=plan, scenarios=n, synthesised=synthesised, costs=costs, **fields)


def test_the_plan_names_models_budget_fixed_costs_scenarios_iterations_and_the_final_share():
    text = cli_plan.plan_text(view())
    assert "task claude-haiku-4-5-20251001, judge claude-opus-5-5" in text
    assert "reflection claude-opus-5-5, target claude-sonnet-5-5" in text
    assert "100 calls (ceiling 300)" in text
    assert "16 before the search, 18 after it, 66 left for the search" in text
    assert "scenarios: 12, synthesised by one call; holdout 4, valset 3, dataset 5" in text
    assert "about 5 GEPA iterations (worst case, 13 calls each) to 7" in text
    assert "the clock may end the search sooner" in text
    assert "the final steps keep 11 min 15 s" in text
    assert "would refuse" not in text


def test_the_plan_says_why_a_real_run_would_refuse_or_keep_the_original():
    assert "a real run would refuse: too few" in cli_plan.plan_text(view(refusal="too few"))
    seven = view(n=7, synthesised=False, keeps_original="7 scenarios")
    text = cli_plan.plan_text(seven)
    assert "scenarios: 7, from --examples; no holdout" in text
    assert "a real run would keep the original without a model call: 7 scenarios" in text


def test_the_plan_object_carries_the_same_numbers():
    obj = cli_plan.plan_object(view(refusal="too few"))
    assert obj["status"] == "dry" and obj["plan"] == dataclasses.asdict(PLAN)
    assert (obj["scenarios"], obj["holdout"], obj["valset"], obj["dataset"]) == (12, 4, 3, 5)
    assert (obj["calls_before_search"], obj["calls_after_search"]) == (16, 18)
    assert (obj["search_calls"], obj["iterations"], obj["iterations_best"]) == (66, 5, 7)
    assert (obj["search_clock_s"], obj["final_clock_s"]) == (2025, 675)
    assert (obj["refusal"], obj["keeps_original"], obj["synthesised"]) == ("too few", None, True)


# --- the one writer (SPEC R2) --------------------------------------------------------------------


def emitter(json_mode: bool) -> tuple[report.Emitter, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    return report.Emitter(out, err, json_mode=json_mode), out, err


def test_without_json_the_prompt_alone_goes_to_stdout_and_the_report_to_stderr():
    emit, out, err = emitter(False)
    emit.outcome(IMPROVED, ORIGINAL, CONTRACT, PLAN)
    assert out.getvalue() == BETTER + "\n"
    assert err.getvalue() == report.render(IMPROVED, ORIGINAL, CONTRACT, PLAN)


def test_a_model_written_prompt_is_cleaned_and_the_users_own_original_is_kept_verbatim():
    emit, out, _ = emitter(False)
    emit.outcome(dataclasses.replace(IMPROVED, prompt=f"{BETTER}{HOSTILE}"), ORIGINAL, None, PLAN)
    assert out.getvalue() == f"{BETTER}abcdefghi\n"
    original = "Keep\tthis \x1b[1mexactly\x1b[0m\n"
    emit, out, _ = emitter(False)
    emit.outcome(dataclasses.replace(UNCHANGED, prompt=original), original, None, PLAN)
    assert out.getvalue() == original


def test_with_json_stdout_holds_one_object_and_stderr_only_the_notices():
    emit, out, err = emitter(True)
    clock = dataclasses.replace(IMPROVED, stop="clock", verified=False, prompt=BETTER + HOSTILE)
    emit.outcome(clock, ORIGINAL, CONTRACT, PLAN)
    assert json.loads(out.getvalue()) == report.outcome_object(clock, ORIGINAL, CONTRACT)
    assert out.getvalue().count("\n") == 1 and "\x1b" not in out.getvalue()
    assert "\x7f" not in out.getvalue()
    notices = err.getvalue()
    assert "cut short by the clock" in notices and "NOT VERIFIED" in notices
    assert f"run folder: {RUN}" in notices and "word diff" not in notices


def test_a_second_result_is_refused_and_stdout_keeps_the_first():
    emit, out, _ = emitter(True)
    emit.outcome(UNCHANGED, ORIGINAL, None, PLAN)
    with pytest.raises(RuntimeError, match="second result"):
        emit.outcome(IMPROVED, ORIGINAL, None, PLAN)
    assert json.loads(out.getvalue())["status"] == "unchanged"


def test_an_error_after_the_result_never_writes_a_second_object():
    emit, out, err = emitter(True)
    emit.outcome(UNCHANGED, ORIGINAL, None, PLAN)
    assert emit.error(1, "internal error: OSError: disk full", RUN) == 1
    assert len(out.getvalue().splitlines()) == 1
    assert json.loads(out.getvalue())["status"] == "unchanged"
    assert "error: internal error: OSError: disk full" in err.getvalue()


@pytest.mark.parametrize("json_mode", [False, True])
def test_an_error_writes_its_lines_to_stderr_and_with_json_one_object(json_mode: bool):
    emit, out, err = emitter(json_mode)
    resume = "autoimprover --resume 20261004-120000-abcdef12"
    assert emit.error(3, "backend \x1b[31mdown", RUN, resume) == 3
    assert err.getvalue() == (f"error: backend down\nrun folder: {RUN}\nresume with: {resume}\n")
    expected = json.dumps(report.error_object(3, "backend down", RUN)) + "\n"
    assert out.getvalue() == (expected if json_mode else "")


def test_a_usage_error_prints_no_run_folder_line():
    emit, _, err = emitter(False)
    emit.error(2, "--budget: too high", RUN)
    assert err.getvalue() == "error: --budget: too high\n"


@pytest.mark.parametrize("json_mode", [False, True])
def test_the_dry_plan_goes_to_stdout_and_a_real_runs_plan_to_stderr(json_mode: bool):
    emit, out, err = emitter(json_mode)
    emit.plan(view(), dry=True)
    if json_mode:
        assert json.loads(out.getvalue()) == cli_plan.plan_object(view())
    else:
        assert out.getvalue().startswith("dry run: no model call made, nothing written\n")
    assert err.getvalue() == ""
    emit, out, err = emitter(json_mode)
    emit.plan(view(), dry=False)
    assert out.getvalue() == "" and err.getvalue() == cli_plan.plan_text(view())


@pytest.mark.parametrize("json_mode", [False, True])
def test_clean_reports_on_stderr_and_with_json_one_object(json_mode: bool):
    emit, out, err = emitter(json_mode)
    emit.cleaned(2, 1, "/state/autoimprover/runs")
    assert "removed 2 run folders from /state/autoimprover/runs" in err.getvalue()
    assert "skipped 1 run that is still running" in err.getvalue()
    expected = {"status": "cleaned", "removed": 2, "skipped": 1}
    assert (json.loads(out.getvalue()) if json_mode else out.getvalue()) == (
        expected if json_mode else ""
    )
    emit, _, err = emitter(json_mode)
    emit.cleaned(1, 0, "/state/autoimprover/runs")
    assert err.getvalue() == "removed 1 run folder from /state/autoimprover/runs\n"
