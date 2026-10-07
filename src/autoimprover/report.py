"""What the tool prints (SPEC R2): `render` builds the human report of an Outcome (scores before
and after, labelled holdout or search score, noise and the bar a result had to clear, length
ratio, calls used, why the search ended, whether the result is verified, the reason in plain
language, what changed and the word diff, the intent contract with its checks and the rules it
learned from the examples (ADR-013), and the run folder; SPEC R3, R5, R7, R11, R12, R13, R14a,
R17, R23), `outcome_object` its `--json` object, `error_object` the object of a failed run.
`Emitter` is the only writer to stdout: at most one result per run, so `--json` gives exactly one
object whatever fails after it, and notices stay on stderr; the plan of `--dry` (SPEC R4) it
prints is built in `cli_plan.py`.

The time tiers (SPEC R25): a report names its tier and the seconds of the run's clock; a quick or
fast result says, in its verified line, its meaning, its notices and its JSON, that it is not
verified on held-out scenarios; its two numbers are the shares of the scenarios it was picked on
that the pairwise judge gave to the original and to the rewrite (a preference, never a score or a
holdout score), its noise the share where the original's two runs had a winner, and its bar that
noise (`fast_stages`, ADR-012); nothing of the GEPA search is said of a run that did not search.
An `--ungated` result (fast or checked) has its own reason code and label, is never verified, and
shows the same two shares as a preference on the scenarios it was picked on, with its lead, which
it did not need. A reference-scored result (every example carried a reference, WP21; its reason
holds `reference_text.MARK`) says so in place of the preference: its two numbers are mean reference
scores, its noise the difference of the original's two runs, its margin the gain over that noise.

A reference-scored reason ends with the reflection rounds and the pick examples passed (WP23,
`reference_text.ROUNDS`); the report prints them on a line of their own.

Model-written text (an improved prompt, its "what changed" lines, the contract, messages that
may quote a reply) is data, never instructions to the terminal: escape sequences and control
characters are removed before it is printed (SPEC R19, `report_clean`). The user's own original
prompt is printed as given."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol, TextIO

from autoimprover import reference_text
from autoimprover.report_clean import clean_text as clean_text
from autoimprover.report_clean import one_line as one_line
from autoimprover.report_clean import word_diff as word_diff
from autoimprover.report_text import (
    FAST_REASON_LINES,
    FAST_UNVERIFIED,
    FAST_UNVERIFIED_NOISE,
    FAST_VERIFIED,
    REASON_LINES,
)
from autoimprover.runner import MIN_THRESHOLD
from autoimprover.types import Contract, Outcome, Plan

# The tiers that run the fast pipeline, not the GEPA search (SPEC R25); quick and fast are never
# verified on held-out scenarios, checked is.
FAST_TIERS = ("quick", "fast", "checked")
_UNHELD = ("quick", "fast")
# What each reason code means is REASON_LINES and FAST_REASON_LINES (`report_text`, re-exported).
_UNGATED = "ungated_best_candidate"

# An improved result that no holdout checked (--trust-search, SPEC R11) says only what was done.
_UNVERIFIED_MEANING = (
    "a rewrite scored higher than the original on the search's own validation set; there is no "
    "holdout and no noise was measured, so the gain is not verified"
)
_STOPS = {
    "budget": "it used its share of the calls (the normal ending)",
    "clock": "the clock ended it, at its share of the wall clock",
    None: "no search ran",
}
# The same for the fast pipeline, which has stages, not a search (SPEC R25).
_FAST_STOPS = {
    None: "every stage ran as planned",
    "clock": "the clock cut a stage short or shrank it",
    "budget": "the call limit cut a stage short or shrank it",
}
_PICKED = "the scenarios it was picked on"
_FAST_CUT_SHORT = "notice: the clock cut the run short; the result comes from what it scored"


# --- the outcome ---------------------------------------------------------------------------------


def render(
    outcome: Outcome,
    original: str,
    contract: Contract | None,
    plan: Plan,
    elapsed_s: float | None = None,
) -> str:
    """The human report of `outcome` (stderr), one fact per line; `original` is the user's prompt,
    `contract` the run's intent contract when one was extracted, `elapsed_s` the seconds on the
    run's clock, shown beside the tier when the outcome has one (SPEC R25)."""
    improved = outcome.status == "improved"
    fast = outcome.mode in FAST_TIERS
    head = "improved" if improved else "unchanged, the original prompt is returned"
    reason, rounds = _reason(outcome)
    lines = [
        f"result: {head} ({outcome.reason_code})",
        f"reason: {one_line(reason)}",
        f"meaning: {_meaning(outcome)}",
    ]
    if improved:
        lines.append(_verified(outcome, plan))
    if outcome.mode is not None:
        lines.append(f"mode: {outcome.mode} ({elapsed_s or 0.0:.0f} s)")
    lines += _scores(outcome, plan)
    if rounds:  # the reflection rounds of a reference-scored result (WP23)
        lines.append(f"rounds: {one_line(rounds)}")
    if improved and outcome.length_ratio is not None:
        lines.append(f"length: {outcome.length_ratio:.2f}x the original's tokens")
    lines.append(f"calls used: {outcome.calls_used} of {plan.budget}")
    if fast:
        lines.append(f"stages: {_FAST_STOPS[outcome.stop]}")
    else:
        lines.append(f"search ended: {_STOPS[outcome.stop]}")
    if outcome.stop == "clock":
        lines.append(_FAST_CUT_SHORT if fast else _CUT_SHORT)
    if improved:
        if outcome.changes:
            lines.append("what changed and why:")
            lines += [f"  - {one_line(change)}" for change in outcome.changes]
        lines.append(f"word diff: {clean_text(word_diff(original, outcome.prompt))}")
    if contract is not None:
        lines += _contract_lines(contract)
        if contract.kind == "task":
            lines.append(
                "note: a task prompt runs single-turn without tools here, so tool use is not "
                "exercised"
            )
    if outcome.stop is not None and not fast:
        lines.append(
            f"note: {plan.budget} calls is a very low budget for GEPA (the paper's runs used 400 "
            "to 7,000 rollouts); the first few reflective updates carry most of the gain"
        )
    lines += _folder(outcome.run_dir)
    return "".join(f"{line}\n" for line in lines)


_UNVERIFIED = (
    "NOT VERIFIED on a holdout: with --trust-search the result only beat the original on the "
    "scenarios the search itself used"
)
_CUT_SHORT = "notice: the search was cut short by the clock; the result comes from what it scored"


def _meaning(outcome: Outcome) -> str:
    """What the outcome means for the user, in the words of its tier."""
    if outcome.reason_code == _UNGATED:
        return REASON_LINES[_UNGATED]
    if outcome.mode in FAST_TIERS:
        ref = _referenced(outcome)
        if outcome.status == "improved" and outcome.verified:
            return reference_text.MEANING_VERIFIED if ref else FAST_VERIFIED
        if outcome.status == "improved" and ref:
            return reference_text.MEANING_UNVERIFIED
        if outcome.status == "improved":
            return FAST_UNVERIFIED if outcome.noise is None else FAST_UNVERIFIED_NOISE
        if ref and outcome.reason_code == "no_reliable_improvement":
            return reference_text.MEANING_NO_WIN
        return FAST_REASON_LINES.get(outcome.reason_code, REASON_LINES[outcome.reason_code])
    if outcome.status == "improved" and not outcome.verified:
        return _UNVERIFIED_MEANING
    return REASON_LINES[outcome.reason_code]


def _verified(outcome: Outcome, plan: Plan | None) -> str:
    """Whether an improved result is verified; a quick or fast one is not, in its own label."""
    if outcome.verified:
        target = f" {plan.models.target}" if plan is not None else ""
        return f"verified: yes, on the holdout, on the target model{target}"
    if outcome.mode in FAST_TIERS:
        return f"verified: no. NOT VERIFIED ({one_line(_reason(outcome)[0])})"
    return f"verified: no. {_UNVERIFIED}"


def _reason(outcome: Outcome) -> tuple[str, str]:
    """The reason of `outcome` and, for a reference-scored one, the rounds it ends with (WP23,
    `reference_text.ROUNDS`, "" without them)."""
    if not _referenced(outcome):
        return outcome.reason, ""
    reason, _mark, rounds = outcome.reason.partition(reference_text.ROUNDS_MARK)
    return reason, rounds


def _scores(outcome: Outcome, plan: Plan) -> list[str]:
    lines = []
    fast, ref = outcome.mode in FAST_TIERS, _referenced(outcome)
    kind, shown = ("reference score", _before_after) if ref else ("preference", _preference)
    picked = f"{kind} on {_PICKED} (judge {plan.models.judge}, not held out)"
    if outcome.score_before is not None:
        if outcome.mode in _UNHELD:
            lines.append(f"{picked}: {shown(outcome.score_before, outcome.score_after)}")
        else:
            where = f"holdout score (target model {plan.models.target})"
            lines.append(f"{where}: {_before_after(outcome.score_before, outcome.score_after)}")
    margin = _margin_text(outcome)
    if outcome.noise is not None and fast:
        said = (
            f"noise: {outcome.noise:.2f} of {_PICKED} had a winner between the original's two runs"
        )
        bar = f"; a rewrite had to lead by more than {outcome.noise:.2f}"
        if ref:
            runs = "two held-out runs" if _held_out(outcome) else "two runs"
            said = f"noise: {outcome.noise:.2f} between the mean reference scores of the original's"
            said += f" {runs}"
            bar = f"; a rewrite had to gain more than {outcome.noise:.2f}"
        lines.append(said if margin is not None else said + bar)
    elif outcome.noise is not None:
        bar = max(MIN_THRESHOLD, 2 * outcome.noise)
        said = f"noise: {outcome.noise:.2f} between the original's two holdout runs"
        lines.append(f"{said}; {margin or f'a result had to gain more than {bar:.2f}'}")
    if outcome.search_score_before is not None:
        if fast:
            preferred = shown(outcome.search_score_before, outcome.search_score_after)
            lines.append(f"{picked}: {preferred}")
        else:
            where = f"search score (valset, search model {plan.models.task})"
            searched = _before_after(outcome.search_score_before, outcome.search_score_after)
            lines.append(f"{where}: {searched}")
    if fast and margin is not None:
        lines.append(f"margin: {margin}")
    return lines


def _margin_text(outcome: Outcome) -> str | None:
    """How far a returned result cleared what it had to: a fast result with noise as its lead
    (the share of the scenarios it won minus the share it lost) against that noise; a fast or
    checked result without noise as its margin over the least gain; a deep result as the bar of
    its noise; an ungated result as its lead on the scenarios it was picked on, which it did not
    need. None without a margin."""
    before, after, margin, noise = (
        outcome.score_before,
        outcome.score_after,
        outcome.margin,
        outcome.noise,
    )
    if margin is None:
        return None
    ref = " in the mean reference score" if _referenced(outcome) else ""
    lead = "gain" if ref else "lead"
    if outcome.reason_code == _UNGATED:
        before, after = outcome.search_score_before, outcome.search_score_after
        if before is None or after is None:
            return None
        against = "" if noise is None else f" vs noise {noise:.2f}"
        return (
            f"{lead} {after - before:.2f}{against}{ref} on {_PICKED}; ungated, so no {lead} was "
            "required"
        )
    if outcome.mode not in FAST_TIERS:
        if noise is None:
            return None
        return f"the result cleared the bar of {max(MIN_THRESHOLD, 2 * noise):.2f} by {margin:.2f}"
    if before is None or after is None:
        return None
    if noise is not None:
        where = "the held-out examples" if _held_out(outcome) else _PICKED
        return f"{lead} {after - before:.2f} vs noise {noise:.2f}{ref} on {where}"
    held = "held-out scenarios" if outcome.verified else "scenarios it was picked on"
    return (
        f"{margin:.2f} above the least gain of {after - before - margin:.2f} on the {held} (no "
        "noise measured)"
    )


def _referenced(outcome: Outcome) -> bool:
    """Whether a fast or checked result was decided by the user's references (WP21)."""
    return outcome.mode in FAST_TIERS and reference_text.MARK in outcome.reason


def _held_out(outcome: Outcome) -> bool:
    """Whether a reference-scored result's noise is that of stage E (checked, not ungated)."""
    return outcome.mode == "checked" and outcome.reason_code != _UNGATED


def _preference(before: float, after: float | None) -> str:
    """The shares of the scenarios the pairwise judge gave to the original and to the rewrite."""
    if after is None:
        return f"the original won {before:.2f}"
    return f"the original won {before:.2f}, the rewrite {after:.2f}"


def _before_after(before: float, after: float | None) -> str:
    if after is None:
        return f"{before:.2f} for the original"
    return f"{before:.2f} before, {after:.2f} after"


def _contract_lines(contract: Contract) -> list[str]:
    def listed(items: tuple[str, ...], none: str) -> str:
        return "; ".join(one_line(item) for item in items) if items else none

    learned = contract.from_examples  # the rules the user's examples showed (ADR-013)
    lines = [
        f"intent contract ({contract.kind}):",
        f"  goal: {one_line(contract.goal)}",
        f"  keep: {listed(contract.keep, 'nothing listed')}",
        f"  constraints: {listed(contract.constraints, 'none listed')}",
        *([f"  rules learned from the examples: {listed(learned, '')}"] if learned else []),
        f"  output format: {one_line(contract.output_format) or 'none required'}; "
        f"language: {one_line(contract.language) or 'not stated'}; "
        f"tone: {one_line(contract.tone) or 'not stated'}",
    ]
    for check in contract.checks:
        how = "judged" if check.rule is None else f"{check.rule} {one_line(check.arg or '')}"
        lines.append(f"  check {one_line(check.id)} ({check.group}, {how}): {one_line(check.text)}")
    return lines


def _folder(run_dir: str) -> list[str]:
    if not run_dir:
        return []
    run_id = run_dir.rstrip("/").rsplit("/", 1)[-1]
    return [f"run folder: {run_dir} (remove it with: autoimprover clean {run_id})"]


def notices(outcome: Outcome) -> list[str]:
    """The lines that go to stderr with `--json` too: a clock stop, an unverified result, the run
    folder (SPEC R2, R11, R23)."""
    fast = outcome.mode in FAST_TIERS
    lines = [_FAST_CUT_SHORT if fast else _CUT_SHORT] if outcome.stop == "clock" else []
    if outcome.status == "improved" and not outcome.verified:
        unverified = f"NOT VERIFIED ({one_line(_reason(outcome)[0])})" if fast else _UNVERIFIED
        lines.append(f"notice: {unverified}")
    return lines + _folder(outcome.run_dir)


def outcome_object(
    outcome: Outcome,
    original: str,
    contract: Contract | None,
    plan: Plan | None = None,
    elapsed_s: float | None = None,
) -> dict[str, Any]:
    """The `--json` object of a finished run (SPEC R2; ARCHITECTURE section 8). After `mode` it
    carries the seconds of the run's clock and the human report's own words, each its line
    without the label: `meaning`, `verified_text` (None when the report prints no verified line)
    and `margin_text` (None without a margin), so a reader shows what the command line says."""
    improved = outcome.status == "improved"
    return {
        "status": outcome.status,
        "prompt": outcome.prompt,
        "verified": outcome.verified,
        "stop": outcome.stop,
        "changes": list(outcome.changes),
        "reason": outcome.reason,
        "reason_code": outcome.reason_code,
        "diff": word_diff(original, outcome.prompt) if improved else "",
        "contract": None if contract is None else json.loads(json.dumps(asdict(contract))),
        "score_before": outcome.score_before,
        "score_after": outcome.score_after,
        "search_score_before": outcome.search_score_before,
        "search_score_after": outcome.search_score_after,
        "noise": outcome.noise,
        "margin": outcome.margin,
        "length_ratio": outcome.length_ratio,
        "calls_used": outcome.calls_used,
        "run_dir": outcome.run_dir,
        "mode": outcome.mode,
        "elapsed_s": elapsed_s,
        "meaning": _meaning(outcome),
        "verified_text": _verified(outcome, plan).removeprefix("verified: ") if improved else None,
        "margin_text": _margin_text(outcome),
    }


def error_object(code: int, message: str, run_dir: str) -> dict[str, Any]:
    """The `--json` object of a run that failed (SPEC R2)."""
    return {"status": "error", "code": code, "error": message, "run_dir": run_dir}


class Shown(Protocol):
    """A plan as `--dry` shows it (SPEC R4; built in `cli_plan.py` for every tier): its lines
    (`text`) and its `--json` object (`object`)."""

    def text(self) -> str: ...

    def object(self) -> dict[str, Any]: ...


# --- the one writer (SPEC R2) --------------------------------------------------------------------


class Emitter:
    """The only writer to stdout, and the writer of every notice on stderr. One result reaches
    stdout per run: with `--json` exactly one object, without it the prompt alone (or the plan of
    `--dry`); an error after the result goes to stderr only."""

    def __init__(self, out: TextIO, err: TextIO, *, json_mode: bool) -> None:
        self._out = out
        self._err = err
        self.json_mode = json_mode
        self._done = False

    def notice(self, text: str) -> None:
        for line in text.splitlines():
            self._err.write(f"{clean_text(line)}\n")
        self._err.flush()

    def outcome(
        self,
        outcome: Outcome,
        original: str,
        contract: Contract | None,
        plan: Plan,
        elapsed_s: float | None = None,
    ) -> None:
        """A finished run: without `--json` the report to stderr and the prompt to stdout, cleaned
        when a model wrote it; with it the object, and the notices to stderr."""
        if self.json_mode:
            self._result(_dumps(outcome_object(outcome, original, contract, plan, elapsed_s)))
            self.notice("\n".join(notices(outcome)))
            return
        self.notice(render(outcome, original, contract, plan, elapsed_s))
        prompt = outcome.prompt if outcome.status == "unchanged" else clean_text(outcome.prompt)
        self._result(prompt if prompt.endswith("\n") else f"{prompt}\n")

    def plan(self, view: Shown, *, dry: bool) -> None:
        """The plan: the result of `--dry` (stdout), or a notice before a real run (stderr)."""
        if not dry:
            self.notice(view.text())
        elif self.json_mode:
            self._result(_dumps(view.object()))
        else:
            self._result(f"dry run: no model call made, nothing written\n{view.text()}")

    def cleaned(self, removed: int, skipped: int, root: str) -> None:
        self.notice(f"removed {removed} run folder{'s' * (removed != 1)} from {root}")
        if skipped:
            runs = "run that is" if skipped == 1 else "runs that are"
            self.notice(f"skipped {skipped} {runs} still running (they hold their lock)")
        if self.json_mode:
            self._result(_dumps({"status": "cleaned", "removed": removed, "skipped": skipped}))

    def help(self, text: str) -> None:
        self._result(_dumps({"status": "help", "help": text}) if self.json_mode else text)

    def error(self, code: int, message: str, run_dir: str = "", resume: str = "") -> int:
        """`error: <message>` on stderr, then for a run that has a folder (not a usage error) the
        folder and, when given, the resume line; with `--json` the error object, unless a result
        was already written. Returns `code`."""
        message = one_line(message)
        lines = [f"error: {message}"]
        if run_dir and code != 2:
            lines.append(f"run folder: {run_dir}")
            if resume:
                lines.append(f"resume with: {resume}")
        self.notice("\n".join(lines))
        if self.json_mode and not self._done:
            self._result(_dumps(error_object(code, message, run_dir)))
        return code

    def result(self, text: str) -> None:
        """`text` as the one result on stdout, as it is (the summary of `autoimprover bench`,
        SPEC R26); a second result raises RuntimeError."""
        self._result(text)

    def _result(self, text: str) -> None:
        if self._done:
            raise RuntimeError("a second result for stdout")
        self._done = True
        self._out.write(text)
        self._out.flush()


def _dumps(obj: object) -> str:
    """One line of JSON, every character outside printable ASCII escaped (`ensure_ascii`), so no
    control character reaches the terminal raw."""
    return json.dumps(obj, allow_nan=False) + "\n"
