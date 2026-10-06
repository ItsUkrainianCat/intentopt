"""What the tool prints (SPEC R2): `render` builds the human report of an Outcome (scores before
and after, labelled holdout or search score, noise and the bar a result had to clear, length
ratio, calls used, why the search ended, whether the result is verified, the reason in plain
language, what changed and the word diff, the intent contract with its checks, and the run
folder; SPEC R3, R5, R7, R11, R12, R13, R14a, R17, R23), `outcome_object` its `--json` object,
`error_object` the object of a failed run. `Emitter` is the only writer to stdout: at most one
result per run, so `--json` gives exactly one object whatever fails after it, and notices stay on
stderr; the plan of `--dry` (SPEC R4) it prints is built in `cli_plan.py`.

The time tiers (SPEC R25): a report names its tier and the seconds of the run's clock; a quick or
fast result says, in its verified line, its meaning, its notices and its JSON, that it is not
verified on held-out scenarios, and its scores are labelled as taken on the scenarios it was
picked on, never as holdout scores; nothing of the GEPA search is said of a run that did not
search.

Model-written text (an improved prompt, its "what changed" lines, the contract, messages that
may quote a reply) is data, never instructions to the terminal: escape sequences and control
characters are removed before it is printed (SPEC R19). The user's own original prompt is
printed as given."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any, Protocol, TextIO

from autoimprover.runner import MIN_THRESHOLD
from autoimprover.types import Contract, Outcome, Plan

# The tiers that run the fast pipeline, not the GEPA search (SPEC R25); quick and fast are never
# verified on held-out scenarios, checked is.
FAST_TIERS = ("quick", "fast", "checked")
_UNHELD = ("quick", "fast")

# What each reason code means for the user, and what to do about it (SPEC R2, R3, R11, R13).
REASON_LINES = {
    "improved": "a rewrite scored higher than the original on scenarios the search never saw, "
    "by more than the measured noise",
    "no_reliable_improvement": "no rewrite beat the original by more than the measured noise, so "
    "the original is kept; this is the normal result for a prompt that already works",
    "already_strong": "the original already passes nearly every check on the held-out scenarios, "
    "so no search ran; high scores can also mean weak checks, so read the checks below",
    "no_holdout": "with fewer than 8 scenarios there is nothing held out to verify a result on, "
    "so the original is kept; give 8 or more examples with --examples, or pass --trust-search "
    "to accept an unverified result",
    "no_candidate_beat_seed": "no rewrite beat the original on the scenarios the search used "
    "(--trust-search, no holdout), so the original is kept",
    "unconfirmed_out_of_budget": "the calls or the clock ran out before a rewrite was confirmed "
    "on the holdout, so the original is kept; a larger --budget leaves more for the final steps",
}

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
FAST_REASON_LINES = {
    "no_reliable_improvement": "no rewrite kept the contract and beat the original clearly "
    "enough on the scenarios, so the original is kept; this is the normal result for a prompt "
    "that already works",
    "no_holdout": "the checked tier holds out the examples after the ones it picks on, and none "
    "were left, so the original is kept; give more examples, or a shorter --time",
    "unconfirmed_out_of_budget": "the clock or the calls ran out before a rewrite passed every "
    "gate, so the original is kept; a longer --time leaves more room",
}
_FAST_VERIFIED = (
    "a rewrite beat the original on held-out scenarios, on the target model; no noise was "
    "measured, so a small margin is a weak signal"
)
_FAST_UNVERIFIED = (
    "a rewrite kept the intent contract and passed the free gates in a short run; it is not "
    "verified on held-out scenarios and no noise was measured, so read it before you use it"
)
_FAST_STOPS = {
    None: "every stage ran as planned",
    "clock": "the clock cut a stage short or shrank it",
    "budget": "the call limit cut a stage short or shrank it",
}
_FAST_CUT_SHORT = "notice: the clock cut the run short; the result comes from what it scored"

# A terminal escape sequence: CSI (ESC [ or the one-byte CSI, parameters, a final byte), OSC
# (ESC ] up to BEL or ESC \), or ESC and one more character. Every part is linear: the ranges
# of neighbouring quantified parts do not overlap.
_ESCAPE = re.compile(
    r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[@-Z\\-_]?"
)
# Control characters other than newline and tab, DEL, the C1 controls, and the bidirectional
# overrides and isolates that can reorder what a terminal shows.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def clean_text(text: str) -> str:
    """`text` without terminal escape sequences and control characters; newlines, tabs and
    ordinary Unicode stay (SPEC R19)."""
    return _CONTROL.sub("", _ESCAPE.sub("", text))


def one_line(text: str) -> str:
    """`clean_text` on one line: every run of whitespace becomes one space."""
    return " ".join(clean_text(text).split())


def word_diff(before: str, after: str) -> str:
    """A word-level diff: removed words in `[-...-]`, added words in `{+...+}` (SPEC R2)."""
    old, new = before.split(), after.split()
    parts: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            parts += old[i1:i2]
            continue
        if i2 > i1:
            parts.append(f"[-{' '.join(old[i1:i2])}-]")
        if j2 > j1:
            parts.append(f"{{+{' '.join(new[j1:j2])}+}}")
    return " ".join(parts)


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
    lines = [
        f"result: {head} ({outcome.reason_code})",
        f"reason: {one_line(outcome.reason)}",
        f"meaning: {_meaning(outcome)}",
    ]
    if improved:
        lines.append(_verified(outcome, plan))
    if outcome.mode is not None:
        lines.append(f"mode: {outcome.mode} ({elapsed_s or 0.0:.0f} s)")
    lines += _scores(outcome, plan)
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
    if outcome.mode in FAST_TIERS:
        if outcome.status == "improved":
            return _FAST_VERIFIED if outcome.verified else _FAST_UNVERIFIED
        return FAST_REASON_LINES.get(outcome.reason_code, REASON_LINES[outcome.reason_code])
    if outcome.status == "improved" and not outcome.verified:
        return _UNVERIFIED_MEANING
    return REASON_LINES[outcome.reason_code]


def _verified(outcome: Outcome, plan: Plan) -> str:
    """Whether an improved result is verified; a quick or fast one is not, in its own label."""
    if outcome.verified:
        return f"verified: yes, on the holdout, on the target model {plan.models.target}"
    if outcome.mode in FAST_TIERS:
        return f"verified: no. NOT VERIFIED ({one_line(outcome.reason)})"
    return f"verified: no. {_UNVERIFIED}"


def _scores(outcome: Outcome, plan: Plan) -> list[str]:
    lines = []
    picked = (
        f"score on the scenarios it was picked on (task model {plan.models.task}, not held out)"
    )
    if outcome.score_before is not None:
        where = (
            picked
            if outcome.mode in _UNHELD
            else f"holdout score (target model {plan.models.target})"
        )
        lines.append(f"{where}: {_before_after(outcome.score_before, outcome.score_after)}")
    if outcome.noise is not None:
        bar = max(MIN_THRESHOLD, 2 * outcome.noise)
        said = f"noise: {outcome.noise:.2f} between the original's two holdout runs; "
        if outcome.margin is not None:
            lines.append(f"{said}the result cleared the bar of {bar:.2f} by {outcome.margin:.2f}")
        else:
            lines.append(f"{said}a result had to gain more than {bar:.2f}")
    if outcome.search_score_before is not None:
        where = (
            picked
            if outcome.mode in FAST_TIERS
            else f"search score (valset, search model {plan.models.task})"
        )
        lines.append(
            f"{where}: {_before_after(outcome.search_score_before, outcome.search_score_after)}"
        )
    before, after, margin = outcome.score_before, outcome.score_after, outcome.margin
    if outcome.mode in FAST_TIERS and before is not None and after is not None:
        if margin is not None:  # a fast-pipeline win: its margin over the least gain (fast.py)
            held = "held-out scenarios" if outcome.verified else "scenarios it was picked on"
            lines.append(
                f"margin: {margin:.2f} above the least gain of {after - before - margin:.2f} on "
                f"the {held} (no noise measured)"
            )
    return lines


def _before_after(before: float, after: float | None) -> str:
    if after is None:
        return f"{before:.2f} for the original"
    return f"{before:.2f} before, {after:.2f} after"


def _contract_lines(contract: Contract) -> list[str]:
    def listed(items: tuple[str, ...], none: str) -> str:
        return "; ".join(one_line(item) for item in items) if items else none

    lines = [
        f"intent contract ({contract.kind}):",
        f"  goal: {one_line(contract.goal)}",
        f"  keep: {listed(contract.keep, 'nothing listed')}",
        f"  constraints: {listed(contract.constraints, 'none listed')}",
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
        unverified = f"NOT VERIFIED ({one_line(outcome.reason)})" if fast else _UNVERIFIED
        lines.append(f"notice: {unverified}")
    return lines + _folder(outcome.run_dir)


def outcome_object(outcome: Outcome, original: str, contract: Contract | None) -> dict[str, Any]:
    """The `--json` object of a finished run (SPEC R2; ARCHITECTURE section 8)."""
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
            self._result(_dumps(outcome_object(outcome, original, contract)))
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
