"""What `autoimprover bench` prints (SPEC R26, with R2 and R4): `Summary`, the measure as text or as
one JSON object, and `DryView`, the plan of `--dry` with the models, efforts and strictness the
flags chose (the run's flags, `cli_bench`); both are a `report.Shown`.

The summary holds the prompts in the set and the ones measured (a run that ended in a backend
failure is an error, not measured), the improved rate among the measured, the tool's comparisons
with the original (an unchanged prompt is a tie; a comparison that judged no scenario is an error,
counted apart), the win rate among the compared and among the improved compared, the naive
baseline's comparisons, the median and 90th percentile (nearest rank) of the runs' seconds, the
calls of the runs and of the bench itself, and a row per prompt. Contract violations in returned
prompts are null: the pipeline never returns a rewrite its contract check vetoed (SPEC R6) and the
pairwise judge sees answers, not prompts, so the bench has no count of its own. Everything shown is
an id, a fixed code or a number, never a prompt (SPEC R26).

A bench of `--ungated` runs says so in its header and its plan (its returned prompts are the
best-ranked candidates, not gate-verified), keeps the same measures, and adds the tally of the
improved rows compared: their wins, ties and losses against the original, and the share of wins
among the decided ones (wins and losses) with the count behind it."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from autoimprover.bench import Measured, Row
from autoimprover.bench_judge import NAIVE_EFFORT, Comparison
from autoimprover.report import one_line
from autoimprover.types import Efforts, Models, Strictness, Tier

_COUNTED = ("win", "tie", "loss")
# What the header and the plan of an `--ungated` bench say of its returned prompts.
_UNGATED_HEAD = (
    ", ungated (--ungated): a returned prompt is the best-ranked candidate, not gate-verified"
)


@dataclass(frozen=True)
class Summary:
    """The measure of a bench: what was measured, the prompts in the set (after `--limit`), the
    tier and clock of every run, whether the naive baseline ran, the bench's folder, and whether
    its runs were `--ungated`."""

    measured: Measured
    prompts: int
    tier: Tier
    time_s: int
    baseline: bool
    folder: str
    ungated: bool = False

    def object(self) -> dict[str, Any]:
        rows = self.measured.rows
        done = [row for row in rows if row.status != "error"]
        improved = [row for row in done if row.status == "improved"]
        tool = _tally([row.tool for row in done])
        returned = _tally([row.tool for row in improved])
        decided = returned["wins"] + returned["losses"]
        tally = {
            "rows": decided + returned["ties"],
            "wins": returned["wins"],
            "ties": returned["ties"],
            "losses": returned["losses"],
            "decided": decided,
            "win_share_of_decided": _rate(returned["wins"], decided),
        }
        seconds = [row.seconds for row in done if row.seconds is not None]
        run_calls = sum(row.calls for row in rows)
        bench_calls = sum(row.bench_calls for row in rows)
        naive = None
        if self.baseline:
            naive = {"kind": "naive", **_tally([row.naive for row in done])}
        return {
            "status": "bench",
            "interrupted": self.measured.interrupted,
            "folder": self.folder,
            "tier": self.tier,
            "time_s": self.time_s,
            "prompts": self.prompts,
            "measured": len(done),
            "errors": len(rows) - len(done),
            "improved": len(improved),
            "improved_rate": _rate(len(improved), len(done)),
            "wins": tool["wins"],
            "ties": tool["ties"],
            "losses": tool["losses"],
            "compare_errors": tool["errors"],
            "win_rate": tool["win_rate"],
            "win_rate_of_improved": returned["win_rate"],
            "ungated": self.ungated,
            "ungated_tally": tally if self.ungated else None,
            "baseline": naive,
            "contract_violations": None,
            "seconds_median": statistics.median(seconds) if seconds else None,
            "seconds_p90": percentile(seconds, 90) if seconds else None,
            "calls": run_calls + bench_calls,
            "run_calls": run_calls,
            "bench_calls": bench_calls,
            "rows": [_row(row) for row in rows],
        }

    def text(self) -> str:
        found = self.object()
        measured, prompts = found["measured"], found["prompts"]
        lines = [
            f"bench: {measured} of {prompts} prompts measured, tier {self.tier} (--time "
            f"{self.time_s} s){_UNGATED_HEAD if self.ungated else ''}",
        ]
        if self.measured.interrupted:
            lines.append(
                f"interrupted: Ctrl-C ended the bench after {len(self.measured.rows)} of "
                f"{prompts} prompts; the prompt it cut is not counted"
            )
        lines += [
            f"runs: {found['improved']} improved, {measured - found['improved']} unchanged, "
            f"{found['errors']} error(s); improved rate {_percent(found['improved_rate'])}",
            "the tool's prompt against the original (blind pairwise judge, both orders; an "
            f"unchanged prompt is a tie): {_counts(found)}; win rate "
            f"{_percent(found['win_rate'])} of the compared, "
            f"{_percent(found['win_rate_of_improved'])} of the "
            f"{_compared_improved(self.measured.rows)} improved compared",
        ]
        if found["compare_errors"]:
            lines.append(
                f"  {found['compare_errors']} comparison(s) judged no scenario: not counted"
            )
        if self.ungated:
            tally = found["ungated_tally"]
            lines.append(
                "ungated: the returned prompts against the original over the "
                f"{tally['rows']} improved rows compared: {_counts(tally)}; wins among the "
                f"decided: {_percent(tally['win_share_of_decided'])} ({tally['wins']} of "
                f"{tally['decided']})"
            )
        if found["baseline"] is not None:
            base = found["baseline"]
            lines.append(
                f"naive baseline against the original: {_counts(base)}; win rate "
                f"{_percent(base['win_rate'])} of the compared"
            )
        lines += [
            "contract violations in returned prompts: not measured (the pipeline returns no "
            "vetoed rewrite, and the pairwise judge sees answers, not prompts)",
            f"seconds per run: median {_seconds(found['seconds_median'])}, p90 "
            f"{_seconds(found['seconds_p90'])}",
            f"calls: {found['calls']} (runs {found['run_calls']}, bench {found['bench_calls']})",
            f"folder: {self.folder}",
            "",
            *_table(self.measured.rows),
        ]
        return "".join(f"{line}\n" for line in lines)


@dataclass(frozen=True)
class DryRow:
    """One prompt of the `--dry` plan: its run's calls and seconds (the tier's plan), and the most
    the bench's own calls and seconds can be (when the run returns a rewrite)."""

    id: str
    run_calls: int
    run_seconds: float
    bench_calls: int
    bench_seconds: float


@dataclass(frozen=True)
class DryView:
    """The plan of `autoimprover bench --dry` (SPEC R4, R26): no call, nothing written. The models,
    efforts, strictness and `--ungated` are every run's; the bench's own comparisons answer on the
    target model at the task effort and judge with the judge model at the judge effort."""

    path: str
    tier: Tier
    time_s: int
    workers: int
    models: Models
    efforts: Efforts
    strictness: Strictness
    baseline: bool
    scenarios: int
    rows: tuple[DryRow, ...]
    refusal: str | None = None
    ungated: bool = False

    def object(self) -> dict[str, Any]:
        run_calls = sum(row.run_calls for row in self.rows)
        bench_calls = sum(row.bench_calls for row in self.rows)
        run_seconds = math.fsum(row.run_seconds for row in self.rows)
        bench_seconds = math.fsum(row.bench_seconds for row in self.rows)
        return {
            "status": "dry",
            "path": self.path,
            "prompts": len(self.rows),
            "tier": self.tier,
            "time_s": self.time_s,
            "workers": self.workers,
            "models": asdict(self.models),
            "efforts": asdict(self.efforts),
            "strictness": self.strictness,
            "scenarios": self.scenarios,
            "baseline": "naive" if self.baseline else "none",
            "est_calls": run_calls + bench_calls,
            "est_seconds": run_seconds + bench_seconds,
            "run_calls": run_calls,
            "run_seconds": run_seconds,
            "bench_calls": bench_calls,
            "bench_seconds": bench_seconds,
            "refusal": self.refusal,
            "ungated": self.ungated,
            "per_prompt": [
                {
                    "id": row.id,
                    "run_calls": row.run_calls,
                    "run_seconds": row.run_seconds,
                    "bench_calls": row.bench_calls,
                    "bench_seconds": row.bench_seconds,
                }
                for row in self.rows
            ],
        }

    def text(self) -> str:
        found, models, efforts = self.object(), self.models, self.efforts
        naive = f", the naive rewrite {NAIVE_EFFORT}" if self.baseline else ""
        lines = [
            f"bench: {found['prompts']} prompts from {self.path}; tier {self.tier} (--time "
            f"{self.time_s} s), {self.workers} calls at a time"
            f"{_UNGATED_HEAD if self.ungated else ''}",
            f"models: task {models.task}, judge {models.judge} (also the pairwise judge), "
            f"reflection {models.reflect} (also the fresh scenarios), target {models.target} "
            "(the pairwise answers)",
            f"effort: task {efforts.task or 'default'}, judge {efforts.judge or 'default'}, "
            f"reflection {efforts.reflect or 'default'} (the pairwise answers take the task "
            f"effort{naive})",
            f"strictness: {self.strictness} (every run's rewrites)",
            f"pairwise: {self.scenarios} fresh scenarios per prompt, each judged in both orders; "
            f"baseline: {found['baseline']}",
            f"estimate, if every prompt is improved: at most {found['est_calls']} calls (runs "
            f"{found['run_calls']}, bench {found['bench_calls']}) in about "
            f"{math.ceil(found['est_seconds'] / 60)} min ({found['est_seconds']:.0f} s)",
        ]
        if self.refusal is not None:
            lines.append(f"a real bench would refuse: {one_line(self.refusal)}")
        width = max([len("id"), *(len(row.id) for row in self.rows)])
        lines.append(f"{'id':<{width}}  run calls  run s  bench calls  bench s")
        lines += [
            f"{row.id:<{width}}  {row.run_calls:>9}  {row.run_seconds:>5.1f}  "
            f"{row.bench_calls:>11}  {row.bench_seconds:>7.1f}"
            for row in self.rows
        ]
        return "".join(f"{line}\n" for line in lines)


def percentile(values: Sequence[float], p: int) -> float:
    """The p-th percentile of `values` by the nearest rank: the smallest value with at least p %
    of the values at or below it."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p * len(ordered) / 100)) - 1]


def _tally(found: Sequence[Comparison | None]) -> dict[str, Any]:
    """Wins, ties and losses of the comparisons that judged something, the errors apart, and the
    win rate among the counted (None without any)."""
    verdicts = [comparison.verdict for comparison in found if comparison is not None]
    wins, ties, losses = (verdicts.count(name) for name in _COUNTED)
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "errors": verdicts.count("error"),
        "win_rate": _rate(wins, wins + ties + losses),
    }


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _row(row: Row) -> dict[str, Any]:
    return {
        "id": row.id,
        "status": row.status,
        "reason_code": row.reason_code,
        "verified": row.verified,
        "seconds": row.seconds,
        "calls": row.calls,
        "bench_calls": row.bench_calls,
        "noise": row.noise,
        "verdict": None if row.tool is None else row.tool.verdict,
        "votes": _votes(row.tool),
        "naive_verdict": None if row.naive is None else row.naive.verdict,
        "naive_votes": _votes(row.naive),
        "error": _error(row),
    }


def _votes(comparison: Comparison | None) -> dict[str, int] | None:
    if comparison is None:
        return None
    return {"wins": comparison.wins, "ties": comparison.ties, "losses": comparison.losses}


def _error(row: Row) -> str | None:
    """What failed: the run, else the tool's comparison, else the naive one; one clean line."""
    for why in (row.error, *(c.why for c in (row.tool, row.naive) if c is not None)):
        if why:
            return one_line(why)
    return None


def _compared_improved(rows: Sequence[Row]) -> int:
    return sum(
        row.status == "improved" and row.tool is not None and row.tool.verdict in _COUNTED
        for row in rows
    )


def _counts(found: dict[str, Any]) -> str:
    def said(n: int, word: str, plural: str) -> str:
        return f"{n} {word if n == 1 else plural}"

    return ", ".join(
        (
            said(found["wins"], "win", "wins"),
            said(found["ties"], "tie", "ties"),
            said(found["losses"], "loss", "losses"),
        )
    )


def _percent(rate: float | None) -> str:
    return "n/a" if rate is None else f"{100 * rate:.0f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def _table(rows: Sequence[Row]) -> list[str]:
    """A line per prompt: id, status, reason code, seconds, the run's and the bench's calls, the
    verdict with its scenario votes (wins-ties-losses), the naive verdict."""
    width = max([len("id"), *(len(row.id) for row in rows)])
    lines = [
        f"{'id':<{width}}  {'status':<9}  {'code':<25}  {'s':>5}  calls  bench  verdict  "
        "w-t-l  naive"
    ]
    for row in rows:
        seconds = "-" if row.seconds is None else f"{row.seconds:.1f}"
        verdict = "-" if row.tool is None else row.tool.verdict
        votes = "-" if row.tool is None else f"{row.tool.wins}-{row.tool.ties}-{row.tool.losses}"
        naive = "-" if row.naive is None else row.naive.verdict
        lines.append(
            f"{row.id:<{width}}  {row.status:<9}  {row.reason_code or '-':<25}  {seconds:>5}  "
            f"{row.calls:>5}  {row.bench_calls:>5}  {verdict:<7}  {votes:<5}  {naive}"
        )
    return lines
