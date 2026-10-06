"""`improve` through the real stack (SPEC R3, R5, R6, R7, R11, R12, R13, R14a, R17, R22, R24): a run
folder in tmp_path, `Cached(Resilient(Budgeted(raw)))` over a scripted raw layer, the pinned GEPA
and a fake clock. Every test that shows a candidate is NOT returned has a twin on the same script
showing one IS, so it cannot pass for the wrong reason (ARCHITECTURE section 3)."""

import io
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import MARKER, FakeClock, happy_backend, judge_reply

from autoimprover import runner
from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.runstore import RunStore, cache_key
from autoimprover.scenarios import split
from autoimprover.types import (
    DEFAULT_MODELS,
    SEARCH_CLOCK_SHARE,
    SYNTH_COUNT,
    BackendError,
    Call,
    CallError,
    Outcome,
    Plan,
    Reply,
    Scenario,
)

PROMPT = "Answer the user's request."
BETTER = f"Answer the user's request {MARKER}."
WORSE = f"{PROMPT} Be brief."
WHY = ("tightened the wording", "kept the output format", "kept every literal")
MODELS = DEFAULT_MODELS
PLAN = Plan(models=MODELS)
SYNTHESISED = [Scenario(id=f"s{i}", input=f"situation {i}") for i in range(1, SYNTH_COUNT + 1)]
PARTS = split(SYNTHESISED, seed=0)  # holdout s2 s6 s9 s10, valset s3 s4 s11
Script = Callable[[Call], str]


def happy(improved: str = BETTER) -> Script:
    """`happy_backend`'s answers: a task output is good exactly when its prompt holds MARKER."""
    model = happy_backend(improved)
    return lambda call: model.complete(call).text


def is_contract_check(call: Call) -> bool:
    return call.role == "judge" and json.loads(call.user)["scenarios"][0]["scenario"] == "contract"


def text_of(call: Call) -> str:
    """The prompt a task call ran: the system prompt (template) or what follows the situation."""
    return call.system or call.user.split("\n\n", 1)[1]


def scenario_of(call: Call) -> str:
    """The scenario input a task call ran on."""
    return call.user if call.system else call.user.split("\n\n", 1)[0]


class Cut(BaseException):
    """The process dies in the middle of a live call (a kill, a crash, Ctrl-C)."""


@dataclass
class Raw:
    """The raw model layer: answers by `script`; dies with Cut at live call `cut_at`; fails every
    attempt of a call `fails` picks. It records every live call, the call limit and deadline in
    force for it, and the successful ones by cache key."""

    script: Script = field(default_factory=happy)
    cut_at: int = 0
    fails: Callable[[Call], bool] = lambda _call: False
    clock: FakeClock = field(default_factory=FakeClock)
    duration_s: float = 0.0
    budgeted: BudgetedBackend | None = None
    calls: list[Call] = field(default_factory=list)
    limits: list[tuple[int, float]] = field(default_factory=list)
    ok: list[str] = field(default_factory=list)

    def complete(self, call: Call) -> Reply:
        self.calls.append(call)
        if self.budgeted is not None:
            self.limits.append((self.budgeted.limit, self.budgeted.deadline))
        if len(self.calls) == self.cut_at:
            raise Cut
        if self.fails(call):
            raise CallError("exit 1")
        self.clock.advance(self.duration_s)
        self.ok.append(cache_key(call))
        return Reply(text=self.script(call), duration_s=self.duration_s)

    def count(self, role: str, model: str | None = None) -> int:
        return sum(c.role == role and model in (None, c.model) for c in self.calls)


@pytest.fixture(autouse=True)
def _no_disk_flush(monkeypatch: pytest.MonkeyPatch):
    """Writes stay atomic but are not flushed: these runs write hundreds of cache entries, and
    durability is tested with the run store."""
    monkeypatch.setattr(os, "fsync", lambda _fd: None)


def new_run(root: Path, plan: Plan = PLAN, prompt: str = PROMPT) -> str:
    now = datetime(2026, 10, 4, 12, tzinfo=UTC)
    store = RunStore.open_or_create(root, plan, prompt, utcnow=lambda: now)
    store.close()
    return store.run_id


def improve(
    root: Path,
    raw: Raw,
    run_id: str = "",
    *,
    plan: Plan = PLAN,
    prompt: str = PROMPT,
    examples: Sequence[Scenario] | None = None,
    trust_search: bool = False,
) -> Outcome:
    """One process: open the run folder (new unless `run_id`), build the real stack over `raw` the
    way the CLI does (limit budget - final, deadline the search's clock share) and improve."""
    store = RunStore.resume(root, run_id or new_run(root, plan, prompt))
    try:
        n = len(store.scenarios() or examples or SYNTHESISED)
        costs = runner.fixed_costs(plan, n, synthesising=examples is None)
        clock = Clock(now=raw.clock.now, elapsed=store.elapsed_s)
        budgeted = BudgetedBackend(
            raw,
            limit=plan.budget - costs.final,
            used=store.calls_used,
            clock=clock,
            deadline=SEARCH_CLOCK_SHARE * plan.wall_clock_s,
            on_call=store.save_progress,
        )
        raw.budgeted = budgeted
        outcome = runner.improve(
            prompt,
            plan,
            backend=(cache := CachedBackend(ResilientBackend(budgeted), store)),
            cache=cache,
            budgeted=budgeted,
            clock=clock,
            store=store,
            scenarios=examples,
            kind=None,
            trust_search=trust_search,
            log=io.StringIO(),
        )
        assert outcome.calls_used == budgeted.used <= plan.budget
        assert outcome.run_dir == str(store.path)
        return outcome
    finally:
        store.close()


# --- the three outcomes of a run on happy_backend (SPEC R3, R13) ---------------------------------


def test_a_candidate_that_wins_on_the_holdout_is_returned_verified_with_its_report(tmp_path):
    raw = Raw()
    outcome = improve(tmp_path, raw)
    assert (outcome.status, outcome.prompt, outcome.reason_code) == ("improved", BETTER, "improved")
    assert (outcome.verified, outcome.stop, outcome.changes) == (True, "budget", WHY)
    assert (outcome.score_before, outcome.score_after, outcome.noise) == (0.0, 1.0, 0.0)
    assert (outcome.search_score_before, outcome.search_score_after) == (0.0, 1.0)
    assert outcome.margin == pytest.approx(0.95)
    ratio = runner.count_tokens(BETTER) / runner.count_tokens(PROMPT)
    assert outcome.length_ratio == ratio and outcome.calls_used == len(raw.calls)


def test_a_candidate_that_never_wins_leaves_the_original_unchanged(tmp_path):
    raw = Raw(happy(f"{PROMPT} Be brief."))
    outcome = improve(tmp_path, raw)
    assert (outcome.status, outcome.prompt) == ("unchanged", PROMPT)
    assert (outcome.reason_code, outcome.verified, outcome.stop) == (
        "no_reliable_improvement",
        False,
        "budget",
    )
    assert (outcome.score_before, outcome.noise, outcome.search_score_before) == (0.0, 0.0, 0.0)
    assert raw.count("reflect") > 0 and outcome.calls_used == len(raw.calls)


def test_an_original_that_already_scores_095_stops_before_the_search(tmp_path):
    raw = Raw(happy(BETTER))
    outcome = improve(tmp_path, raw, prompt=BETTER)
    assert (outcome.status, outcome.prompt, outcome.reason_code) == (
        "unchanged",
        BETTER,
        "already_strong",
    )
    assert (outcome.stop, outcome.score_before, outcome.noise) == (None, 1.0, 0.0)
    assert raw.count("reflect") == 0 and len(raw.calls) == 1 + 1 + 2 * (4 + 1)


# --- target-model confirmation (SPEC R14a) and the models of each step ----------------------------


def test_a_candidate_that_wins_on_the_search_model_but_not_on_the_target_is_not_returned(
    tmp_path,
):
    def script(call: Call) -> str:
        if call.role == "task":
            good = MARKER in call.user and call.model == MODELS.task
            return "GOOD answer" if good else "BAD answer"
        return happy()(call)

    outcome = improve(tmp_path, Raw(script))
    assert (outcome.reason_code, outcome.search_score_after) == ("no_reliable_improvement", None)
    assert outcome.prompt == PROMPT


def test_seed_and_finalist_runs_use_the_target_model_and_the_search_the_task_model(tmp_path):
    raw = Raw()
    improve(tmp_path, raw)
    holdout = {s.input for s in PARTS.holdout}
    searched = {s.input for s in (*PARTS.train, *PARTS.val)}
    tasks = [c for c in raw.calls if c.role == "task"]
    on_target = [c for c in tasks if c.model == MODELS.target]
    assert {scenario_of(c) for c in on_target} == holdout
    assert [text_of(c) for c in on_target] == [PROMPT] * 8 + [BETTER] * 4
    assert [c.sample for c in on_target] == [0] * 4 + [1] * 4 + [0] * 4
    on_task = [c for c in tasks if c.model == MODELS.task]
    assert on_task and {scenario_of(c) for c in on_task} <= searched
    assert {c.model for c in tasks} == {MODELS.task, MODELS.target}
    assert {c.model for c in raw.calls if c.role == "judge"} == {MODELS.judge}
    assert {c.model for c in raw.calls if c.role in ("intake", "synth", "reflect")} == {
        MODELS.reflect
    }


def test_the_holdout_never_reaches_the_reflection_model(tmp_path):
    raw = Raw()
    improve(tmp_path, raw)
    reflections = [c.user for c in raw.calls if c.role == "reflect"]
    assert reflections and "Keep every item of the intent contract" in reflections[0]
    assert "BAD answer" in reflections[0] and "answers the request" in reflections[0]
    for scenario in PARTS.holdout:
        pattern = rf"\b({re.escape(scenario.id)}|{re.escape(scenario.input)})\b"
        assert not any(re.search(pattern, text) for text in reflections), scenario.id


# --- the gates every answer passes (SPEC R6, R7, R9) ----------------------------------------------

LITERAL_PROMPT = "Answer the user's request about {topic}."


def test_a_candidate_that_drops_a_literal_is_never_returned_even_when_it_scores_best(tmp_path):
    raw = Raw(happy(f"Answer the user's request {MARKER}."))
    outcome = improve(tmp_path, raw, prompt=LITERAL_PROMPT)
    assert (outcome.reason_code, outcome.prompt) == ("no_reliable_improvement", LITERAL_PROMPT)
    assert not any(is_contract_check(c) for c in raw.calls)
    assert not any(c.model == MODELS.target and MARKER in c.user for c in raw.calls)


def test_its_twin_that_keeps_the_literal_is_returned(tmp_path):
    keeps = f"Answer the user's request about {{topic}} {MARKER}."
    outcome = improve(tmp_path, Raw(happy(keeps)), prompt=LITERAL_PROMPT)
    assert (outcome.reason_code, outcome.prompt) == ("improved", keeps)


def failing_contract_check(call: Call) -> str:
    if is_contract_check(call):
        return judge_reply(call, lambda *_: False)
    return happy()(call)


def test_a_candidate_that_fails_the_judged_contract_check_is_never_returned(tmp_path):
    raw = Raw(failing_contract_check)
    outcome = improve(tmp_path, raw)
    assert (outcome.reason_code, outcome.prompt) == ("no_reliable_improvement", PROMPT)
    assert sum(is_contract_check(c) for c in raw.calls) == 1
    assert not any(c.model == MODELS.target and MARKER in c.user for c in raw.calls)


LONG = f"{PROMPT} {MARKER} " + " ".join(f"word{i}" for i in range(40))


def test_a_candidate_over_the_length_cap_is_never_returned(tmp_path):
    raw = Raw(happy(LONG))
    outcome = improve(tmp_path, raw)
    assert (outcome.reason_code, outcome.prompt) == ("no_reliable_improvement", PROMPT)
    assert not any(is_contract_check(c) for c in raw.calls)


def test_its_twin_with_allow_growth_is_returned(tmp_path):
    outcome = improve(tmp_path, Raw(happy(LONG)), plan=replace(PLAN, allow_growth=True))
    assert (outcome.reason_code, outcome.prompt) == ("improved", LONG)
    assert outcome.length_ratio == runner.count_tokens(LONG) / runner.count_tokens(PROMPT)


# --- --trust-search without a holdout (SPEC R11) --------------------------------------------------

SEVEN = [Scenario(id=f"e{i}", input=f"example {i}") for i in range(1, 8)]


@pytest.mark.parametrize(
    ("improved", "code"), [(BETTER, "improved"), (WORSE, "no_candidate_beat_seed")]
)
def test_trust_search_returns_only_a_candidate_above_the_seed_on_the_valset_unverified(
    tmp_path, improved: str, code: str
):
    raw = Raw(happy(improved))
    outcome = improve(tmp_path, raw, examples=SEVEN, trust_search=True)
    assert (outcome.reason_code, outcome.verified, outcome.score_before) == (code, False, None)
    assert outcome.prompt == (BETTER if code == "improved" else PROMPT)
    assert outcome.search_score_before == 0.0 and raw.count("reflect") > 0
    assert not any(c.model == MODELS.target for c in raw.calls)  # no holdout, no target runs


# --- the budget: the search's share, the count kept across a resume (SPEC R17, R22, R24) ----------


def test_the_search_spends_only_its_share_and_the_final_steps_get_the_rest(tmp_path):
    raw = Raw()
    improve(tmp_path, raw)
    final = runner.fixed_costs(PLAN, SYNTH_COUNT, synthesising=True).final
    share = (PLAN.budget - final, SEARCH_CLOCK_SHARE * PLAN.wall_clock_s)
    searching = [c.role == "reflect" or c.model == MODELS.task for c in raw.calls]
    finishing = [
        is_contract_check(c) or (MARKER in c.user and c.model == MODELS.target) for c in raw.calls
    ]
    assert any(searching) and all(
        lim == share for lim, s in zip(raw.limits, searching, strict=True) if s
    )
    rest = (PLAN.budget, PLAN.wall_clock_s)
    assert any(finishing) and all(
        lim == rest for lim, f in zip(raw.limits, finishing, strict=True) if f
    )
    assert raw.limits.count(share) <= share[0] and set(raw.limits) == {share, rest}


def test_a_call_that_failed_outside_the_search_is_tried_again_on_resume_and_still_counts(tmp_path):
    run_id = new_run(tmp_path)
    with pytest.raises(BackendError, match="intake"):
        improve(tmp_path, Raw(fails=lambda call: call.role == "intake"), run_id)
    resumed = Raw()
    outcome = improve(tmp_path, resumed, run_id)
    assert resumed.calls[0].role == "intake" and outcome.reason_code == "improved"
    assert outcome.calls_used == 3 + len(resumed.calls)  # three failed attempts, then this run


def test_time_running_out_in_the_final_steps_keeps_the_original_and_its_measured_scores(tmp_path):
    clock, healthy = FakeClock(), happy()

    def script(call: Call) -> str:  # a finalist run on the target model takes 1000 s a call
        clock.advance(1000.0 if call.model == MODELS.target and MARKER in call.user else 0.0)
        return healthy(call)

    raw = Raw(script, clock=clock)
    outcome = improve(tmp_path, raw)
    assert (outcome.reason_code, outcome.prompt, outcome.stop) == (
        "unconfirmed_out_of_budget",
        PROMPT,
        "budget",
    )
    assert (outcome.score_before, outcome.noise, outcome.search_score_before) == (0.0, 0.0, 0.0)


def test_a_search_cut_short_by_its_clock_share_still_confirms_its_finalist(tmp_path):
    raw = Raw(duration_s=10.0)
    outcome = improve(tmp_path, raw, plan=replace(PLAN, wall_clock_s=600))
    assert (outcome.reason_code, outcome.stop, outcome.prompt) == ("improved", "clock", BETTER)
    assert raw.clock.t < 600


def test_a_call_that_failed_inside_the_search_is_replayed_as_failed_after_a_resume(tmp_path):
    probe = Raw()
    improve(tmp_path / "probe", probe)
    reflect = next(c for c in probe.calls if c.role == "reflect")  # skipped, then asked again
    reference_raw = Raw(fails=lambda call: call == reflect)
    reference = improve(tmp_path / "reference", reference_raw)
    after = max(i for i, c in enumerate(reference_raw.calls, start=1) if c == reflect) + 1
    run_id = new_run(tmp_path / "run")
    with pytest.raises(Cut):
        improve(tmp_path / "run", Raw(fails=lambda call: call == reflect, cut_at=after), run_id)
    resumed = Raw()  # the model would answer it now: only the tombstone keeps the original path
    outcome = improve(tmp_path / "run", resumed, run_id)
    assert reflect not in resumed.calls and reference.reason_code == "improved"
    assert replace(outcome, calls_used=0, run_dir="") == replace(
        reference, calls_used=0, run_dir=""
    )


def stages(calls: list[Call]) -> dict[str, int]:
    """The 1-based index of a live call in each stage of a run."""
    reflects = [i for i, c in enumerate(calls, start=1) if c.role == "reflect"]

    def first(match: Callable[[Call], bool]) -> int:
        return next(i for i, c in enumerate(calls, start=1) if match(c))

    return {
        "intake": 1,
        "synthesis": 2,
        "seed run 2": first(lambda c: c.model == MODELS.target and c.sample == 1),
        "first reflection": reflects[0],
        "mid-search": (reflects[0] + reflects[-1]) // 2,
        "last reflection": reflects[-1],
        "contract check": first(is_contract_check),
        "finalist run": first(lambda c: c.model == MODELS.target and MARKER in c.user),
        "last call": len(calls),
    }


def test_a_run_cut_at_any_stage_resumes_to_the_same_outcome_without_a_paid_call_twice(tmp_path):
    reference_raw = Raw()
    reference = improve(tmp_path / "reference", reference_raw)
    for stage, k in stages(reference_raw.calls).items():
        root = tmp_path / stage.replace(" ", "-")
        run_id = new_run(root)
        cut = Raw(cut_at=k)
        with pytest.raises(Cut):
            improve(root, cut, run_id)
        resumed = Raw()
        outcome = improve(root, resumed, run_id)
        assert replace(outcome, calls_used=0, run_dir="") == replace(
            reference, calls_used=0, run_dir=""
        ), stage
        assert max(Counter(cut.ok + resumed.ok).values()) == 1, stage  # no success paid twice
        assert outcome.calls_used == k + len(resumed.calls), stage  # the count goes on
