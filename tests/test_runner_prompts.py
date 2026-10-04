"""Which prompt `improve` returns, on a given search result (SPEC R3, R6, R7, R9, R11, R12, R13,
R17, R24): a stand-in replaces `run_search`, so the valset scores are exact, while the contract,
the scenarios, the seed and finalist runs and the contract checks go through the real stack over
a scripted model. The finalists, the noise threshold, the 0.95 check, `--trust-search`, the
search's arguments and its share of the budget, and the failures around the search. The whole
flow on GEPA is tested in `test_runner_flow.py`."""

import io
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import MARKER, FakeClock, ScriptedBackend, intake_reply, judge_reply, synth_reply

from autoimprover import runner
from autoimprover.backend import BudgetedBackend, CachedBackend, Clock, ResilientBackend
from autoimprover.evaluator import Evaluator
from autoimprover.runstore import RunStore
from autoimprover.scenarios import split
from autoimprover.search import Candidate, SearchResult
from autoimprover.types import (
    DEFAULT_MODELS,
    SEARCH_CLOCK_SHARE,
    BackendError,
    Call,
    CallError,
    Check,
    Contract,
    Kind,
    Outcome,
    Plan,
    Scenario,
    SessionNotLockedDown,
)

# --- what improve does with a search result (SPEC R3, R6, R7, R9, R11, R17, R24) ------------------

PROMPT = "Answer the user's request about {topic}."  # "{topic}" is a literal (SPEC R9)
MODELS = DEFAULT_MODELS
PLAN = Plan(models=MODELS)
EXAMPLES = [Scenario(id=f"e{i}", input=f"example {i}") for i in range(1, 9)]
SYNTHESISED = [Scenario(id=f"s{i}", input=f"situation {i}") for i in range(1, 13)]  # synth_reply
HOLDOUT = split(SYNTHESISED, seed=0).holdout  # s2, s6, s9, s10


def cand(text: str, score: float) -> Candidate:
    return Candidate(text, score, (f"note for {text[-1]}",))


def found(*candidates: Candidate, seed: float | None = 0.0) -> SearchResult:
    return SearchResult((cand(PROMPT, seed or 0.0), *candidates), seed, "budget", 3)


def contract_checked(call: Call) -> str | None:
    """The candidate a contract-check call asks about, or None for another call."""
    scenario = json.loads(call.user)["scenarios"][0] if call.role == "judge" else {}
    return scenario.get("output") if scenario.get("scenario") == "contract" else None


def ran(call: Call) -> str:
    """The prompt a task call ran: its system prompt (template) or what follows the situation."""
    return call.system or call.user.split("\n\n", 1)[1]


def model(winners: Sequence[str] = (), vetoed: Sequence[str] = ()) -> Callable[[Call], str]:
    """Task outputs are good for the `winners`; the judge passes good outputs, and every contract
    check except for the `vetoed` candidates."""

    def script(call: Call) -> str:
        if call.role == "intake":
            return intake_reply()
        if call.role == "synth":
            return synth_reply()
        if call.role == "task":
            return "GOOD answer" if ran(call) in winners else "BAD answer"
        if (checked := contract_checked(call)) is not None:
            return judge_reply(call, lambda *_: checked not in vetoed)
        return judge_reply(call, lambda _s, _c, output: output.startswith("GOOD"))

    return script


class Searched:
    """Stands in for `search.run_search`: returns `result`, or raises `error`, and keeps the
    arguments and the call limit it saw."""

    def __init__(self, result: SearchResult, error: Exception | None = None) -> None:
        self.result, self.error = result, error
        self.kwargs: dict[str, Any] = {}
        self.budgeted: BudgetedBackend | None = None
        self.limit = 0

    def __call__(self, **kwargs: Any) -> SearchResult:
        self.kwargs = kwargs
        self.limit = self.budgeted.limit if self.budgeted else -1
        if self.error is not None:
            raise self.error
        return self.result


def decide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    searched: Searched,
    raw: ScriptedBackend,
    *,
    plan: Plan = PLAN,
    examples: Sequence[Scenario] | None = None,
    trust_search: bool = False,
    fake_clock: FakeClock | None = None,
    kind: Kind | None = None,
    prepare: Callable[[RunStore], None] = lambda _store: None,
) -> Outcome:
    """`improve` on the real stack over `raw`, with `searched` in place of the search; `prepare`
    may fill the new run folder first, as a resumed run finds it."""
    monkeypatch.setattr(runner, "run_search", searched)
    now = datetime(2026, 10, 4, 12, tzinfo=UTC)
    store = RunStore.open_or_create(tmp_path / "runs", plan, PROMPT, utcnow=lambda: now)
    try:
        prepare(store)
        costs = runner.fixed_costs(plan, len(examples or ()) or 12, examples is None)
        clock = Clock(now=(fake_clock or FakeClock()).now)
        deadline = SEARCH_CLOCK_SHARE * plan.wall_clock_s
        budgeted = BudgetedBackend(raw, plan.budget - costs.final, 0, clock, deadline)
        searched.budgeted = budgeted
        return runner.improve(
            PROMPT,
            plan,
            backend=CachedBackend(ResilientBackend(budgeted), store),
            budgeted=budgeted,
            clock=clock,
            store=store,
            scenarios=examples,
            kind=kind,
            trust_search=trust_search,
            log=io.StringIO(),
        )
    finally:
        store.close()


A, B, C, D = (f"{PROMPT} {MARKER} {name}" for name in "ABCD")
LONG = f"{PROMPT} {MARKER} " + " ".join(f"w{i}" for i in range(60)) + " E"
DROPS = f"Answer the user's request {MARKER} F"


def test_finalists_pass_the_free_gates_then_the_best_three_are_contract_checked_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = ScriptedBackend(model(winners=(B, D, LONG, DROPS), vetoed=(B,)))
    result = found(cand(A, 0.5), cand(B, 0.7), cand(C, 0.7), cand(D, 0.6))
    result = replace(result, candidates=(*result.candidates, cand(LONG, 0.9), cand(DROPS, 0.8)))
    outcome = decide(tmp_path, monkeypatch, Searched(result), raw)
    checked = [text for call in raw.calls if (text := contract_checked(call)) is not None]
    assert checked == [C, B, D]  # 0.7 twice (the later first), then 0.6; A is fourth
    on_target = [c.user.split("\n\n", 1)[1] for c in raw.calls if c.model == MODELS.target]
    assert on_target == [PROMPT] * 8 + [C] * 4 + [D] * 4  # B was vetoed, D wins
    assert (outcome.prompt, outcome.verified, outcome.search_score_after) == (D, True, 0.6)
    assert outcome.changes == ("note for D",)


def test_a_failed_finalist_run_or_contract_check_ends_the_run_and_is_never_a_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def down(match: Callable[[Call], bool]) -> ScriptedBackend:
        healthy = model(winners=(A,))
        return ScriptedBackend(lambda call: CallError("exit 1") if match(call) else healthy(call))

    first = split(SYNTHESISED, seed=0).holdout[0].input  # one call fails, not three in a row
    finalist = down(lambda c: c.model == MODELS.target and c.user == f"{first}\n\n{A}")
    with pytest.raises(BackendError, match="holdout"):
        decide(tmp_path / "1", monkeypatch, Searched(found(cand(A, 0.5))), finalist)
    checking = down(lambda c: contract_checked(c) == A)
    with pytest.raises(BackendError, match="judge call"):
        decide(tmp_path / "2", monkeypatch, Searched(found(cand(A, 0.5))), checking)
    nothing = down(lambda _call: False)
    outcome = decide(tmp_path / "3", monkeypatch, Searched(found(cand(A, 0.5))), nothing)
    assert outcome.prompt == A  # the twin: the same model with nothing down


@pytest.mark.parametrize("error", [BackendError("3 failed"), SessionNotLockedDown("tools")])
def test_an_aborted_search_raises_its_own_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
):
    with pytest.raises(type(error)):
        decide(tmp_path, monkeypatch, Searched(found(), error), ScriptedBackend(model()))


def test_running_out_of_time_in_the_final_steps_keeps_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def late(path: Path, jump: float) -> tuple[Outcome, ScriptedBackend]:
        clock, healthy = FakeClock(), model(winners=(A,))

        def script(call: Call) -> str:
            if contract_checked(call) == A:
                clock.advance(jump)  # the clock runs out while the contract is checked
            return healthy(call)

        raw, searched = ScriptedBackend(script), Searched(found(cand(A, 0.5)))
        return decide(path, monkeypatch, searched, raw, fake_clock=clock), raw

    outcome, raw = late(tmp_path / "late", 2700.0)
    assert (outcome.status, outcome.reason_code) == ("unchanged", "unconfirmed_out_of_budget")
    assert (outcome.prompt, outcome.stop, outcome.calls_used) == (PROMPT, "budget", len(raw.calls))
    assert not any(c.model == MODELS.target and A in c.user for c in raw.calls)
    twin, _ = late(tmp_path / "twin", 600.0)
    assert (twin.prompt, twin.reason_code) == (A, "improved")


def test_running_out_before_the_search_keeps_the_original_with_no_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clock, healthy = FakeClock(), model()

    def script(call: Call) -> str:
        clock.advance(2025.0 if call.role == "intake" else 0.0)
        return healthy(call)

    searched = Searched(found(cand(A, 0.5)))
    outcome = decide(tmp_path, monkeypatch, searched, ScriptedBackend(script), fake_clock=clock)
    assert (outcome.reason_code, outcome.stop, outcome.calls_used) == (
        "unconfirmed_out_of_budget",
        None,
        1,
    )
    assert searched.kwargs == {}


def test_the_search_gets_train_and_val_only_its_share_of_the_budget_and_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = Plan(models=MODELS, strictness="balanced", budget=120, seed=7, merge=True)
    plan = replace(plan, allow_growth=True)
    raw, searched = ScriptedBackend(model()), Searched(found())
    decide(tmp_path, monkeypatch, searched, raw, plan=plan)
    kw = searched.kwargs
    store = RunStore.resume(tmp_path / "runs", next((tmp_path / "runs").iterdir()).name)
    contract, scenarios = store.contract(), store.scenarios()
    store.close()
    assert contract is not None and scenarios is not None
    parts = split(scenarios, seed=7)
    assert (kw["seed"], kw["train"], kw["val"]) == (PROMPT, parts.train, parts.val)
    assert not set(parts.holdout) & {*kw["train"], *kw["val"]}
    assert kw["search_start"] == (12, 0.0)  # intake, synthesis, two seed runs of 4 + 1
    assert (kw["calls_left_at_start"], kw["iter_cost"]) == (120 - 18 - 12, 13)
    assert searched.limit == 120 - 18 and searched.budgeted and searched.budgeted.limit == 120
    assert (kw["rng_seed"], kw["merge"], kw["reflect_model"]) == (7, True, MODELS.reflect)
    assert (kw["wall_clock_s"], kw["clock_share"]) == (plan.wall_clock_s, SEARCH_CLOCK_SHARE)
    template = runner.reflection_template(contract, "balanced", runner.count_tokens(PROMPT), True)
    assert kw["reflection_template"] == template
    assert isinstance(kw["backend"], CachedBackend) and isinstance(kw["log"], io.StringIO)
    evaluator = kw["make_evaluator"](probe := ScriptedBackend(model()))
    assert isinstance(evaluator, Evaluator)
    evaluator(PROMPT, parts.val[:1])
    assert [(c.role, c.model, c.sample) for c in probe.calls] == [
        ("task", MODELS.task, 0),
        ("judge", MODELS.judge, 0),
    ]


# --- --trust-search without a holdout (SPEC R11) --------------------------------------------------


@pytest.mark.parametrize(
    ("candidates", "seed", "expected"),
    [
        ((cand(A, 0.2),), 0.9, None),  # the round 2 probe: the seed 0.9, a candidate 0.2
        ((cand(A, 0.9),), 0.9, None),  # a tie does not beat the seed
        ((cand(A, 0.2), cand(D, 0.95)), 0.9, D),
        ((cand(A, 0.95),), None, None),  # a seed without a valset score cannot be beaten
        ((cand(B, 0.99), cand(C, 0.97), cand(A, 0.95)), 0.5, C),  # B is vetoed
        ((cand(LONG, 0.99), cand(DROPS, 0.98), cand(A, 0.6)), 0.5, A),  # the free gates
    ],
)
def test_trust_search_returns_only_the_best_gated_candidate_above_the_seed_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[Candidate, ...],
    seed: float | None,
    expected: str | None,
):
    raw = ScriptedBackend(model(vetoed=(B,)))
    result = found(*candidates, seed=seed)
    outcome = decide(
        tmp_path, monkeypatch, Searched(result), raw, examples=EXAMPLES[:5], trust_search=True
    )
    assert not any(c.model == MODELS.target for c in raw.calls)  # no holdout runs at all
    checked = [text for call in raw.calls if (text := contract_checked(call)) is not None]
    assert all(next(c.val_score for c in candidates if c.text == t) > (seed or 1) for t in checked)
    if expected is None:
        assert (outcome.status, outcome.reason_code) == ("unchanged", "no_candidate_beat_seed")
        assert (outcome.prompt, outcome.search_score_before) == (PROMPT, seed)
    else:
        assert (outcome.status, outcome.prompt, outcome.verified) == ("improved", expected, False)
        assert "not verified on a holdout" in outcome.reason
        assert (outcome.search_score_before, outcome.score_before) == (seed, None)


# --- the seed runs: noise, threshold and the 0.95 check (SPEC R12, R13, R14a) ---------------------


def scored(passes: Callable[[str, str, str], bool], checks: Sequence[dict] | None = None):
    """Task outputs name the prompt ("seed" for the original, else its last character) and the
    sample; the judge passes a scoring check when `passes(scenario, check, output)`, and every
    contract check."""

    def script(call: Call) -> str:
        if call.role == "intake":
            return intake_reply(checks=checks)
        if call.role == "synth":
            return synth_reply()
        if call.role == "task":
            text = ran(call)
            return f"{'seed' if text == PROMPT else text[-1]} answer {call.sample}"
        if contract_checked(call) is not None:
            return judge_reply(call)
        return judge_reply(call, passes)

    return ScriptedBackend(script)


@pytest.mark.parametrize("swap", [False, True])  # the second run may score higher
@pytest.mark.parametrize(("fails", "code"), [(0, "improved"), (1, "no_reliable_improvement")])
def test_a_candidate_must_beat_the_original_by_twice_the_noise_of_its_two_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: int, code: str, swap: bool
):
    runs = [{HOLDOUT[0].id, HOLDOUT[1].id}, {HOLDOUT[0].id}]  # 0.5 and 0.25
    seed_passes = dict(zip("10" if swap else "01", runs, strict=True))
    failing = {s.id for s in HOLDOUT[:fails]}  # 1.0, or 0.75: above 0.375 + 0.25, not + 0.5

    def passes(scenario: str, _check: str, output: str) -> bool:
        name, _, sample = output.split()
        return scenario in seed_passes[sample] if name == "seed" else scenario not in failing

    raw = scored(passes)
    outcome = decide(tmp_path, monkeypatch, Searched(found(cand(A, 0.5))), raw)
    assert (outcome.reason_code, outcome.score_before, outcome.noise) == (code, 0.375, 0.25)
    assert (outcome.score_after, outcome.margin) == ((1.0, 0.125) if fails == 0 else (None, None))
    seeds = [c for c in raw.calls if c.role == "task" and c.user.endswith(f"\n\n{PROMPT}")]
    assert [(c.model, c.sample) for c in seeds] == [(MODELS.target, 0)] * 4 + [
        (MODELS.target, 1)
    ] * 4


FIVE_CHECKS = [
    {"id": f"c{i}", "group": "content", "text": f"check {i}", "rule": None, "arg": None}
    for i in range(1, 6)
]


@pytest.mark.parametrize(("passed", "code"), [(1, "no_reliable_improvement"), (2, "improved")])
def test_a_gain_exactly_at_the_threshold_is_not_a_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: int, code: str
):
    def passes(scenario: str, check: str, output: str) -> bool:  # the original passes nothing
        return output[0] == "A" and scenario == HOLDOUT[0].id and int(check[3:]) <= passed

    raw = scored(passes, FIVE_CHECKS)
    outcome = decide(tmp_path, monkeypatch, Searched(found(cand(A, 0.5))), raw)
    assert (outcome.reason_code, outcome.score_before, outcome.noise) == (code, 0.0, 0.0)
    assert outcome.margin == (None if passed == 1 else 0.05)  # 0.05 or 0.1, against 0.05


@pytest.mark.parametrize(("failed", "strong"), [(2, True), (3, False)])
def test_an_original_at_095_on_the_holdout_is_already_strong_and_is_not_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: int, strong: bool
):
    def passes(scenario: str, check: str, output: str) -> bool:  # run 2 fails some of s10's
        return not (
            output.endswith("1") and scenario == HOLDOUT[-1].id and int(check[3:]) <= failed
        )

    searched = Searched(found(cand(A, 0.5)))
    outcome = decide(tmp_path, monkeypatch, searched, scored(passes, FIVE_CHECKS))
    if strong:  # runs of 1.0 and 0.9
        assert (outcome.reason_code, outcome.score_before, outcome.stop) == (
            "already_strong",
            0.95,
            None,
        )
        assert searched.kwargs == {} and outcome.noise == pytest.approx(0.1)
    else:  # runs of 1.0 and 0.85
        assert outcome.reason_code == "no_reliable_improvement" and searched.kwargs
        assert outcome.score_before == pytest.approx(0.925)


def test_the_095_check_reads_the_holdout_on_the_target_not_the_valset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    searched = Searched(found(cand(A, 0.5), seed=1.0))  # the search model rates the original 1.0
    outcome = decide(tmp_path, monkeypatch, searched, scored(lambda *_: False))
    assert (outcome.reason_code, outcome.score_before) == ("no_reliable_improvement", 0.0)
    assert outcome.search_score_before == 1.0 and searched.kwargs


# --- before the search: the holdout rule, the run folder, failures (SPEC R5, R11, R22, R24) -------


@pytest.mark.parametrize("n", [5, 7, 8])
def test_fewer_than_8_examples_without_trust_search_keep_the_original_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int
):
    raw, examples = ScriptedBackend(model(winners=(A,))), list(EXAMPLES[:n])
    searched = Searched(found(cand(A, 0.5)))
    outcome = decide(tmp_path, monkeypatch, searched, raw, examples=examples)
    assert examples == EXAMPLES[:n]  # the caller's list is left as it was
    if n < 8:
        assert (outcome.reason_code, outcome.prompt, outcome.stop) == ("no_holdout", PROMPT, None)
        assert raw.calls == [] and outcome.calls_used == 0
    else:  # the twin: 8 examples keep a holdout of 3
        assert (outcome.reason_code, outcome.prompt) == ("improved", A)
        kw = searched.kwargs  # final 3 x (3 + 1) + 3, start 1 + 2 x (3 + 1)
        assert (kw["calls_left_at_start"], kw["iter_cost"]) == (100 - 15 - 9, 12)
        assert {c.user.split("\n\n")[0] for c in raw.calls if c.model == MODELS.target} == {
            s.input for s in split(EXAMPLES, seed=0).holdout
        }


def test_the_run_folder_s_contract_and_scenarios_win_over_extraction_kind_and_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    saved = Contract(goal="answer", kind="template", checks=(Check("c1", "content", "answers"),))
    scenarios = [Scenario(id=f"r{i}", input=f"saved {i}") for i in range(1, 10)]

    def prepare(store: RunStore) -> None:
        store.save_contract(saved)
        store.save_scenarios(scenarios)

    raw, searched = ScriptedBackend(model()), Searched(found())
    decide(tmp_path, monkeypatch, searched, raw, examples=EXAMPLES, kind="task", prepare=prepare)
    assert not any(c.role in ("intake", "synth") for c in raw.calls)
    parts = split(scenarios, seed=0)
    assert (searched.kwargs["train"], searched.kwargs["val"]) == (parts.train, parts.val)
    tasks = [c for c in raw.calls if c.role == "task"]
    assert tasks and all(c.system == PROMPT and c.user.startswith("saved ") for c in tasks)


def test_kind_given_by_the_user_replaces_the_guess_and_the_contract_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw, searched = ScriptedBackend(model()), Searched(found())
    decide(tmp_path, monkeypatch, searched, raw, kind="template")  # the intake guesses "task"
    tasks = [c for c in raw.calls if c.role == "task"]
    assert tasks and all(c.system == PROMPT for c in tasks)
    store = RunStore.resume(tmp_path / "runs", next((tmp_path / "runs").iterdir()).name)
    try:
        assert store.contract() is not None and store.contract().kind == "template"  # type: ignore[union-attr]
        assert store.scenarios() == SYNTHESISED
    finally:
        store.close()


@pytest.mark.parametrize("step", ["intake", "synth", "seed run 2"])
def test_a_failed_call_before_the_search_ends_the_run_and_is_never_a_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
):
    def failed(call: Call) -> bool:
        if step == "seed run 2":
            return call.sample == 1 and call.user == f"{HOLDOUT[0].input}\n\n{PROMPT}"
        return call.role == step

    healthy = model()
    raw = ScriptedBackend(lambda call: CallError("exit 1") if failed(call) else healthy(call))
    searched = Searched(found(cand(A, 0.5)))
    with pytest.raises(BackendError, match="intake|synth|holdout"):
        decide(tmp_path, monkeypatch, searched, raw)
    assert searched.kwargs == {} and sum(map(failed, raw.calls)) == 3  # tried 3 times, then out
