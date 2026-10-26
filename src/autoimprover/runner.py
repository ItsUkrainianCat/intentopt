"""The improve flow of ARCHITECTURE section 1, from the intent contract to the Outcome, and the
arithmetic it is planned with: `fixed_costs` and `refusal` decide before the first paid call
whether a budget pays for a search (SPEC R4, R17). `improve` extracts the contract (SPEC R5),
splits the scenarios (SPEC R11, R15), scores the original twice on the holdout on the target model
(SPEC R12, R13, R14a), runs GEPA with the reflection prompt built here (SPEC R8; ADR-006) on the
search's share of the budget, then opens the final steps' share (SPEC R17). An answer passes the
free gates (length, literals), the judged contract check and a target-model holdout run that beats
the original by more than the noise (SPEC R3, R6, R7, R9); without a holdout only `--trust-search`
returns one, unverified (SPEC R11). A failed call outside the search is a BackendError, never a
score (SPEC R24); the flow reads only the run folder and the call cache, so a resume reaches the
same Outcome (SPEC R22)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TextIO

from autoimprover import deep_pairwise
from autoimprover.backend import BudgetedBackend, CachedBackend, Clock
from autoimprover.contract import check, extract_contract, literals_preserved
from autoimprover.evaluator import Evaluator
from autoimprover.fast_pairwise import prefers
from autoimprover.lengths import count_tokens as count_tokens  # re-exported: many importers
from autoimprover.lengths import length_ok as length_ok
from autoimprover.reference_score import reference_of
from autoimprover.reflection_text import reflection_template as reflection_template
from autoimprover.runstore import RunStore
from autoimprover.scenarios import MIN_SCENARIOS_FOR_HOLDOUT, split, split_sizes, synthesize
from autoimprover.search import Candidate, SearchResult, run_search
from autoimprover.types import (
    JUDGE_BATCH_MAX,
    MINIBATCH_SIZE,
    SEARCH_CLOCK_SHARE,
    Backend,
    BackendError,
    BatchEvaluator,
    BudgetExhausted,
    CallFailed,
    Contract,
    Kind,
    Outcome,
    Plan,
    Scenario,
)

# A real run refuses a budget that affords fewer search iterations than this, worst case, unless
# `--force-low-budget` is given (SPEC R4).
MIN_ITERATIONS = 4
# Candidates that reach the final steps: each is contract-checked and run on the holdout (SPEC R17).
FINALISTS = 3


@dataclass(frozen=True)
class FixedCosts:
    """The calls a plan spends outside GEPA's iterations and what is left for them (SPEC R4, R17;
    ARCHITECTURE section 1). `pre` is paid before the search (intake, synthesis, the two seed runs
    on the holdout, GEPA's scoring of the original on the valset), `final` after it (up to
    FINALISTS holdout runs and contract checks; with `pairwise`, each finalist is only run, not
    scored, and judged against the original in two orders, after the original's two runs were
    judged against each other: ADR-014). `iter_cost` is an accepted child (a reflection,
    the parent's and the child's minibatch, a valset pass); `iterations` is what `search_calls`
    affords at that cost, the worst case, and `iterations_best` at the cost of a rejected child.
    `search_calls` is negative when the fixed costs alone exceed the budget."""

    holdout: int
    valset: int
    minibatch: int
    pre: int
    final: int
    iter_cost: int
    search_calls: int
    iterations: int
    iterations_best: int


def _pass(k: int) -> int:
    """Calls for one scoring pass over k scenarios: a task call each, plus ceil(k /
    JUDGE_BATCH_MAX) judge calls, in whole numbers."""
    return k - (-k // JUDGE_BATCH_MAX)


def fixed_costs(plan: Plan, n: int, synthesising: bool, *, pairwise: bool = True) -> FixedCosts:
    """The costs of `plan` with n scenarios, synthesised by one call or given by the user. From 8
    scenarios on the split of SPEC R15 holds out some; below 8 there is no holdout and every
    scenario is both dataset and valset (SPEC R11). `pairwise`: the final decision is the pairwise
    preference of ADR-014, not the reference score. Whole-number arithmetic only."""
    if n < 1:
        raise ValueError("a plan needs at least one scenario")
    holdout, valset, dataset = split_sizes(n) if n >= MIN_SCENARIOS_FOR_HOLDOUT else (0, n, n)
    minibatch = min(MINIBATCH_SIZE, dataset)
    pre = 1 + int(synthesising) + 2 * _pass(holdout) + _pass(valset)  # _pass(0) is 0
    if pairwise and holdout:
        final = FINALISTS * holdout + FINALISTS + 2 * FINALISTS + 2
    else:
        final = FINALISTS * _pass(holdout) + FINALISTS
    rejected = 1 + 2 * (minibatch + 1)
    iter_cost = rejected + _pass(valset)
    search_calls = plan.budget - pre - final
    share = max(0, search_calls)
    iterations, best = share // iter_cost, share // rejected
    return FixedCosts(
        holdout, valset, minibatch, pre, final, iter_cost, search_calls, iterations, best
    )


def refusal(costs: FixedCosts, *, force_low_budget: bool) -> str | None:
    """Why a real run with these costs must not start (exit 2 before any paid call), or None: the
    fixed costs alone exceed the budget, which `--force-low-budget` cannot override, or the budget
    affords fewer than MIN_ITERATIONS iterations, worst case (SPEC R4, R17). The message names the
    budget that would afford MIN_ITERATIONS."""
    budget = costs.pre + costs.final + costs.search_calls
    enough = costs.pre + costs.final + MIN_ITERATIONS * costs.iter_cost
    if costs.search_calls < 0:
        return (
            f"the fixed costs ({costs.pre} calls before the search and {costs.final} after it) "
            f"exceed the budget of {budget}; use --budget {enough} or more"
        )
    if costs.iterations < MIN_ITERATIONS and not force_low_budget:
        return (
            f"the budget of {budget} calls affords {costs.iterations} search iterations in the "
            f"worst case, fewer than {MIN_ITERATIONS}; use --budget {enough} or more, or pass "
            "--force-low-budget"
        )
    return None


# --- the holdout (SPEC R3, R12, R14a, R24) --------------------------------------------------------


def score_holdout(evaluator: BatchEvaluator, candidate: str, holdout: Sequence[Scenario]) -> float:
    """The mean score of `candidate` on the holdout. A failed call leaves an incomplete entry, and
    that raises BackendError: a failed seed or finalist run is never a score (SPEC R24)."""
    if not holdout:
        raise ValueError("there is no holdout to score on")
    return _mean_score(evaluator(candidate, holdout))


def _mean_score(entries: Sequence[tuple[float, dict[str, Any]]]) -> float:
    """The mean of the entries' scores; an incomplete entry raises BackendError (SPEC R24)."""
    for _score, info in entries:
        if info.get("incomplete"):
            why = info.get("error", "no reason given")
            raise BackendError(f"a call failed while scoring on the holdout: {why}")
    return math.fsum(score for score, _info in entries) / len(entries)


# --- the flow (SPEC R3, R5, R6, R11, R12, R13, R14a, R17, R22, R24) -------------------------------

# An original that scores at least this on the holdout is kept without a search (SPEC R13).
ALREADY_STRONG = 0.95
# The least a candidate must gain on the holdout; else twice the seed runs' difference (SPEC R12).
MIN_THRESHOLD = 0.05

_REASONS = {
    "improved": "beats the original on the holdout, on the target model, by more than the noise",
    "no_reliable_improvement": "no reliable improvement: no candidate beat the original on the "
    "holdout by more than the noise",
    "already_strong": "already strong, relative to the generated checks: the original scores at "
    f"least {ALREADY_STRONG} on the holdout",
    "no_holdout": f"fewer than {MIN_SCENARIOS_FOR_HOLDOUT} scenarios leave no holdout to verify "
    "a result on, so the original is kept; --trust-search accepts an unverified result",
    "no_candidate_beat_seed": "no candidate beat the original on the valset",
    "unconfirmed_out_of_budget": "ran out of budget or time before a candidate was confirmed",
}
# The reasons of a deep result decided by the pairwise judge (ADR-014); `deep_pairwise.MARK` tells
# the report to word it as a preference.
_PAIRWISE_REASONS = {
    "improved": f"preferred over the original by the {deep_pairwise.MARK}, on the target model, "
    "by more than the noise of the original's two runs",
    "no_reliable_improvement": "no reliable improvement: no candidate was preferred over the "
    f"original by the {deep_pairwise.MARK} by more than the noise of the original's two runs",
    "unconfirmed_out_of_budget": "ran out of budget or time before the "
    f"{deep_pairwise.MARK} had confirmed a candidate",
}
_UNVERIFIED = "beats the original on the valset; not verified on a holdout (--trust-search)"


class _Seed(NamedTuple):
    """The original's two holdout runs on the target model: their mean, their difference and the
    gain a candidate must exceed (SPEC R12), and their outputs, which the pairwise decision
    compares (ADR-014)."""

    score: float
    noise: float
    threshold: float
    outputs: tuple[dict[str, str | CallFailed], ...] = ()


def improve(
    prompt: str,
    plan: Plan,
    *,
    backend: Backend,
    cache: CachedBackend,
    budgeted: BudgetedBackend,
    clock: Clock,
    store: RunStore,
    scenarios: Sequence[Scenario] | None,
    kind: Kind | None,
    trust_search: bool,
    log: TextIO,
) -> Outcome:
    """Improve `prompt` under `plan` (ARCHITECTURE section 1, from the contract on). `backend` is
    the run's stack, `cache` its `Cached(Resilient(Budgeted))` layer, the tombstone cache (R22);
    `budgeted` the Budgeted inside it, its limit still `budget - final`; `clock` the run's clock,
    `store` its open run folder. `scenarios` are the user's examples, or None to synthesise them;
    the run folder's contract and scenarios win over a new extraction, `kind` and `scenarios` (a
    resumed run). `log` takes GEPA's output.

    Fewer than 8 scenarios without `trust_search` keep the original before any call (SPEC R11).
    A BudgetExhausted outside the search keeps the original (SPEC R17); a CallFailed outside it
    raises BackendError (SPEC R24); an abort inside it raises its own exception."""
    run = _Run(prompt, plan, backend, cache, budgeted, store)
    given = store.scenarios() or (None if scenarios is None else list(scenarios))
    if given is not None and len(given) < MIN_SCENARIOS_FOR_HOLDOUT and not trust_search:
        return run.outcome("no_holdout")
    try:
        return run.flow(given, kind, clock, log)
    except BudgetExhausted:  # before the search or in the final steps: never inside it
        return run.outcome("unconfirmed_out_of_budget")
    except CallFailed as error:
        raise BackendError(str(error)) from error


@dataclass
class _Run:
    """One improve run: its fixed inputs, the steps of the flow, and what every ending reports:
    calls used, run folder and, in `measured`, what was measured before it: the original's
    holdout score and noise once both seed runs ended, the stop cause and the original's valset
    score once the search ended (SPEC R12, R17)."""

    prompt: str
    plan: Plan
    backend: Backend
    cache: CachedBackend
    budgeted: BudgetedBackend
    store: RunStore
    measured: dict[str, Any] = field(default_factory=dict)
    pairwise: bool = False  # the final decision is the pairwise preference (ADR-014)
    decided: bool = False  # ... and it was made, so the reason says so

    def flow(
        self, given: list[Scenario] | None, kind: Kind | None, clock: Clock, log: TextIO
    ) -> Outcome:
        plan, prompt, models = self.plan, self.prompt, self.plan.models
        contract = self.store.contract()
        if contract is None:
            contract = extract_contract(self.backend, models.reflect, prompt, kind)
            self.store.save_contract(contract)
        template = reflection_template(
            contract, plan.strictness, count_tokens(prompt), plan.allow_growth
        )
        synthesising = given is None
        if given is None:
            given = synthesize(self.backend, models.reflect, prompt, contract)
        if self.store.scenarios() is None:
            self.store.save_scenarios(given)
        self.pairwise = reference_of(given) is None
        costs = fixed_costs(plan, len(given), synthesising, pairwise=self.pairwise)
        parts = split(given, plan.seed)
        seed = self.seed_runs(contract, parts.holdout) if parts.holdout else None
        if seed is not None and seed.score >= ALREADY_STRONG:
            return self.outcome("already_strong")
        start = self.store.search_start_or_record(self.budgeted.used, clock.elapsed())
        result = run_search(
            seed=prompt,
            train=parts.train,
            val=parts.val,
            make_evaluator=lambda b: Evaluator(b, contract, models.task, models.judge),
            backend=self.backend,
            reflect_model=models.reflect,
            reflection_template=template,
            rng_seed=plan.seed,
            calls_left_at_start=(plan.budget - costs.final) - start[0],
            iter_cost=costs.iter_cost,
            search_start=start,
            wall_clock_s=plan.wall_clock_s,
            clock_share=SEARCH_CLOCK_SHARE,
            log=log,
            merge=plan.merge,
            cache=self.cache,
        )
        self.measured |= {"stop": result.stop, "search_score_before": result.seed_val_score}
        self.budgeted.raise_limit(plan.budget, plan.wall_clock_s)
        if seed is None:
            return self.trusted(contract, result)
        return self.confirmed(contract, parts.holdout, seed, result)

    def on_target(
        self, contract: Contract, text: str, holdout: Sequence[Scenario], sample: int = 0
    ) -> float:
        """`text` scored on the holdout on the target model, as the user will run it (SPEC R14a)."""
        models = self.plan.models
        evaluator = Evaluator(self.backend, contract, models.target, models.judge, sample)
        return score_holdout(evaluator, text, holdout)

    def seed_run(
        self, contract: Contract, holdout: Sequence[Scenario], sample: int
    ) -> tuple[float, dict[str, str | CallFailed]]:
        """The original on the holdout on the target model: its mean score and its outputs."""
        models = self.plan.models
        evaluator = Evaluator(self.backend, contract, models.target, models.judge, sample)
        outputs = evaluator.outputs(self.prompt, holdout)
        return _mean_score(evaluator.score(holdout, outputs)), outputs

    def seed_runs(self, contract: Contract, holdout: Sequence[Scenario]) -> _Seed:
        """The original scored twice on the holdout, the second run under sample 1 so it is a new
        call, not a cache hit (SPEC R12)."""
        (first, out_first), (second, out_second) = (
            self.seed_run(contract, holdout, s) for s in (0, 1)
        )
        noise = abs(first - second)
        seed = _Seed(
            (first + second) / 2, noise, max(MIN_THRESHOLD, 2 * noise), (out_first, out_second)
        )
        self.measured |= {"score_before": seed.score, "noise": seed.noise}
        return seed

    def finalists(
        self, contract: Contract, result: SearchResult, above: float | None = None
    ) -> list[tuple[Candidate, float]]:
        """The candidates that may become the answer, with their length ratios, best valset score
        first and the later of two equal ones first: never the original, and only those above
        `above` when it is given; the free gates first (length cap, literals), then the best
        FINALISTS of those, each kept only when the judged contract check finds no violation
        (SPEC R6, R7, R9)."""
        plan, prompt = self.plan, self.prompt
        gated: list[tuple[float, int, Candidate, float]] = []
        for index, candidate in enumerate(result.candidates):
            if candidate.text == prompt or (above is not None and candidate.val_score <= above):
                continue
            fits, ratio = length_ok(prompt, candidate.text, plan.strictness, plan.allow_growth)
            if fits and literals_preserved(prompt, candidate.text):
                gated.append((candidate.val_score, index, candidate, ratio))
        best = sorted(gated, key=lambda item: item[:2], reverse=True)[:FINALISTS]
        return [
            (candidate, ratio)
            for _score, _index, candidate, ratio in best
            if not check(self.backend, plan.models.judge, contract, prompt, candidate.text)
        ]

    def confirmed(
        self, contract: Contract, holdout: Sequence[Scenario], seed: _Seed, result: SearchResult
    ) -> Outcome:
        """The first finalist whose holdout score on the target model beats the original's by more
        than the threshold, verified; else the original (SPEC R3, R14a). Without references the
        finalists are judged by preference instead (`preferred`, ADR-014)."""
        if self.pairwise:
            return self.preferred(contract, holdout, seed, result)
        for candidate, ratio in self.finalists(contract, result):
            after = self.on_target(contract, candidate.text, holdout)
            margin = after - seed.score - seed.threshold
            if margin > 0:
                return self.outcome(
                    "improved",
                    candidate,
                    verified=True,
                    score_after=after,
                    margin=margin,
                    length_ratio=ratio,
                )
        return self.outcome("no_reliable_improvement")

    def preferred(
        self, contract: Contract, holdout: Sequence[Scenario], seed: _Seed, result: SearchResult
    ) -> Outcome:
        """The first finalist the blind pairwise judge prefers to the original on the holdout, on
        the target model, in both orders, by more scenarios than the original's two runs
        disagree on; else the original (ADR-014, SPEC R25). Each finalist is run, never scored:
        what decides is the judge's preference, so the scores of the original's two runs only
        serve the check of SPEC R13. `score_*`, `noise` and `margin` of the result are shares of
        the holdout scenarios, as in the fast tiers."""
        judge, request = self.plan.models.judge, self.prompt
        total = len(holdout)
        bar: int | None = None
        for candidate, ratio in self.finalists(contract, result):
            if bar is None:
                bar = deep_pairwise.noise(self.backend, judge, request, holdout, *seed.outputs)
            answers = self.target_outputs(contract, candidate.text, holdout)
            found = deep_pairwise.preferred(
                self.backend, judge, request, holdout, seed.outputs[0], answers
            )
            if prefers(found, bar):
                self.settle(bar, total)
                return self.outcome(
                    "improved",
                    candidate,
                    verified=True,
                    score_before=found.losses / total,
                    score_after=found.wins / total,
                    margin=(found.wins - found.losses - bar) / total,
                    length_ratio=ratio,
                )
        if bar is not None:
            self.settle(bar, total)
        return self.outcome("no_reliable_improvement")

    def settle(self, bar: int, total: int) -> None:
        """The decision was made by preference: the noise is the share of the holdout scenarios
        where the original's two runs had a winner, and the original's absolute score is no part
        of the result. A run cut short before this keeps what it measured (SPEC R12)."""
        self.decided = True
        self.measured.pop("score_before", None)
        self.measured["noise"] = bar / total

    def target_outputs(
        self, contract: Contract, text: str, holdout: Sequence[Scenario]
    ) -> dict[str, str | CallFailed]:
        """`text` run on the holdout on the target model, as the user will run it (SPEC R14a); a
        failed call is a BackendError, never a missing answer (SPEC R24)."""
        models = self.plan.models
        outputs = Evaluator(self.backend, contract, models.target, models.judge).outputs(
            text, holdout
        )
        for failure in outputs.values():
            if isinstance(failure, CallFailed):
                raise BackendError(f"a call failed while running on the holdout: {failure}")
        return outputs

    def trusted(self, contract: Contract, result: SearchResult) -> Outcome:
        """Without a holdout (`--trust-search`): the best finalist that beats the original on the
        valset, marked unverified; else the original (SPEC R11)."""
        before = result.seed_val_score
        finalists = [] if before is None else self.finalists(contract, result, above=before)
        if not finalists:
            return self.outcome("no_candidate_beat_seed")
        candidate, ratio = finalists[0]
        return self.outcome("improved", candidate, reason=_UNVERIFIED, length_ratio=ratio)

    def outcome(
        self, code: str, candidate: Candidate | None = None, reason: str = "", **fields: Any
    ) -> Outcome:
        """The Outcome `code`: the candidate as the answer, with its lineage notes and valset
        score, or the original unchanged."""
        fields = {**self.measured, **fields}
        if candidate is not None:
            fields |= {"changes": candidate.notes, "search_score_after": candidate.val_score}
        return Outcome(
            status="improved" if candidate is not None else "unchanged",
            prompt=self.prompt if candidate is None else candidate.text,
            reason=reason or (_PAIRWISE_REASONS if self.decided else _REASONS)[code],
            reason_code=code,
            calls_used=self.budgeted.used,
            run_dir=str(self.store.path),
            **fields,
        )
