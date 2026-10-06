"""The pipeline of the quick, fast and checked tiers (SPEC R25; ADR-011) and the gates every rewrite
it returns has passed (SPEC R6, R7, R9, R14a, R17, R24); the free gates, stages B to D and the
second generation are in `fast_stages`.

Stage A is one wave: the intake, the synthesis of the scenarios (the user's examples replace it) and
the K rewrites (clarify, structure, tighten, specify, in that order), none of which needs another's
reply. A rewrite that fails a free gate is dropped there, before it costs a call: one with no change
in meaning words, one over the length cap, one that lost a literal. Stages B to D, with a second
generation from 45 s when the plan has one, pick a rewrite by pairwise preference against the noise
of two runs of the original (`fast_stages`, `fast_pairwise`). Stage E (checked tier) runs the winner
against the original once on the held-out scenarios on the target model, decided as `runner` decides
with MIN_THRESHOLD as the margin (SPEC R14a); a second held-out run of the original would cost more
calls than that, so the checked tier measures no held-out noise. The quick tier only checks its one
rewrite's contract. With `--ungated` (fast and checked tiers, a measuring aid), a pick that found
no winner returns the best-ranked rewrite that passed every gate (`fast_pairwise.best_ungated`),
under its own reason code and label, never verified; the checked tier then never runs stage E,
and returns a winner under that code too, so the measure sees the ranking's choice, not E's.
When every example of the user carries a reference (`reference_score.reference_of`), stages C to
E decide by agreement with the references instead (`fast_reference`; stage E then runs the
original twice and compares as the pick does), and every reason says so (`reference_text`, WP21).

After stage A the latency model is fitted to its replies (`fast_calibrate`), and the time it
leaves within PLAN_SHARE of the clock, never past the deadline, buys more pick scenarios, up to
`fastplan.MAX_SCENARIOS`: the user's examples first, else one more synthesis call (stage A2, its
own sample, its scenarios under ids of their own and kept in the run folder); a failed or cut
stage A2 keeps the plan's scenarios. Before stage E the time and calls left are compared with its
estimate (`fastplan`); a rewrite that has not passed every gate is never returned (SPEC R17). A
failed intake, synthesis, contract check or held-out run ends the run as BackendError (SPEC R24).

DEBT, private names used here, in `fast_stages` and in `fast_prompts` until their owners add public
seams: `evaluator.Evaluator._task_call` (overridden by `fast_prompts.FastEvaluator`),
`contract._ask`, `runner._token_cap`, `scenarios._loads`, `scenarios._text_problem`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, TextIO, cast

from autoimprover import reference_text as said
from autoimprover.backend import BudgetedBackend, Clock
from autoimprover.contract import _ask, check, extract_contract
from autoimprover.fast_calibrate import grow
from autoimprover.fast_pairwise import Rewrite, Win
from autoimprover.fast_prompts import (
    FastEvaluator,
    parse_synth,
    rewrite_call,
    synth_call,
)
from autoimprover.fast_reference import ReferenceStages, shares
from autoimprover.fast_stages import Dropped, dropped
from autoimprover.fastplan import PLAN_SHARE, FastPlan, contract_stage, tail
from autoimprover.parallel import parallel_map
from autoimprover.reference_score import Margin, beats, reference_of
from autoimprover.runner import MIN_THRESHOLD, count_tokens, score_holdout
from autoimprover.runstore import RunStore
from autoimprover.types import (
    CALL_RETRIES,
    Backend,
    BackendError,
    BudgetExhausted,
    CallFailed,
    Contract,
    Kind,
    Outcome,
    Plan,
    Scenario,
)

_FAST_LABEL = (
    "fast check: preferred over the original by a pairwise judge on the same few scenarios it was "
    "picked on, noise measured by comparing the original with itself, not verified on held-out "
    "scenarios"
)
_QUICK_LABEL = (
    "quick check: passed the contract check and the free gates, not scored on any scenario, not "
    "verified on held-out scenarios, no noise measured"
)
_NO_WIN = (
    "no rewrite kept the contract and was preferred over the original on more scenarios than it "
    "lost, by more than the noise of the original's two runs"
)
_UNGATED_LABEL = "ungated: the best-ranked candidate; no win over the original was shown"
_CUT = {
    "clock": "the clock ran out before a rewrite passed every gate",
    "budget": "the call limit ran out before a rewrite passed every gate",
}
# The sample of the second synthesis wave: past the first wave's sample 0 and its retries.
MORE_SAMPLE = 1 + CALL_RETRIES


def improve_fast(
    prompt: str,
    plan: Plan,
    fplan: FastPlan,
    *,
    backend: Backend,
    budgeted: BudgetedBackend,
    clock: Clock,
    store: RunStore,
    scenarios: Sequence[Scenario] | None,
    kind: Kind | None,
    log: TextIO,
    workers: int,
    ungated: bool = False,
) -> Outcome:
    """Improve `prompt` by `fplan` (SPEC R25). `backend` is the run's stack (the cache, or a layer
    above it), `budgeted` the Budgeted inside it, whose limit and deadline bind the run; `clock`
    the run's clock, `store` its open run folder. `scenarios` are the user's examples, or None to
    synthesise them; the run folder's contract and scenarios win over a new intake, `kind` and
    `scenarios` (a resumed run). `log` gets a line per stage and per dropped call; `workers` is
    how many calls run at a time; `ungated` is `--ungated` (fast and checked tiers): with no win,
    the best-ranked rewrite that passed every gate is returned, unverified, and the checked tier
    skips stage E, also for a winner. The checked tier with nothing left to hold out keeps the
    original before any call."""
    given = store.scenarios() or (None if scenarios is None else list(scenarios))
    ref = None if given is None else reference_of(given)
    run = _Fast(prompt, plan, fplan, backend, budgeted, clock, store, log, workers, ungated, ref)
    if fplan.tier == "checked" and given is not None and len(given) <= fplan.scenarios:
        return run.outcome(
            "no_holdout",
            why=f"the checked tier holds out the examples after the first {fplan.scenarios}, "
            "and there are none; give more examples, or a shorter --time",
        )
    try:
        return run.flow(given, kind, synthesising=given is None)
    except BudgetExhausted as error:  # the intake, the synthesis, the contract check or stage E
        run.stop = run.stop or dropped(error).cut
        return run.kept("")
    except CallFailed as error:
        raise BackendError(str(error)) from error


class _Fast(ReferenceStages):
    """One fast run: stages B to D and the run's state are `Stages` (`ReferenceStages` when every
    example carries a reference); here stage A, stage E, the quick tier and the endings."""

    def flow(self, given: list[Scenario] | None, kind: Kind | None, synthesising: bool) -> Outcome:
        contract, scenarios, rewrites = self.stage_a(kind, given)
        if self.fplan.tier == "quick":
            return self.quick(contract, rewrites)
        if self.store.scenarios() is None:
            self.store.save_scenarios(scenarios)
        if not rewrites:
            return self.kept("no rewrite passed the free gates (length cap, literals)")
        self.calibrate("A")
        m, scenarios = self.more(len(rewrites), scenarios, synthesising)
        h = self.fplan.holdout
        win = self.contest(contract, rewrites, scenarios[:m], h)
        referenced = self.ref is not None
        if win is None:
            return self.kept(said.NO_WIN if referenced else _NO_WIN)
        checked = self.fplan.tier == "checked"
        if not win.gated or (self.ungated and checked):  # --ungated: no stage E, never verified
            return self.outcome(
                "ungated_best_candidate",
                win.rewrite,
                why=said.UNGATED_LABEL if referenced else _UNGATED_LABEL,
                search_score_before=win.before,
                search_score_after=win.after,
                noise=win.bar,
                margin=win.gain - win.bar,
            )
        if checked:
            return self.confirm(contract, win, scenarios[m : m + h])
        return self.outcome(
            "improved",
            win.rewrite,
            why=said.FAST_LABEL if referenced else _FAST_LABEL,
            score_before=win.before,
            score_after=win.after,
            margin=win.gain - win.bar,
        )

    # --- stage A -----------------------------------------------------------------------------

    def stage_a(
        self, kind: Kind | None, given: list[Scenario] | None
    ) -> tuple[Contract, list[Scenario], list[Rewrite]]:
        """In one wave: the contract (the run folder's, or a new intake), the scenarios (`given`,
        else one synthesis call; none in the quick tier) and the rewrites that pass the free
        gates."""
        saved, fp, reflect = self.store.contract(), self.fplan, self.plan.models.reflect
        k, count = fp.rewrites, 0 if given else fp.scenarios + fp.holdout  # quick: 0 + 0
        jobs = (["intake"] if saved is None else []) + (["synth"] if count else []) + [*range(k)]
        self.note(f"stage A: {', '.join(map(str, jobs))}")

        def job(name: str | int) -> Contract | list[Scenario] | str | Dropped:
            if name == "intake":
                return extract_contract(self.backend, reflect, self.prompt, kind)
            if name == "synth":
                call = synth_call(self.prompt, count, reflect)
                return _ask(self.backend, call, lambda text: parse_synth(text, count))
            return self.draft(cast(int, name))

        results = dict(zip(jobs, parallel_map(job, jobs, self.workers), strict=True))
        contract = saved or cast(Contract, results["intake"])
        if saved is None:
            self.store.save_contract(contract)
        drafts = [cast(str | Dropped, results[v]) for v in range(k)]
        self.absorb([(f"rewrite {v}", draft) for v, draft in enumerate(drafts)])
        rewrites = self.gates([(v, d) for v, d in enumerate(drafts) if isinstance(d, str)])
        return contract, given or cast(list[Scenario], results.get("synth", [])), rewrites

    def more(
        self, k: int, scenarios: list[Scenario], can_synthesise: bool
    ) -> tuple[int, list[Scenario]]:
        """The pick count and the scenarios after stage A for `k` rewrites: the plan's, or more
        when the fitted model leaves time for them (`fast_calibrate.grow`), with stage A2 when
        the scenarios at hand are too few and the run `can_synthesise`."""
        fp, budgeted = self.fplan, self.budgeted
        if self.latency is None:
            return fp.scenarios, scenarios
        grown = grow(
            tier=fp.tier,
            rewrites=k,
            scenarios=fp.scenarios,
            holdout=fp.holdout,
            have=len(scenarios),
            can_synthesise=can_synthesise,
            rewrites2=fp.rewrites2,
            workers=self.workers,
            prompt_tokens=count_tokens(self.prompt),
            model=self.latency,
            seconds_left=min(PLAN_SHARE * fp.time_s, budgeted.deadline) - self.clock.elapsed(),
            calls_left=budgeted.limit - budgeted.used,
            ref=self.ref,
        )
        if grown is None:
            return fp.scenarios, scenarios
        count = grown.synthesise
        if not count:
            self.note(f"re-plan: pick on {grown.scenarios} scenarios")
        else:
            self.note(f"stage A2: {count} more scenarios, to pick on {grown.scenarios}")
            call = synth_call(self.prompt, count, self.plan.models.reflect, sample=MORE_SAMPLE)
            try:
                found = _ask(self.backend, call, lambda text: parse_synth(text, count))
            except (CallFailed, BudgetExhausted) as error:
                self.absorb([("more scenarios", dropped(error))])
                return fp.scenarios, scenarios
            scenarios = [*scenarios, *renamed(found, scenarios)]
            self.store.save_scenarios(scenarios)
        return grown.scenarios, scenarios

    def draft(self, variant: int) -> str | Dropped:
        plan = self.plan
        call = rewrite_call(
            self.prompt, variant, plan.models.reflect, plan.strictness, plan.allow_growth
        )
        return self.parsed(self.ask(call))

    # --- stage E and the quick tier ----------------------------------------------------------

    def confirm(self, contract: Contract, win: Win, holdout: Sequence[Scenario]) -> Outcome:
        """Stage E: the winner is returned, verified, only when it beats the original on the
        held-out scenarios on the target model by more than MIN_THRESHOLD (SPEC R3, R14a); with
        references, by the pick's rule against the original's two runs there (`held`)."""
        self.measured["search_score_before"] = win.before
        h, p = len(holdout), count_tokens(self.prompt)
        if not self.fits(tail(0, 0, h, self.workers, p, ref=self.ref)):
            return self.kept("")
        self.note(f"stage E: the winner and the original on {h} held-out scenarios")
        if self.ref is not None:
            return self.held(win, self.held_out(contract, win.rewrite.text, holdout), h)
        models, inner = self.plan.models, max(1, self.workers // 2)

        def on_target(text: str) -> float:
            evaluator = FastEvaluator(self.backend, contract, models.target, models.judge, 0, inner)
            return score_holdout(evaluator, text, holdout)

        before, after = parallel_map(on_target, [self.prompt, win.rewrite.text], self.workers)
        self.measured["score_before"] = before
        margin = after - before - MIN_THRESHOLD
        held = f"{len(holdout)} held-out scenarios, on the target model, by more than"
        if margin <= 0:
            return self.kept(f"the winner did not beat the original on {held} {MIN_THRESHOLD}")
        return self.outcome(
            "improved",
            win.rewrite,
            why=f"beats the original on {held} {MIN_THRESHOLD}; no noise measured",
            verified=True,
            score_after=after,
            search_score_after=win.after,
            margin=margin,
        )

    def held(self, win: Win, margin: Margin, h: int) -> Outcome:
        """Stage E's ending with references: the winner is returned, verified, only when it
        beats the original's two held-out runs on the target model (`reference_score.beats`)."""
        found = shares(win.rewrite, margin)
        self.measured |= {"score_before": found.before, "noise": found.bar}
        held = f"{h} held-out examples, on the target model"
        if not beats(margin):
            return self.kept(said.HELD_LOSS.format(held=held))
        return self.outcome(
            "improved",
            win.rewrite,
            why=said.HELD_WIN.format(held=held),
            verified=True,
            score_after=found.after,
            search_score_after=win.after,
            margin=found.gain - found.bar,
        )

    def quick(self, contract: Contract, rewrites: list[Rewrite]) -> Outcome:
        """The quick tier: its rewrite is returned when it passes the contract check (SPEC R6)."""
        if not rewrites:
            return self.kept("the rewrite did not pass the free gates (length cap, literals)")
        if not self.fits((contract_stage(),)):
            return self.kept("")
        self.note("the contract check of the rewrite")
        if check(self.backend, self.plan.models.judge, contract, self.prompt, rewrites[0].text):
            return self.kept("the rewrite did not pass the contract check")
        return self.outcome("improved", rewrites[0], why=_QUICK_LABEL)

    # --- the endings ---------------------------------------------------------------------------

    def kept(self, why: str) -> Outcome:
        """The original, unchanged: for `why`, or out of time or calls when a stage was shrunk
        or cut and no rewrite passed every gate (SPEC R17)."""
        if self.stop:
            return self.outcome("unconfirmed_out_of_budget", why=_CUT[self.stop])
        return self.outcome("no_reliable_improvement", why=why)

    def outcome(
        self, code: str, rewrite: Rewrite | None = None, why: str = "", **fields: Any
    ) -> Outcome:
        fields = {**self.measured, **fields}
        if rewrite is not None:
            fields["changes"] = (rewrite.note,)
            fields["length_ratio"] = rewrite.ratio
        return Outcome(
            status="unchanged" if rewrite is None else "improved",
            prompt=self.prompt if rewrite is None else rewrite.text,
            reason=f"tier {self.fplan.tier}: {why}",
            reason_code=code,
            stop=self.stop,
            calls_used=self.budgeted.used,
            run_dir=str(self.store.path),
            mode=self.fplan.tier,
            **fields,
        )


def renamed(found: Sequence[Scenario], earlier: Sequence[Scenario]) -> list[Scenario]:
    """`found` under ids no earlier scenario has: a taken id gets a "+" until it is free."""
    taken = {scenario.id for scenario in earlier}
    named = []
    for scenario in found:
        name = scenario.id
        while name in taken:
            name += "+"
        taken.add(name)
        named.append(dataclasses.replace(scenario, id=name))
    return named
