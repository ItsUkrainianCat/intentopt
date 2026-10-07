"""The reflection rounds of a fast or checked run whose examples all carry a reference (SPEC R16,
R17, R25; ADR-006, ADR-011, ADR-013, WP23): the second generation of such a run is up to
MAX_ROUNDS rounds of stages R, B2 and C2, each starting from the best candidate so far.

A round reflects on the best rewrite so far that kept the contract, by mean gain over the original
(`fast_reference.leading`; the original when none did), and the pick examples it failed, with
their references, outputs and failed checks (`refine.failures`); its K2 reflections each fix them
their own way (`refine.refine_strategy`), pass the same free gates as every rewrite (the meaning
words of the original and of every earlier candidate, the length cap, the literals, no copy of a
pick example's input), run on the pick examples (B2) and are judged with the contract check
against the original's scores of stage C (C2); the pick of stage D is over every candidate of
every round. Held-out examples are never read: they are not scored before stage E.

The first round runs when the plan has a second generation and it fits what is left, as the plan's
second generation always did (a misfit is a stop of the run). A round that found no better
candidate than the best before it, or was cut, ends the rounds; after one that did, the latency
model is fitted again to every reply so far (`fast_calibrate`), and another round runs only when
the best candidate still fails a pick example and a round of K2 reflections, or of fewer, and
stage E after it fit within PLAN_SHARE of the clock, never past the deadline, and the calls left,
by that model (a misfit here is the normal ending, no stop). Stage E stays last. Each round logs
how many pick examples the best candidate passes in full, before and after it, and the reason of
the outcome ends with the rounds and the pick examples passed (`said_rounds`,
`reference_text.ROUNDS`). Without references the second generation is the pairwise one of
`fast_stages`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from autoimprover import reference_text
from autoimprover.fast_pairwise import Judged, Rewrite
from autoimprover.fast_prompts import reflect_call
from autoimprover.fast_reference import ReferenceStages, gain_of, leading
from autoimprover.fastplan import PLAN_SHARE, Stage, misfit, second_stages, tail
from autoimprover.parallel import parallel_map
from autoimprover.reference_score import scored
from autoimprover.refine import MAX_ROUNDS, refine_note
from autoimprover.runner import count_tokens
from autoimprover.types import CallFailed, Contract, Scenario

# Scores are compared at this many decimals (`reference_score`).
_DIGITS = 9


@dataclass
class RoundStages(ReferenceStages):
    """`ReferenceStages` whose second generation, with references, is the rounds above: `leader` is
    the best judged rewrite so far (None: the original), `rounds` the rounds that reflected and
    `picked` the ids of the pick examples stage C judged."""

    leader: Judged | None = field(default=None, init=False)
    rounds: int = field(default=0, init=False)
    picked: tuple[str, ...] = field(default=(), init=False)

    def stage_c(
        self,
        contract: Contract,
        rewrites: list[Rewrite],
        answers: list[dict[str, str | CallFailed]],
        original: dict[str, str | CallFailed],
        pick: list[Scenario],
        noise_run: dict[str, str | CallFailed] | None,
    ) -> tuple[list[Judged], float] | None:
        found = super().stage_c(contract, rewrites, answers, original, pick, noise_run)
        if self.ref is not None and noise_run is not None and found is not None:
            self.picked, self.leader = tuple(s.id for s in pick), leading(found[0])
        return found

    def second(
        self,
        contract: Contract,
        pick: list[Scenario],
        original: dict[str, str | CallFailed],
        noise: float,
        first: list[Judged],
        holdout: int,
    ) -> list[Judged]:
        """With references, the rewrites of every round as judged; without, the pairwise second
        generation."""
        if self.ref is None:
            return super().second(contract, pick, original, noise, first, holdout)
        judged: list[Judged] = []
        for number in range(1, MAX_ROUNDS + 1):
            before, earlier = self.leader, [*first, *judged]
            if number > 1 and not self.parents(earlier, pick)[0]["scenarios"]:
                self.note(f"round {number}: the best candidate fails no pick example")
                break
            k2 = self.room(number, len(pick), holdout)
            if not k2:
                break
            judged += self.round(number, k2, contract, pick, original, earlier, holdout)
            self.leader = leading([*first, *judged])
            self.note(
                f"round {number}: best passes {self.passes(self.leader)} of {len(pick)} pick "
                f"examples, was {self.passes(before)}"
            )
            gained = round(gain_of(self.leader), _DIGITS) > round(gain_of(before), _DIGITS)
            if self.ended or not gained:
                break
            self.calibrate(f"C2 of round {number}")
        return judged

    def room(self, number: int, m: int, holdout: int) -> int:
        """How many reflections round `number` on `m` pick examples writes: the plan's K2 when
        its stages and stage E fit what is left (round 1; a misfit is kept in `stop`), from round
        2 the most, down to 1, that fit `within` what is left; 0 when none does."""
        k2 = self.fplan.rewrites2
        if number == 1:
            return k2 if self.fits(self.later(k2, m, holdout)) else 0
        cause = None
        for k in range(k2, 0, -1):
            if (cause := self.within(self.later(k, m, holdout))) is None:
                return k
        left = f"time left within {PLAN_SHARE} of the clock" if cause == "clock" else "calls left"
        self.note(f"round {number}: no {left} for it and stage E")
        return 0

    def later(self, k2: int, m: int, holdout: int, start: int = 0) -> tuple[Stage, ...]:
        """Stages R (from `start` 0; 1 leaves it out), B2 and C2 of `k2` reflections on `m` pick
        examples, then stage E of `holdout` held out, by the latency model at hand."""
        w, p, lat, ref = self.workers, count_tokens(self.prompt), self.latency, self.ref
        return (
            *second_stages(k2, m, w, p, lat, ref)[start:],
            *tail(0, 0, holdout, w, p, lat, ref),
        )

    def within(self, stages: Sequence[Stage]) -> str | None:
        """Why `stages` do not fit within PLAN_SHARE of the clock, never past the deadline, and
        the calls left ("clock" or "budget"); None when they fit. A misfit is no stop."""
        budgeted = self.budgeted
        seconds = min(PLAN_SHARE * self.fplan.time_s, budgeted.deadline) - self.clock.elapsed()
        return misfit(stages, seconds, budgeted.limit - budgeted.used)

    def round(
        self,
        number: int,
        k2: int,
        contract: Contract,
        pick: list[Scenario],
        original: dict[str, str | CallFailed],
        earlier: list[Judged],
        holdout: int,
    ) -> list[Judged]:
        """Round `number`: K2 reflections on the best of `earlier` and its failures (R), those
        that pass the free gates run (B2) and are judged (C2); none when the clock, the calls or
        the gates leave none, or a stage was cut."""
        plan = self.plan
        parents = self.parents(earlier, pick)
        self.rounds = number
        self.note(
            f"stage R, round {number}: {k2} reflection(s) on the best candidate and the "
            f"{len(parents[0]['scenarios'])} pick example(s) it failed"
        )
        calls = [
            reflect_call(
                self.prompt,
                parents,
                contract,
                v,
                plan.models.reflect,
                plan.strictness,
                plan.allow_growth,
                reference=True,
                round_number=number,
            )
            for v in range(k2)
        ]
        drafts = parallel_map(lambda call: self.parsed(self.ask(call)), calls, self.workers)
        self.absorb([(f"reflection {v}", draft) for v, draft in enumerate(drafts)])
        texts = [(v, d) for v, d in enumerate(drafts) if isinstance(d, str)]
        kept = self.gates(texts, [j.rewrite for j in earlier], second=True, pick=pick)
        rewrites = [r._replace(note=refine_note(r.variant, number)) for r in kept]
        rest = self.later(k2, len(pick), holdout, start=1)
        fits = self.fits(rest) if number == 1 else self.within(rest) is None
        if self.ended or not rewrites or not fits:
            return []
        self.note(f"stage B2: {len(rewrites)} reflection(s) on {len(pick)} scenarios")
        outputs = self.stage_b(contract, [(r.text, 0) for r in rewrites], pick)
        if self.ended:
            return []
        found = self.stage_c(contract, rewrites, outputs, original, pick, None)
        if found is None or self.ended:
            return []
        return found[0]

    def passes(self, judged: Judged | None) -> int:
        """The pick examples `judged`'s rewrite passed in full, every check; with None, those
        both runs of the original passed."""
        return self.passed(None if judged is None else judged.rewrite.text)

    def passed(self, text: str | None) -> int:
        """The pick examples the rewrite `text` passed in full; None: the original's two runs."""
        runs: tuple[Mapping[str, float], ...] = (
            self.base or ({}, {}) if text is None else (scored(self.graded.get(text, [])),)
        )
        return sum(all(_full(run.get(name)) for run in runs) for name in self.picked)

    def said_rounds(self, rewrite: Rewrite | None) -> str:
        """The rounds of a reference-scored outcome and the pick examples passed in full by
        `rewrite`, when one is returned, else by the best candidate, and by the original
        (`reference_text.ROUNDS`); "" without references or before stage C scored."""
        if self.ref is None or self.base is None:
            return ""
        best = None if self.leader is None else self.leader.rewrite
        chosen = best if rewrite is None else rewrite
        return reference_text.ROUNDS.format(
            rounds=self.rounds,
            passed=self.passed(None if chosen is None else chosen.text),
            original=self.passed(None),
            total=len(self.picked),
        )


def _full(score: float | None) -> bool:
    return score is not None and round(score, _DIGITS) >= 1
