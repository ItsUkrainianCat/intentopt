"""The quick, fast and checked tiers on the command line (SPEC R4, R21, R22, R25; ADR-011): why a
real run of a fast plan would refuse or keep the original, the fast plan a resumed run rebuilds
from its saved plan, and `Progress`, the log `improve_fast` writes, which reaches stderr line by
line while the run goes."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import TextIO

from autoimprover.cli_plan import duration
from autoimprover.fastplan import FastPlan, fast_plan, tier_for
from autoimprover.report import Emitter
from autoimprover.runner import count_tokens
from autoimprover.runstore import RunStore, RunStoreError
from autoimprover.types import Scenario


def fast_refusal(fplan: FastPlan) -> str | None:
    """Why a real run of `fplan` would not start (SPEC R4, R25): even its smallest shape needs
    more seconds than `--time` gives."""
    if fplan.est_seconds <= fplan.time_s:
        return None
    return (
        f"the {fplan.tier} plan needs about {fplan.est_seconds:.0f} s, more than --time "
        f"{duration(fplan.time_s)}; give a longer --time, or more --workers (now {fplan.workers})"
    )


def checked_keeps(fplan: FastPlan, examples: Sequence[Scenario] | None) -> str | None:
    """Why a checked run would keep the original without a call: the user's examples leave
    nothing to hold out after the ones it picks on (as `fast.improve_fast` decides)."""
    if fplan.tier != "checked" or examples is None or len(examples) > fplan.scenarios:
        return None
    return (
        f"the checked tier holds out the examples after the first {fplan.scenarios}, and "
        f"{len(examples)} examples leave none; give more examples, or a shorter --time"
    )


def saved_fast_plan(store: RunStore, had_examples: bool) -> FastPlan:
    """The fast plan of a resumed run, rebuilt from its saved plan and whether it had the user's
    examples, so it asks the very calls the run asked (SPEC R22); a saved tier that its clock
    does not select is a damaged manifest."""
    plan = store.plan
    try:
        tier = tier_for(plan.wall_clock_s)
    except ValueError:
        tier = None
    if tier != plan.tier:
        raise RunStoreError(
            f"manifest.json in the run folder {store.path} is damaged (its tier {plan.tier} does "
            f"not go with its {plan.wall_clock_s} s); start a new run, or remove this one with "
            f"`autoimprover clean {store.run_id}`"
        )
    return fast_plan(plan.wall_clock_s, plan.workers, count_tokens(store.prompt), had_examples)


class Progress(io.StringIO):
    """The log `improve_fast` writes, a line per stage and per dropped call (SPEC R25): each
    line goes to the run's log file and, cleaned and flushed, to stderr, so the mod's status line
    shows the last one (SPEC R21) and stdout keeps only the result (SPEC R2). A StringIO only to
    be a TextIO: it keeps nothing."""

    def __init__(self, emit: Emitter, log: TextIO) -> None:
        super().__init__()
        self._emit = emit
        self._log = log
        self._partial = ""

    def write(self, text: str) -> int:
        self._log.write(text)
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self._emit.notice(line)
        return len(text)
