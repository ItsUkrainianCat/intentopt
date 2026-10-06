"""`claude --effort` per role, applied in one place (SPEC R25; ADR-011 "Customisation"):
`EffortBackend` sits above the disk cache and gives a call that names no effort the level of its
role, the task level to task calls, the judge level to judge calls and the reflection level to
intake, synthesis and reflection calls. Being above the cache, the effort is part of the cache key
(SPEC R22), and no module that builds calls needs to know the levels."""

from __future__ import annotations

import dataclasses

from autoimprover.types import Backend, Call, Efforts, Reply, Role

_LEVEL_OF: dict[Role, str] = {
    "task": "task",
    "judge": "judge",
    "intake": "reflect",
    "synth": "reflect",
    "reflect": "reflect",
}


class EffortBackend:
    """`Backend` that sets `Call.effort` from `efforts` by the call's role, when the call has
    none and the role's level is not None, then asks `inner`."""

    def __init__(self, inner: Backend, efforts: Efforts) -> None:
        self._inner = inner
        self._efforts = efforts

    def complete(self, call: Call) -> Reply:
        level = getattr(self._efforts, _LEVEL_OF[call.role])
        if call.effort is None and level is not None:
            call = dataclasses.replace(call, effort=level)
        return self._inner.complete(call)
