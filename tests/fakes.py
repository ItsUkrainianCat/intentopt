"""Test doubles shared by unit and acceptance tests; import them as `from fakes import ...`.

Owned by the lead. `ScriptedBackend` implements the `Backend` protocol: it answers from a function,
records every call and can inject failures, so no test needs a real model.
"""

from collections.abc import Callable, Mapping, Sequence

from autoimprover.types import BackendError, Call, Reply

Script = Callable[[Call], "str | Exception"]


class ScriptedBackend:
    def __init__(self, script: Script) -> None:
        self._script = script
        self.calls: list[Call] = []

    def complete(self, call: Call) -> Reply:
        self.calls.append(call)
        answer = self._script(call)
        if isinstance(answer, Exception):
            raise answer
        return Reply(text=answer, tokens_in=len(call.user), tokens_out=len(answer))

    def count(self, role: str | None = None) -> int:
        return sum(1 for c in self.calls if role is None or c.role == role)


def by_role(replies: Mapping[str, str | Exception | Sequence[str | Exception]]) -> ScriptedBackend:
    """Answer per role: a constant, or a list served in order (the last entry repeats)."""
    served: dict[str, int] = {}

    def script(call: Call) -> str | Exception:
        answer = replies[call.role]
        if isinstance(answer, str | Exception):
            return answer
        i = served.get(call.role, 0)
        served[call.role] = i + 1
        return answer[min(i, len(answer) - 1)]

    return ScriptedBackend(script)


def failing(message: str = "backend down") -> ScriptedBackend:
    """A backend whose every call fails with BackendError (SPEC R24)."""
    return ScriptedBackend(lambda _call: BackendError(message))
