"""Fixtures for the black-box acceptance tests (WP8), written from docs/SPEC.md, ADR-007, ADR-008
and docs/ARCHITECTURE.md sections 8 and 9 only.

Every test drives the public entry `autoimprover.cli.main(argv, backend=..., now=...)` the way a
user's shell would, with the scripted raw backend of `tests/fakes.py` in place of `claude -p` and a
`FakeClock` in place of the monotonic clock. The run folder lives under the `XDG_STATE_HOME` that
the R20 guards of `tests/conftest.py` point into the test's own `tmp_path`.
"""

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fakes import FakeClock, ScriptedBackend

from autoimprover import cli
from autoimprover.types import RUN_ID_PATTERN, Call, Reply


@dataclass
class Result:
    """What a user sees after one `autoimprover` invocation."""

    code: int
    out: str
    err: str
    calls: list[Call] = field(default_factory=list)

    def json(self) -> dict:
        """stdout must be exactly one JSON object (SPEC R2)."""
        decoder = json.JSONDecoder()
        text = self.out.strip()
        obj, end = decoder.raw_decode(text)
        assert end == len(text), f"more than one JSON value on stdout: {self.out!r}"
        assert isinstance(obj, dict), f"stdout is not a JSON object: {self.out!r}"
        return obj

    def resume_id(self) -> str:
        """The run id from the `resume with: ... autoimprover --resume <id>` line."""
        match = re.search(r"resume with: .*autoimprover --resume (\S+)", self.err)
        assert match, f"no resume line in stderr: {self.err!r}"
        run_id = match.group(1)
        assert re.fullmatch(RUN_ID_PATTERN, run_id), run_id
        return run_id

    def run_folder(self) -> Path:
        """The folder named on the `run folder: <path>` line."""
        match = re.search(r"run folder: (.+)", self.err)
        assert match, f"no run folder line in stderr: {self.err!r}"
        return Path(match.group(1).strip())


class TimedBackend:
    """A raw backend whose calls take `duration(call)` seconds on a FakeClock. It records the clock
    reading at the start of every call, so a test can prove no call starts after a deadline."""

    def __init__(
        self, inner: ScriptedBackend, clock: FakeClock, duration: Callable[[Call], float]
    ) -> None:
        self.inner = inner
        self.clock = clock
        self.duration = duration
        self.starts: list[float] = []

    @property
    def calls(self) -> list[Call]:
        return self.inner.calls

    def count(self, role: str | None = None) -> int:
        return self.inner.count(role)

    def complete(self, call: Call) -> Reply:
        self.starts.append(self.clock.now())
        seconds = self.duration(call)
        self.clock.advance(seconds)
        reply = self.inner.complete(call)
        return Reply(
            text=reply.text,
            tokens_in=reply.tokens_in,
            tokens_out=reply.tokens_out,
            duration_s=seconds,
        )


@pytest.fixture
def run_cli(capsys: pytest.CaptureFixture[str]) -> Callable[..., Result]:
    """Run `cli.main` with an injected raw backend and fake clock; capture exit, stdout, stderr."""

    def run(
        argv: Sequence[str],
        backend: ScriptedBackend | TimedBackend | None = None,
        clock: FakeClock | None = None,
    ) -> Result:
        backend = backend if backend is not None else ScriptedBackend(_no_call_expected)
        clock = clock if clock is not None else FakeClock()
        capsys.readouterr()
        code = cli.main(list(argv), backend=backend, now=clock.now)
        out, err = capsys.readouterr()
        return Result(code=code, out=out, err=err, calls=list(backend.calls))

    return run


def _no_call_expected(call: Call) -> str:
    raise AssertionError(f"no model call expected, got a {call.role} call")


@pytest.fixture
def state_home() -> Path:
    """The XDG_STATE_HOME of this test (set by the R20 guards)."""
    return Path(os.environ["XDG_STATE_HOME"])


@pytest.fixture
def runs_dir(state_home: Path) -> Path:
    """Where run folders go: `$XDG_STATE_HOME/autoimprover/runs/` (SPEC R23)."""
    return state_home / "autoimprover" / "runs"


@pytest.fixture
def examples(tmp_path: Path) -> Callable[..., str]:
    """Write an `--examples` JSONL file with n scenarios `situation 1`..`situation n`."""

    def write(n: int, name: str = "examples.jsonl", **extra: object) -> str:
        path = tmp_path / name
        lines = [json.dumps({"input": f"situation {i}", **extra}) for i in range(1, n + 1)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    return write


@pytest.fixture
def timed() -> Callable[..., TimedBackend]:
    """Wrap a ScriptedBackend so each call advances a FakeClock (R17 clock tests)."""

    def make(
        inner: ScriptedBackend, clock: FakeClock, duration: float | Callable[[Call], float]
    ) -> TimedBackend:
        fn = duration if callable(duration) else (lambda _call: float(duration))
        return TimedBackend(inner, clock, fn)

    return make


@pytest.fixture
def cut_at() -> Callable[[ScriptedBackend, int], ScriptedBackend]:
    """A raw backend that answers like `inner` but raises KeyboardInterrupt (a BaseException, the
    way Ctrl-C arrives) on its k-th call, counting from 0; the calls before it succeeded."""

    def make(inner: ScriptedBackend, k: int) -> ScriptedBackend:
        def script(call: Call) -> str:
            if len(outer.calls) - 1 == k:
                raise KeyboardInterrupt
            return inner.complete(call).text

        outer = ScriptedBackend(script)
        return outer

    return make


@pytest.fixture
def override() -> Callable[..., ScriptedBackend]:
    """A raw backend that answers like `inner` except for the roles given as keyword arguments,
    each a function of the Call returning the reply text or an Exception to raise."""

    def make(inner: ScriptedBackend, **roles: Callable[[Call], "str | Exception"]):
        def script(call: Call) -> "str | Exception":
            if call.role in roles:
                return roles[call.role](call)
            return inner.complete(call).text

        return ScriptedBackend(script)

    return make


@pytest.fixture
def files_under() -> Callable[[Path], list[Path]]:
    """Every file below a folder (empty list when it does not exist)."""

    def walk(root: Path) -> list[Path]:
        return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []

    return walk
