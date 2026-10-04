"""Command-line entry point. Stub from the skeleton commit: work package WP6 replaces the body and
keeps this signature (docs/ARCHITECTURE.md). `backend` replaces the raw model layer and `now` the
monotonic clock, so tests inject a fake model and a fake clock (SPEC R17, R20)."""

from collections.abc import Callable, Sequence

from autoimprover.types import Backend


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: Backend | None = None,
    now: Callable[[], float] | None = None,
) -> int:
    """Run the tool and return the exit code of SPEC R2."""
    raise NotImplementedError("WP6")
