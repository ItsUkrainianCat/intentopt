"""Command-line entry point. Stub from the skeleton commit: work package WP6 replaces the body and
keeps this signature (docs/ARCHITECTURE.md). `backend` lets tests inject a fake (SPEC R20)."""

from collections.abc import Sequence

from autoimprover.types import Backend


def main(argv: Sequence[str] | None = None, *, backend: Backend | None = None) -> int:
    """Run the tool and return the exit code of SPEC R2."""
    raise NotImplementedError("WP6")
