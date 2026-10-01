"""Proves the check loop works and that tests import this checkout's code, not another one's."""

from pathlib import Path

import autoimprover


def test_version_is_set():
    assert autoimprover.__version__


def test_imports_this_checkout_src():
    root = Path(__file__).resolve().parents[1]
    assert autoimprover.__file__ is not None
    assert Path(autoimprover.__file__).resolve().is_relative_to(root / "src")


def test_pinned_gepa_is_installed():
    from importlib.metadata import version

    assert version("gepa") == "0.1.4"
