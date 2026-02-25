"""Placeholder test suite for Task 0. Real tests land in Task 1+."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fda_predictor  # noqa: F401
from fda_predictor.utils.paths import HUB_CACHE


def test_package_imports() -> None:
    import fda_predictor as pkg

    assert pkg.__version__ == "0.1.0"


def test_hub_cache_is_configured() -> None:
    assert str(HUB_CACHE).lower().startswith("d:\\models")
