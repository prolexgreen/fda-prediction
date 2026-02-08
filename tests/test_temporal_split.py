"""Unit tests for the strict temporal split guard.

The guard's contract: max(train dates) <= min(val dates) <= max(val dates)
<= min(test dates). These tests use synthetic date frames so they run
without the TDC download present.
"""

from __future__ import annotations

from datetime import date

import pytest


def _split(dates):
    """Reference implementation of the guard contract used by the loader."""
    train, val, test = dates
    assert max(train) <= min(val), "lookahead: train extends past val"
    assert max(val) <= min(test), "lookahead: val extends past test"
    return True


def test_valid_chronology_passes() -> None:
    d = lambda y, m: date(y, m, 15)
    assert _split(([d(2015, 1), d(2016, 2)], [d(2017, 1), d(2017, 6)], [d(2018, 1), d(2019, 3)]))


def test_train_past_val_fails() -> None:
    d = lambda y, m: date(y, m, 15)
    with pytest.raises(AssertionError):
        _split(([d(2017, 5)], [d(2017, 1)], [d(2018, 1)]))


def test_val_past_test_fails() -> None:
    d = lambda y, m: date(y, m, 15)
    with pytest.raises(AssertionError):
        _split(([d(2015, 1)], [d(2019, 1)], [d(2018, 1)]))


def test_boundary_equality_allowed() -> None:
    boundary = date(2017, 6, 30)
    assert _split(([boundary], [boundary], [date(2017, 7, 1)]))
