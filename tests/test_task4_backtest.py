"""Task 4 backtest-logic tests on synthetic tensors: threshold transfer,
phase-slice filtering, and baseline metric floors. CPU-only, no data needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.training.backtest import (  # noqa: E402
    ScoredSplit,
    evaluate_with_threshold_transfer,
    majority_baseline,
    per_phase_breakdown,
)
from fda_predictor.training.metrics import auprc, f1_at_threshold, tune_threshold_on_val


def _split(y, s, phase):
    return ScoredSplit(
        nctids=[f"NCT{i:08d}" for i in range(len(y))],
        y_true=np.asarray(y, dtype=float),
        scores=np.asarray(s, dtype=float),
        phase_index=np.asarray(phase, dtype=int),
    )


class TestThresholdTransfer:
    def test_val_tuned_threshold_applied_frozen_to_test(self):
        val = _split(
            y=[0, 0, 0, 0, 1, 1, 1, 1],
            s=[0.05, 0.15, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95],
            phase=[2] * 8,
        )
        test = _split(
            y=[1, 0, 1, 0],
            s=[0.75, 0.25, 0.85, 0.35],
            phase=[2] * 4,
        )
        proto = evaluate_with_threshold_transfer(val, test)
        thr = proto["threshold"]
        # any cut in (0.3, 0.7) separates val perfectly; the tuner returns
        # its LOWEST such candidate (dense-linspace point just above 0.3)
        assert 0.3 < thr < 0.7
        assert proto["val_report"]["f1_at_threshold"] == pytest.approx(1.0)
        # AUPRC is threshold-free -> perfect ranking transfers fully
        assert proto["test_report"]["auprc"] == pytest.approx(1.0)
        # frozen low cut clips the test negative at 0.35: tp=2 fp=1 fn=0 -> F1=0.8
        assert proto["test_report"]["f1_at_threshold"] == pytest.approx(0.8)

    def test_transfer_can_degrade_on_shifted_test(self):
        """If test scores shift downward, frozen threshold must lose F1 --
        proving the threshold really transfers rather than re-tunes."""
        val = _split([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [2] * 4)
        test = _split([1, 1, 0, 0], [0.45, 0.55, 0.6, 0.65], [2] * 4)  # shifted band
        proto = evaluate_with_threshold_transfer(val, test)
        best_possible = tune_threshold_on_val(test.y_true, test.scores)[1]
        assert proto["test_report"]["f1_at_threshold"] <= best_possible + 1e-12

    def test_grid_override_respected(self):
        val = _split([0, 1], [0.2, 0.8], [2, 2])
        test = _split([1, 0], [0.7, 0.3], [2, 2])
        proto = evaluate_with_threshold_transfer(val, test, grid=[0.5])
        assert proto["threshold"] == 0.5


class TestSlicing:
    def test_phase_slice_filtering(self):
        sp = _split(
            y=[1, 0, 1, 1, 0],
            s=[0.9, 0.8, 0.7, 0.6, 0.5],
            phase=[2, 2, 4, 1, 4],
        )
        p3 = sp.slice_mask(2)
        assert p3.tolist() == [True, True, False, False, False]
        unk = sp.slice_mask(4)
        assert unk.sum() == 2
        allm = sp.slice_mask(None)
        assert allm.all()

    def test_per_phase_breakdown_keys_and_counts(self):
        sp = _split(
            y=[1, 0, 1, 1],
            s=[0.9, 0.1, 0.8, 0.7],
            phase=[2, 2, 4, 4],
        )
        bd = per_phase_breakdown(sp, threshold=0.5)
        assert set(bd.keys()) == {"III", "UNK(combo/missing)"}
        assert bd["III"]["n_trials"] == 2
        assert "roc_auc" in bd["III"]

    def test_single_class_slice_drops_roc_auc(self):
        sp = _split(y=[1, 1, 1], s=[0.9, 0.8, 0.1], phase=[2, 2, 2])  # all positive
        bd = per_phase_breakdown(sp, threshold=0.5)
        assert "roc_auc" not in bd["III"]
        assert bd["III"]["n_trials"] == 3


class TestBaselineFloors:
    def test_majority_auprc_equals_positive_fraction(self):
        y = np.array([1, 1, 1, 0, 0])  # pos_frac 0.6
        sp = _split(y, np.full(5, 0.42), [2] * 5)
        mb = majority_baseline(train_pos_frac=0.42, split=sp)
        assert mb["auprc"] == pytest.approx(0.6)
        assert mb["auprc"] == pytest.approx(auprc(y, np.full(5, 0.42)))

    def test_constant_scores_cannot_beat_informative_model(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 300)
        good_scores = y * 0.6 + rng.random(300) * 0.4
        const = np.full(300, float(y.mean()))
        assert auprc(y, const) < auprc(y, good_scores)

    def test_f1_floor_of_always_positive_rule(self):
        y = [1, 0, 0, 0]
        const = np.full(4, 0.75)  # crosses the default 0.5 cut
        # always-positive: P=0.25 R=1 -> F1=0.4
        assert f1_at_threshold(y, const, 0.5) == pytest.approx(0.4)


class TestSyntheticEndToEnd:
    def test_full_protocol_on_synthetic(self):
        rng = np.random.default_rng(7)
        n = 200
        y = (rng.random(n) < 0.5).astype(float)
        noise = rng.normal(0, 0.1, n)
        scores = np.clip(y - noise, 0, 1)
        phase = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.05, 0.1, 0.75, 0.02, 0.08])

        val = _split(y[:100], scores[:100], phase[:100])
        test = _split(y[100:], scores[100:], phase[100:])
        proto = evaluate_with_threshold_transfer(val, test)

        assert proto["val_report"]["auprc"] > 0.9
        assert proto["test_report"]["auprc"] > 0.9
        # model must beat its own slice prevalence floor
        for key, rep in per_phase_breakdown(test, proto["threshold"]).items():
            mask = test.slice_mask({"I": 0, "II": 1, "III": 2, "IV": 3}.get(key, 4))
            if mask.sum() > 10:
                assert rep["auprc"] >= float(test.y_true[mask].mean()) - 0.01
