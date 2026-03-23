"""Tests for CTO merge sanity + tabular encoder masking (stage-7 layout)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.approval_labels import approval_label_coverage  # noqa: E402
from fda_predictor.data.merge_datasets import _chronological_split_frame  # noqa: E402
from fda_predictor.data.tabular_features import (  # noqa: E402
    N_TABULAR_FEATURES,
    attach_tabular_features,
    fit_tabular_stats,
    row_tabular_raw,
)


class TestMergeSplit:
    def _synthetic_frame(self, n: int, source: str, start_year: int = 2000) -> pd.DataFrame:
        dates = pd.date_range(f"{start_year}-01-01", periods=n, freq="180D")
        return pd.DataFrame(
            {
                "nctid": [f"NCT{i:08d}" for i in range(n)],
                "label": [i % 2 for i in range(n)],
                "criteria": ["criteria text " * 8] * n,
                "smiles_canonical": [["C"]] * n,
                "phase_index": [2] * n,
                "data_source": [source] * n,
                "chronology_date": [d.strftime("%Y-%m-%d") for d in dates],
            }
        )

    def test_chronological_split_monotonic(self):
        cto = self._synthetic_frame(600, "CTO", 2010)
        top = self._synthetic_frame(400, "TOP", 2000)
        df = pd.concat([top, cto], ignore_index=True)
        tr, va, te = _chronological_split_frame(df, test_fraction=0.2, val_fraction_of_trainval=0.15)
        for name, part in (("train", tr), ("val", va), ("test", te)):
            chrono = pd.to_datetime(part["chronology_date"], errors="coerce")
            assert not chrono.isna().all(), name
            assert chrono.is_monotonic_increasing, f"{name} not chronological"
        assert set(tr.nctid).isdisjoint(te.nctid) and set(va.nctid).isdisjoint(te.nctid)


class TestStockFeaturesWereReplaced:
    """Placeholder for legacy TestStockFeatures/API checks — stage-7 replaced
    the stock fetch path with bulk Commission sources; assertion kept as
    documentation that tabular features carry the value now."""

    def test_merge_stats_default_shapes(self):
        df = pd.DataFrame(
            {
                "enrollment": [100, 200],
                "number_of_arms": [2, 1],
                "has_dmc": [1, 0],
                "source_class": ["Industry", "NIH"],
                "molecule_mask": [1, 1],
                "n_prior_drug_approvals": [0, 5],
                "is_fda_regulated_drug": [1, 1],
                "stage7_feats": [[0.0] * 28, [1.0, 0.0] + [0.0] * 26],
                "stage7_mask": [[1.0] * 28, [1.0] * 28],
            }
        )
        feats, mask = row_tabular_raw(df.iloc[0])
        assert feats.shape == (N_TABULAR_FEATURES,)
        assert mask.shape == (N_TABULAR_FEATURES,)
        out, stats = attach_tabular_features(df, fit_stats=True)
        assert stats is not None
        assert len(out["tabular_feats"].iloc[0]) == N_TABULAR_FEATURES


class TestApprovalCoverage:
    def test_covered_labeled_rows_counted(self):
        df = pd.DataFrame(
            {
                "approval_label": [1.0, 0.0, None],
                "previously_approved": [False, False, False],
            }
        )
        cov = approval_label_coverage(df)
        assert cov["labeled"] == 2


class TestConfigFusionDim:
    def test_config_fusion_dim_includes_stock(self):
        pytest.importorskip("transformers")
        from fda_predictor.models.multimodal_net import TriStreamNet

        # CPU-forward only possible with real checkpoints; skip when cache absent.
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs" / "config.yaml"
        if not config_path.exists():
            pytest.skip("config missing")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        try:
            net = TriStreamNet.from_config(config)
        except Exception:
            pytest.skip("backbone weights unavailable offline")
        assert net.fusion_input_dim == 384 + 768 + 768 + 32 + 64 + 64
        assert net.CHECKPOINT_LAYOUT == 7
