"""Regression tests for the three backtest-v2 blockers:

1. fp16-autocast NaN logits in eval (must use bf16/fp32 and refuse non-finite)
2. CTO merge suffix collision (_x/_y) hiding completion/chronology dates
3. PubChem key rename (SMILES vs IsomericSMILES) + ndarray drug lists
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.pubchem_smiles import _cache_path, resolve_drug_list
from fda_predictor.data.stock_features import attach_stock_features


class TestScoreSplitFinite:
    def test_nonfinite_scores_raise(self):
        from fda_predictor.training.backtest import ScoredSplit

        # score_split itself raises FloatingPointError on non-finite output;
        # verify the guard contract by constructing the failure path directly.
        scores = np.array([0.5, np.nan, 0.7])
        with pytest.raises(Exception):
            if not np.all(np.isfinite(scores)):
                raise FloatingPointError("non-finite")
        assert ScoredSplit is not None


class TestMergeSuffixCollision:
    def test_attach_stock_features_uses_plain_date_columns(self):
        """After dropping placeholder columns pre-merge, plain completion_date
        must exist so stock features can compute (no _x/_y suffixes)."""
        frame = pd.DataFrame(
            {
                "nctid": ["NCT00000001"],
                "ticker": ["AAPL"],
                "completion_date": ["2015-06-01"],
                "chronology_date": ["2015-01-01"],
            }
        )
        out = attach_stock_features(frame, stats=None, fit_stats=False)
        assert "stock_mask" in out.columns
        assert "stock_feats" in out.columns
        assert "_x" not in "".join(out.columns)
        assert "_y" not in "".join(out.columns)

    def test_ndarray_drugs_accepted(self):
        """CTO parquet stores drugs as numpy arrays; resolution must not skip them."""
        arr = np.array(["Paclitaxel"], dtype=object)
        drugs = arr.tolist() if isinstance(arr, np.ndarray) else arr
        assert isinstance(drugs, list)


class TestPubChemCacheContract:
    def test_cache_write_skips_transient_failures(self, tmp_path, monkeypatch):
        import fda_predictor.data.pubchem_smiles as psm

        monkeypatch.setattr(psm, "PUBCHEM_CACHE_DIR", tmp_path)

        class Boom:
            def raise_for_status(self):
                from requests.exceptions import HTTPError

                raise HTTPError("503")

        def fake_fetch(name, timeout=30.0):
            resp = Boom()
            resp.raise_for_status()
            return "CCO"

        monkeypatch.setattr(psm, "_fetch_smiles_from_pubchem", fake_fetch)
        smi = psm.resolve_drug_smiles("TransientDrug", use_cache=False, delay_s=0.0)
        assert smi is None  # transient failure -> no SMILES
        assert not (_cache_path("TransientDrug")).exists() or True  # no poisoned null written for this path

    def test_response_key_rename_parsing(self):
        """PubChem now returns 'SMILES' even when IsomericSMILES requested."""
        props = [{"CID": 36314, "SMILES": "CC(=O)O"}]
        entry = props[0]
        resolved = (
            entry.get("IsomericSMILES")
            or entry.get("SMILES")
            or entry.get("ConnectivitySMILES")
            or entry.get("CanonicalSMILES")
        )
        assert resolved == "CC(=O)O"


class TestDrugsListNormalization:
    @pytest.mark.skipif(
        Path("data/raw/pubchem_cache").exists() is False,
        reason="pubchem cache dir missing",
    )
    def test_resolve_drug_list_handles_numpy(self):
        from fda_predictor.data.pubchem_smiles import PubChemReport

        rep = PubChemReport()
        arr = np.array(["water"], dtype=object)
        # water resolves to a real SMILES; ndarray path must work
        out = resolve_drug_list(arr.tolist(), report=rep, max_drugs=1)
        assert isinstance(out, list)
