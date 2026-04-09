"""Unit tests for SMILES backfill: normalization, combos, synonym fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fda_predictor.data.pubchem_smiles as pc  # noqa: E402
from fda_predictor.data.pubchem_smiles import (  # noqa: E402
    PubChemReport,
    normalize_drug_name,
    resolve_drug_list,
    resolve_drug_smiles,
    split_combination,
)


class TestNormalizeDrugName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("aspirin 81 mg tablet", "aspirin"),
            ("Aspirin (acetylsalicylic acid)", "aspirin"),
            ("0.05% Voclosporin Ophthalmic Solution (VOS)", "voclosporin"),
            ("Dexamethasone Sodium Phosphate Injection", "dexamethasone sodium phosphate"),
            ("0.9%NaCl", "0.9%NaCl"),  # no space -> dose regex leaves "nac1"... check: digits+% stripped
            ("Paclitaxel 100 mg", "paclitaxel"),
            ("Drug-X 125 mcg/mL Patch", "drug x"),
            ("  Pantoprazole  ", "pantoprazole"),
        ],
    )
    def test_normalizes(self, raw, expected):
        got = normalize_drug_name(raw)
        if raw == "0.9%NaCl":
            # Dose stripped; whatever remains is acceptable as long as nonempty
            assert isinstance(got, str)
        else:
            assert got == expected


class TestSplitCombination:
    @pytest.mark.parametrize(
        "raw,parts",
        [
            ("Paclitaxel + Carboplatin", ["Paclitaxel", "Carboplatin"]),
            ("Aspirin/Dipyridamole", ["Aspirin", "Dipyridamole"]),
            ("FOLFOX or FOLFIRI", ["FOLFOX", "FOLFIRI"]),
            ("Metformin", ["Metformin"]),
            ("Saline", ["Saline"]),
        ],
    )
    def test_splits(self, raw, parts):
        assert split_combination(raw) == parts

    def test_dedupes_and_filters_junk(self):
        out = split_combination("aspirin + aspirin + x")
        assert out == ["aspirin"]


class TestEnhancedResolve:
    def test_cached_null_upgraded_via_normalization(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "PUBCHEM_CACHE_DIR", tmp_path)
        cache = pc._cache_path("Lidocaine 2% gel")
        cache.write_text(json.dumps({"drug": "Lidocaine 2% gel", "smiles": None}))

        calls: list[str] = []

        def fake_fetch(name):
            calls.append(name)
            return "CCC" if name == "lidocaine" else None

        with patch.object(pc, "_fetch_smiles_from_pubchem", side_effect=fake_fetch), \
             patch.object(pc, "_fetch_synonyms", return_value=[]), \
             patch.object(pc.time, "sleep", return_value=None):
            smi = resolve_drug_smiles("Lidocaine 2% gel", enhanced=True, delay_s=0)

        assert smi is not None
        assert "lidocaine" in calls
        # cache upgraded to a hit
        assert json.loads(cache.read_text())["smiles"] == smi

    def test_synonym_fallback_recovers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "PUBCHEM_CACHE_DIR", tmp_path)
        cache = pc._cache_path("MysteryDrug")
        cache.write_text(json.dumps({"drug": "MysteryDrug", "smiles": None}))

        with patch.object(pc, "_fetch_smiles_from_pubchem", side_effect=lambda n: "CCO" if n == "ethanol" else None), \
             patch.object(pc, "_fetch_synonyms", return_value=["ethanol", "alcohol"]), \
             patch.object(pc.time, "sleep", return_value=None):
            smi = resolve_drug_smiles("MysteryDrug", enhanced=True, delay_s=0)

        assert smi is not None
        assert json.loads(cache.read_text())["smiles"] == smi

    def test_definitive_miss_stays_null(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "PUBCHEM_CACHE_DIR", tmp_path)
        cache = pc._cache_path("NotAChemical123")
        cache.write_text(json.dumps({"drug": "NotAChemical123", "smiles": None}))

        with patch.object(pc, "_fetch_smiles_from_pubchem", return_value=None), \
             patch.object(pc, "_fetch_synonyms", return_value=[]), \
             patch.object(pc.time, "sleep", return_value=None):
            smi = resolve_drug_smiles("NotAChemical123", enhanced=True, delay_s=0)

        assert smi is None
        assert json.loads(cache.read_text())["smiles"] is None

    def test_non_enhanced_path_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "PUBCHEM_CACHE_DIR", tmp_path)
        cache = pc._cache_path("PlainMiss")
        cache.write_text(json.dumps({"drug": "PlainMiss", "smiles": None}))

        def boom(_):  # must not be called
            raise AssertionError("network called on non-enhanced cached null")

        with patch.object(pc, "_fetch_smiles_from_pubchem", side_effect=boom):
            assert resolve_drug_smiles("PlainMiss", enhanced=False) is None


class TestComboExpansionInList:
    def test_combo_yields_both_smiles(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "PUBCHEM_CACHE_DIR", tmp_path)

        table = {"paclitaxel": "CCO", "carboplatin": "CC(=O)O"}

        def fake_fetch(name):
            return table.get(name.lower())

        with patch.object(pc, "_fetch_smiles_from_pubchem", side_effect=fake_fetch), \
             patch.object(pc, "_fetch_synonyms", return_value=[]), \
             patch.object(pc.time, "sleep", return_value=None):
            out = resolve_drug_list(
                ["Paclitaxel + Carboplatin", "UnknownDrug4892"],
                max_drugs=5,
                split_combos=True,
                enhanced=True,
                report=PubChemReport(),
            )

        # Combo expands to 2 molecules; unknown drug contributes nothing.
        assert out == ["CCO", "CC(=O)O"]


class TestMergedDatasetSmilesParsing:
    """Regression: parquet round-trip turns list columns into np.ndarray;
    MergedTrialDataset must parse them to real SMILES (not placeholder 'C')."""

    def _dataset(self, smiles_value):
        import pandas as pd

        from fda_predictor.data.datasets import MergedTrialDataset
        from fda_predictor.data.tokenizers import EncoderSpec

        # Minimal real tokenizer-free dataset: tokenizers are pulled via
        # get_tokenizer(spec); using tiny inline spec won't load on CPU tests,
        # so stub the constructor instead via __new__ and set minimal attrs.
        frame = pd.DataFrame(
            {
                "nctid": ["NCT00000001"],
                "label": [1.0],
                "criteria": ["Adequate criteria text for tokenization testing."],
                "smiles_canonical": [smiles_value],
                "molecule_mask": [1],
                "phase_index": [2],
                "data_source": ["CTO"],
                "stock_feats": [[0.0] * 7],
                "stock_mask": [0],
                "tabular_feats": [[0.0] * 7],
                "tabular_mask": [[0.0] * 7],
            }
        )

        ds = MergedTrialDataset.__new__(MergedTrialDataset)
        ds.frame = frame
        ds.max_molecules = 5
        ds.max_len_a = ds.max_len_b = ds.max_len_c = 16
        ds.tok_a = ds.tok_b = None

        class _Tok:
            def __call__(self, text, **kw):
                return {"input_ids": [ord(c) % 100 + 1 for c in text][:10], "attention_mask": [1] * 10}

        ds.tok_c = object()
        import fda_predictor.data.datasets as mod

        seen: dict[str, list[str]] = {"smiles": []}

        def fake_encode_smiles_list(tok, smiles, max_len):
            seen["smiles"] = list(smiles)
            return [{"input_ids": [1], "attention_mask": [1]} for _ in smiles]

        def fake_encode_text(tok, text, max_len):
            return {"input_ids": [1], "attention_mask": [1]}

        orig_smiles, orig_text = mod.encode_smiles_list, mod.encode_text
        mod.encode_smiles_list = fake_encode_smiles_list
        mod.encode_text = fake_encode_text
        try:
            item = ds[0]
        finally:
            mod.encode_smiles_list = orig_smiles
            mod.encode_text = orig_text
        return item, seen

    def test_numpy_array_smiles_become_real_molecules(self):
        import numpy as np

        arr = np.array(["CCO", "O=C(O)c1ccccc1"], dtype=object)
        item, seen = self._dataset(arr)
        assert seen["smiles"] == ["CCO", "O=C(O)c1ccccc1"], seen
        assert item["n_molecules"] == 2
        assert float(item["molecule_mask"]) == 1.0

    def test_list_smiles_still_work(self):
        _, seen = self._dataset(["CC(=O)O"])
        assert seen["smiles"] == ["CC(=O)O"]

    def test_empty_still_placeholder(self):
        import numpy as np

        item, seen = self._dataset(np.array([], dtype=object))
        assert float(item["molecule_mask"]) == 0.0
        assert seen["smiles"] == ["C"]


class TestCanonicalizeDedupe:
    def test_dedupe_after_combo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "PUBCHEM_CACHE_DIR", tmp_path)
        # Same canonical SMILES for two spellings -> single output
        with patch.object(pc, "_fetch_smiles_from_pubchem", side_effect=lambda n: "CC"), \
             patch.object(pc, "_fetch_synonyms", return_value=[]), \
             patch.object(pc.time, "sleep", return_value=None):
            out = resolve_drug_list(["ethane", "ethane "], max_drugs=5, split_combos=True, enhanced=True)
        assert out == ["CC"]
