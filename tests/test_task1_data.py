"""Task 1 unit tests: split guards, preprocessing, tokenizer independence,
collate shapes, and invalid-SMILES accounting. CPU-only; the TDC-dependent
integration test is skipped automatically if data/raw has no cached download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data import datasets as ds_mod  # noqa: E402
from fda_predictor.data.preprocessing import (  # noqa: E402
    CanonicalizationReport,
    canonicalize_molecule_list,
    canonicalize_smiles,
    clean_criteria_text,
)
from fda_predictor.data.top_dataset import assert_temporal_order, nct_number  # noqa: E402


def _frame(nctids, labels):
    return pd.DataFrame(
        {
            "nctid": list(nctids),
            "label": list(labels),
            "smiles_canonical": [["CC(=O)OC1=CC=CC=C1C(=O)O"]] * len(nctids),
        }
    )


class TestTemporalGuard:
    def test_valid_chronology_passes(self):
        train = _frame(["NCT00000001", "NCT00000100"], [1, 0])
        val = _frame(["NCT00000200", "NCT00000300"], [1, 1])
        test = _frame(["NCT00010000"], [0])
        assert_temporal_order((train, val, test))

    def test_train_past_val_fails(self):
        train = _frame(["NCT00050000"], [1])
        val = _frame(["NCT00020000"], [0])
        test = _frame(["NCT00090000"], [1])
        with pytest.raises(AssertionError, match="LOOKAHEAD BIAS"):
            assert_temporal_order((train, val, test))

    def test_val_past_test_fails(self):
        train = _frame(["NCT00000001"], [1])
        val = _frame(["NCT00090000"], [0])
        test = _frame(["NCT00080000"], [1])
        with pytest.raises(AssertionError, match="LOOKAHEAD BIAS"):
            assert_temporal_order((train, val, test))

    def test_disjointness_enforced(self):
        train = _frame(["NCT00000001"], [1])
        val = _frame(["NCT00000001", "NCT00000002"], [0, 1])  # duplicate id across splits
        test = _frame(["NCT00090000"], [1])
        with pytest.raises(AssertionError, match="leakage"):
            assert_temporal_order((train, val, test))

    def test_nct_number_parsing(self):
        assert nct_number("NCT04212345") == 4212345
        with pytest.raises(ValueError):
            nct_number("NCT")

    def test_boundary_equality_allowed(self):
        train = _frame(["NCT00001000"], [1])
        val = _frame(["NCT00010000"], [1])
        test = _frame(["NCT00010001"], [0])
        # strict ordering holds: train.max=1000 < val.min=10000 < test.min=10001
        assert_temporal_order((train, val, test))


class TestPreprocessing:
    def test_canonicalize_valid_and_invalid(self):
        report = CanonicalizationReport()
        smiles_in = ["CC(=O)C", "not_a_smiles", "", None, "CC(=O)C"]
        out = canonicalize_molecule_list(smiles_in, report)
        expected = canonicalize_smiles("CC(=O)C")
        assert out == [expected]
        assert report.n_input == 5
        assert report.n_invalid == 3  # unparseable + empty + None
        assert report.n_duplicates_removed == 1
        assert abs(report.validity_rate - 2 / 5) < 1e-9

    def test_canonical_smiles_is_canonical_form(self):
        assert canonicalize_smiles("O=C(C)Oc1ccccc1C(=O)O") == "CC(=O)Oc1ccccc1C(=O)O"

    def test_clean_criteria_markdown_artifacts(self):
        raw = "Inclusion:\n- age \\> 18 years\n- dose \\* 2 mg"
        cleaned = clean_criteria_text(raw)
        assert "\\>" not in cleaned and "\\*" not in cleaned
        assert "> 18 years" in cleaned and "* 2 mg" in cleaned
        assert "\n" in cleaned  # real newlines preserved, blanks collapsed

    def test_clean_criteria_whitespace(self):
        cleaned = clean_criteria_text("a   b\t\tc\n\n\n\nd")
        assert "a b c" in cleaned
        assert "\n\n\n" not in cleaned

    def test_clean_criteria_handles_non_string(self):
        assert clean_criteria_text(None) == ""
        assert clean_criteria_text(float("nan")) == ""


@pytest.mark.integration
class TestRealDataAndTokenizers:
    @pytest.fixture(scope="class")
    def real_splits(self):
        try:
            return load_top_splits_safe()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"TDC TOP not available locally yet: {e!r}")

    def test_real_split_guard(self, real_splits):
        assert_temporal_order((real_splits.train, real_splits.val, real_splits.test))

    def test_tokenizer_independence(self, real_splits):
        batch = real_splits.train.head(2)
        enc = _encode_two_streams(batch.iloc[0]["smiles_canonical"])
        ids_a, ids_b = enc
        assert len(ids_a[0]) > 2 and len(ids_b[0]) > 2
        assert ids_a != ids_b, "ChemBERTa and MoLFormer must tokenize differently"

    def test_collate_shapes(self, real_splits):
        from torch.utils.data import DataLoader

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text()
        )
        specs = __import__(
            "fda_predictor.data.tokenizers", fromlist=["specs_from_config"]
        ).specs_from_config(config)

        ds = ds_mod.TOPTrialDataset(real_splits.train.head(16), **_spec_kwargs(specs))
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ds_mod.build_collate_fn(ds))
        b = next(iter(loader))

        n_flat = b["mol_input_a"]["input_ids"].shape[0]
        assert b["mol_input_b"]["input_ids"].shape[0] == n_flat
        assert b["group_index"].numel() == n_flat
        assert b["group_index"].max().item() == len(b["label"]) - 1
        assert b["crit_input"]["input_ids"].shape[0] == len(b["label"])
        assert set(b["group_index"].tolist()) <= set(range(len(b["label"])))


def load_top_splits_safe():
    from fda_predictor.data.top_dataset import load_top_splits

    return load_top_splits()


def _encode_two_streams(smiles_list):
    from fda_predictor.data.tokenizers import EncoderSpec, encode_smiles_list, get_tokenizer

    spec_a = EncoderSpec("chemberta", "DeepChem/ChemBERTa-77M-MTR",
                         "66b895cab8adebea0cb59a8effa66b2020f204ca", False, 128)
    spec_b = EncoderSpec("molformer", "ibm-research/MoLFormer-XL-both-10pct",
                         "361063d0ad524ef77cf39b08469f6be770dc550f", True, 128)
    ta = get_tokenizer(spec_a)
    tb = get_tokenizer(spec_b)
    ea = encode_smiles_list(ta, smiles_list[:1], 128)
    eb = encode_smiles_list(tb, smiles_list[:1], 128)
    return [m["input_ids"] for m in ea], [m["input_ids"] for m in eb]


def _spec_kwargs(specs):
    return dict(chemberta_spec=specs["chemberta"], molformer_spec=specs["molformer"],
                clinicalbert_spec=specs["clinicalbert"])
