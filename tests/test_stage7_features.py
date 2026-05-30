"""Stage-7 feature tests: rdkit descriptors, modality, sponsor stats."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.chem_features import (  # noqa: E402
    ChemblMechanismIndex,
    MechRecord,
    aggregate_descriptors,
    classify_modality,
    smiles_descriptors,
    trial_mechanism_features,
)
from fda_predictor.data.sponsor_features import (  # noqa: E402
    normalize_sponsor,
    sponsor_prior_stats,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
METFORMIN = "CN(C)C(=N)NC(=N)N"
BEVACIZUMAB_MAB = ""  # no SMILES (biologic) - tested via modality only


class TestDescriptors:
    def test_aspirin_values(self):
        d = smiles_descriptors(ASPIRIN)
        assert d is not None
        assert 179.0 <= d["mw"] <= 181.5
        assert d["hbd"] == 1.0  # COOH
        assert d["hba"] == 3.0  # ester C=O, ether O, carboxyl COOH contribute 3
        assert 60.0 <= d["tpsa"] <= 66.0
        assert 0.40 <= d["arom_frac"] <= 0.50

    def test_invalid_smiles_none(self):
        assert smiles_descriptors("not a smiles ((") is None

    def test_aggregate_single(self):
        feats, mask = aggregate_descriptors([ASPIRIN])
        assert mask.sum() == 8
        assert 179.0 <= feats[0] <= 181.5

    def test_aggregate_empty_masked(self):
        feats, mask = aggregate_descriptors(["this is not", "a molecule"])
        assert feats.sum() == 0.0
        assert mask.sum() == 0.0

    def test_aggregate_multi_mean(self):
        feats, mask = aggregate_descriptors([ASPIRIN, METFORMIN])
        mw1 = smiles_descriptors(ASPIRIN)["mw"]
        mw2 = smiles_descriptors(METFORMIN)["mw"]
        assert abs(feats[0] - (mw1 + mw2) / 2) < 1e-3


class TestModality:
    def test_mab(self):
        assert classify_modality(["bevacizumab"], False)[1] == 1.0

    def test_cell_gene(self):
        assert classify_modality(["tisagenlecleucel CAR-T"], False)[3] == 1.0

    def test_vaccine(self):
        assert classify_modality(["HPV vaccine"], False)[4] == 1.0

    def test_small_molecule_by_smiles(self):
        assert classify_modality(["imatinib"], True)[0] == 1.0

    def test_unknown_no_info(self):
        assert classify_modality(["XY-9901"], False)[5] == 1.0


class TestChemblIndex:
    def _mini_index(self) -> ChemblMechanismIndex:
        idx = ChemblMechanismIndex(db_path=None)
        # Imatinib -> BCR-ABL tyrosine kinase (enzyme bucket via keyword).
        rec = MechRecord(
            tid=1,
            target_type="SINGLE PROTEIN",
            pref_name="Tyrosine-protein kinase ABL1",
            action_type="INHIBITOR",
            max_phase=4,
        )
        idx._by_name = {"imatinib": [rec], "gleevec": [rec]}
        idx._competitors = {1: 7}
        return idx

    def test_lookup_and_buckets(self):
        idx = self._mini_index()
        feats, mask = trial_mechanism_features(idx, ["imatinib"])
        assert mask.all()
        assert feats[0] == 1.0  # mech_known
        assert feats[4] == 1.0  # kinase bucket
        assert feats[7] == 1.0  # inhibitor
        assert feats[11] == pytest.approx(np.log1p(7))

    def test_missing_drugs_masked(self):
        feats, mask = trial_mechanism_features(self._mini_index(), ["flubber"])
        assert not mask.any()
        assert feats.sum() == 0.0


class TestSponsorFeatures:
    def _fda_frame(self) -> pd.DataFrame:
        # Pfizer got approved 2010 + 2015; Novartis 2012 only.
        return pd.DataFrame(
            {
                "sponsor_name": ["Pfizer Inc.", "PFIZER", "Novartis Pharmaceuticals Corporation"],
                "submission_type": ["ORIG", "SUPPL", "ORIG"],
                "approval_date": pd.to_datetime(["2010-01-01", "2015-01-01", "2012-01-01"]),
            }
        )

    def test_normalize_sponsor_collapses_suffixes(self):
        a = normalize_sponsor("Pfizer Inc.")
        b = normalize_sponsor("PFIZER")
        c = normalize_sponsor("Pfizer Pharmaceuticals")
        assert a == b == "pfizer"
        assert c == "pfizer"
        assert normalize_sponsor(None) == ""

    def test_prior_counts_are_leakage_safe(self):
        fda = self._fda_frame()
        trials = pd.DataFrame(
            {
                "nctid": ["NCT00000001", "NCT00000002", "NCT00000003", "NCT00000004"],
                "sponsor": ["Pfizer", "Pfizer", "Novartis Pharmaceuticals", ""],
                "start_date": pd.to_datetime(["2011-01-01", "2016-01-01", "2011-01-01", "2011-01-01"]),
                "phase_index": [2, 2, 2, 2],
            }
        )
        prior_app, has_prior, prior_trials, mask = sponsor_prior_stats(trials, fda)
        assert mask.tolist() == [1.0, 1.0, 1.0, 0.0]
        # 2011 Pfizer: 1 approval prior (2010)
        assert prior_app[0] == pytest.approx(np.log1p(1))
        assert has_prior[0] == 1.0
        # 2016 Pfizer: 2 approvals prior (2010 + 2015)
        assert prior_app[1] == pytest.approx(np.log1p(2))
        # 2011 Novartis: 0 prior (2012 comes after)
        assert prior_app[2] == 0.0
        assert has_prior[2] == 0.0
