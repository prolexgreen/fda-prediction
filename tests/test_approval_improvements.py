"""Unit tests for decision-cost thresholds, approval labels, tabular features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.approval_labels import (  # noqa: E402
    approval_label_coverage,
    earliest_orig_approval_date,
    label_trial_approval,
)
from fda_predictor.data.tabular_features import (  # noqa: E402
    N_TABULAR_FEATURES,
    attach_tabular_features,
    row_tabular_raw,
)
from fda_predictor.training.backtest import (  # noqa: E402
    ScoredSplit,
    dump_scored_split,
    evaluate_per_phase_threshold_transfer,
    load_scored_split,
)
from fda_predictor.training.metrics import (  # noqa: E402
    decision_cost,
    tune_threshold_cost,
)


def _split(y, s, phase, src=None):
    return ScoredSplit(
        nctids=[f"NCT{i:08d}" for i in range(len(y))],
        y_true=np.asarray(y, dtype=float),
        scores=np.asarray(s, dtype=float),
        phase_index=np.asarray(phase, dtype=int),
        data_source=np.asarray(src) if src is not None else None,
    )


class TestDecisionCost:
    def test_cost_prefers_higher_threshold_when_fp_cheap_fn_expensive_is_default(self):
        # Well-separated scores: F1 thr is mid-band; cost with cost_fn=2 still mid.
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        s = [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95]
        thr, cost = tune_threshold_cost(y, s, cost_fp=1.0, cost_fn=2.0)
        assert 0.35 < thr < 0.65
        assert cost == decision_cost(y, s, thr, cost_fp=1.0, cost_fn=2.0)

    def test_high_fp_cost_raises_threshold(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        s = np.array([0.1, 0.2, 0.4, 0.45, 0.55, 0.7, 0.8, 0.9])
        thr_bal, _ = tune_threshold_cost(y, s, cost_fp=1.0, cost_fn=1.0)
        thr_fp, _ = tune_threshold_cost(y, s, cost_fp=10.0, cost_fn=1.0)
        assert thr_fp >= thr_bal


class TestPerPhaseThresholds:
    def test_phase_specific_thresholds_transfer(self):
        # Phase II: low base rate -> higher thr; Phase III: high base rate -> lower thr
        rng = np.random.default_rng(0)
        n = 200
        phase = np.array([1] * 100 + [2] * 100)
        y = np.concatenate(
            [
                (rng.random(100) < 0.2).astype(float),
                (rng.random(100) < 0.7).astype(float),
            ]
        )
        scores = np.clip(y * 0.5 + rng.random(n) * 0.5, 0, 1)
        val = _split(y[:100], scores[:100], phase[:100])
        # rebuild so each phase appears in both
        val = _split(y, scores, phase)
        test = _split(y, scores + 0.01, phase)
        proto = evaluate_per_phase_threshold_transfer(val, test, objective="cost")
        assert "II" in proto["thresholds"] and "III" in proto["thresholds"]
        assert "test" in proto and "III" in proto["test"]


class TestScoreDump:
    def test_roundtrip(self, tmp_path):
        sp = _split([1, 0, 1], [0.9, 0.1, 0.8], [2, 2, 1], src=["TOP", "CTO", "TOP"])
        path = dump_scored_split(sp, tmp_path / "scores.csv")
        loaded = load_scored_split(path)
        assert loaded.nctids == sp.nctids
        assert np.allclose(loaded.scores, sp.scores)
        assert list(loaded.data_source) == ["TOP", "CTO", "TOP"]


class TestApprovalLabels:
    def _payload(self, date: str):
        return {
            "results": [
                {
                    "application_number": "NDA123456",
                    "openfda": {"brand_name": ["WonderDrug"]},
                    "submissions": [
                        {
                            "submission_type": "ORIG",
                            "submission_status": "APPROVED",
                            "submission_status_date": date,
                        }
                    ],
                }
            ]
        }

    def test_earliest_orig_parse(self):
        d = earliest_orig_approval_date(self._payload("20150115"))
        assert d is not None
        assert d.year == 2015

    def test_post_start_approval_is_positive(self):
        cache = {"wonderdrug": self._payload("20180101")}

        def fetch(name, search_field="active_ingredient", **kwargs):
            return cache.get(name.lower())

        lab = label_trial_approval(
            "NCT1",
            drugs=["WonderDrug"],
            start_date="2015-01-01",
            fetch_fn=fetch,
        )
        assert lab.approval_label == 1.0
        assert lab.previously_approved is False

    def test_pre_start_only_is_previously_approved(self):
        cache = {"olddrug": self._payload("20100101")}

        def fetch(name, search_field="active_ingredient", **kwargs):
            return cache.get(name.lower())

        lab = label_trial_approval(
            "NCT2",
            drugs=["OldDrug"],
            start_date="2015-01-01",
            fetch_fn=fetch,
        )
        assert lab.approval_label is None
        assert lab.previously_approved is True
        assert lab.n_prior_drug_approvals == 1

    def test_unresolved_is_unknown(self):
        def fetch(name, search_field="active_ingredient", **kwargs):
            return {"results": [], "not_found": True}

        lab = label_trial_approval("NCT3", drugs=["NoSuchDrugXYZ"], start_date="2015-01-01", fetch_fn=fetch)
        assert lab.approval_label is None
        assert lab.n_resolved == 0

    def test_coverage_stats(self):
        df = pd.DataFrame(
            {
                "approval_label": [1.0, 0.0, np.nan, 1.0],
                "previously_approved": [False, False, True, False],
                "phase_index": [2, 2, 2, 1],
            }
        )
        cov = approval_label_coverage(df)
        assert cov["labeled"] == 3
        assert cov["by_phase"]["III"]["labeled"] == 2


class TestCoverageGate:
    def test_gate_passes_on_fixture(self, tmp_path):
        import subprocess

        rows = []
        for i in range(596):
            rows.append(
                {
                    "nctid": f"NCT{i:08d}",
                    "drugs": ["pembrolizumab"],
                    "phase_index": 2,
                    "split": "train" if i < 496 else ("val" if i < 546 else "test"),
                    "label": 1,
                }
            )
        for drug in ("nivolumab", "macitentan", "erlotinib"):
            rows.append(
                {
                    "nctid": f"NCT9{drug[:5]}",
                    "drugs": [drug],
                    "phase_index": 2,
                    "split": "train",
                    "label": 1,
                }
            )
        merged = pd.DataFrame(rows)
        approval = [1.0 if i < 60 else 0.0 for i in range(len(merged))]
        for idx in range(len(merged) - 3, len(merged)):
            approval[idx] = 1.0
        labels = pd.DataFrame(
            {
                "nctid": merged["nctid"],
                "approval_label": approval,
                "previously_approved": [False] * len(merged),
            }
        )
        mp = tmp_path / "merged.parquet"
        lp = tmp_path / "labels.parquet"
        merged.to_parquet(mp, index=False)
        labels.to_parquet(lp, index=False)
        script = Path(__file__).resolve().parents[1] / "scripts" / "check_approval_coverage.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--merged",
                str(mp),
                "--labels",
                str(lp),
                "--out",
                str(tmp_path / "gate.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout


class TestTabularFeatures:
    def test_row_shape_and_molecule_flag(self):
        feats, mask = row_tabular_raw(
            {
                "enrollment": 100,
                "number_of_arms": 2,
                "has_dmc": True,
                "source_class": "INDUSTRY",
                "molecule_mask": 1,
                "n_prior_drug_approvals": 0,
                "is_fda_regulated_drug": 1,
            }
        )
        assert feats.shape == (N_TABULAR_FEATURES,)
        assert mask.shape == (N_TABULAR_FEATURES,)
        assert feats[4] == 1.0  # molecule_present

    def test_attach_roundtrip(self):
        df = pd.DataFrame(
            {
                "enrollment": [50, None],
                "number_of_arms": [2, 3],
                "has_dmc": [True, False],
                "source_class": ["INDUSTRY", "OTHER"],
                "molecule_mask": [1, 0],
                "n_prior_drug_approvals": [0, 1],
                "is_fda_regulated_drug": [1, None],
            }
        )
        out, stats = attach_tabular_features(df, fit_stats=True)
        assert stats is not None
        assert len(out["tabular_feats"].iloc[0]) == N_TABULAR_FEATURES
        assert len(out["tabular_mask"].iloc[0]) == N_TABULAR_FEATURES


class TestMultiTaskLoss:
    def _logits(self, n, fill=0.0):
        return torch.full((n, 1), fill)

    def test_zero_mask_degenerates_to_success_loss(self):
        from fda_predictor.training.losses import MultiTaskLoss, WeightedBCE

        torch.manual_seed(0)
        succ = torch.randn(8, 1)
        y = torch.randint(0, 2, (8, 1)).float()
        appr = torch.randn(8, 1)
        mask = torch.zeros(8)
        mt = MultiTaskLoss(success_w=1.5, approval_w=3.0)
        expected = 1.5 * WeightedBCE(pos_weight=1.0)(succ, y)
        actual = mt(succ, y, appr, torch.zeros(8), mask)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_masked_rows_excluded_from_approval_term(self):
        from fda_predictor.training.losses import MultiTaskLoss

        succ = self._logits(4)
        y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        # Labeled rows get logit 0 (BCE 0.693); masked rows get huge logits
        # that must not influence the approval term at all.
        appr = torch.tensor([[0.0], [-100.0], [0.0], [100.0]])
        y_app = torch.tensor([1.0, 0.0, 1.0, 0.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        mt = MultiTaskLoss(approval_w=1.0)
        loss = mt(succ, y, appr, y_app, mask)
        bce01 = float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.zeros(1, 1), torch.ones(1, 1)
        ))
        # success: 4 rows of BCE(0->y) = 0.693 each -> mean 0.693
        # approval: 2 labeled rows of BCE(0->1) = 0.693 each -> mean 0.693
        assert loss.item() == pytest.approx(2 * bce01, rel=1e-4)

        # Flipping labels/logits on masked rows must not change the loss.
        appr_flipped = torch.tensor([[0.0], [100.0], [0.0], [-100.0]])
        y_app_flipped = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss2 = mt(succ, y, appr_flipped, y_app_flipped, mask)
        assert loss2.item() == pytest.approx(loss.item(), abs=1e-6)

    def test_weighting_arithmetic(self):
        from fda_predictor.training.losses import MultiTaskLoss

        succ = self._logits(2)
        y = torch.tensor([[1.0], [0.0]])
        appr = self._logits(2)
        y_app = torch.tensor([1.0, 1.0])
        mask = torch.ones(2)
        mt = MultiTaskLoss(
            success_w=2.0,
            approval_w=0.5,
            success_pos_weight=3.0,
            approval_pos_weight=4.0,
        )
        e_succ = 2.0 * float(torch.nn.functional.binary_cross_entropy_with_logits(
            succ, y, pos_weight=torch.tensor(3.0)
        ))
        e_appr = 0.5 * float(torch.nn.functional.binary_cross_entropy_with_logits(
            appr, y_app.view(2, 1), pos_weight=torch.tensor(4.0)
        ))
        assert mt(succ, y, appr, y_app, mask).item() == pytest.approx(e_succ + e_appr, rel=1e-5)

    def test_build_loss_dual_mode(self):
        import yaml

        from fda_predictor.training.losses import MultiTaskLoss, build_loss

        config = {
            "training": {"loss": "weighted_bce"},
            "labels": {"target": "dual", "success_weight": 1.0, "approval_weight": 2.0},
        }
        crit, info = build_loss(
            config,
            train_labels=[1.0, 0.0, 1.0, 0.0],
            train_approval_labels=[1.0, -1.0, -1.0, 0.0],
            train_approval_mask=[1.0, 0.0, 0.0, 1.0],
        )
        assert isinstance(crit, MultiTaskLoss)
        assert info["loss_kind"] == "multitask"
        assert info["success_weight"] == 1.0 and info["approval_weight"] == 2.0
        assert info["n_approval_labeled"] == 2


class TestDualHeadModel:
    def _tiny_net(self):
        from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

        from fda_predictor.models.encoders import (
            MoleculeEncoder,
            ProtocolEncoder,
            StockEncoder,
            TabularEncoder,
        )
        from fda_predictor.models.multimodal_net import TriStreamNet

        cfg_r = RobertaConfig(
            vocab_size=120, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=132, pad_token_id=1,
        )
        cfg_b = BertConfig(
            vocab_size=120, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=512,
        )
        return TriStreamNet(
            chemberta=MoleculeEncoder(RobertaModel(cfg_r), 16),
            molformer=MoleculeEncoder(RobertaModel(cfg_r), 16),
            clinicalbert=ProtocolEncoder(BertModel(cfg_b), 16),
            stock_encoder=StockEncoder(n_features=7, out_dim=8),
            tabular_encoder=TabularEncoder(n_features=7, out_dim=8),
            fusion_hidden_dims=(16,),
            fusion_dropout=0.0,
            phase_emb_dim=4,
            stock_emb_dim=8,
            tabular_emb_dim=8,
            n_tabular_features=7,
            approval_hidden_dims=(8,),
        )

    def _batch(self, n=6):
        torch.manual_seed(0)
        ids_a = torch.randint(20, 100, (n * 2, 10))
        ids_b = torch.randint(20, 100, (n * 2, 10))
        crit_ids = torch.randint(5, 100, (n, 24))
        ones_a = torch.ones(n * 2, 10, dtype=torch.long)
        ones_b = torch.ones(n * 2, 10, dtype=torch.long)
        return {
            "label": torch.randint(0, 2, (n, 1)).float(),
            "mol_input_a": {"input_ids": ids_a, "attention_mask": ones_a},
            "mol_input_b": {"input_ids": ids_b, "attention_mask": ones_b},
            "group_index": torch.arange(n).repeat_interleave(2),
            "batch_size": n,
            "crit_input": {"input_ids": crit_ids, "attention_mask": torch.ones(n, 24, dtype=torch.long)},
            "phase_index": torch.full((n,), 2),
            "molecule_mask": torch.ones(n),
            "stock_feats": torch.zeros(n, 7),
            "stock_mask": torch.zeros(n),
            "tabular_feats": torch.zeros(n, 7),
            "tabular_mask": torch.zeros(n, 7),
            "approval_label": torch.tensor([1.0, -1.0, -1.0, 0.0, 1.0, -1.0]),
            "approval_mask": torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        }

    def test_layout_and_forward_shapes(self):
        from fda_predictor.models.multimodal_net import TriStreamNet

        assert TriStreamNet.CHECKPOINT_LAYOUT == 7
        net = self._tiny_net()
        b = self._batch()
        with torch.no_grad():
            s, a = net.forward_with_approval(
                mol_input_a=b["mol_input_a"],
                mol_input_b=b["mol_input_b"],
                group_index=b["group_index"],
                batch_size=b["batch_size"],
                crit_input=b["crit_input"],
                phase_index=b["phase_index"],
                stock_feats=b["stock_feats"],
                stock_mask=b["stock_mask"],
                molecule_mask=b["molecule_mask"],
                tabular_feats=b["tabular_feats"],
                tabular_mask=b["tabular_mask"],
            )
        assert s.shape == (6, 1)
        assert a is not None and a.shape == (6, 1)

    def test_forward_returns_success_only(self):
        net = self._tiny_net()
        b = self._batch()
        out = net(
            mol_input_a=b["mol_input_a"],
            mol_input_b=b["mol_input_b"],
            group_index=b["group_index"],
            batch_size=b["batch_size"],
            crit_input=b["crit_input"],
            phase_index=b["phase_index"],
            stock_feats=b["stock_feats"],
            stock_mask=b["stock_mask"],
            molecule_mask=b["molecule_mask"],
            tabular_feats=b["tabular_feats"],
            tabular_mask=b["tabular_mask"],
        )
        assert out.shape == (6, 1)


class TestScoredSplitApproval:
    def test_dump_load_roundtrip_with_approval(self, tmp_path):
        split = ScoredSplit(
            nctids=["NCT1", "NCT2", "NCT3"],
            y_true=np.array([1.0, 0.0, 1.0]),
            scores=np.array([0.9, 0.2, 0.7]),
            phase_index=np.array([2, 2, 1]),
            data_source=np.array(["TOP", "CTO", "TOP"]),
            approval_true=np.array([1.0, -1.0, 0.0]),
            approval_mask=np.array([1.0, 0.0, 1.0]),
            approval_scores=np.array([0.8, np.nan, 0.3]),
        )
        path = dump_scored_split(split, tmp_path / "scores.csv")
        loaded = load_scored_split(path)
        assert loaded.approval_true is not None
        np.testing.assert_allclose(loaded.approval_true, split.approval_true)
        np.testing.assert_allclose(loaded.approval_mask, split.approval_mask)
        assert loaded.approval_labeled_mask().tolist() == [True, False, True]

    def test_labeled_mask_empty_when_absent(self):
        split = ScoredSplit(
            nctids=["NCT1"],
            y_true=np.array([1.0]),
            scores=np.array([0.5]),
            phase_index=np.array([2]),
        )
        assert split.approval_labeled_mask().tolist() == [False]


class TestCollatorApproval:
    def test_collator_stacks_approval_fields(self):
        from fda_predictor.data.datasets import TrialCollator

        item = {
            "nctid": "NCT1",
            "label": torch.tensor(1.0),
            "mol_input_a": [{"input_ids": torch.tensor([5, 6]), "attention_mask": torch.tensor([1, 1])}],
            "mol_input_b": [{"input_ids": torch.tensor([7, 8]), "attention_mask": torch.tensor([1, 1])}],
            "crit_input": {"input_ids": torch.tensor([9, 10]), "attention_mask": torch.tensor([1, 1])},
            "n_molecules": 1,
            "phase_index": torch.tensor(2),
            "molecule_mask": torch.tensor(1.0),
            "stock_feats": torch.zeros(7),
            "stock_mask": torch.tensor(0.0),
            "tabular_feats": torch.zeros(7),
            "tabular_mask": torch.zeros(7),
            "approval_label": torch.tensor(1.0),
            "approval_mask": torch.tensor(1.0),
        }
        collator = TrialCollator(tok_a_pad_id=1, tok_b_pad_id=1, tok_c_pad_id=1)
        batch = collator([item])
        assert batch["approval_label"].shape == (1,)
        assert batch["approval_mask"].shape == (1,)
        assert float(batch["approval_label"][0]) == 1.0
