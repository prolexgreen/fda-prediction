"""Task 3 training tests: metrics math, weighted loss correctness, and a
trainer smoke run on a tiny synthetic dataset. GPU tests auto-skip when
CUDA is absent; everything here also runs on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.training.losses import (  # noqa: E402
    FocalLoss,
    WeightedBCE,
    positive_weight_from_labels,
)
from fda_predictor.training.metrics import (  # noqa: E402
    auprc,
    classification_report,
    f1_at_threshold,
    roc_auc,
    tune_threshold_on_val,
)
from fda_predictor.training.trainer import Trainer  # noqa: E402


class TestMetrics:
    def test_auprc_perfect_and_inverted(self):
        y = [0, 0, 1, 1]
        assert auprc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
        assert auprc(y, [0.9, 0.8, 0.2, 0.1]) < 0.5

    def test_auprc_random_is_positive_rate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        scores = rng.random(200)
        ap = auprc(y, scores)
        assert abs(ap - y.mean()) < 0.12  # E[AP] ~= prevalence for random scores

    def test_f1_at_threshold(self):
        y = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.3, 0.8]
        # threshold .5: preds [1,0,0,1] -> tp=1 fp=1 fn=1 -> F1 = 0.5
        assert f1_at_threshold(y, scores, 0.5) == pytest.approx(0.5)
        # threshold .35: preds [1,1,0,1] -> tp=2 fp=1 fn=0 -> F1 = 0.8
        assert f1_at_threshold(y, scores, 0.35) == pytest.approx(0.8)

    def test_tune_threshold_finds_best_on_val_only(self):
        y_val = [1, 1, 1, 0, 0, 0, 0, 0]
        s_val = [0.95, 0.85, 0.7, 0.55, 0.45, 0.4, 0.2, 0.1]
        best_t, best_f1 = tune_threshold_on_val(y_val, s_val)
        assert best_f1 == pytest.approx(1.0)  # perfect separation exists
        # any cut in (0.55, 0.70] works; the tuner returns the lowest such
        # grid point (dense linspace lands just above 0.55)
        assert 0.55 < best_t <= 0.625 + 1e-9

    def test_roc_auc_none_for_single_class(self):
        assert roc_auc([1, 1, 1], [0.2, 0.5, 0.9]) is None
        # perfectly anti-aligned scores -> AUC 0
        assert roc_auc([0, 1, 0], [0.9, 0.05, 0.8]) == pytest.approx(0.0)

    def test_classification_report_counts(self):
        rep = classification_report([1, 1, 0, 0], [0.9, 0.4, 0.3, 0.8], threshold=0.5)
        assert (rep["tp"], rep["fp"], rep["fn"]) == (1, 1, 1)
        assert rep["precision"] == pytest.approx(0.5)
        assert rep["recall"] == pytest.approx(0.5)

    def test_no_accuracy_metric_exists(self):
        import fda_predictor.training.metrics as m

        public = {n for n in dir(m) if not n.startswith("_")}
        assert not any("accuracy" in n.lower() for n in public)


class TestLosses:
    def test_pos_weight_formula(self):
        labels = [1, 1, 0, 0, 0, 0]  # 2 pos, 4 neg
        assert positive_weight_from_labels(labels) == pytest.approx(2.0)
        assert positive_weight_from_labels([1, 1, 1]) == pytest.approx(1.0)  # degenerate
        assert positive_weight_from_labels([]) == pytest.approx(1.0)

    def test_weighted_bce_matches_manual(self):
        logits = torch.tensor([[2.0], [-1.5], [0.3]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])
        w = 3.0
        manual = []
        for z, t in zip(logits.ravel(), targets.ravel()):
            bce = torch.nn.functional.binary_cross_entropy_with_logits(z, t, reduction="sum")
            manual.append(bce * (w if t == 1 else 1))
        expected = sum(manual) / len(manual)
        crit = WeightedBCE(pos_weight=w)
        assert crit(logits, targets).item() == pytest.approx(expected.item(), rel=1e-5)

    def test_weighted_bce_upweights_positives(self):
        torch.manual_seed(0)
        z = torch.randn(64, 1)
        t = torch.randint(0, 2, (64, 1)).float()
        plain = WeightedBCE(pos_weight=1.0)(z, t).item()
        up = WeightedBCE(pos_weight=5.0)(z, t).item()
        assert up > plain

    def test_focal_downweights_easy_examples(self):
        torch.manual_seed(1)
        z_good = torch.tensor([[6.0], [-6.0]])   # easy: confident & correct
        t = torch.tensor([[1.0], [0.0]])
        bce = WeightedBCE(pos_weight=1.0)(z_good, t).item()
        focal = FocalLoss(gamma=2.0)(z_good, t).item()
        assert focal < bce * 0.05

    def test_focal_focuses_hard_examples(self):
        """On ambiguous inputs (p=0.5) focal scales BCE by (1-0.5)^gamma=0.25:
        hard examples are down-weighted far less than easy ones (near-zero)."""
        torch.manual_seed(2)
        z_hard = torch.zeros(32, 1)              # p = 0.5 for every example
        t = torch.randint(0, 2, (32, 1)).float()
        focal = FocalLoss(gamma=2.0)(z_hard, t).item()
        bce = WeightedBCE(pos_weight=1.0)(z_hard, t).item()
        assert focal == pytest.approx(0.25 * bce, rel=0.05)


def _tiny_trainer(device_str: str):
    from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

    from fda_predictor.models.encoders import MoleculeEncoder, ProtocolEncoder, StockEncoder, TabularEncoder
    from fda_predictor.models.multimodal_net import TriStreamNet
    from fda_predictor.training.losses import WeightedBCE
    from fda_predictor.training.trainer import TrainConfig, Trainer
    from fda_predictor.utils.cuda_utils import resolve_device

    cfg_r = RobertaConfig(
        vocab_size=120, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=132, pad_token_id=1,
    )
    cfg_b = BertConfig(
        vocab_size=120, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=512,
    )
    net = TriStreamNet(
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
    )
    criterion = WeightedBCE(pos_weight=1.0)
    tcfg = TrainConfig(
        max_epochs=25, patience=25, lr=5e-3, batch_size=4, grad_accum_steps=1,
        num_workers=0, pin_memory=False, warmup_ratio=0.0,
        amp_requested="bfloat16", checkpoint_dir=None, tensorboard_dir=None,
    )
    return net, criterion, tcfg, resolve_device(device_str)


class _SyntheticTrialDataset(torch.utils.data.Dataset):
    """Linearly-separable synthetic trials: criteria token 10 count == label."""

    def __init__(self, n=32):
        g = torch.Generator().manual_seed(11)
        self.rows = []
        for i in range(n):
            label = float(i % 2)
            crit = torch.full((24,), int(label), dtype=torch.long)
            pos = torch.arange(2, 24)
            flip = pos[torch.rand(24 - 2, generator=g) < 0.15]
            crit[flip] = 5
            crit[:4] = torch.tensor([2, 3, 4, 10 * int(label) + 1])
            self.rows.append({
                "label": torch.tensor(label),
                "crit_input": {"input_ids": crit, "attention_mask": torch.ones(24, dtype=torch.long)},
                "mol_input_a": self._mol(g),
                "mol_input_b": self._mol(g),
                "group_index": torch.zeros(1, dtype=torch.long),
                "n_molecules": 1,
                "phase_index": torch.tensor(2, dtype=torch.long),
                "molecule_mask": torch.tensor(1.0),
                "stock_feats": torch.zeros(7),
                "stock_mask": torch.tensor(0.0),
                "tabular_feats": torch.zeros(7),
                "tabular_mask": torch.zeros(7),
            })

    @staticmethod
    def _mol(g):
        ids = torch.randint(20, 100, (10,), generator=g)
        return [{"input_ids": ids, "attention_mask": torch.ones(10, dtype=torch.long)}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _collate_synth(batch):
    out_ids = torch.stack([b["crit_input"]["input_ids"] for b in batch])
    out_mask = torch.stack([b["crit_input"]["attention_mask"] for b in batch])

    def flat(key):
        ids = [m["input_ids"] for b in batch for m in b[key]]
        masks = [m["attention_mask"] for b in batch for m in b[key]]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=1),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0),
        }

    return {
        "label": torch.stack([b["label"] for b in batch]),
        "crit_input": {"input_ids": out_ids, "attention_mask": out_mask},
        "mol_input_a": flat("mol_input_a"),
        "mol_input_b": flat("mol_input_b"),
        "group_index": torch.cat([b["group_index"] + i for i, b in enumerate(batch)]),
        "phase_index": torch.stack([b["phase_index"] for b in batch]),
        "molecule_mask": torch.stack([b["molecule_mask"] for b in batch]),
        "stock_feats": torch.stack([b["stock_feats"] for b in batch]),
        "stock_mask": torch.stack([b["stock_mask"] for b in batch]),
        "tabular_feats": torch.stack([b["tabular_feats"] for b in batch]),
        "tabular_mask": torch.stack([b["tabular_mask"] for b in batch]),
    }


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
class TestTrainerSmoke:
    @pytest.fixture
    def trained(self, device_str):
        if device_str == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        net, criterion, tcfg, device = _tiny_trainer(device_str)
        trainer = Trainer(net, criterion, tcfg, device)
        ds = _SyntheticTrialDataset(n=32)
        result = trainer.fit(ds, ds, collate_fn=_collate_synth)
        return trainer, result, device

    def test_overfits_separable_subset(self, trained):
        trainer, result, device = trained
        val_loss, y, p, _phase = trainer.evaluate(
            torch.utils.data.DataLoader(_SyntheticTrialDataset(n=32), batch_size=8, collate_fn=_collate_synth)
        )
        assert result.best_val_auprc > 0.9
        assert auprc(y, p) > 0.9

    def test_history_and_checkpoint_contract(self, trained):
        _, result, _ = trained
        assert result.epochs_run >= 1
        assert all(np.isfinite(h["train_loss"]) for h in result.history)
        assert result.best_epoch >= 1


def test_early_stopping_fires_when_patience_exhausted():
    """Regression: bad_epochs must increment on non-improving epochs.

    On perfectly separable data AUPRC reaches 1.0 and can no longer
    strictly improve, so with patience=2 training must stop well before
    max_epochs. (The original bug ran every epoch regardless.)
    """
    from dataclasses import replace

    net, criterion, tcfg, device = _tiny_trainer("cpu")
    tcfg = replace(tcfg, patience=2, max_epochs=25)
    trainer = Trainer(net, criterion, tcfg, device)
    ds = _SyntheticTrialDataset(n=32)
    result = trainer.fit(ds, ds, collate_fn=_collate_synth)
    assert result.epochs_run < 25, "early stopping never fired"


class _DualSyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dual-target rows: success from crit tokens, approval from
    molecule token 42 count; half the approval labels are masked."""

    def __init__(self, n=32):
        g = torch.Generator().manual_seed(13)
        self.rows = []
        for i in range(n):
            label = float(i % 2)
            appr = float((i // 2) % 2)
            masked = float(i % 4 == 1)  # ~25% unlabeled approval rows
            crit = torch.full((24,), int(label), dtype=torch.long)
            crit[:4] = torch.tensor([2, 3, 4, 10 * int(label) + 1])
            ids = torch.randint(50, 100, (10,), generator=g)
            if appr > 0:
                ids[0] = 42
                ids[1] = 42
            self.rows.append({
                "label": torch.tensor(label),
                "approval_label": torch.tensor(-1.0 if masked else appr),
                "approval_mask": torch.tensor(0.0 if masked else 1.0),
                "crit_input": {"input_ids": crit, "attention_mask": torch.ones(24, dtype=torch.long)},
                "mol_input_a": [{"input_ids": ids, "attention_mask": torch.ones(10, dtype=torch.long)}],
                "mol_input_b": [{"input_ids": ids + 1, "attention_mask": torch.ones(10, dtype=torch.long)}],
                "group_index": torch.zeros(1, dtype=torch.long),
                "n_molecules": 1,
                "phase_index": torch.tensor(2, dtype=torch.long),
                "molecule_mask": torch.tensor(1.0),
                "stock_feats": torch.zeros(7),
                "stock_mask": torch.tensor(0.0),
                "tabular_feats": torch.zeros(7),
                "tabular_mask": torch.zeros(7),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _collate_dual(batch):
    out = _collate_synth(batch)
    out["approval_label"] = torch.stack([b["approval_label"] for b in batch])
    out["approval_mask"] = torch.stack([b["approval_mask"] for b in batch])
    return out


@pytest.mark.parametrize("device_str", ["cpu"])
class TestTrainerDualSmoke:
    @pytest.fixture
    def trained(self, device_str):
        if device_str == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from fda_predictor.training.losses import MultiTaskLoss

        net, _criterion, tcfg, device = _tiny_trainer(device_str)
        net.approval_head = type(net.approval_head)(
            input_dim=16 + 16 + 16 + 4 + 8 + 8, hidden_dims=(8,), dropout=0.0
        )
        criterion = MultiTaskLoss(success_w=1.0, approval_w=1.0)
        trainer = Trainer(net, criterion, tcfg, device)
        ds = _DualSyntheticDataset(n=32)
        result = trainer.fit(ds, ds, collate_fn=_collate_dual)
        return trainer, result

    def test_overfits_dual_targets(self, trained):
        trainer, result = trained
        loader = torch.utils.data.DataLoader(
            _DualSyntheticDataset(n=32), batch_size=8, collate_fn=_collate_dual
        )
        loss, arrs = trainer.evaluate_dual(loader)
        am = arrs["approval_mask"] > 0
        assert result.best_val_auprc > 0.9, "success head must fit dense signal"
        assert am.sum() >= 10
        assert np.isfinite(loss)
        assert auprc(arrs["y_approval"][am], arrs["p_approval"][am]) > 0.9, (
            "approval head must fit its sparse labeled subset"
        )
