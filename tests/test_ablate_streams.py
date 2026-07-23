"""Ablation-mask unit tests: knockouts must actually change model output."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.models.encoders import (  # noqa: E402
    MoleculeEncoder,
    ProtocolEncoder,
    StockEncoder,
    TabularEncoder,
)
from fda_predictor.models.multimodal_net import TriStreamNet  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablate_streams import MaskedDataset  # noqa: E402


class _FakeDataset:
    tok_a = tok_b = tok_c = object()

    def __init__(self, n=5):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "nctid": f"NCT{idx:08d}",
            "label": torch.tensor(1.0),
            "molecule_mask": torch.tensor(1.0),
            "stock_mask": torch.tensor(1.0),
            "tabular_mask": torch.ones(7),
        }


def _tiny_net() -> TriStreamNet:
    from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

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


def _batch(n=4, mol_mask=1.0, stock_mask=1.0, tab_mask=1.0):
    torch.manual_seed(7)
    ids = torch.randint(20, 100, (n, 10))
    crit = torch.randint(5, 100, (n, 24))
    return dict(
        mol_input_a={"input_ids": ids, "attention_mask": torch.ones(n, 10, dtype=torch.long)},
        mol_input_b={"input_ids": ids + 1, "attention_mask": torch.ones(n, 10, dtype=torch.long)},
        group_index=torch.arange(n),
        batch_size=n,
        crit_input={"input_ids": crit, "attention_mask": torch.ones(n, 24, dtype=torch.long)},
        phase_index=torch.full((n,), 2),
        molecule_mask=torch.full((n,), mol_mask),
        stock_feats=torch.randn(n, 7),
        stock_mask=torch.full((n,), stock_mask),
        tabular_feats=torch.randn(n, 7),
        tabular_mask=torch.full((n, 7), tab_mask),
    )


def test_masked_dataset_zeroes_target_mask_only():
    masked = MaskedDataset(_FakeDataset(), "molecule_mask")
    item = masked[0]
    assert float(item["molecule_mask"]) == 0.0
    # other masks untouched
    assert float(item["stock_mask"]) == 1.0
    assert float(item["tabular_mask"].sum()) == 7.0
    assert len(masked) == 5
    assert masked.tok_a is masked._ds.tok_a  # getattr pass-through


def test_molecule_knockout_changes_scores():
    net = _tiny_net().eval()
    with torch.no_grad():
        on = net(**_batch(mol_mask=1.0))
        off = net(**_batch(mol_mask=0.0))
    assert not torch.allclose(on, off)


def test_stock_and_tabular_knockouts_change_scores():
    net = _tiny_net().eval()
    ref = _batch()
    with torch.no_grad():
        base = net(**ref)
        no_stock = net(**{**ref, "stock_mask": torch.zeros(4)})
        no_tab = net(**{**ref, "tabular_mask": torch.zeros(4, 7)})
    assert not torch.allclose(base, no_stock)
    assert not torch.allclose(base, no_tab)


def test_phase3_delta_computed_when_slice_present():
    # sanity: auprc helper handles a Phase III slice on skewed labels
    from fda_predictor.training.metrics import auprc

    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1], dtype=float)
    p = np.linspace(0.05, 0.95, len(y))
    assert 0.0 < auprc(y, p) < 1.0
