"""Stage-5 tests: freeze_mol_bottom_layers + per-encoder LLRD param groups."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.models.encoders import (  # noqa: E402
    MoleculeEncoder,
    ProtocolEncoder,
    StockEncoder,
    TabularEncoder,
)
from fda_predictor.models.multimodal_net import TriStreamNet  # noqa: E402
from fda_predictor.training.trainer import _encoder_param_groups  # noqa: E402


def _shrink_to_cpu_cpu() -> TriStreamNet:
    """Tiny 2-layer all-transformer net; no HF hub download needed."""
    from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

    cfg_r = RobertaConfig(
        vocab_size=120, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=132, pad_token_id=1,
    )
    cfg_b = BertConfig(
        vocab_size=120, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
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


class TestFreezeMolBottomLayers:
    def test_freezes_embeddings_and_bottom_layers_both_mol_encoders(self):
        net = _shrink_to_cpu_cpu()
        n = net.freeze_mol_bottom_layers(1)
        assert n > 0
        for name, backbone in (
            ("chemberta", net.mol_encoder_a.backbone),
            ("molformer", net.mol_encoder_b.backbone),
        ):
            assert not any(
                p.requires_grad for p in backbone.embeddings.parameters()
            ), f"{name} embeddings must be frozen"
            bottom = backbone.encoder.layer[0]
            assert not any(p.requires_grad for p in bottom.parameters()), f"{name} layer0 frozen"
            top = backbone.encoder.layer[1]
            assert any(p.requires_grad for p in top.parameters()), f"{name} top layer stays trainable"

    def test_n_zero_freezes_nothing(self):
        net = _shrink_to_cpu_cpu()
        assert net.freeze_mol_bottom_layers(0) == 0

    def test_idempotent(self):
        net = _shrink_to_cpu_cpu()
        first = net.freeze_mol_bottom_layers(1)
        second = net.freeze_mol_bottom_layers(1)
        assert first == second


class TestEncoderLLRDGroups:
    def test_per_encoder_decay_descending_by_depth(self):
        net = _shrink_to_cpu_cpu()
        groups = _encoder_param_groups(
            net,
            base_lr=1e-3,
            llrd_map={"chemberta": 0.5, "molformer": 0.5, "clinicalbert": 1.0},
            weight_decay=0.01,
        )
        by_name = {g["name"]: g["lr"] for g in groups}
        # embeddings deepest
        assert by_name["chemberta.embeddings"] == 1e-3 * 0.5**3
        assert by_name["molformer.embeddings"] == 1e-3 * 0.5**3
        # layer1 (top) > layer0 (bottom)
        assert by_name["chemberta.layer.1"] > by_name["chemberta.layer.0"]
        assert by_name["molformer.layer.1"] > by_name["molformer.layer.0"]
        # clinicalbert at 1.0 -> lands in rest (base lr), not in per-layer groups
        non_encoder = [n for n in by_name if n.startswith("clinicalbert.")]
        assert not non_encoder
        assert by_name["rest"] == 1e-3

    def test_frozen_encoder_skipped(self):
        net = _shrink_to_cpu_cpu()
        net.apply_freeze(chemberta=True)  # no trainable params in stream A
        groups = _encoder_param_groups(
            net,
            base_lr=1e-3,
            llrd_map={"chemberta": 0.5, "molformer": 0.5, "clinicalbert": 1.0},
            weight_decay=0.01,
        )
        names = {g["name"] for g in groups}
        assert not any(n.startswith("chemberta.") for n in names)
        assert any(n.startswith("molformer.") for n in names)

    def test_every_trainable_param_assigned_once(self):
        net = _shrink_to_cpu_cpu()
        groups = _encoder_param_groups(
            net,
            base_lr=1e-3,
            llrd_map={"chemberta": 0.8, "molformer": 0.8, "clinicalbert": 0.9},
            weight_decay=0.01,
        )
        seen: set[int] = set()
        total = 0
        for g in groups:
            for p in g["params"]:
                assert id(p) not in seen, f"param double-assigned ({g['name']})"
                seen.add(id(p))
            total += len(g["params"])
        trainable = [p for p in net.parameters() if p.requires_grad]
        assert total == len(trainable)

    def test_no_overlap_when_llrd_all_unity(self):
        net = _shrink_to_cpu_cpu()
        groups = _encoder_param_groups(
            net,
            base_lr=1e-3,
            llrd_map={"chemberta": 1.0, "molformer": 1.0, "clinicalbert": 1.0},
            weight_decay=0.01,
        )
        assert [g["name"] for g in groups] == ["rest"]
