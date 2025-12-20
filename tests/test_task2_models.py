"""Task 2 model tests (CPU-only mechanics + real-checkpoint integration).

Mechanical tests build tiny random Roberta/Bert backbones locally -- no
network, no pinned checkpoints. The integration test loads the real three
checkpoints and a real Task 1 batch end-to-end, skipping automatically if
either is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import fda_predictor.utils.paths BEFORE transformers so HF cache env vars
# are set before huggingface_hub bakes its constants.
from fda_predictor.models.encoders import (  # noqa: E402
    MoleculeEncoder,
    ProtocolEncoder,
    StockEncoder,
    TabularEncoder,
    masked_mean,
)
from fda_predictor.models.fusion import FusionClassifier  # noqa: E402
from fda_predictor.models.multimodal_net import TriStreamNet  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel  # noqa: E402


def _roberta_tiny(hidden: int = 16):
    cfg = RobertaConfig(
        vocab_size=300,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=2 * hidden,
        max_position_embeddings=132,
        pad_token_id=1,
    )
    return RobertaModel(cfg), hidden


def _bert_tiny(hidden: int = 12):
    cfg = BertConfig(
        vocab_size=300,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=2 * hidden,
        max_position_embeddings=512,
    )
    return BertModel(cfg), hidden


def _mol_input(vocab: int, n_mols: int, lengths: list[int]) -> dict[str, torch.Tensor]:
    ids, masks = [], []
    g = torch.Generator().manual_seed(7)
    for L in lengths:
        row = torch.randint(5, vocab, (L,), generator=g)
        ids.append(row)
        masks.append(torch.ones(L, dtype=torch.long))
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=1),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        ),
    }


class TestMaskedMean:
    def test_ignores_padding(self):
        h = torch.randn(2, 5, 4)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        pooled = masked_mean(h, mask)
        expected0 = h[0, :3].mean(dim=0)
        assert torch.allclose(pooled[0], expected0, atol=1e-6)

    def test_all_padding_is_safe(self):
        h = torch.randn(1, 3, 4)
        mask = torch.zeros(1, 3, dtype=torch.long)
        assert torch.isfinite(masked_mean(h, mask)).all()


class TestFusion:
    def test_output_shape_and_dim(self):
        fusion = FusionClassifier(input_dim=48, hidden_dims=(24,), dropout=0.0)
        out = fusion(torch.randn(4, 16), torch.randn(4, 20), torch.randn(4, 12))
        assert out.shape == (4, 1)

    def test_returns_unbounded_raw_logits(self):
        """Contract: no sigmoid inside forward; logits may exceed [0,1]."""
        fusion = FusionClassifier(input_dim=6, hidden_dims=(8,), dropout=0.0)
        with torch.no_grad():
            # Deterministic large weights/bias: avoids RNG-dependent flakiness.
            fusion.net[-1].weight.fill_(50.0)
            fusion.net[-1].bias.fill_(10.0)
        out = fusion(torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        assert out.abs().max() > 1.0, "expected raw logits, got sigmoid-bounded values"

    def test_gradients_flow_to_all_streams(self):
        fusion = FusionClassifier(input_dim=9, hidden_dims=(8,), dropout=0.0)
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3, requires_grad=True)
        c = torch.randn(2, 3, requires_grad=True)
        fusion(a, b, c).sum().backward()
        assert a.grad is not None and b.grad is not None and c.grad is not None


class TestMoleculeEncoder:
    def _encoder(self, hidden=16):
        backbone, h = _roberta_tiny(hidden)
        return MoleculeEncoder(backbone, h, name="test")

    def test_group_pooling_matches_manual(self):
        enc = self._encoder()
        enc.eval()
        group_index = torch.tensor([0, 0, 1])
        mol_input = _mol_input(300, 3, [10, 7, 12])

        with torch.no_grad():
            out = enc(mol_input, group_index, batch_size=2)
            per_mol = masked_mean(
                enc.backbone(**mol_input).last_hidden_state, mol_input["attention_mask"]
            )
        expected = torch.stack([per_mol[:2].mean(dim=0), per_mol[2]])
        assert out.shape == (2, enc.hidden_size)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_variable_counts_including_single(self):
        enc = self._encoder()
        enc.eval()
        group_index = torch.tensor([0, 1, 2, 2, 2, 3])
        mol_input = _mol_input(300, 6, [9, 11, 5, 6, 7, 8])
        with torch.no_grad():
            out = enc(mol_input, group_index, batch_size=4)
        assert out.shape == (4, enc.hidden_size)
        # single-molecule trials pass their vector through unchanged
        per_mol = masked_mean(
            enc.backbone(**mol_input).last_hidden_state, mol_input["attention_mask"]
        )
        assert torch.allclose(out[1], per_mol[1], atol=1e-6)
        assert torch.allclose(out[3], per_mol[5], atol=1e-6)


class TestTriStreamNet:
    def _net(self):
        bb_a, h_a = _roberta_tiny(16)   # deliberately different sizes to prove
        bb_b, h_b = _roberta_tiny(20)   # the fusion dim is computed dynamically
        bb_c, h_c = _bert_tiny(12)
        net = TriStreamNet(
            chemberta=MoleculeEncoder(bb_a, h_a),
            molformer=MoleculeEncoder(bb_b, h_b),
            clinicalbert=ProtocolEncoder(bb_c, h_c),
            stock_encoder=StockEncoder(n_features=7, out_dim=16),
            tabular_encoder=TabularEncoder(n_features=7, out_dim=16),
            fusion_hidden_dims=(24,),
            fusion_dropout=0.0,
            phase_emb_dim=8,
            stock_emb_dim=16,
            tabular_emb_dim=16,
            n_tabular_features=7,
        )
        return net, h_a, h_b, h_c

    def _batch(self, b_sizes=(2, 1, 3)):
        vocab = 300
        flat_lens = [9, 7] + [11] + [5, 6, 8]
        mol_a = _mol_input(vocab, sum(b_sizes), flat_lens)
        mol_b = _mol_input(vocab, sum(b_sizes), [l + 1 for l in flat_lens])
        crit_ids = [torch.randint(5, vocab, (30,)) for _ in range(len(b_sizes))]
        crit = {
            "input_ids": torch.stack(crit_ids),
            "attention_mask": torch.ones(len(b_sizes), 30, dtype=torch.long),
        }
        return {
            "mol_input_a": mol_a,
            "mol_input_b": mol_b,
            "group_index": torch.tensor([i for i, n in enumerate(b_sizes) for _ in range(n)]),
            "batch_size": len(b_sizes),
            "crit_input": crit,
        }

    def test_fusion_input_dim_is_sum_of_hidden_sizes(self):
        net, ha, hb, hc = self._net()
        assert net.fusion_input_dim == ha + hb + hc + 8 + 16 + 16  # phase + stock + tabular

    def test_forward_logits_shape(self):
        net, *_ = self._net()
        net.eval()
        batch = self._batch()
        with torch.no_grad():
            logits = net(**batch)
        assert logits.shape == (batch["batch_size"], 1)
        assert torch.isfinite(logits).all()

    def test_forward_with_explicit_phase_indices(self):
        net, *_ = self._net()
        net.eval()
        batch = self._batch()
        phase = torch.tensor([0, 4, 2])  # incl. UNK
        with torch.no_grad():
            logits = net(**batch, phase_index=phase)
        assert logits.shape == (batch["batch_size"], 1)

    def test_freeze_toggle_hits_only_target_stream(self):
        net, *_ = self._net()
        counts = net.apply_freeze(chemberta=True)
        assert counts["chemberta"] > 0
        assert all(not p.requires_grad for p in net.mol_encoder_a.backbone.parameters())
        assert all(p.requires_grad for p in net.mol_encoder_b.backbone.parameters())
        assert all(p.requires_grad for p in net.protocol_encoder.backbone.parameters())
        assert all(p.requires_grad for p in net.fusion.parameters())

    def test_all_frozen_leaves_head_trainable(self):
        net, *_ = self._net()
        net.apply_freeze(chemberta=True, molformer=True, clinicalbert=True)
        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        assert trainable and all(
            n.startswith("fusion.")
            or n.startswith("approval_head.")
            or n.startswith("phase_embedding.")
            or n.startswith("stock_encoder.")
            or n.startswith("tabular_encoder.")
            or n.startswith("mol_null_")
            for n in trainable
        )

    def test_parameter_summary_reports_totals(self):
        net, *_ = self._net()
        summary = net.parameter_summary()
        assert "TOTAL" in summary and f"fusion input dim: {net.fusion_input_dim}" in summary

    def test_checkpointing_probe_degrades_gracefully(self):
        net, *_ = self._net()
        results = net.enable_gradient_checkpointing()
        # roberta/bert expose the standard HF interface
        assert results["chemberta"] is True and results["clinicalbert"] is True
        # a plain module without the interface must report False, not raise
        net.mol_encoder_b.backbone = torch.nn.Module()
        results = net.enable_gradient_checkpointing()
        assert results["molformer"] is False


@pytest.mark.integration
class TestRealCheckpointsEndToEnd:
    @pytest.fixture(scope="class")
    def net_and_batch(self):
        try:
            import yaml

            root = Path(__file__).resolve().parents[1]
            config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
            net = TriStreamNet.from_config(config)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"real checkpoints unavailable: {e!r}")

        try:
            from fda_predictor.data.datasets import TOPTrialDataset, build_collate_fn
            from fda_predictor.data.tokenizers import specs_from_config
            from fda_predictor.data.top_dataset import load_top_splits

            splits = load_top_splits()
            specs = specs_from_config(config)
            common = dict(
                chemberta_spec=specs["chemberta"],
                molformer_spec=specs["molformer"],
                clinicalbert_spec=specs["clinicalbert"],
            )
            ds = TOPTrialDataset(splits.train.head(8), **common)
            loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=build_collate_fn(ds))
            batch = next(iter(loader))
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"TOP data unavailable: {e!r}")

        return net, batch

    def test_real_forward_produces_logits(self, net_and_batch):
        net, batch = net_and_batch
        assert net.fusion_input_dim == 384 + 768 + 768 + 32 + 64 + 64
        net.eval()
        with torch.no_grad():
            logits = net(
                mol_input_a=batch["mol_input_a"],
                mol_input_b=batch["mol_input_b"],
                group_index=batch["group_index"],
                batch_size=len(batch["label"]),
                crit_input=batch["crit_input"],
                phase_index=batch.get("phase_index"),
                stock_feats=batch.get("stock_feats"),
                stock_mask=batch.get("stock_mask"),
                molecule_mask=batch.get("molecule_mask"),
            )
        assert logits.shape == (len(batch["label"]), 1)
        assert torch.isfinite(logits).all()

    def test_default_config_is_stage5_chem_unfrozen(self, net_and_batch):
        """Stage-5 default: all three transformers unfrozen; ClinicalBERT bottom
        4 layers frozen; fusion trainable."""
        net, _ = net_and_batch
        assert any(p.requires_grad for p in net.mol_encoder_a.backbone.parameters())
        assert any(p.requires_grad for p in net.mol_encoder_b.backbone.parameters())
        assert any(p.requires_grad for p in net.protocol_encoder.backbone.parameters())
        assert any(p.requires_grad for p in net.fusion.parameters())
