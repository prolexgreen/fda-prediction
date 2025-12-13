"""TriStreamNet (+ phase + stock + tabular): multimodal classifier."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from fda_predictor.data.tokenizers import EncoderSpec
from fda_predictor.models.encoders import (
    MoleculeEncoder,
    ProtocolEncoder,
    StockEncoder,
    TabularEncoder,
)
from fda_predictor.models.fusion import FusionClassifier

logger = logging.getLogger(__name__)


class TriStreamNet(nn.Module):
    """Transformer streams + phase + stock + tabular fusion head.

    CHECKPOINT_LAYOUT=7 widens the tabular stream to include the stage-8
    TxGNN knowledge-graph embedding block (PCA-64 projector).
    6 added stage-7 (chemistry descriptors, modality, mechanism, sponsor)
    tabular features; 5 added the approval head; 4 added tabular;
    3 had stock; 2 had phase only.
    """

    CHECKPOINT_LAYOUT = 7

    def __init__(
        self,
        chemberta: MoleculeEncoder,
        molformer: MoleculeEncoder,
        clinicalbert: ProtocolEncoder,
        stock_encoder: StockEncoder,
        tabular_encoder: TabularEncoder | None = None,
        fusion_hidden_dims: tuple[int, ...] = (768,),
        fusion_dropout: float = 0.2,
        gradient_checkpointing: bool = False,
        num_phase_categories: int = 5,
        phase_emb_dim: int = 32,
        stock_emb_dim: int = 64,
        n_stock_features: int = 7,
        tabular_emb_dim: int = 64,
        n_tabular_features: int = 7,
        use_mol_null: bool = True,
        approval_hidden_dims: tuple[int, ...] | None = (256,),
    ):
        super().__init__()
        self.mol_encoder_a = chemberta
        self.mol_encoder_b = molformer
        self.protocol_encoder = clinicalbert
        self.stock_encoder = stock_encoder
        self.tabular_encoder = tabular_encoder
        self.num_phase_categories = num_phase_categories
        self.phase_unk_index = num_phase_categories - 1
        self.phase_embedding = nn.Embedding(num_phase_categories, phase_emb_dim)
        self.stock_emb_dim = stock_emb_dim
        self.n_stock_features = n_stock_features
        self.tabular_emb_dim = tabular_emb_dim if tabular_encoder is not None else 0
        self.n_tabular_features = n_tabular_features
        self.use_mol_null = use_mol_null
        if use_mol_null:
            self.mol_null_a = nn.Parameter(torch.zeros(chemberta.hidden_size))
            self.mol_null_b = nn.Parameter(torch.zeros(molformer.hidden_size))
        else:
            self.register_parameter("mol_null_a", None)
            self.register_parameter("mol_null_b", None)

        fusion_input_dim = (
            chemberta.hidden_size
            + molformer.hidden_size
            + clinicalbert.hidden_size
            + phase_emb_dim
            + stock_emb_dim
            + (tabular_emb_dim if tabular_encoder is not None else 0)
        )
        self.fusion_input_dim = fusion_input_dim
        self.fusion = FusionClassifier(fusion_input_dim, fusion_hidden_dims, fusion_dropout)
        self.approval_head = (
            FusionClassifier(fusion_input_dim, approval_hidden_dims, fusion_dropout)
            if approval_hidden_dims
            else None
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        gradient_checkpointing: bool | None = None,
        force_layout: int | None = None,
    ) -> "TriStreamNet":
        specs = _specs_from_config(config)
        enc_cfg = config["encoders"]
        fusion_cfg = config.get("fusion", {})
        stock_cfg = config.get("stock", {})
        tab_cfg = config.get("features", {}).get("tabular", {})
        n_stock = int(stock_cfg.get("n_features", 7))
        stock_dim = int(stock_cfg.get("emb_dim", 64))
        n_tab = int(tab_cfg.get("n_features", 7))
        tab_dim = int(tab_cfg.get("emb_dim", 64))
        # Layout 3 checkpoints lack the tabular stream; force_layout=3 skips it.
        tabular_enabled = bool(tab_cfg.get("enabled", True)) and force_layout != 3
        tabular_encoder = (
            TabularEncoder(n_features=n_tab, out_dim=tab_dim) if tabular_enabled else None
        )
        net = cls(
            chemberta=MoleculeEncoder.from_spec(specs["chemberta"]),
            molformer=MoleculeEncoder.from_spec(specs["molformer"]),
            clinicalbert=ProtocolEncoder.from_spec(specs["clinicalbert"]),
            stock_encoder=StockEncoder(n_features=n_stock, out_dim=stock_dim),
            tabular_encoder=tabular_encoder,
            fusion_hidden_dims=tuple(fusion_cfg["hidden_dims"]),
            fusion_dropout=float(fusion_cfg["dropout"]),
            approval_hidden_dims=tuple(fusion_cfg.get("approval_hidden_dims", [256])) or None,
            phase_emb_dim=int(fusion_cfg.get("phase_emb_dim", 32)),
            stock_emb_dim=stock_dim,
            n_stock_features=n_stock,
            tabular_emb_dim=tab_dim if tabular_enabled else 0,
            n_tabular_features=n_tab,
            use_mol_null=bool(tab_cfg.get("use_mol_null", True)) and tabular_enabled,
        )
        net.apply_freeze(
            chemberta=bool(enc_cfg["chemberta"]["freeze"]),
            molformer=bool(enc_cfg["molformer"]["freeze"]),
            clinicalbert=bool(enc_cfg["clinicalbert"]["freeze"]),
            stock_encoder=bool(stock_cfg.get("freeze", False)),
        )
        freeze_bottom = int(config.get("training", {}).get("freeze_clinicalbert_bottom_layers", 0) or 0)
        if freeze_bottom > 0 and not bool(enc_cfg["clinicalbert"]["freeze"]):
            net.freeze_clinicalbert_bottom_layers(freeze_bottom)
        # OOM fallback lever for stage-5: freeze bottom N of the molecule encoders.
        # Idempotent over apply_freeze; re-freezing frozen params is a no-op.
        freeze_mol = int(config.get("training", {}).get("freeze_mol_bottom_layers", 0) or 0)
        if freeze_mol > 0:
            n = net.freeze_mol_bottom_layers(freeze_mol)
            logger.info("froze %s params across mol encoders (bottom %s layers)", f"{n:,}", freeze_mol)
        gc = bool(config.get("compute", {}).get("gradient_checkpointing", False))
        if gradient_checkpointing is not None:
            gc = gradient_checkpointing
        if gc:
            net.enable_gradient_checkpointing()
        return net

    def forward(
        self,
        mol_input_a: dict[str, Any],
        mol_input_b: dict[str, Any],
        group_index,
        batch_size: int,
        crit_input: dict[str, Any],
        phase_index=None,
        stock_feats=None,
        stock_mask=None,
        molecule_mask=None,
        tabular_feats=None,
        tabular_mask=None,
    ):
        success_logits, _approval_logits = self.forward_with_approval(
            mol_input_a=mol_input_a,
            mol_input_b=mol_input_b,
            group_index=group_index,
            batch_size=batch_size,
            crit_input=crit_input,
            phase_index=phase_index,
            stock_feats=stock_feats,
            stock_mask=stock_mask,
            molecule_mask=molecule_mask,
            tabular_feats=tabular_feats,
            tabular_mask=tabular_mask,
        )
        return success_logits

    def forward_with_approval(
        self,
        mol_input_a: dict[str, Any],
        mol_input_b: dict[str, Any],
        group_index,
        batch_size: int,
        crit_input: dict[str, Any],
        phase_index=None,
        stock_feats=None,
        stock_mask=None,
        molecule_mask=None,
        tabular_feats=None,
        tabular_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (success_logits, approval_logits). Raw logits, shape (B, 1).

        approval_logits is None when the approval head is disabled.
        """
        vec_a = self.mol_encoder_a(mol_input_a, group_index, batch_size)
        vec_b = self.mol_encoder_b(mol_input_b, group_index, batch_size)
        vec_c = self.protocol_encoder(crit_input)

        if molecule_mask is not None:
            m = molecule_mask.to(vec_a.dtype).unsqueeze(-1)
            if self.use_mol_null and self.mol_null_a is not None:
                vec_a = vec_a * m + self.mol_null_a.unsqueeze(0) * (1.0 - m)
                vec_b = vec_b * m + self.mol_null_b.unsqueeze(0) * (1.0 - m)
            else:
                vec_a = vec_a * m
                vec_b = vec_b * m

        if phase_index is None:
            phase_index = torch.full(
                (batch_size,),
                self.phase_unk_index,
                dtype=torch.long,
                device=vec_a.device,
            )
        elif phase_index.device != vec_a.device:
            phase_index = phase_index.to(vec_a.device)
        phase_vec = self.phase_embedding(phase_index)

        if stock_feats is None:
            stock_feats = torch.zeros(
                batch_size, self.n_stock_features, dtype=vec_a.dtype, device=vec_a.device
            )
        elif stock_feats.device != vec_a.device:
            stock_feats = stock_feats.to(vec_a.device)
        if stock_mask is None:
            stock_mask = torch.zeros(batch_size, dtype=vec_a.dtype, device=vec_a.device)
        elif stock_mask.device != vec_a.device:
            stock_mask = stock_mask.to(vec_a.device)
        stock_vec = self.stock_encoder(stock_feats, stock_mask)

        if self.tabular_encoder is None:
            tabular_vec = None
        else:
            if tabular_feats is None:
                tabular_feats = torch.zeros(
                    batch_size,
                    self.n_tabular_features,
                    dtype=vec_a.dtype,
                    device=vec_a.device,
                )
            elif tabular_feats.device != vec_a.device:
                tabular_feats = tabular_feats.to(vec_a.device)
            if tabular_mask is None:
                tabular_mask = torch.zeros(
                    batch_size,
                    self.n_tabular_features,
                    dtype=vec_a.dtype,
                    device=vec_a.device,
                )
            elif tabular_mask.device != vec_a.device:
                tabular_mask = tabular_mask.to(vec_a.device)
            tabular_vec = self.tabular_encoder(tabular_feats, tabular_mask)

        success_logits = self.fusion(vec_a, vec_b, vec_c, phase_vec, stock_vec, tabular_vec)
        approval_logits = (
            self.approval_head(vec_a, vec_b, vec_c, phase_vec, stock_vec, tabular_vec)
            if self.approval_head is not None
            else None
        )
        return success_logits, approval_logits

    def apply_freeze(
        self,
        chemberta: bool | None = None,
        molformer: bool | None = None,
        clinicalbert: bool | None = None,
        stock_encoder: bool | None = None,
    ) -> dict[str, int]:
        frozen_counts: dict[str, int] = {}
        mapping = {
            "chemberta": (self.mol_encoder_a, chemberta),
            "molformer": (self.mol_encoder_b, molformer),
            "clinicalbert": (self.protocol_encoder, clinicalbert),
        }
        for key, (encoder, flag) in mapping.items():
            if flag is None:
                continue
            n_frozen = 0
            for p in encoder.backbone.parameters():
                p.requires_grad = not flag
                if flag:
                    n_frozen += p.numel()
            frozen_counts[key] = n_frozen

        if stock_encoder is not None:
            n_frozen = 0
            for p in self.stock_encoder.parameters():
                p.requires_grad = not stock_encoder
                if stock_encoder:
                    n_frozen += p.numel()
            frozen_counts["stock_encoder"] = n_frozen
        return frozen_counts

    def freeze_clinicalbert_bottom_layers(self, n_layers: int) -> int:
        """Freeze the bottom `n_layers` of ClinicalBERT (embeddings + early encoder)."""
        return _freeze_backbone_bottom_layers(self.protocol_encoder.backbone, n_layers)

    def freeze_mol_bottom_layers(self, n_layers: int) -> int:
        """Freeze the bottom `n_layers` of BOTH molecule encoders.

        ChemBERTa is a RoBERTa variant and MoLFormer's remote-code module also
        exposes `.embeddings` + `.encoder.layer` (verified on the pinned
        revision), so one generic helper covers all three transformer streams.
        OOM fallback lever for stage-5 unfrozen training.
        """
        return _freeze_backbone_bottom_layers(
            self.mol_encoder_a.backbone, n_layers
        ) + _freeze_backbone_bottom_layers(self.mol_encoder_b.backbone, n_layers)

    def enable_gradient_checkpointing(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        encoders = {
            "chemberta": self.mol_encoder_a.backbone,
            "molformer": self.mol_encoder_b.backbone,
            "clinicalbert": self.protocol_encoder.backbone,
        }
        for name, backbone in encoders.items():
            results[name] = _try_enable_checkpointing(name, backbone)
        return results

    def parameter_summary(self) -> str:
        lines = [f"fusion input dim: {self.fusion_input_dim}"]
        total_trainable = 0
        total_all = 0
        streams: "OrderedDict[str, nn.Module]" = OrderedDict(
            [
                ("mol_encoder_a(chemberta)", self.mol_encoder_a),
                ("mol_encoder_b(molformer)", self.mol_encoder_b),
                ("protocol_encoder(clinicalbert)", self.protocol_encoder),
                ("phase_embedding", self.phase_embedding),
                ("stock_encoder", self.stock_encoder),
            ]
        )
        if self.tabular_encoder is not None:
            streams["tabular_encoder"] = self.tabular_encoder
        if self.approval_head is not None:
            streams["approval_head"] = self.approval_head
        streams["fusion"] = self.fusion
        for name, module in streams.items():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            all_p = sum(p.numel() for p in module.parameters())
            total_trainable += trainable
            total_all += all_p
            lines.append(f"{name:<32} trainable {trainable:>12,} | total {all_p:>12,}")
        lines.append(f"{'TOTAL':<32} trainable {total_trainable:>12,} | total {total_all:>12,}")
        return "\n".join(lines)


def _freeze_backbone_bottom_layers(backbone: nn.Module, n_layers: int) -> int:
    """Freeze embeddings + the bottom `n_layers` transformer layers.

    Works for RoBERTa, BERT, and MoLFormer-style modules (all expose
    `.embeddings` and `.encoder.layer`). Returns frozen parameter count.
    """
    if int(n_layers) <= 0:
        return 0
    frozen = 0
    embeddings = getattr(backbone, "embeddings", None)
    if embeddings is not None:
        for p in embeddings.parameters():
            p.requires_grad = False
            frozen += p.numel()
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is not None:
        for layer in list(layers)[: max(0, int(n_layers))]:
            for p in layer.parameters():
                p.requires_grad = False
                frozen += p.numel()
    return frozen


def _try_enable_checkpointing(name: str, backbone: nn.Module) -> bool:
    method = getattr(backbone, "gradient_checkpointing_enable", None)
    if method is None or not callable(method):
        logger.warning(
            "%s backbone does not expose gradient_checkpointing_enable(); "
            "continuing without checkpointing for this stream.",
            name,
        )
        return False
    try:
        method()
        return True
    except Exception as exc:
        logger.warning(
            "%s.gradient_checkpointing_enable() failed (%s); continuing without.",
            name,
            exc,
        )
        return False


def _specs_from_config(config: dict[str, Any]) -> dict[str, EncoderSpec]:
    from fda_predictor.data.tokenizers import specs_from_config

    return specs_from_config(config)
