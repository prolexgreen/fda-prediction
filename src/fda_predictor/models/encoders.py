"""Transformer stream encoders for the three-stream model.

`fda_predictor.utils.paths` must be imported before this module (any
package import path through fda_predictor.data or fda_predictor.models
guarantees it) so HF resolution targets D:\\Models\\hub.

MoleculeEncoder consumes the flattened molecule tensors produced by
TrialCollator -- (total_molecules_in_batch, L) -- runs ONE forward pass
over every molecule, mask-mean-pools tokens into per-molecule vectors,
then aggregates each trial's molecules into a single (batch, H) vector
via group_index (variable molecule counts incl. single-molecule trials).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from fda_predictor.data.tokenizers import EncoderSpec


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Token-mean pooling that ignores padding positions."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def load_backbone(spec: EncoderSpec) -> tuple[nn.Module, int]:
    """Load a pinned HF backbone; returns (model, hidden_size).

    cache_dir is passed explicitly (not relied on env vars) so resolution
    targets D:\\Models even when transformers was imported before
    fda_predictor.utils.paths in some entry point.
    """
    from fda_predictor.utils.paths import HUB_CACHE

    kwargs = dict(
        trust_remote_code=spec.trust_remote_code,
        cache_dir=str(HUB_CACHE),
    )
    if spec.revision:
        kwargs["revision"] = spec.revision
    cfg = AutoConfig.from_pretrained(spec.name, **kwargs)
    model = AutoModel.from_pretrained(spec.name, **kwargs)
    return model, int(cfg.hidden_size)


class MoleculeEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, name: str = "molecule"):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.stream_name = name

    @classmethod
    def from_spec(cls, spec: EncoderSpec) -> "MoleculeEncoder":
        backbone, hidden_size = load_backbone(spec)
        return cls(backbone, hidden_size, name=f"mol:{spec.key}")

    def forward(
        self,
        mol_input: dict[str, torch.Tensor],
        group_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """(M, L) tokens -> one (batch_size, hidden) vector per trial."""
        out = self.backbone(
            input_ids=mol_input["input_ids"],
            attention_mask=mol_input["attention_mask"],
        )
        mol_vecs = masked_mean(out.last_hidden_state, mol_input["attention_mask"])  # (M, H)

        sums = torch.zeros(
            batch_size, self.hidden_size, dtype=mol_vecs.dtype, device=mol_vecs.device
        )
        sums.index_add_(0, group_index, mol_vecs)
        counts = (
            torch.bincount(group_index, minlength=batch_size)
            .to(mol_vecs.dtype)
            .clamp(min=1)
            .unsqueeze(1)
        )
        return sums / counts


class ProtocolEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, name: str = "protocol"):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.stream_name = name

    @classmethod
    def from_spec(cls, spec: EncoderSpec) -> "ProtocolEncoder":
        backbone, hidden_size = load_backbone(spec)
        return cls(backbone, hidden_size, name=f"text:{spec.key}")

    def forward(self, crit_input: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, Lc) criteria tokens -> (batch_size, hidden)."""
        out = self.backbone(
            input_ids=crit_input["input_ids"],
            attention_mask=crit_input["attention_mask"],
        )
        return masked_mean(out.last_hidden_state, crit_input["attention_mask"])


class StockEncoder(nn.Module):
    """Pre-event market trend vector -> embedding (mask zeros invalid rows)."""

    def __init__(self, n_features: int = 7, out_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        stock_feats: torch.Tensor,
        stock_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.net(stock_feats)
        if stock_mask is not None:
            m = stock_mask.to(out.dtype).unsqueeze(-1)
            out = out * m
        return out


class TabularEncoder(nn.Module):
    """Design-time tabular vector -> embedding; per-feature mask zeros missing dims."""

    def __init__(self, n_features: int = 7, out_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.out_dim = out_dim
        self.missing = nn.Parameter(torch.zeros(n_features))
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        tabular_feats: torch.Tensor,
        tabular_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = tabular_feats
        if tabular_mask is not None:
            m = tabular_mask.to(x.dtype)
            # Replace missing dims with a learned null embedding coordinate
            x = x * m + self.missing.unsqueeze(0) * (1.0 - m)
        return self.net(x)
