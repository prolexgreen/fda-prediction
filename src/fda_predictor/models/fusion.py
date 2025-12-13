"""Fusion classifier: concatenate stream embeddings -> MLP -> 1 logit.

Contract: forward() returns RAW LOGITS (B, 1). Sigmoid is applied only at the
loss boundary (BCEWithLogitsLoss) or at inference. No sigmoid inside.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (768,),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        mol_vec_a: torch.Tensor,
        mol_vec_b: torch.Tensor,
        crit_vec: torch.Tensor,
        phase_vec: torch.Tensor | None = None,
        stock_vec: torch.Tensor | None = None,
        tabular_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [mol_vec_a, mol_vec_b, crit_vec]
        if phase_vec is not None:
            parts.append(phase_vec)
        if stock_vec is not None:
            parts.append(stock_vec)
        if tabular_vec is not None:
            parts.append(tabular_vec)
        fused = torch.cat(parts, dim=-1)
        return self.net(fused)
