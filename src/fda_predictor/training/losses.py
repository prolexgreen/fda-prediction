"""Loss functions: positively-weighted BCE and focal loss (ablation).

Class weights are computed from the TRAIN split only -- val/test statistics
must never leak into training.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def positive_weight_from_labels(labels, max_weight_cap: float = 10.0) -> float:
    """w = n_negative / n_positive (standard BCE pos_weight).

    Degenerate label sets (all-positive / all-negative / empty) get a
    neutral weight of 1.0; capped for stability when positives are rare.
    """
    labels = torch.as_tensor(labels, dtype=torch.float32)
    n_pos = float(labels.sum())
    n_neg = float(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return min(n_neg / n_pos, max_weight_cap)


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float().view_as(logits), pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """Binary focal loss on logits with optional positive weighting.

    fl = -alpha * (1-p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view_as(targets).float()
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """Dense success loss + masked sparse approval loss.

    total = success_w * BCE(success_logits, y_success)
          + approval_w * BCE(approval_logits[mask], y_approval[mask])

    Rows with mask == 0 (unlabeled / previously-approved) contribute nothing
    to the approval term; the success term always covers every row.
    """

    def __init__(
        self,
        success_w: float = 1.0,
        approval_w: float = 1.0,
        success_pos_weight: float = 1.0,
        approval_pos_weight: float = 1.0,
    ):
        super().__init__()
        self.success_w = float(success_w)
        self.approval_w = float(approval_w)
        self.register_buffer("success_pos_weight", torch.tensor(float(success_pos_weight)))
        self.register_buffer("approval_pos_weight", torch.tensor(float(approval_pos_weight)))

    def forward(
        self,
        success_logits: torch.Tensor,
        y_success: torch.Tensor,
        approval_logits: torch.Tensor | None,
        y_approval: torch.Tensor,
        approval_mask: torch.Tensor,
    ) -> torch.Tensor:
        succ = nn.functional.binary_cross_entropy_with_logits(
            success_logits, y_success.float().view_as(success_logits),
            pos_weight=self.success_pos_weight,
        )
        if approval_logits is None or self.approval_w <= 0:
            return self.success_w * succ
        mask = approval_mask.float().view(-1) > 0
        if mask.any():
            app_logits = approval_logits.view(-1)[mask]
            app_targets = y_approval.float().view(-1)[mask]
            appr = nn.functional.binary_cross_entropy_with_logits(
                app_logits, app_targets.view_as(app_logits),
                pos_weight=self.approval_pos_weight,
            )
        else:
            appr = success_logits.new_zeros(())
        return self.success_w * succ + self.approval_w * appr


def build_loss(
    config: dict,
    train_labels,
    train_approval_labels=None,
    train_approval_mask=None,
) -> tuple[nn.Module, dict]:
    """Factory honoring config['training']['loss'] (+ dual-head multitask).

    Per-head pos_weights come from TRAIN split statistics only.
    """
    tcfg = config["training"]
    kind = tcfg.get("loss", "weighted_bce")
    target_mode = str(config.get("labels", {}).get("target", "success")).lower()

    if target_mode == "dual" and train_approval_labels is not None:
        pos_w_succ = positive_weight_from_labels(train_labels)
        mask = np.asarray(train_approval_mask or [], dtype=float) > 0
        labeled_app = [
            float(y)
            for y, keep in zip(train_approval_labels, mask)
            if keep and y is not None and not (isinstance(y, float) and math.isnan(y))
        ]
        pos_w_app = positive_weight_from_labels(labeled_app) if labeled_app else 1.0
        succ_w = float(config.get("labels", {}).get("success_weight", 1.0))
        appr_w = float(config.get("labels", {}).get("approval_weight", 1.0))
        criterion = MultiTaskLoss(
            success_w=succ_w,
            approval_w=appr_w,
            success_pos_weight=pos_w_succ,
            approval_pos_weight=pos_w_app,
        )
        info = {
            "loss_kind": "multitask",
            "success_weight": succ_w,
            "approval_weight": appr_w,
            "success_pos_weight": pos_w_succ,
            "approval_pos_weight": pos_w_app,
            "n_approval_labeled": int(mask.sum()),
        }
        return criterion, info

    pos_w = positive_weight_from_labels(train_labels)
    info = {"loss_kind": kind, "pos_weight": pos_w}
    if kind == "weighted_bce":
        criterion = WeightedBCE(pos_weight=pos_w)
    elif kind == "focal":
        alpha = min(0.5 * pos_w / (1 + 0.5 * pos_w), 0.9)  # bounded alpha from train stats
        criterion = FocalLoss(gamma=float(tcfg.get("focal_gamma", 2.0)), alpha=alpha)
        info["focal_alpha"] = alpha
        info["focal_gamma"] = float(tcfg.get("focal_gamma", 2.0))
    else:
        raise ValueError(f"unknown loss: {kind!r}")
    return criterion, info
