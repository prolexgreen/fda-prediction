"""AMP CUDA trainer with val-AUPRC early stopping, grad accumulation,
TensorBoard logging, and best-checkpoint persistence.

Contract with the model: TriStreamNet.forward returns raw logits (B, 1);
losses use BCEWithLogits semantics. Sigmoid is applied only when scoring.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fda_predictor.training.metrics import auprc, classification_report
from fda_predictor.utils.cuda_utils import amp_dtype, seed_worker


@dataclass
class TrainConfig:
    max_epochs: int = 30
    patience: int = 5
    lr: float = 2.0e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    amp_requested: str = "bfloat16"
    num_workers: int = 2
    pin_memory: bool = True
    metric_for_best: str = "val_auprc"  # or val_phase3_auprc
    llrd: float = 1.0  # layer-wise LR decay factor for ClinicalBERT (<1 enables)
    llrd_chemberta: float = 1.0  # same, for ChemBERTa (stage-5 unfrozen)
    llrd_molformer: float = 1.0  # same, for MoLFormer (stage-5 unfrozen)
    checkpoint_dir: Path | None = None
    tensorboard_dir: Path | None = None
    run_name: str = "run"
    seed: int = 42
    log_every_steps: int = 20


@dataclass
class TrainResult:
    best_val_auprc: float
    best_epoch: int
    stopped_epoch: int
    epochs_run: int
    history: list[dict] = field(default_factory=list)
    best_checkpoint_path: Path | None = None
    train_seconds: float = 0.0
    best_metric_name: str = "val_auprc"
    best_metric_value: float = 0.0


def _encoder_param_groups(
    net,
    base_lr: float,
    llrd_map: dict[str, float],
    weight_decay: float,
) -> list[dict]:
    """Per-encoder layer-wise LR decay groups.

    Each backbone's bottom layers get base_lr * llrd^(n_layers - layer_idx);
    fusion/heads/encoders-without-LLRD keep base_lr. Works for RoBERTa, BERT,
    and MoLFormer-style modules (all expose .embeddings + .encoder.layer).
    """
    groups: list[dict] = []
    assigned: set[int] = set()

    def add_group(p_iter, lr_mult: float, name: str):
        params = [p for p in p_iter if p.requires_grad]
        if not params:
            return
        for p in params:
            assigned.add(id(p))
        groups.append(
            {
                "params": params,
                "lr": base_lr * lr_mult,
                "weight_decay": weight_decay,
                "name": name,
            }
        )

    backbones = {
        "chemberta": getattr(net.mol_encoder_a, "backbone", None),
        "molformer": getattr(net.mol_encoder_b, "backbone", None),
        "clinicalbert": getattr(net.protocol_encoder, "backbone", None),
    }
    for enc_name, llrd in llrd_map.items():
        backbone = backbones.get(enc_name)
        if backbone is None or llrd >= 1.0 - 1e-12:
            continue
        encoder = getattr(backbone, "encoder", None)
        layers = list(getattr(encoder, "layer", [])) if encoder is not None else []
        n_layers = len(layers)
        if hasattr(backbone, "embeddings"):
            add_group(
                backbone.embeddings.parameters(), llrd ** (n_layers + 1), f"{enc_name}.embeddings"
            )
        for i, layer in enumerate(layers):
            add_group(layer.parameters(), llrd ** (n_layers - i), f"{enc_name}.layer.{i}")
        # Encoder-level extras (poolers etc.) keep base_lr unless layer-assigned.

    rest = [p for p in net.parameters() if p.requires_grad and id(p) not in assigned]
    if rest:
        groups.append({"params": rest, "lr": base_lr, "weight_decay": weight_decay, "name": "rest"})
    return groups


class Trainer:
    def __init__(self, net, criterion, config: TrainConfig, device: torch.device):
        self.net = net.to(device)
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device

        trainable = [p for p in self.net.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("no trainable parameters -- every stream is frozen incl. fusion?")

        llrd_map = {
            "clinicalbert": float(config.llrd),
            "chemberta": float(config.llrd_chemberta),
            "molformer": float(config.llrd_molformer),
        }
        if any(v < 1.0 - 1e-12 for v in llrd_map.values()):
            param_groups = _encoder_param_groups(
                self.net, config.lr, llrd_map, config.weight_decay
            )
            self.optimizer = torch.optim.AdamW(param_groups)
            print(
                "LLRD enabled: "
                + ", ".join(f"{g.get('name', '?')} lr={g['lr']:.2e}" for g in param_groups)
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable, lr=config.lr, weight_decay=config.weight_decay
            )
        self.amp_dtype, self.use_scaler = amp_dtype(device, config.amp_requested)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_scaler)
            if self.use_scaler
            else None
        )

        self.writer = None
        if config.tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            tb_path = Path(config.tensorboard_dir) / config.run_name
            tb_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_path))
            self.tb_path = tb_path
        else:
            self.tb_path = None

    # ------------------------------------------------------------------ utils

    def _lr_lambda(self, step: int) -> float:
        warmup = max(1, int(self.warmup_steps))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, self.total_steps - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        for key in ("mol_input_a", "mol_input_b", "crit_input"):
            out[key] = {
                k: v.to(self.device, non_blocking=True) for k, v in batch[key].items()
            }
        out["group_index"] = batch["group_index"].to(self.device, non_blocking=True)
        out["label"] = batch["label"].to(self.device, non_blocking=True)
        if "phase_index" in batch:
            out["phase_index"] = batch["phase_index"].to(self.device, non_blocking=True)
        if "molecule_mask" in batch:
            out["molecule_mask"] = batch["molecule_mask"].to(self.device, non_blocking=True)
        if "stock_feats" in batch:
            out["stock_feats"] = batch["stock_feats"].to(self.device, non_blocking=True)
        if "stock_mask" in batch:
            out["stock_mask"] = batch["stock_mask"].to(self.device, non_blocking=True)
        if "tabular_feats" in batch:
            out["tabular_feats"] = batch["tabular_feats"].to(self.device, non_blocking=True)
        if "tabular_mask" in batch:
            out["tabular_mask"] = batch["tabular_mask"].to(self.device, non_blocking=True)
        if "approval_label" in batch:
            out["approval_label"] = batch["approval_label"].to(self.device, non_blocking=True)
        if "approval_mask" in batch:
            out["approval_mask"] = batch["approval_mask"].to(self.device, non_blocking=True)
        return out

    def _is_multitask(self) -> bool:
        from fda_predictor.training.losses import MultiTaskLoss

        return isinstance(self.criterion, MultiTaskLoss)

    def _forward_logits(self, batch: dict[str, Any]):
        b = len(batch["label"])
        return self.net(
            mol_input_a=batch["mol_input_a"],
            mol_input_b=batch["mol_input_b"],
            group_index=batch["group_index"],
            batch_size=b,
            crit_input=batch["crit_input"],
            phase_index=batch.get("phase_index"),
            stock_feats=batch.get("stock_feats"),
            stock_mask=batch.get("stock_mask"),
            molecule_mask=batch.get("molecule_mask"),
            tabular_feats=batch.get("tabular_feats"),
            tabular_mask=batch.get("tabular_mask"),
        )

    def _compute_loss(self, batch: dict[str, Any]):
        """Returns (total_loss, success_logits). Handles single- and dual-head."""
        if self._is_multitask():
            succ_logits, appr_logits = self.net.forward_with_approval(
                mol_input_a=batch["mol_input_a"],
                mol_input_b=batch["mol_input_b"],
                group_index=batch["group_index"],
                batch_size=len(batch["label"]),
                crit_input=batch["crit_input"],
                phase_index=batch.get("phase_index"),
                stock_feats=batch.get("stock_feats"),
                stock_mask=batch.get("stock_mask"),
                molecule_mask=batch.get("molecule_mask"),
                tabular_feats=batch.get("tabular_feats"),
                tabular_mask=batch.get("tabular_mask"),
            )
            loss = self.criterion(
                succ_logits,
                batch["label"],
                appr_logits,
                batch.get("approval_label", torch.full_like(batch["label"], -1.0)),
                batch.get("approval_mask", torch.zeros_like(batch["label"])),
            )
            return loss, succ_logits
        return self.criterion(self._forward_logits(batch), batch["label"]), None

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean_loss, y_true, scores, phase_index).

        Scores are success-head probabilities; when a MultiTaskLoss criterion
        is attached, the loss includes the masked approval term and approval
        outputs are available via evaluate_dual.
        """
        self.net.eval()
        losses, ys, ps, phs = [], [], [], []
        for batch in loader:
            batch = self._to_device(batch)
            with torch.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None
            ):
                logits_or_loss = (
                    self._compute_loss(batch) if self._is_multitask() else None
                )
                if logits_or_loss is not None:
                    loss, succ_logits = logits_or_loss
                    logits = succ_logits
                else:
                    logits = self._forward_logits(batch)
                    loss = self.criterion(logits, batch["label"])
            losses.append(loss.item())
            ys.append(batch["label"].float().cpu().numpy())
            ps.append(torch.sigmoid(logits.float()).cpu().numpy())
            if "phase_index" in batch:
                phs.append(batch["phase_index"].detach().cpu().numpy())
        y = np.concatenate(ys)
        p = np.concatenate(ps).ravel()
        phase = np.concatenate(phs) if phs else np.full(len(y), -1, dtype=int)
        return float(np.mean(losses)), y, p, phase

    @torch.no_grad()
    def evaluate_dual(
        self, loader: DataLoader
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Dual-head evaluation.

        Returns (mean_loss, arrays) where arrays holds: y_success,
        p_success, y_approval (sentinel -1 where unlabeled), approval_mask,
        p_approval, phase_index.
        """
        from fda_predictor.training.losses import MultiTaskLoss

        is_dual = isinstance(self.criterion, MultiTaskLoss)
        self.net.eval()
        losses = []
        acc: dict[str, list[np.ndarray]] = {
            "y_success": [], "p_success": [],
            "y_approval": [], "approval_mask": [], "p_approval": [],
            "phase": [],
        }
        for batch in loader:
            batch = self._to_device(batch)
            with torch.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None
            ):
                if is_dual:
                    succ_logits, appr_logits = self.net.forward_with_approval(
                        mol_input_a=batch["mol_input_a"],
                        mol_input_b=batch["mol_input_b"],
                        group_index=batch["group_index"],
                        batch_size=len(batch["label"]),
                        crit_input=batch["crit_input"],
                        phase_index=batch.get("phase_index"),
                        stock_feats=batch.get("stock_feats"),
                        stock_mask=batch.get("stock_mask"),
                        molecule_mask=batch.get("molecule_mask"),
                        tabular_feats=batch.get("tabular_feats"),
                        tabular_mask=batch.get("tabular_mask"),
                    )
                    loss = self.criterion(
                        succ_logits,
                        batch["label"],
                        appr_logits,
                        batch.get("approval_label", torch.full_like(batch["label"], -1.0)),
                        batch.get("approval_mask", torch.zeros_like(batch["label"])),
                    )
                    p_app = (
                        torch.sigmoid(appr_logits.float()).cpu().numpy().ravel()
                        if appr_logits is not None
                        else np.full(len(batch["label"]), np.nan)
                    )
                else:
                    succ_logits = self._forward_logits(batch)
                    loss = self.criterion(succ_logits, batch["label"])
                    p_app = np.full(len(batch["label"]), np.nan)
            losses.append(loss.item())
            acc["y_success"].append(batch["label"].float().cpu().numpy())
            acc["p_success"].append(torch.sigmoid(succ_logits.float()).cpu().numpy().ravel())
            app_label = batch.get(
                "approval_label", torch.full_like(batch["label"], -1.0)
            )
            app_mask = batch.get("approval_mask", torch.zeros_like(batch["label"]))
            acc["y_approval"].append(app_label.float().cpu().numpy().ravel())
            acc["approval_mask"].append(app_mask.float().cpu().numpy().ravel())
            acc["p_approval"].append(p_app)
            acc["phase"].append(
                batch["phase_index"].detach().cpu().numpy()
                if "phase_index" in batch
                else np.full(len(batch["label"]), -1, dtype=int)
            )
        arrays = {k: np.concatenate(v) for k, v in acc.items()}
        return float(np.mean(losses)), arrays

    # ------------------------------------------------------------------ train

    def fit(
        self,
        train_dataset,
        val_dataset,
        collate_fn=None,
        steps_per_epoch_cap: int | None = None,
        subset_train_len: int | None = None,
    ) -> TrainResult:
        cfg = self.config
        if subset_train_len is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, range(subset_train_len))

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(8, cfg.batch_size),
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
        )

        steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
        if steps_per_epoch_cap:
            steps_per_epoch = min(steps_per_epoch, steps_per_epoch_cap)
        self.total_steps = steps_per_epoch * cfg.max_epochs
        self.warmup_steps = cfg.warmup_ratio * self.total_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)

        best_metric, best_epoch, bad_epochs = -1.0, -1, 0
        history: list[dict] = []
        global_step = 0
        ckpt_path = (
            Path(cfg.checkpoint_dir) / f"{cfg.run_name}_best.pt"
            if cfg.checkpoint_dir
            else None
        )
        t_start = time.time()
        stopped_epoch = cfg.max_epochs

        for epoch in range(1, cfg.max_epochs + 1):
            self.net.train()
            epoch_losses = []
            self.optimizer.zero_grad(set_to_none=True)

            micro = 0
            capped = False
            for batch in train_loader:
                batch = self._to_device(batch)
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    loss, _succ = self._compute_loss(batch)
                    loss = loss / cfg.grad_accum_steps
                self.scaler.scale(loss).backward() if self.use_scaler else loss.backward()

                micro += 1
                if micro % cfg.grad_accum_steps == 0:
                    if self.use_scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.net.parameters() if p.requires_grad], cfg.max_grad_norm
                    )
                    if self.use_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    epoch_losses.append(loss.item() * cfg.grad_accum_steps)

                    if self.writer and global_step % cfg.log_every_steps == 0:
                        self.writer.add_scalar("train/loss_step", epoch_losses[-1], global_step)
                    if steps_per_epoch_cap and global_step >= steps_per_epoch_cap:
                        capped = True
                        break

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            val_loss, y_val, p_val, phase_val = self.evaluate(val_loader)
            val_auprc = auprc(y_val, p_val)
            p3_mask = phase_val == 2
            if p3_mask.sum() >= 10 and len(np.unique(y_val[p3_mask])) >= 2:
                val_phase3_auprc = auprc(y_val[p3_mask], p_val[p3_mask])
            else:
                val_phase3_auprc = val_auprc

            # Dual-head approval tracking (masked to labeled val rows).
            val_p3_approval_auprc: float | None = None
            val_approval_auprc: float | None = None
            if self._is_multitask():
                _vl, arrs = self.evaluate_dual(val_loader)
                am = arrs["approval_mask"] > 0
                if am.sum() >= 10:
                    y_app = arrs["y_approval"][am]
                    p_app = arrs["p_approval"][am]
                    finite = np.isfinite(p_app)
                    if finite.any() and len(np.unique(y_app[finite])) >= 2:
                        val_approval_auprc = auprc(y_app[finite], p_app[finite])
                        ap3 = am & (arrs["phase"] == 2) & np.isfinite(arrs["p_approval"])
                        if ap3.sum() >= 10:
                            ya3 = arrs["y_approval"][ap3]
                            pa3 = arrs["p_approval"][ap3]
                            if len(np.unique(ya3)) >= 2:
                                val_p3_approval_auprc = auprc(ya3, pa3)

            metric_name = cfg.metric_for_best or "val_auprc"
            metric_candidates = {
                "val_auprc": val_auprc,
                "val_phase3_auprc": val_phase3_auprc,
                "val_p3_approval_auprc": (
                    val_p3_approval_auprc
                    if val_p3_approval_auprc is not None
                    else -1.0
                ),
                "val_approval_auprc": (
                    val_approval_auprc if val_approval_auprc is not None else -1.0
                ),
            }
            metric_value = metric_candidates.get(metric_name, val_auprc)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auprc": val_auprc,
                "val_phase3_auprc": val_phase3_auprc,
                "val_p3_approval_auprc": val_p3_approval_auprc,
                "val_approval_auprc": val_approval_auprc,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            if self.writer:
                self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("val/auprc", val_auprc, epoch)
                self.writer.add_scalar("val/phase3_auprc", val_phase3_auprc, epoch)
                if val_p3_approval_auprc is not None:
                    self.writer.add_scalar(
                        "val/p3_approval_auprc", val_p3_approval_auprc, epoch
                    )
                if val_approval_auprc is not None:
                    self.writer.add_scalar(
                        "val/approval_auprc", val_approval_auprc, epoch
                    )
                self.writer.add_scalar("train/lr", record["lr"], epoch)

            improved = metric_value > best_metric
            marker = ""
            if improved:
                best_metric, best_epoch, bad_epochs = metric_value, epoch, 0
                marker = " *best*"
                if ckpt_path is not None:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "model_state": self.net.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
                        "epoch": epoch,
                        "val_auprc": val_auprc,
                        "val_phase3_auprc": val_phase3_auprc,
                        "val_p3_approval_auprc": val_p3_approval_auprc,
                        "val_approval_auprc": val_approval_auprc,
                        "metric_for_best": metric_name,
                        "best_metric_value": metric_value,
                        "checkpoint_layout": getattr(self.net, "CHECKPOINT_LAYOUT", 1),
                        "config": {
                            "run_name": cfg.run_name,
                            "batch_size": cfg.batch_size,
                            "grad_accum_steps": cfg.grad_accum_steps,
                            "lr": cfg.lr,
                            "llrd": cfg.llrd,
                            "amp_dtype": str(self.amp_dtype),
                            "use_scaler": self.use_scaler,
                            "seed": cfg.seed,
                        },
                    }
                    torch.save(payload, ckpt_path)
            else:
                bad_epochs += 1
            print(
                f"epoch {epoch:>3} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
                f"| val_AUPRC {val_auprc:.4f} | P3_AUPRC {val_phase3_auprc:.4f}"
                + (
                    f" | P3_APPR {val_p3_approval_auprc:.4f}"
                    if val_p3_approval_auprc is not None
                    else ""
                )
                + marker,
                flush=True,
            )

            if bad_epochs >= cfg.patience:
                stopped_epoch = epoch
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no {metric_name} gain in {cfg.patience} epochs)"
                )
                break
            if capped:
                stopped_epoch = epoch
                break

        elapsed = time.time() - t_start
        result = TrainResult(
            best_val_auprc=best_metric if cfg.metric_for_best != "val_phase3_auprc" else best_metric,
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            epochs_run=len(history),
            history=history,
            best_checkpoint_path=ckpt_path,
            train_seconds=elapsed,
            best_metric_name=cfg.metric_for_best or "val_auprc",
            best_metric_value=best_metric,
        )
        if self.writer:
            self.writer.flush()
        return result

    # ------------------------------------------------------------------ resume

    def load_checkpoint(self, path: Path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scheduler_state"):
            self.scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state") and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler_state"])
        return payload
