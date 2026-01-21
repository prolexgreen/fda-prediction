"""Training entry point.

Examples:
    uv run python scripts/train.py                       # staged recipe from config.yaml
    uv run python scripts/train.py --max-epochs 2        # quick proof run
    uv run python scripts/train.py --subset 64 --max-epochs 30   # overfit sanity gate
    uv run python scripts/train.py --run-name stage1-head-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.datasets import MergedTrialDataset, TOPTrialDataset, build_collate_fn  # noqa: E402
from fda_predictor.data.merge_datasets import load_merged_splits  # noqa: E402
from fda_predictor.data.tokenizers import specs_from_config  # noqa: E402
from fda_predictor.data.top_dataset import load_top_splits  # noqa: E402
from fda_predictor.models.multimodal_net import TriStreamNet  # noqa: E402
from fda_predictor.training.losses import build_loss  # noqa: E402
from fda_predictor.training.trainer import TrainConfig, Trainer  # noqa: E402
from fda_predictor.training.backtest import filter_compatible_state_dict  # noqa: E402
from fda_predictor.utils.cuda_utils import resolve_device, seed_everything  # noqa: E402
from fda_predictor.utils.paths import CHECKPOINTS_DIR, RUNS_DIR, ensure_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TriStreamNet on TOP Phase III")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--subset", type=int, default=None, help="train on first N trials (sanity)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps-per-epoch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None, help="override early-stopping patience")
    p.add_argument("--merged", action="store_true", help="train on TOP+CTO merged corpus")
    p.add_argument("--top-only", action="store_true", help="TOP only (ignore merged flag in config)")
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="warm-start from an older checkpoint (strict=False; e.g. stage2 for tabular bump)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    config = yaml.safe_load(Path(args.config).read_text())
    tcfg = config["training"]
    ccfg = config["compute"]

    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(ccfg.get("device", "cuda"))
    amp_req = ccfg.get("amp_dtype", "bfloat16")

    run_name = args.run_name or tcfg.get("run_name") or f"stage1_{device.type}"
    print(f"run: {run_name} | device: {device} | amp requested: {amp_req}")

    specs = specs_from_config(config)
    common = dict(
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    max_mols = int(config["data"]["max_molecules_per_trial"])
    use_merged = (args.merged or bool(config["data"].get("use_merged", False))) and not args.top_only

    print("Loading splits and building datasets ...")
    label_target = str(config.get("labels", {}).get("target", "success")).lower()
    exclude_prev = bool(config.get("labels", {}).get("exclude_previously_approved", True))

    def _prepare_frame(df):
        """Attach training target; dual mode keeps every row (dense head)."""
        out = df.copy()
        has_approval = "approval_label" in out.columns and out["approval_label"].notna().any()
        if label_target == "dual":
            if not has_approval:
                print(
                    "WARN: labels.target=dual but approval_label missing; "
                    "success-only training. Run build_approval_labels.py."
                )
            return out.reset_index(drop=True)
        if label_target == "approval" and has_approval:
            out["_train_target"] = pd.to_numeric(out["approval_label"], errors="coerce")
            if exclude_prev and "previously_approved" in out.columns:
                prev = out["previously_approved"].fillna(False).astype(bool)
                out = out.loc[~prev].copy()
            out = out.loc[out["_train_target"].notna()].copy()
            out["_train_target"] = out["_train_target"].astype(float)
            print(f"  approval-target rows: {len(out)} (from {len(df)})")
            if len(out) == 0:
                raise RuntimeError(
                    "labels.target=approval but no labeled rows remain; "
                    "run scripts/build_approval_labels.py first."
                )
        else:
            if label_target == "approval" and not has_approval:
                print(
                    "WARN: labels.target=approval but approval_label missing; "
                    "falling back to success labels. Run build_approval_labels.py."
                )
            out["_train_target"] = out["label"].astype(float)
        return out.reset_index(drop=True)

    def _dual_loss_arrays(df) -> tuple[list[float], list[float]]:
        labels = pd.to_numeric(df.get("label"), errors="coerce").astype(float).tolist()
        if "approval_label" in df.columns:
            app = pd.to_numeric(df["approval_label"], errors="coerce")
            mask = app.notna().astype(float).tolist()
            app_vals = app.fillna(-1.0).astype(float).tolist()
            if exclude_prev and "previously_approved" in df.columns:
                prev = df["previously_approved"].fillna(False).astype(bool).to_numpy()
                mask = [0.0 if p else m for p, m in zip(prev, mask)]
        else:
            app_vals = [-1.0] * len(labels)
            mask = [0.0] * len(labels)
        n_labeled = int(sum(mask))
        print(f"  dual mode: {n_labeled}/{len(labels)} rows approval-labeled")
        if n_labeled == 0:
            raise RuntimeError(
                "labels.target=dual but no approval-labeled rows; "
                "run scripts/build_approval_labels.py first."
            )
        return app_vals, mask

    import pandas as pd

    if use_merged:
        splits_m = load_merged_splits(
            val_fraction_of_trainval=float(config["data"]["val_fraction_of_trainval"]),
            min_criteria_chars=int(config["data"]["min_criteria_chars"]),
            fetch_ctgov=bool(config.get("cto", {}).get("fetch_ctgov", True)),
            resolve_smiles=bool(config.get("cto", {}).get("resolve_smiles", True)),
        )
        train_df = _prepare_frame(splits_m.train)
        val_df = _prepare_frame(splits_m.val)
        train_labels = train_df["_train_target"].tolist() if "_train_target" in train_df.columns else train_df["label"].astype(float).tolist()
        ds_train = MergedTrialDataset(train_df, max_molecules_per_trial=max_mols, **common)
        ds_val = MergedTrialDataset(val_df, max_molecules_per_trial=max_mols, **common)
    else:
        splits = load_top_splits(
            val_fraction_of_trainval=float(config["data"]["val_fraction_of_trainval"]),
            seed=seed,
            min_criteria_chars=int(config["data"]["min_criteria_chars"]),
            split_mode=config["data"].get("split_mode", "chronological"),
        )
        train_df = _prepare_frame(splits.train)
        val_df = _prepare_frame(splits.val)
        train_labels = train_df["_train_target"].tolist() if "_train_target" in train_df.columns else train_df["label"].astype(float).tolist()
        ds_train = TOPTrialDataset(train_df, max_molecules_per_trial=max_mols, **common)
        ds_val = TOPTrialDataset(val_df, max_molecules_per_trial=max_mols, **common)
    collate = build_collate_fn(ds_train)

    print("Building model (pinned checkpoints from D:\\Models) ...")
    net = TriStreamNet.from_config(config)
    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.exists():
            ckpt_path = CHECKPOINTS_DIR / args.init_checkpoint
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        compatible, skipped = filter_compatible_state_dict(net, payload["model_state"])
        missing, unexpected = net.load_state_dict(compatible, strict=False)
        print(
            f"init from {ckpt_path}: layout={payload.get('checkpoint_layout')} "
            f"loaded={len(compatible)} skipped_shape={len(skipped)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    print(net.parameter_summary())

    if label_target == "dual":
        app_vals, app_mask = _dual_loss_arrays(train_df)
        criterion, loss_info = build_loss(
            config, train_labels, train_approval_labels=app_vals, train_approval_mask=app_mask
        )
    else:
        criterion, loss_info = build_loss(config, train_labels)
    print(f"loss: {loss_info}")
    print(f"label target: {label_target} | metric_for_best: {tcfg.get('metric_for_best')}")

    train_cfg = TrainConfig(
        max_epochs=args.max_epochs or int(tcfg["max_epochs"]),
        patience=args.patience if args.patience is not None else int(tcfg["early_stopping_patience"]),
        lr=args.lr or float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        batch_size=args.batch_size or int(tcfg["batch_size"]),
        grad_accum_steps=int(tcfg["grad_accum_steps"]),
        max_grad_norm=float(tcfg["max_grad_norm"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        amp_requested=amp_req,
        num_workers=int(ccfg.get("num_workers", 2)),
        pin_memory=bool(ccfg.get("pin_memory", True)),
        metric_for_best=str(tcfg.get("metric_for_best", "val_auprc")),
        llrd=float(tcfg.get("llrd", 1.0)),
        llrd_chemberta=float(tcfg.get("llrd_chemberta", 1.0)),
        llrd_molformer=float(tcfg.get("llrd_molformer", 1.0)),
        checkpoint_dir=CHECKPOINTS_DIR,
        tensorboard_dir=RUNS_DIR,
        run_name=run_name,
        seed=seed,
    )

    trainer = Trainer(net, criterion, train_cfg, device)
    if ccfg.get("gradient_checkpointing", False):
        gc = net.enable_gradient_checkpointing()
        print(f"gradient checkpointing: {gc}")

    result = trainer.fit(
        ds_train,
        ds_val,
        collate_fn=collate,
        steps_per_epoch_cap=args.max_steps_per_epoch,
        subset_train_len=args.subset,
    )

    if args.subset:
        subset_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds_train, range(min(args.subset, len(ds_train)))),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        subset_loss, y_sub, p_sub, _ph = trainer.evaluate(subset_loader)
        from fda_predictor.training.metrics import auprc

        print(f"\n[sanity gate] train-subset: loss={subset_loss:.4f} AUPRC={auprc(y_sub, p_sub):.4f}")

    print("\n== Run summary ==")
    print(
        f"best {result.best_metric_name}: {result.best_metric_value:.4f} "
        f"(epoch {result.best_epoch}); pooled val AUPRC tracked={result.best_val_auprc:.4f}"
    )
    print(f"epochs run: {result.epochs_run} | wall time: {result.train_seconds:.1f}s")
    if result.best_checkpoint_path:
        print(f"best checkpoint: {result.best_checkpoint_path}")
    if trainer.tb_path:
        print(f"tensorboard events: {trainer.tb_path}")
        print("view: uv run tensorboard --logdir artifacts/runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
