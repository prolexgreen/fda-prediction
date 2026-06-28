"""Reproduce backtest eval degeneracy: compare fp16-default autocast vs bf16."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.datasets import MergedTrialDataset, build_collate_fn
from fda_predictor.data.tokenizers import specs_from_config
from fda_predictor.training.backtest import load_model_from_checkpoint
from fda_predictor.utils.cuda_utils import resolve_device, seed_everything


def main() -> int:
    seed_everything(42)
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    device = resolve_device("cuda")

    ckpt_path = Path("artifacts/checkpoints/stage1_merged_best_best.pt")
    if not ckpt_path.exists():
        print(f"checkpoint missing: {ckpt_path}")
        return 1
    net, _ = load_model_from_checkpoint(ckpt_path, config, device)

    merged = pd.read_parquet("data/processed/merged_trials.parquet")
    val_df = merged[merged["split"] == "val"].head(32).reset_index(drop=True)
    specs = specs_from_config(config)
    common = dict(
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    ds = MergedTrialDataset(
        val_df,
        max_molecules_per_trial=int(config["data"]["max_molecules_per_trial"]),
        **common,
    )
    collate = build_collate_fn(ds)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)

    def run(dtype):
        logits_all = []
        with torch.no_grad():
            for batch in loader:
                mol_a = {k: v.to(device) for k, v in batch["mol_input_a"].items()}
                mol_b = {k: v.to(device) for k, v in batch["mol_input_b"].items()}
                crit = {k: v.to(device) for k, v in batch["crit_input"].items()}
                if dtype is None:
                    ctx = torch.autocast(device_type="cuda", enabled=False)
                else:
                    ctx = torch.autocast(device_type="cuda", dtype=dtype)
                with ctx:
                    lg = net(
                        mol_input_a=mol_a,
                        mol_input_b=mol_b,
                        group_index=batch["group_index"].to(device),
                        batch_size=len(batch["label"]),
                        crit_input=crit,
                        phase_index=batch["phase_index"].to(device),
                        stock_feats=batch["stock_feats"].to(device),
                        stock_mask=batch["stock_mask"].to(device),
                        molecule_mask=batch["molecule_mask"].to(device),
                    )
                logits_all.append(lg.float().cpu().numpy().ravel())
        return np.concatenate(logits_all)

    for label, dtype in (
        ("fp32 (no autocast)", None),
        ("bf16", torch.bfloat16),
        ("fp16 (backtest default)", torch.float16),
    ):
        lg = run(dtype)
        n_nan = int(np.isnan(lg).sum())
        finite = np.where(np.isfinite(lg), lg, 0.0)
        p = 1 / (1 + np.exp(-finite))
        spread = float(p.max() - p.min())
        print(
            f"{label:>26}: nan={n_nan} | logit range "
            f"[{finite.min():.4f}, {finite.max():.4f}] | prob spread {spread:.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
