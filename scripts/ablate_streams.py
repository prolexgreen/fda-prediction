"""Stream-knockout ablation: measure each stream's contribution to AUPRC.

Scores a checkpoint on a frozen split twice per stream: unmodified vs that
stream knocked out by forcing its mask to zero (chemistry: molecule_mask=0
so mol_null_* substitutes; stock: stock_mask=0; tabular: tabular_mask=0 so
the learned missing embedding substitutes).

Writes artifacts/runs/ablation_<tag>/ablation.json with per-stream and
Phase III AUPRC deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.datasets import MergedTrialDataset, build_collate_fn  # noqa: E402
from fda_predictor.data.tabular_features import attach_tabular_features  # noqa: E402
from fda_predictor.data.tokenizers import specs_from_config  # noqa: E402
from fda_predictor.training.backtest import (  # noqa: E402
    PHASE_PURE3_INDEX,
    load_model_from_checkpoint,
    score_split,
)
from fda_predictor.training.metrics import auprc, roc_auc  # noqa: E402
from fda_predictor.utils.cuda_utils import resolve_device, seed_everything  # noqa: E402
from fda_predictor.utils.paths import (  # noqa: E402
    CHECKPOINTS_DIR,
    MERGED_PROCESSED_PARQUET,
    RUNS_DIR,
    ensure_dirs,
)

# mask key -> which stream it disables (tabular_mask is per-feature: zero the
# whole vector and the tabular encoder substitutes its learned nulls).
KNOCKOUTS: dict[str, str] = {
    "chemistry": "molecule_mask",
    "stock": "stock_mask",
    "tabular": "tabular_mask",
}

# Named feature blocks inside the 36-dim tabular vector (order fixed by
# TABULAR_FEATURE_NAMES). Indices computed from names at import time.
from fda_predictor.data.tabular_features import TABULAR_FEATURE_NAMES  # noqa: E402

_TAB_IDX = {name: i for i, name in enumerate(TABULAR_FEATURE_NAMES)}


def _block_indices(prefixes: tuple[str, ...]) -> list[int]:
    return [i for name, i in _TAB_IDX.items() if name.startswith(prefixes)]


TABULAR_BLOCKS: dict[str, list[int]] = {
    "block_legacy": _block_indices(("enrollment_log1p", "number_of_arms", "has_dmc", "source_class_industry", "molecule_present", "n_prior_drug_approvals", "is_fda_regulated_drug")),
    "block_chemistry": _block_indices(("chem_",)),
    "block_modality": _block_indices(("mod_",)),
    "block_mechanism": _block_indices(("mech_", "act_")),
    "block_sponsor": _block_indices(("sponsor_",)),
    "block_kg": _block_indices(("kg_",)),
}


class MaskedDataset:
    """Dataset wrapper that forces a mask key to zero for every sample."""

    def __init__(self, dataset, mask_key: str):
        self._ds = dataset
        self._mask_key = mask_key

    def __len__(self) -> int:
        return len(self._ds)

    def __getattr__(self, name: str):  # tokenizers read through: tok_a/tok_b/...
        ds = self.__dict__.get("_ds")  # avoid recursion during copy/pickle
        if ds is None:
            raise AttributeError(name)
        return getattr(ds, name)

    def __getitem__(self, idx: int) -> dict:
        item = self._ds[idx]
        mask = item[self._mask_key]
        item[self._mask_key] = torch.zeros_like(mask)
        return item


class TabularBlockMaskedDataset:
    """Dataset wrapper that zeroes selected feature indices of the tabular
    vector AND its mask (mask=0 => tabular encoder substitutes learned nulls)."""

    def __init__(self, dataset, indices: list[int]):
        self._ds = dataset
        self._idx = indices

    def __len__(self) -> int:
        return len(self._ds)

    def __getattr__(self, name: str):
        ds = self.__dict__.get("_ds")
        if ds is None:
            raise AttributeError(name)
        return getattr(ds, name)

    def __getitem__(self, idx: int) -> dict:
        item = self._ds[idx]
        for j in self._idx:
            item["tabular_feats"][j] = 0.0
            item["tabular_mask"][j] = 0.0
        return item


def _auprc_bundle(scores) -> dict:
    y = scores.y_true
    p = scores.scores
    out = {
        "auprc": auprc(y, p),
        "roc_auc": roc_auc(y, p),
        "n": int(len(y)),
        "pos_frac": float(np.mean(y)) if len(y) else None,
    }
    m3 = scores.phase_index == PHASE_PURE3_INDEX
    if m3.sum() >= 10 and len(np.unique(y[m3])) >= 2:
        out["phase3_auprc"] = auprc(y[m3], p[m3])
        out["phase3_roc_auc"] = roc_auc(y[m3], p[m3])
        out["phase3_n"] = int(m3.sum())
    else:
        out["phase3_auprc"] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="stage4_dual_head_best.pt")
    parser.add_argument("--run-name", default="ablation_v7")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--blocks",
        action="store_true",
        help="also ablate named tabular feature blocks (block_chemistry/mechanism/sponsor/modality/legacy)",
    )
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(42)
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    device = resolve_device(config["compute"].get("device", "cuda"))

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        ckpt = CHECKPOINTS_DIR / args.checkpoint
    print(f"Loading checkpoint: {ckpt}", flush=True)
    net, payload = load_model_from_checkpoint(ckpt, config, device)

    print(f"Loading {MERGED_PROCESSED_PARQUET} ...", flush=True)
    pooled = pd.read_parquet(MERGED_PROCESSED_PARQUET)
    if "tabular_feats" not in pooled.columns:
        pooled, _ = attach_tabular_features(pooled, fit_stats=True)
    approval_path = MERGED_PROCESSED_PARQUET.parent / "approval_labels.parquet"
    if approval_path.exists() and "approval_label" not in pooled.columns:
        side = pd.read_parquet(approval_path)
        pooled = pooled.merge(side.drop_duplicates("nctid"), on="nctid", how="left")

    split_name = args.split
    frame = pooled[pooled["split"] == split_name].reset_index(drop=True)
    print(f"split={split_name} n={len(frame)}", flush=True)

    specs = specs_from_config(config)
    common = dict(
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    max_mols = int(config["data"]["max_molecules_per_trial"])
    ds = MergedTrialDataset(frame, max_molecules_per_trial=max_mols, **common)
    collate = build_collate_fn(ds)

    results: dict[str, dict] = {"baseline": {}}
    print("Scoring baseline ...", flush=True)
    base = score_split(net, ds, collate, device, batch_size=args.batch_size)
    results["baseline"] = _auprc_bundle(base)

    # Fraction of rows where each stream actually carries real data, so deltas
    # are interpretable (a stream near-fully masked can't contribute anyway).
    coverage: dict[str, float] = {}
    for stream, key in KNOCKOUTS.items():
        if key == "tabular_mask":
            vals = frame["tabular_mask"].apply(lambda m: float(np.mean(m)) if m is not None else 0.0)
            coverage[stream] = float((vals > 0).mean())
        elif key in frame.columns:
            coverage[stream] = float(pd.to_numeric(frame[key], errors="coerce").fillna(0).mean())

    for stream, key in KNOCKOUTS.items():
        print(f"Scoring knockout: {stream} ({key}=0) ...", flush=True)
        knocked = MaskedDataset(ds, key)
        scores = score_split(net, knocked, collate, device, batch_size=args.batch_size)
        results[stream] = _auprc_bundle(scores)
        base_p3 = results["baseline"].get("phase3_auprc")
        ko_p3 = results[stream].get("phase3_auprc")
        results[stream]["delta_auprc"] = results["baseline"]["auprc"] - results[stream]["auprc"]
        results[stream]["delta_phase3_auprc"] = (
            (base_p3 - ko_p3) if (base_p3 is not None and ko_p3 is not None) else None
        )
        results[stream]["mask_coverage_frac"] = coverage.get(stream)
        print(
            f"  {stream}: P3 AUPRC "
            f"{base_p3:.4f} -> {ko_p3:.4f} "
            f"(delta {results[stream]['delta_phase3_auprc']:+.4f})",
            flush=True,
        )

    # Tabular feature-block ablation (stage-7 groups).
    if args.blocks:
        if not {"tabular_feats", "tabular_mask"} <= set(frame.columns):
            print("No tabular columns; skipping block ablation.", flush=True)
        else:
            results["blocks"] = {}
            base_p3 = results["baseline"].get("phase3_auprc")
            for block_name, idxs in TABULAR_BLOCKS.items():
                print(f"Scoring block knockout: {block_name} ({idxs}) ...", flush=True)
                knocked = TabularBlockMaskedDataset(ds, idxs)
                scores = score_split(net, knocked, collate, device, batch_size=args.batch_size)
                got = _auprc_bundle(scores)
                p_now = got.get("phase3_auprc")
                results["blocks"][block_name] = {
                    **got,
                    "indices": idxs,
                    "delta_phase3_auprc": (base_p3 - p_now) if base_p3 is not None and p_now is not None else None,
                }
                print(
                    f"  {block_name}: P3 AUPRC {base_p3:.4f} -> "
                    f"{p_now:.4f} (delta {results['blocks'][block_name]['delta_phase3_auprc']:+.4f})",
                    flush=True,
                )

    out = {
        "checkpoint": str(ckpt),
        "checkpoint_layout": payload.get("checkpoint_layout"),
        "split": split_name,
        "streams": results,
    }
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'ablation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
