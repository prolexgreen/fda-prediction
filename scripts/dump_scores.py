"""Fast score dump from existing merged parquet (no stock re-fetch).

Uses data/processed/merged_trials.parquet split column + a trained
checkpoint. Writes scores_val.csv / scores_test.csv / thresholds.json
under artifacts/runs/<run-name>/.
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
    dump_scored_split,
    evaluate_per_phase_threshold_transfer,
    evaluate_with_threshold_transfer,
    load_model_from_checkpoint,
    score_split,
)
from fda_predictor.utils.cuda_utils import resolve_device, seed_everything  # noqa: E402
from fda_predictor.utils.paths import (  # noqa: E402
    CHECKPOINTS_DIR,
    MERGED_PROCESSED_PARQUET,
    RUNS_DIR,
    ensure_dirs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="stage2_clinicalbert_best.pt")
    parser.add_argument("--run-name", default="backtest_v5_op")
    parser.add_argument(
        "--threshold-objective",
        default=None,
        choices=("cost", "f1", "precision"),
    )
    parser.add_argument("--cost-fp", type=float, default=None)
    parser.add_argument("--cost-fn", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
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

    thr_cfg = config.get("threshold", {})
    thr_objective = args.threshold_objective or thr_cfg.get("objective", "precision")
    cost_fp = float(
        args.cost_fp if args.cost_fp is not None else thr_cfg.get("cost_fp", 1.0)
    )
    cost_fn = float(
        args.cost_fn if args.cost_fn is not None else thr_cfg.get("cost_fn", 2.0)
    )
    min_precision = float(
        args.min_precision
        if args.min_precision is not None
        else thr_cfg.get("min_precision", 0.80)
    )

    print(f"Loading {MERGED_PROCESSED_PARQUET} ...", flush=True)
    pooled = pd.read_parquet(MERGED_PROCESSED_PARQUET)
    # Ensure tabular columns exist (idempotent)
    if "tabular_feats" not in pooled.columns:
        pooled, _ = attach_tabular_features(pooled, fit_stats=True)
    approval_path = MERGED_PROCESSED_PARQUET.parent / "approval_labels.parquet"
    if approval_path.exists() and "approval_label" not in pooled.columns:
        side = pd.read_parquet(approval_path)
        keep = [c for c in side.columns if c != "nctid" or c == "nctid"]
        pooled = pooled.merge(side[keep].drop_duplicates("nctid"), on="nctid", how="left")

    from fda_predictor.training.backtest import ScoredSplit as _ScoredSplit  # noqa: F401

    val_df = pooled[pooled["split"] == "val"].reset_index(drop=True)
    test_df = pooled[pooled["split"] == "test"].reset_index(drop=True)
    print(f"val={len(val_df)} test={len(test_df)}", flush=True)

    specs = specs_from_config(config)
    common = dict(
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    max_mols = int(config["data"]["max_molecules_per_trial"])
    ds_val = MergedTrialDataset(val_df, max_molecules_per_trial=max_mols, **common)
    ds_test = MergedTrialDataset(test_df, max_molecules_per_trial=max_mols, **common)
    collate = build_collate_fn(ds_val)

    print("Scoring ...", flush=True)
    val_scores = score_split(net, ds_val, collate, device)
    test_scores = score_split(net, ds_test, collate, device)

    f1_proto = evaluate_with_threshold_transfer(val_scores, test_scores, objective="f1")
    tuned_proto = evaluate_per_phase_threshold_transfer(
        val_scores,
        test_scores,
        objective=thr_objective,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        min_precision=min_precision,
    )

    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_scored_split(val_scores, out_dir / "scores_val.csv")
    dump_scored_split(test_scores, out_dir / "scores_test.csv")

    # Dual-head approval thresholds: tune on the model's approval head output,
    # restricted to rows the approval head actually scored (mask=1). Must run
    # before thresholds.json / metrics.json assembly below.
    def _approval_slices(split: ScoredSplit) -> ScoredSplit | None:
        lab = split.approval_labeled_mask()
        if split.approval_scores is None or lab.sum() < 20:
            return None
        return _ScoredSplit(
            nctids=[n for n, k in zip(split.nctids, lab) if k],
            y_true=split.approval_true[lab].astype(float),
            scores=split.approval_scores[lab].astype(float),
            phase_index=split.phase_index[lab],
            data_source=split.data_source[lab] if split.data_source is not None else None,
        )

    def _approval_test_slice(split: ScoredSplit) -> ScoredSplit | None:
        lab = split.approval_labeled_mask()
        if split.approval_scores is None or lab.sum() < 5:
            return None
        return _ScoredSplit(
            nctids=[n for n, k in zip(split.nctids, lab) if k],
            y_true=split.approval_true[lab].astype(float),
            scores=split.approval_scores[lab].astype(float),
            phase_index=split.phase_index[lab],
            data_source=split.data_source[lab] if split.data_source is not None else None,
        )

    app_val_slice = _approval_slices(val_scores)
    app_test_slice = _approval_test_slice(test_scores)
    approval_thresholds: dict = {}
    appr_proto = None
    if app_val_slice is not None and app_test_slice is not None:
        appr_proto = evaluate_per_phase_threshold_transfer(
            app_val_slice,
            app_test_slice,
            objective="precision",
            min_precision=float(config.get("threshold", {}).get("min_precision", 0.80)),
        )
        approval_thresholds = {
            "objective": "precision",
            "global_threshold": appr_proto["global_threshold"],
            "per_phase": appr_proto["thresholds"],
        }

    thresholds = {
        "objective": thr_objective,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "min_precision": min_precision,
        "global_threshold": tuned_proto["global_threshold"],
        "global_f1_threshold": f1_proto["threshold"],
        "per_phase": tuned_proto["thresholds"],
        "abstain_band": float(config.get("threshold", {}).get("abstain_band", 0.05)),
    }
    if approval_thresholds:
        thresholds["approval_per_phase"] = approval_thresholds["per_phase"]
        thresholds["approval_global"] = approval_thresholds["global_threshold"]
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    metrics = {
        "checkpoint": str(ckpt),
        "checkpoint_layout": payload.get("checkpoint_layout"),
        "checkpoint_epoch": payload.get("epoch"),
        "threshold_transfer": thresholds,
        "per_phase_tuned": {
            "objective": thr_objective,
            "thresholds": tuned_proto["thresholds"],
            "test": tuned_proto["test"],
            "val": tuned_proto["val"],
        },
        "model_f1_threshold": {
            "val": f1_proto["val_report"],
            "test": f1_proto["test_report"],
        },
        "model": {
            "val": tuned_proto["global"]["val_report"],
            "test": tuned_proto["global"]["test_report"],
        },
    }

    if "approval_label" in val_df.columns:
        from fda_predictor.data.approval_labels import approval_label_coverage
        from fda_predictor.training.metrics import classification_report as cr

        def _approval_block(scores, frame):
            y, s, ph = [], [], []
            labeled = scores.approval_labeled_mask()
            for take, nct in zip(labeled, scores.nctids):
                if not take:
                    continue
                row = frame.loc[frame["nctid"].astype(str) == str(nct)]
                al = row["approval_label"].iloc[0] if not row.empty else np.nan
                if pd.isna(al):
                    continue
                idx = scores.nctids.index(nct)
                y.append(float(al))
                s.append(float(scores.approval_scores[idx]))
                ph.append(int(scores.phase_index[idx]))
            if not y:
                return {"n": 0}
            y_arr = np.array(y)
            s_arr = np.array(s)
            ph_arr = np.array(ph)
            p3_thr = float(tuned_proto["thresholds"].get("III", tuned_proto["global_threshold"]))
            p3 = ph_arr == 2
            block = {
                "coverage": approval_label_coverage(frame),
                "test_all": cr(y_arr, s_arr, tuned_proto["global_threshold"]),
                "test_phase3": cr(y_arr[p3], s_arr[p3], p3_thr) if p3.any() else {"n": 0},
            }
            return block

        metrics["approval_target"] = {
            "val": _approval_block(val_scores, val_df),
            "test": _approval_block(test_scores, test_df),
        }
        if appr_proto is not None:
            metrics["approval_target"]["approval_head"] = {
                "test_phase3": appr_proto["test"].get("III"),
            }

    if app_val_slice is not None and app_test_slice is not None and appr_proto is not None:
        metrics["approval_head"] = {
            "n_val_labeled": int(len(app_val_slice.y_true)),
            "n_test_labeled": int(len(app_test_slice.y_true)),
            "thresholds": approval_thresholds,
            "test_phase3": appr_proto["test"].get("III"),
            "test_all_phases": appr_proto["global"]["test_report"],
            "val_all_phases": appr_proto["global"]["val_report"],
        }
        r3 = appr_proto["test"].get("III")
        if r3:
            print(
                f"Approval head TEST P3 @prec-thr {appr_proto['thresholds'].get('III', float('nan')):.3f}: "
                f"AUPRC={r3['auprc']:.4f} P={r3['precision']:.3f} R={r3['recall']:.3f} "
                f"TP/FP/FN={r3['tp']}/{r3['fp']}/{r3['fn']} (n={r3.get('n_trials', '?')})",
                flush=True,
            )

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    print(f"Wrote scores + thresholds -> {out_dir}", flush=True)
    print(f"per-phase thresholds ({thr_objective}): {tuned_proto['thresholds']}", flush=True)
    if "III" in tuned_proto["test"]:
        r = tuned_proto["test"]["III"]
        print(
            f"Phase III TEST @{thr_objective}: P={r['precision']:.3f} R={r['recall']:.3f} "
            f"TP/FP/FN={r['tp']}/{r['fp']}/{r['fn']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
