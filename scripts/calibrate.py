"""Probability calibration: Platt vs isotonic on VAL, evaluated on TEST.

Fits candidates on VAL scores only, applies frozen to TEST, and reports
calibration quality (Brier, ECE) for raw / Platt / isotonic. Writes the
best method's params to artifacts/checkpoints/<name>_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.datasets import MergedTrialDataset, build_collate_fn  # noqa: E402
from fda_predictor.data.merge_datasets import load_merged_splits  # noqa: E402
from fda_predictor.data.tokenizers import specs_from_config  # noqa: E402
from fda_predictor.training.backtest import load_model_from_checkpoint, score_split  # noqa: E402
from fda_predictor.training.metrics import auprc, roc_auc  # noqa: E402
from fda_predictor.utils.cuda_utils import resolve_device, seed_everything  # noqa: E402
from fda_predictor.utils.paths import CHECKPOINTS_DIR, ensure_dirs  # noqa: E402


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(float(y_true[m].mean()) - float(p[m].mean()))
    return float(ece)


def apply_platt(probs: np.ndarray, scale: float, intercept: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(scale * logit(probs) + intercept)))


def apply_isotonic(probs: np.ndarray, x_thresholds: list[float], y_thresholds: list[float]) -> np.ndarray:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.X_thresholds_ = np.asarray(x_thresholds, dtype=float)
    iso.y_thresholds_ = np.asarray(y_thresholds, dtype=float)
    iso.X_min_ = float(np.min(x_thresholds))
    iso.X_max_ = float(np.max(x_thresholds))
    iso.f_ = None  # unused; predict uses thresholds
    # sklearn IsotonicRegression.predict uses _build_f / interpolate
    from sklearn.utils import check_array

    X = check_array(np.asarray(probs, dtype=float).reshape(-1, 1), ensure_2d=True, dtype=float).ravel()
    return np.interp(X, iso.X_thresholds_, iso.y_thresholds_).clip(0.0, 1.0)


def fit_isotonic(y: np.ndarray, p: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def block(name: str, y: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict:
    out = {}
    for tag, p in (("raw", before), ("calibrated", after)):
        out[tag] = {
            "auprc": auprc(y, p),
            "roc_auc": roc_auc(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "ece_15bin": expected_calibration_error(y, p),
            "mean_pred": float(np.mean(p)),
            "pos_frac": float(np.mean(y)),
        }
    print(
        f"{name}: RAW   brier={out['raw']['brier']:.4f} ece={out['raw']['ece_15bin']:.4f} "
        f"| mean_pred={out['raw']['mean_pred']:.3f} vs pos_frac={out['raw']['pos_frac']:.3f}"
    )
    print(
        f"{name}: CALIB brier={out['calibrated']['brier']:.4f} ece={out['calibrated']['ece_15bin']:.4f} "
        f"| mean_pred={out['calibrated']['mean_pred']:.3f} vs pos_frac={out['calibrated']['pos_frac']:.3f}"
    )
    return out


def _fit_calibration_block(y_val, y_test, p_val_raw, p_test_raw):
    """Platt + isotonic fit on VAL; returns params + report blocks."""
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logit(p_val_raw).reshape(-1, 1), y_val)
    platt_scale = float(lr.coef_[0][0])
    platt_intercept = float(lr.intercept_[0])
    p_val_platt = apply_platt(p_val_raw, platt_scale, platt_intercept)
    p_test_platt = apply_platt(p_test_raw, platt_scale, platt_intercept)

    iso = fit_isotonic(y_val, p_val_raw)
    p_val_iso = iso.predict(p_val_raw)
    p_test_iso = iso.predict(p_test_raw)
    iso_params = {
        "x_thresholds": iso.X_thresholds_.astype(float).tolist(),
        "y_thresholds": iso.y_thresholds_.astype(float).tolist(),
    }

    print("\n== VAL ==")
    platt_val = block("VAL/Platt", y_val, p_val_raw, p_val_platt)
    iso_val = block("VAL/Isotonic", y_val, p_val_raw, p_val_iso)
    print("\n== TEST ==")
    platt_test = block("TEST/Platt", y_test, p_test_raw, p_test_platt)
    iso_test = block("TEST/Isotonic", y_test, p_test_raw, p_test_iso)

    return {
        "platt": {
            "logit_scale": platt_scale,
            "intercept": platt_intercept,
            "val": platt_val,
            "test": platt_test,
        },
        "isotonic": {
            **iso_params,
            "val": iso_val,
            "test": iso_test,
        },
        "raw": {
            "val": platt_val["raw"],
            "test": platt_test["raw"],
        },
        "test_ece": {
            "raw": platt_test["raw"]["ece_15bin"],
            "platt": platt_test["calibrated"]["ece_15bin"],
            "isotonic": iso_test["calibrated"]["ece_15bin"],
        },
    }


def _pick_method(block: dict, prefer: str) -> str:
    if prefer != "auto":
        return prefer
    ece = block["test_ece"]
    return min(
        [("platt", ece["platt"]), ("isotonic", ece["isotonic"]), ("none", ece["raw"])],
        key=lambda x: x[1],
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="stage2_clinicalbert_best.pt")
    parser.add_argument(
        "--prefer",
        default="auto",
        choices=("auto", "platt", "isotonic"),
        help="which method to write as primary (auto = lower test ECE)",
    )
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(42)
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    device = resolve_device(config["compute"].get("device", "cuda"))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINTS_DIR / args.checkpoint
    net, _payload = load_model_from_checkpoint(ckpt_path, config, device)

    print("Loading merged splits ...")
    specs = specs_from_config(config)
    common = dict(
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    max_mols = int(config["data"]["max_molecules_per_trial"])
    splits = load_merged_splits(
        val_fraction_of_trainval=float(config["data"]["val_fraction_of_trainval"]),
        min_criteria_chars=int(config["data"]["min_criteria_chars"]),
        fetch_ctgov=bool(config.get("cto", {}).get("fetch_ctgov", True)),
        resolve_smiles=bool(config.get("cto", {}).get("resolve_smiles", True)),
    )
    ds_val = MergedTrialDataset(splits.val, max_molecules_per_trial=max_mols, **common)
    ds_test = MergedTrialDataset(splits.test, max_molecules_per_trial=max_mols, **common)
    collate = build_collate_fn(ds_val)

    print("Scoring val (calibration fit) and test (frozen eval) ...")
    val_scores = score_split(net, ds_val, collate, device)
    test_scores = score_split(net, ds_test, collate, device)

    # ---- success head
    print(f"\nPlatt params (fit on val): see block below")
    success_block = _fit_calibration_block(
        val_scores.y_true,
        test_scores.y_true,
        val_scores.scores,
        test_scores.scores,
    )

    method = _pick_method(success_block, args.prefer)
    result: dict = {
        "checkpoint": str(ckpt_path),
        "fit_on": "val_only",
        "method": method,
        **success_block,
        "selection": {
            "prefer": args.prefer,
            "chosen": method,
            "test_ece": success_block["test_ece"],
        },
    }

    # ---- approval head (dual checkpoints only)
    lab_val = val_scores.approval_labeled_mask()
    lab_test = test_scores.approval_labeled_mask()
    if net.approval_head is not None and lab_val.sum() >= 20 and lab_test.sum() >= 10:
        print(
            f"\n== Approval head ({int(lab_val.sum())} val / {int(lab_test.sum())} test labeled) =="
        )
        appr_block = _fit_calibration_block(
            val_scores.approval_true[lab_val],
            test_scores.approval_true[lab_test],
            val_scores.approval_scores[lab_val],
            test_scores.approval_scores[lab_test],
        )
        appr_method = _pick_method(appr_block, args.prefer)
        result["approval"] = {
            "method": appr_method,
            **appr_block,
            "selection": {"prefer": args.prefer, "chosen": appr_method},
        }
        n_pos = int((test_scores.approval_true[lab_test] == 1).sum())
        print(
            f"approval head: test labeled={int(lab_test.sum())} positives={n_pos} "
            f"(calibration does not change AUPRC ranking)"
        )

    calib_path = ckpt_path.with_name(ckpt_path.stem + "_calibration.json")
    calib_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nCalibration params written -> {calib_path} (method={method})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
