"""Full backtest report driver: classical baselines + REPORT.md + PR curves.

Reads the score dumps written by dump_scores.py (`<run_dir>/scores_val.csv`,
`scores_test.csv`, `thresholds.json`, `metrics.json`), recomputes per-slice
metrics at transferred thresholds, and writes a markdown report.
Use dump_scores.py first (GPU scoring is slow) — this script is CPU-only.

Back-compat: `scripts/evaluate.py --merged --checkpoint X --run-name Y` first
runs the dump if scores are missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.training.backtest import (  # noqa: E402
    PHASE_PURE3_INDEX,
    ScoredSplit,
    load_scored_split,
    tune_threshold_for_precision,
)
from fda_predictor.training.metrics import (  # noqa: E402
    auprc,
    classification_report,
    roc_auc,
    tune_threshold_on_val,
)
from fda_predictor.utils.paths import CHECKPOINTS_DIR, RUNS_DIR  # noqa: E402


def _fmt(v, digits=4):
    return "n/a" if v is None else f"{v:.{digits}f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--run-name", default="backtest")
    p.add_argument("--merged", action="store_true")
    p.add_argument("--top-only", action="store_true")
    p.add_argument("--min-precision", type=float, default=0.80)
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_name
    val_csv = run_dir / "scores_val.csv"
    test_csv = run_dir / "scores_test.csv"

    if not (val_csv.exists() and test_csv.exists()):
        ckpt = args.checkpoint or (
            "stage2_clinicalbert_best.pt" if not args.merged else "stage4_dual_head_best.pt"
        )
        print(f"scores missing; running dump_scores first (checkpoint={ckpt})")
        dump = ROOT / "scripts" / "dump_scores.py"
        subprocess.check_call(
            [
                sys.executable,
                str(dump),
                "--checkpoint",
                ckpt,
                "--run-name",
                args.run_name,
                "--min-precision",
                str(args.min_precision),
            ],
            cwd=str(ROOT),
        )

    val = load_scored_split(val_csv)
    test = load_scored_split(test_csv)

    m_val_p3 = val.slice_mask(PHASE_PURE3_INDEX)
    m_test_p3 = test.slice_mask(PHASE_PURE3_INDEX)

    # Global F1 threshold (tuned on val) + Phase-III precision-target threshold
    thr_f1, _ = tune_threshold_on_val(val.y_true, val.scores)
    thr_p3_prec, _ = tune_threshold_for_precision(
        val.y_true[m_val_p3], val.scores[m_val_p3], min_precision=args.min_precision
    )

    overall = {
        "val": classification_report(val.y_true, val.scores, thr_f1),
        "test": classification_report(test.y_true, test.scores, thr_f1),
    }
    phase3 = {
        "val_global": classification_report(val.y_true[m_val_p3], val.scores[m_val_p3], thr_f1),
        "test_global": classification_report(test.y_true[m_test_p3], test.scores[m_test_p3], thr_f1),
        "val_phase": classification_report(val.y_true[m_val_p3], val.scores[m_val_p3], thr_p3_prec),
        "test_phase": classification_report(test.y_true[m_test_p3], test.scores[m_test_p3], thr_p3_prec),
    }

    lines = [
        f"# Backtest Report - {args.run_name}",
        "",
        f"- overall F1 threshold (VAL-tuned): **{thr_f1:.4f}**",
        f"- Phase-III precision>={args.min_precision:.2f} threshold (VAL-tuned): **{thr_p3_prec:.4f}**",
        f"- val n={len(val.y_true)}, test n={len(test.y_true)}",
        f"- Phase III n: val={int(m_val_p3.sum())}, test={int(m_test_p3.sum())}",
        "",
        "## Overall",
        "",
        "| era | AUPRC | ROC-AUC | precision | recall | F1 | TP/FP/FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for era, rep in (("val", overall["val"]), ("test", overall["test"])):
        lines.append(
            f"| {era} | {rep['auprc']:.4f} | {_fmt(rep['roc_auc'])} | {rep['precision']:.4f} "
            f"| {rep['recall']:.4f} | {rep['f1_at_threshold']:.4f} | "
            f"{rep['tp']}/{rep['fp']}/{rep['fn']} |"
        )
    lines += [
        "",
        "## Phase III slice",
        "",
        "| setting | AUPRC | precision | recall | specificity | TP/FP/FN |",
        "|---|---|---|---|---|---|",
    ]
    for name, rep in (
        ("VAL @global-F1", phase3["val_global"]),
        ("TEST @global-F1", phase3["test_global"]),
        ("VAL @P3-precision", phase3["val_phase"]),
        ("TEST @P3-precision", phase3["test_phase"]),
    ):
        lines.append(
            f"| {name} | {rep['auprc']:.4f} | {rep['precision']:.4f} | {rep['recall']:.4f} "
            f"| {_fmt(rep['specificity'])} | {rep['tp']}/{rep['fp']}/{rep['fn']} |"
        )

    if test.data_source is not None:
        lines += ["", "## Per-source (TEST)", "", "| source | n | AUPRC | ROC-AUC |", "|---|---|---|---|"]
        for src in ("TOP", "CTO"):
            mask = test.source_mask(src)
            if mask.any():
                lines.append(
                    f"| {src} | {int(mask.sum())} | {auprc(test.y_true[mask], test.scores[mask]):.4f} "
                    f"| {_fmt(roc_auc(test.y_true[mask], test.scores[mask]))} |"
                )

    report_path = run_dir / "REPORT.md"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
