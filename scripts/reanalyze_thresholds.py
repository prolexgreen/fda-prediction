"""Re-analyze operating points from dumped score CSVs (no GPU).

Compares legacy F1-global vs precision-targeted per-phase thresholds and
FP-penalizing decision cost on the Phase III slice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.training.backtest import (  # noqa: E402
    PHASE_PURE3_INDEX,
    evaluate_per_phase_threshold_transfer,
    evaluate_with_threshold_transfer,
    load_scored_split,
)
from fda_predictor.training.metrics import classification_report  # noqa: E402
from fda_predictor.utils.paths import RUNS_DIR  # noqa: E402


def _row(name, before, after):
    return (
        f"| {name} | thr={before['threshold']:.3f} "
        f"P={before['precision']:.3f} R={before['recall']:.3f} "
        f"TP/FP/FN={before['tp']}/{before['fp']}/{before['fn']} "
        f"| thr={after['threshold']:.3f} "
        f"P={after['precision']:.3f} R={after['recall']:.3f} "
        f"TP/FP/FN={after['tp']}/{after['fp']}/{after['fn']} "
        f"| FP {after['fp'] - before['fp']:+d} / FN {after['fn'] - before['fn']:+d} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--cost-fp", type=float, default=3.0)
    parser.add_argument("--cost-fn", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / "backtest_v5_op"
    if not (run_dir / "scores_val.csv").exists():
        candidates = sorted(
            RUNS_DIR.glob("*/scores_val.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("ERROR: no scores_val.csv; run scripts/dump_scores.py first", file=sys.stderr)
            return 1
        run_dir = candidates[0].parent
        print(f"Using run dir: {run_dir}")

    val = load_scored_split(run_dir / "scores_val.csv")
    test = load_scored_split(run_dir / "scores_test.csv")
    p3_test = test.slice_mask(PHASE_PURE3_INDEX)

    f1_proto = evaluate_with_threshold_transfer(val, test, objective="f1")
    prec_proto = evaluate_per_phase_threshold_transfer(
        val, test, objective="precision", min_precision=args.min_precision
    )
    cost_proto = evaluate_per_phase_threshold_transfer(
        val, test, objective="cost", cost_fp=args.cost_fp, cost_fn=args.cost_fn
    )

    f1_thr = float(f1_proto["threshold"])
    prec_thr = float(prec_proto["thresholds"].get("III", prec_proto["global_threshold"]))
    cost_thr = float(cost_proto["thresholds"].get("III", cost_proto["global_threshold"]))

    baseline = classification_report(test.y_true[p3_test], test.scores[p3_test], f1_thr)
    # Also report original backtest_v5 threshold if known
    legacy_thr = 0.28322964906692505
    legacy = classification_report(test.y_true[p3_test], test.scores[p3_test], legacy_thr)
    after_prec = classification_report(test.y_true[p3_test], test.scores[p3_test], prec_thr)
    after_cost = classification_report(test.y_true[p3_test], test.scores[p3_test], cost_thr)

    thresholds_out = {
        "objective": "precision",
        "min_precision": args.min_precision,
        "cost_fp": args.cost_fp,
        "cost_fn": args.cost_fn,
        "global_threshold": prec_proto["global_threshold"],
        "global_f1_threshold": f1_thr,
        "per_phase": prec_proto["thresholds"],
        "per_phase_cost_fp_heavy": cost_proto["thresholds"],
        "abstain_band": 0.05,
    }
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds_out, indent=2), encoding="utf-8")

    payload = {
        "run_dir": str(run_dir),
        "phase3_n_test": int(p3_test.sum()),
        "legacy_v5_threshold": legacy_thr,
        "f1_global_threshold": f1_thr,
        "phase3_precision_threshold": prec_thr,
        "phase3_cost_fp_heavy_threshold": cost_thr,
        "phase3_test_legacy_v5": legacy,
        "phase3_test_f1": baseline,
        "phase3_test_precision_target": after_prec,
        "phase3_test_cost_fp_heavy": after_cost,
        "thresholds": thresholds_out,
    }
    out = run_dir / "operating_point_comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase III operating-point comparison",
        "",
        f"n_test Phase III = {int(p3_test.sum())} | base rate = {legacy['base_rate']:.3f} | AUPRC = {legacy['auprc']:.3f}",
        "",
        "| setting | before (F1-global) | after | delta |",
        "|---|---|---|---|",
        _row(f"precision>={args.min_precision:.2f}", baseline, after_prec),
        _row(f"cost fp={args.cost_fp:g}/fn={args.cost_fn:g}", baseline, after_cost),
        _row("vs original v5 thr=0.283", legacy, after_prec),
        "",
        f"Per-phase precision thresholds: `{prec_proto['thresholds']}`",
        f"Artifacts: `{out.name}`, `thresholds.json`",
    ]
    md = run_dir / "operating_point_comparison.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
