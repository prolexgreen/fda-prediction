"""Compare backtest_v5 / v5_op / v6 (approval-only retrain) / v7_dual (dual-head)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.utils.paths import RUNS_DIR


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _phase3_test(metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    p3 = metrics.get("phase3_slice", {}).get("test")
    if p3:
        return p3
    tuned = metrics.get("per_phase_tuned", {}).get("test", {})
    return tuned.get("III")


def main() -> int:
    v5 = RUNS_DIR / "backtest_v5" / "metrics.json"
    v5op = RUNS_DIR / "backtest_v5_op" / "metrics.json"
    v6 = RUNS_DIR / "backtest_v6" / "metrics.json"
    v7 = RUNS_DIR / "backtest_v7_dual" / "metrics.json"
    v8 = RUNS_DIR / "backtest_v8_chem" / "metrics.json"
    op_cmp = RUNS_DIR / "backtest_v5_op" / "operating_point_comparison.json"

    m5 = _load_metrics(v5)
    mop = _load_metrics(v5op)
    m6 = _load_metrics(v6)
    m7 = _load_metrics(v7)
    m8 = _load_metrics(v8)
    cmp = _load_metrics(op_cmp) if op_cmp.exists() else None

    def _abl(name: str) -> dict | None:
        p = RUNS_DIR / name / "ablation.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        streams = d.get("streams", {})
        return {
            "chemistry_delta_phase3_auprc": (streams.get("chemistry") or {}).get(
                "delta_phase3_auprc"
            ),
            "stock_delta_phase3_auprc": (streams.get("stock") or {}).get(
                "delta_phase3_auprc"
            ),
            "tabular_delta_phase3_auprc": (streams.get("tabular") or {}).get(
                "delta_phase3_auprc"
            ),
        }

    out = {
        "backtest_v5": {
            "path": str(v5),
            "phase3_test_success": _phase3_test(m5),
            "threshold": (m5 or {}).get("threshold_transfer", {}).get("threshold"),
        },
        "backtest_v5_op": {
            "path": str(v5op),
            "phase3_test_precision": (
                (cmp or {}).get("phase3_test_precision_target")
                or _phase3_test(mop)
            ),
            "thresholds": (mop or {}).get("threshold_transfer", {}).get("per_phase"),
        },
        "backtest_v6": {
            "path": str(v6),
            "checkpoint": (m6 or {}).get("checkpoint"),
            "phase3_test_success": _phase3_test(m6),
            "phase3_test_approval": (
                (m6 or {}).get("approval_target", {}).get("test", {}).get("test_phase3")
            ),
            "thresholds": (m6 or {}).get("threshold_transfer", {}).get("per_phase"),
        },
        "backtest_v7_dual": {
            "path": str(v7),
            "checkpoint": (m7 or {}).get("checkpoint"),
            "phase3_test_success": _phase3_test(m7),
            "approval_head_phase3": (m7 or {}).get("approval_head", {}).get("test_phase3"),
            "success_thresholds": (m7 or {}).get("threshold_transfer", {}).get("per_phase"),
            "approval_thresholds": (m7 or {}).get("threshold_transfer", {}).get(
                "approval_per_phase"
            ),
        },
        "backtest_v8_chem": {
            "path": str(v8),
            "checkpoint": (m8 or {}).get("checkpoint"),
            "phase3_test_success": _phase3_test(m8),
            "approval_head_phase3": (m8 or {}).get("approval_head", {}).get("test_phase3"),
            "success_thresholds": (m8 or {}).get("threshold_transfer", {}).get("per_phase"),
        },
        "ablation_v7": _abl("ablation_v7"),
        "ablation_v8": _abl("ablation_v8"),
        "chemistry_did_learn": None,
        "operating_point_comparison_v5op": cmp,
        "delta_v6_vs_v5_phase3_success": None,
        "delta_v6_vs_v5_phase3_approval": None,
        "delta_v7_vs_v5op_phase3_success": None,
        "delta_v8_vs_v7_phase3_success": None,
        "v7_gate_vs_v5op": None,
    }

    p5 = out["backtest_v5"]["phase3_test_success"]
    p6 = out["backtest_v6"]["phase3_test_success"]
    if p5 and p6:
        out["delta_v6_vs_v5_phase3_success"] = {
            k: round(float(p6.get(k, 0)) - float(p5.get(k, 0)), 4)
            for k in ("auprc", "roc_auc", "precision", "recall", "f1")
            if k in p5 and k in p6
        }

    app6 = out["backtest_v6"]["phase3_test_approval"]
    if app6 and app6.get("n", 0) > 0:
        out["delta_v6_vs_v5_phase3_approval"] = {
            "n_labeled": app6.get("n"),
            "auprc": app6.get("auprc"),
            "roc_auc": app6.get("roc_auc"),
            "precision_at_phase_thr": app6.get("precision"),
            "recall_at_phase_thr": app6.get("recall"),
        }

    # v7 dual-head vs v5_op gate (plan gate: success AUPRC within ~0.03,
    # precision >=0.80 operating point, recall >= v5_op's 0.68).
    p5op = out["backtest_v5_op"]["phase3_test_precision"]
    p7 = out["backtest_v7_dual"]["phase3_test_success"]
    if p5op and p7:
        delta = {
            k: round(float(p7.get(k, 0)) - float(p5op.get(k, 0)), 4)
            for k in ("auprc", "roc_auc", "precision", "recall", "f1")
            if k in p5op and k in p7
        }
        out["delta_v7_vs_v5op_phase3_success"] = delta
        appr3 = out["backtest_v7_dual"]["approval_head_phase3"]
        out["v7_gate_vs_v5op"] = {
            "success_auprc_within_0.03_of_0.815": abs(delta.get("auprc", -1)) <= 0.03,
            "recall_not_below_v5op": delta.get("recall", -1) >= 0,
            "approval_head_phase3_auprc": (appr3 or {}).get("auprc"),
            "approval_head_phase3_n": (appr3 or {}).get("n_trials"),
            "pass": abs(delta.get("auprc", -1)) <= 0.03 and delta.get("recall", -1) >= 0,
        }

    # v8 chemistry-finetuned vs v7 dual-head: plan gate = overall Phase III
    # AUPRC must not regress below v7 (0.809); chemistry knockout delta must
    # grow vs v7's ~0.0.
    p8 = out["backtest_v8_chem"]["phase3_test_success"]
    if p7 and p8:
        out["delta_v8_vs_v7_phase3_success"] = {
            k: round(float(p8.get(k, 0)) - float(p7.get(k, 0)), 4)
            for k in ("auprc", "roc_auc", "precision", "recall", "f1")
            if k in p7 and k in p8
        }
    a_v7 = out.get("ablation_v7")
    a_v8 = out.get("ablation_v8")
    if a_v7 and a_v8:
        d7 = a_v7.get("chemistry_delta_phase3_auprc")
        d8 = a_v8.get("chemistry_delta_phase3_auprc")
        p7_auprc = (p7 or {}).get("auprc")
        p8_auprc = (p8 or {}).get("auprc")
        out["chemistry_did_learn"] = {
            "v7_chemistry_knockout_delta": d7,
            "v8_chemistry_knockout_delta": d8,
            "delta_grew": (d8 or 0) > (d7 or 0),
            "v8_phase3_auprc_not_below_v7": (p8_auprc or 0) >= (p7_auprc or 0),
        }

    dest = RUNS_DIR / "backtest_v6_comparison.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
