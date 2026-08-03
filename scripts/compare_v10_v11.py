"""Stage-7 acceptance summary: v11 vs v10 vs dual-head v7 (all on P3 test)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fda_predictor.utils.paths import RUNS_DIR


def _metrics(run: str) -> dict | None:
    p = RUNS_DIR / run / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _p3(m: dict) -> dict:
    tuned = m.get("per_phase_tuned", {}).get("test", {})
    if "III" in tuned:
        return tuned["III"]
    return m.get("phase3_slice", {}).get("test", {})


def main() -> int:
    runs = {
        "v5_op": "backtest_v5_op",
        "v7_dual": "backtest_v7_dual",
        "v10_molfix": "backtest_v10_molfix",
        "v11_mecha_sponsor": "backtest_v11_mecha",
    }
    tbl = ["| run | checkpoint | phase3 AUPRC | roc_auc | precision | recall |", "|---|---|---|---|---|---|"]
    for label, run in runs.items():
        m = _metrics(run)
        if not m:
            tbl.append(f"| {label} | (missing) | - | - | - | - |")
            continue
        p3 = _p3(m)
        tbl.append(
            f"| {label} | {Path(m.get('checkpoint','')).name} | {p3.get('auprc', -1):.4f} "
            f"| {p3.get('roc_auc') or 0:.4f} | {p3.get('precision', 0):.4f} | {p3.get('recall', 0):.4f} |"
        )

    v10 = _metrics("backtest_v10_molfix")
    v11 = _metrics("backtest_v11_mecha")
    ab11 = json.loads((RUNS_DIR / "ablation_v11" / "ablation.json").read_text(encoding="utf-8"))
    s11 = ab11.get("streams", {})
    b11 = s11.get("blocks", {})
    p10 = _p3(v10) if v10 else {}
    p11 = _p3(v11) if v11 else {}
    d = {
        "delta_p3_auprc_v11_minus_v10": round(p11.get("auprc", 0) - p10.get("auprc", 0), 4),
        "block_knockout_deltas": {k: v.get("delta_phase3_auprc") for k, v in b11.items()},
        "stream_knockout_deltas": {k: v.get("delta_phase3_auprc") for k, v in s11.items()},
    }
    out = {"table": tbl, "comparison": d}
    dest = RUNS_DIR / "v11_vs_v10_summary.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for line in tbl:
        print(line)
    print()
    print(json.dumps(d, indent=2))
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
