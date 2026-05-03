"""Coverage + sanity gate for FDA approval labels.

Reads approval_label_coverage.json (or recomputes from merged parquet +
approval sidecar) and verifies minimum coverage per split/phase plus known
blockbuster drug positives.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.approval_labels import approval_label_coverage  # noqa: E402
from fda_predictor.utils.paths import (  # noqa: E402
    MERGED_PROCESSED_PARQUET,
    PROCESSED_DATA_DIR,
    ensure_dirs,
)

COVERAGE_PATH = PROCESSED_DATA_DIR / "approval_label_coverage.json"
LABELS_PATH = PROCESSED_DATA_DIR / "approval_labels.parquet"
GATE_PATH = PROCESSED_DATA_DIR / "approval_coverage_gate.json"

# Minimum gates — tuned to current corpus without being trivially loose.
GATES = {
    "phase3_coverage_min": 0.25,
    "phase3_labeled_min": 500,
    "overall_positives_min": 50,
    "split_labeled_min": {"train": 100, "val": 50, "test": 50},
}

SANITY_DRUGS = (
    "pembrolizumab",
    "nivolumab",
    "macitentan",
    "erlotinib",
)


def _parse_drugs(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                raw = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return [s] if s else []
        else:
            return [s] if s else []
    if isinstance(raw, (list, tuple)):
        return [str(d).strip() for d in raw if str(d).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _drug_hits(frame: pd.DataFrame, drug: str) -> pd.DataFrame:
    needle = drug.lower()

    def _has(raw) -> bool:
        return any(needle in d.lower() for d in _parse_drugs(raw))

    return frame[frame["drugs"].apply(_has)]


def run_sanity_checks(frame: pd.DataFrame) -> dict:
    """Known post-start approvals should appear as positives when labeled."""
    pos = frame[pd.to_numeric(frame["approval_label"], errors="coerce") == 1.0]
    checks = {}
    for drug in SANITY_DRUGS:
        hits = _drug_hits(frame, drug)
        pos_hits = _drug_hits(pos, drug)
        checks[drug] = {
            "trials_with_drug": int(len(hits)),
            "positive_labeled": int(len(pos_hits)),
            "pass": len(pos_hits) >= 1,
            "example_nctids": pos_hits["nctid"].astype(str).head(3).tolist(),
        }
    return checks


def evaluate_gates(cov: dict, sanity: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    by_phase = cov.get("by_phase", {})
    p3 = by_phase.get("III", {})
    if p3.get("coverage", 0.0) < GATES["phase3_coverage_min"]:
        failures.append(
            f"Phase III coverage {p3.get('coverage', 0):.3f} < {GATES['phase3_coverage_min']}"
        )
    if p3.get("labeled", 0) < GATES["phase3_labeled_min"]:
        failures.append(
            f"Phase III labeled {p3.get('labeled', 0)} < {GATES['phase3_labeled_min']}"
        )
    pos_total = int(round(cov.get("pos_frac_labeled", 0) * cov.get("labeled", 0)))
    if pos_total < GATES["overall_positives_min"]:
        failures.append(
            f"overall positives ~{pos_total} < {GATES['overall_positives_min']}"
        )
    for split, min_n in GATES["split_labeled_min"].items():
        split_cov = cov.get("by_split", {}).get(split, {})
        if split_cov.get("labeled", 0) < min_n:
            failures.append(
                f"split {split} labeled {split_cov.get('labeled', 0)} < {min_n}"
            )
    for drug, info in sanity.items():
        if not info.get("pass"):
            failures.append(f"sanity drug {drug}: no positive labeled trials")
    return len(failures) == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", default=str(COVERAGE_PATH))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--merged", default=str(MERGED_PROCESSED_PARQUET))
    parser.add_argument("--out", default=str(GATE_PATH))
    args = parser.parse_args()

    ensure_dirs()
    labels_path = Path(args.labels)
    merged_path = Path(args.merged)
    if not labels_path.exists():
        print(f"ERROR: missing {labels_path}; run build_approval_labels.py", file=sys.stderr)
        return 1
    if not merged_path.exists():
        print(f"ERROR: missing {merged_path}", file=sys.stderr)
        return 1

    side = pd.read_parquet(labels_path)
    merged = pd.read_parquet(merged_path)
    frame = merged.merge(
        side.drop_duplicates(subset="nctid", keep="last"),
        on="nctid",
        how="left",
        suffixes=("", "_side"),
    )
    cov = approval_label_coverage(frame)
    sanity = run_sanity_checks(frame)
    ok, failures = evaluate_gates(cov, sanity)

    report = {
        "pass": ok,
        "failures": failures,
        "gates": GATES,
        "coverage": cov,
        "sanity_drugs": sanity,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.coverage).write_text(json.dumps(cov, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote gate report -> {out_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
