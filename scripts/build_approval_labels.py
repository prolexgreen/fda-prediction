"""Batch-build FDA approval labels for the merged trial corpus.

Resumable: writes data/processed/approval_labels.parquet keyed by nctid.
Uses the openFDA disk cache so re-runs are cheap after the first pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.approval_labels import (  # noqa: E402
    approval_label_coverage,
    attach_approval_labels,
)
from fda_predictor.utils.paths import (  # noqa: E402
    MERGED_PROCESSED_PARQUET,
    PROCESSED_DATA_DIR,
    ensure_dirs,
)

OUT_PATH = PROCESSED_DATA_DIR / "approval_labels.parquet"
COVERAGE_PATH = PROCESSED_DATA_DIR / "approval_label_coverage.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(MERGED_PROCESSED_PARQUET))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--max-drugs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="optional row cap for smoke runs")
    parser.add_argument("--resume", action="store_true", help="skip nctids already in --out")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--phase3-only", action="store_true", help="label Phase III trials only")
    args = parser.parse_args()

    ensure_dirs()
    src = Path(args.input)
    if not src.exists():
        print(f"ERROR: missing input parquet {src}", file=sys.stderr)
        return 1

    frame = pd.read_parquet(src)
    if args.phase3_only and "phase_index" in frame.columns:
        frame = frame[frame["phase_index"] == 2].copy()
        print(f"phase3-only filter: {len(frame)} trials")
    if args.limit:
        frame = frame.head(int(args.limit)).copy()

    out_path = Path(args.out)
    if args.resume and out_path.exists():
        existing = pd.read_parquet(out_path)
        done = set(existing["nctid"].astype(str))
        before = len(frame)
        frame = frame[~frame["nctid"].astype(str).isin(done)].copy()
        print(f"resume: {before - len(frame)} already labeled, {len(frame)} remaining")
        if frame.empty:
            cov = approval_label_coverage(existing)
            COVERAGE_PATH.write_text(json.dumps(cov, indent=2), encoding="utf-8")
            print(json.dumps(cov, indent=2))
            return 0
    else:
        existing = None

    print(f"Labeling {len(frame)} trials (max_drugs={args.max_drugs}, workers={args.workers}) ...")
    labeled = attach_approval_labels(
        frame, max_drugs=int(args.max_drugs), workers=int(args.workers)
    )
    cols = [
        "nctid",
        "approval_label",
        "previously_approved",
        "n_drugs",
        "n_resolved",
        "n_prior_drug_approvals",
        "earliest_post_start_approval",
    ]
    side = labeled[cols].copy()
    if existing is not None:
        side = pd.concat([existing[cols], side], ignore_index=True)
        side = side.drop_duplicates(subset="nctid", keep="last")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    side.to_parquet(out_path, index=False)

    # Coverage over the full input (join for split/phase if present)
    full = pd.read_parquet(src)
    merged = full.merge(side, on="nctid", how="left")
    cov = approval_label_coverage(merged)
    COVERAGE_PATH.write_text(json.dumps(cov, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(side)} rows)")
    print(f"Coverage -> {COVERAGE_PATH}")
    print(json.dumps(cov, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
