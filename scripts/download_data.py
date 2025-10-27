"""Fetch the TOP Phase III benchmark into project-local data/raw/.

Model weights never land here -- they live on D:\\Models (see
src/fda_predictor/utils/paths.py). We pull the benchmark authors'
predefined Phase III train/valid/test CSVs (see top_dataset.py for why the
installed PyTDC package cannot provide this dataset with intact identifiers).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.top_dataset import (  # noqa: E402
    SPLIT_FILES,
    fetch_hint_splits,
)
from fda_predictor.utils.paths import RAW_DATA_DIR, ensure_dirs  # noqa: E402


def main() -> int:
    ensure_dirs()
    print(f"Benchmark cache directory: {RAW_DATA_DIR}")
    frames = fetch_hint_splits()

    total_rows = 0
    for split, fname in SPLIT_FILES.items():
        df = frames[split]
        total_rows += len(df)
        pos = int(df["label"].sum()) if "label" in df.columns else -1
        print(f"{fname}: {len(df)} rows | label=1: {pos} ({pos / len(df):.1%})")
    print(f"TOTAL: {total_rows} trials")

    sample_cols = [c for c in ("nctid", "label", "drugs", "smiless", "criteria", "phase", "status") if c in frames["train"].columns]
    print(f"\nTrain schema columns: {list(frames['train'].columns)}")
    print(f"Expected core columns present: {sample_cols}")

    if not {"nctid", "label", "smiless", "criteria"}.issubset(frames["train"].columns):
        print("[WARN] expected core columns missing -- check source repo layout")
        return 1
    print("\nDownload OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
