"""Build Stage-8 KG feature sidecar keyed by nctid.

Pools TxGNN drug-node embeddings over each trial's drug list, z-scores with
TRAIN-split stats only (leakage-safe), writes
data/processed/stage8_kg_features.parquet (nctid -> kg_emb list[512], kg_mask).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.data.kg_features import (  # noqa: E402
    KG_EMB_DIM,
    KGEmbeddingIndex,
    fit_kg_stats,
)
from fda_predictor.utils.paths import (  # noqa: E402
    MERGED_PROCESSED_PARQUET,
    PROCESSED_DATA_DIR,
    ensure_dirs,
)

OUT_PATH = PROCESSED_DATA_DIR / "stage8_kg_features.parquet"


def _parse_drugs(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, str):
        import ast

        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                raw = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return [s] if s else []
        else:
            return [s] if s else []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(MERGED_PROCESSED_PARQUET))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit).copy()
    n = len(df)
    print(f"rows: {n}", flush=True)

    print("loading TxGNN drug embedding index ...", flush=True)
    index = KGEmbeddingIndex()
    print(f"  indexed drug names: {index.n_drug_names}", flush=True)

    print("pooling per-trial embeddings ...", flush=True)
    vecs = np.zeros((n, KG_EMB_DIM), dtype=np.float32)
    masks = np.zeros(n, dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        drugs = _parse_drugs(row_dict.get("drugs"))
        vec, m = index.pooled(drugs)
        vecs[i] = vec
        masks[i] = m
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n} (mapped so far: {masks.sum():.0f})", flush=True)

    # Train-split stats only
    split_col = df["split"] if "split" in df.columns else pd.Series(["train"] * n)
    train_mask = (split_col == "train").to_numpy() & (masks > 0.5)
    if train_mask.any():
        means = vecs[train_mask].mean(axis=0)
        stds = vecs[train_mask].std(axis=0)
        stds[stds < 1e-6] = 1.0
    else:
        means, stds = np.zeros(KG_EMB_DIM), np.ones(KG_EMB_DIM)
    vecs[masks > 0.5] = (vecs[masks > 0.5] - means) / stds

    out = pd.DataFrame(
        {
            "nctid": df["nctid"].astype(str).values,
            "kg_emb": [v.tolist() for v in vecs],
            "kg_mask": masks,
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    cov = float(masks.mean())
    cov_p3 = float(masks[(df["phase_index"] == 2).to_numpy()].mean()) if "phase_index" in df.columns else None
    print(f"\ncoverage: overall={cov:.1%} phase3={cov_p3 if cov_p3 is None else f'{cov_p3:.1%}'}", flush=True)
    print(f"wrote {out_path} ({len(out)} rows, {KG_EMB_DIM} dims)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
