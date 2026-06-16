"""PCA-compress stage8 KG sidecar 512 -> 64 dims (train-split fit only).

Fits sklearn PCA on the train subset of kg-mask=1 rows in
data/processed/stage8_kg_features.parquet, writes
data/processed/stage8_kg_pca.parquet (nctid -> kg_pca list[64], kg_pca_mask),
and dumps the fitted PCA to data/processed/stage8_kg_pca.joblib so val/test/
inference reuse the same projection without refitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.utils.paths import PROCESSED_DATA_DIR  # noqa: E402

KG_IN = PROCESSED_DATA_DIR / "stage8_kg_features.parquet"
KG_OUT = PROCESSED_DATA_DIR / "stage8_kg_pca.parquet"
PCA_OUT = PROCESSED_DATA_DIR / "stage8_kg_pca.joblib"
MERGED = PROCESSED_DATA_DIR / "merged_trials.parquet"
N_COMPONENTS = 64


def main() -> int:
    if not KG_IN.exists():
        print(f"ERROR: missing {KG_IN}; run build_stage8_kg_features.py first")
        return 1
    kg = pd.read_parquet(KG_IN)
    merged = pd.read_parquet(MERGED, columns=["nctid", "split"])
    kg = kg.merge(merged, on="nctid", how="left")
    x = np.stack(kg["kg_emb"].apply(np.asarray).tolist()).astype(np.float32)
    mask = kg["kg_mask"].astype(float).to_numpy() > 0.5
    is_train = (kg["split"] == "train").to_numpy()
    fit_sel = mask & is_train
    if fit_sel.sum() < N_COMPONENTS:
        print(f"ERROR: only {fit_sel.sum()} train rows mapped; need >= {N_COMPONENTS}")
        return 1

    from sklearn.decomposition import PCA

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(x[fit_sel])
    ev = float(pca.explained_variance_ratio_.sum())
    print(f"PCA fit on {fit_sel.sum()} train rows; explained variance={ev:.1%}", flush=True)

    proj = pca.transform(x).astype(np.float32)
    out = pd.DataFrame(
        {
            "nctid": kg["nctid"].astype(str).values,
            "kg_pca": [v.tolist() for v in proj],
            "kg_pca_mask": mask.astype(float),
        }
    )
    out.to_parquet(KG_OUT, index=False)
    joblib.dump(pca, PCA_OUT)
    print(f"wrote {KG_OUT} ({len(out)} rows x {N_COMPONENTS})", flush=True)
    print(f"pca model saved -> {PCA_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
