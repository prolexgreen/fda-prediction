"""Path and cache governance. Import this module FIRST, before any
transformers / huggingface_hub / tdc import, so the environment defaults
are in place in fresh shells (no profile variables required).

Model weights live on D:\Models (explicit user requirement); TDC datasets
stay inside the project under data/raw.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --- Model weight cache (D:\Models, override with FDA_MODELS_ROOT) --------
FDA_MODELS_ROOT = Path(os.environ.get("FDA_MODELS_ROOT", "D:/Models"))
HUB_CACHE = FDA_MODELS_ROOT / "hub"

os.environ.setdefault("HF_HOME", str(FDA_MODELS_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(HUB_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- Project data locations (TDC and processed artifacts) -----------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CTO_RAW_DIR = RAW_DATA_DIR / "cto"
PUBCHEM_CACHE_DIR = RAW_DATA_DIR / "pubchem_cache"
STOCK_CACHE_DIR = RAW_DATA_DIR / "stock_cache"
OPENFDA_BULK_DIR = RAW_DATA_DIR / "openfda_bulk"
CHEMBL_DIR = FDA_MODELS_ROOT / "chembl"
CTO_PROCESSED_PARQUET = PROCESSED_DATA_DIR / "cto_human.parquet"
MERGED_PROCESSED_PARQUET = PROCESSED_DATA_DIR / "merged_trials.parquet"
STAGE7_FEATURES_PARQUET = PROCESSED_DATA_DIR / "stage7_features.parquet"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def ensure_dirs() -> None:
    for d in (
        FDA_MODELS_ROOT,
        HUB_CACHE,
        DATA_DIR,
        RAW_DATA_DIR,
        CTO_RAW_DIR,
        PUBCHEM_CACHE_DIR,
        STOCK_CACHE_DIR,
        OPENFDA_BULK_DIR,
        CHEMBL_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        ARTIFACTS_DIR,
        CHECKPOINTS_DIR,
        RUNS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
