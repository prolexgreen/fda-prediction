"""Stage-8 knowledge-graph features from TxGNN node embeddings.

DrugBank drug nodes: 7957 x 512 embedding matrix + id/name mappings, all
precomputed artifacts of the TxGNN Explorer package — no DGL dependency.

Per trial: pool drug embeddings across mapped drugs (mean), z-score with
train-split stats, mask=1 when at least one drug mapped.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fda_predictor.utils.paths import FDA_MODELS_ROOT

TXGNN_DIR = FDA_MODELS_ROOT / "TxGNNExplorer_v2" / "TxGNNExplorer"
KG_EMB_DIM = 512


@dataclass
class KGLookup:
    name_to_emb: dict[str, np.ndarray]


def _normalize(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


class KGEmbeddingIndex:
    """Maps drug names -> TxGNN drug-node embeddings."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else TXGNN_DIR
        node_emb_path = self.base_dir / "node_emb.pkl"
        mapping_path = self.base_dir / "name_mapping.pkl"
        with open(mapping_path, "rb") as f:
            mapping: dict[str, Any] = pickle.load(f)
        with open(node_emb_path, "rb") as f:
            node_emb = pickle.load(f)
        drug_emb = node_emb["drug"].detach().cpu().numpy().astype(np.float32)
        idx2id = mapping["idx2id_drug"]
        id2name = mapping["id2name_drug"]

        self.name_to_emb: dict[str, np.ndarray] = {}
        n_map, n_name = 0, 0
        for idx, drug_id in idx2id.items():
            name = id2name.get(drug_id)
            if name is None:
                continue
            n_name += 1
            emb = drug_emb[int(idx)]
            self.name_to_emb.setdefault(_normalize(name), emb)
            n_map += 1
        self.n_drug_names = n_map

    def lookup(self, name: str) -> np.ndarray | None:
        return self.name_to_emb.get(_normalize(name))

    def pooled(self, drug_names: list[str]) -> tuple[np.ndarray, float]:
        """Mean-pool embeddings for the mapped drugs. Returns (512 vec, mask)."""
        embs = []
        for name in drug_names:
            v = self.lookup(name)
            if v is not None:
                embs.append(v)
        if not embs:
            return np.zeros(KG_EMB_DIM, dtype=np.float32), 0.0
        return np.stack(embs).mean(axis=0).astype(np.float32), 1.0


@dataclass
class KGStats:
    means: np.ndarray
    stds: np.ndarray

    def to_dict(self) -> dict:
        return {"means": self.means.tolist(), "stds": self.stds.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "KGStats":
        return cls(np.asarray(d["means"], dtype=np.float32), np.asarray(d["stds"], dtype=np.float32))


def fit_kg_stats(frames: list[np.ndarray], masks: list[float]) -> KGStats:
    X = np.stack(frames)
    M = np.asarray(masks, dtype=bool)
    if not M.any():
        return KGStats(means=np.zeros(KG_EMB_DIM), stds=np.ones(KG_EMB_DIM))
    sub = X[M]
    means = sub.mean(axis=0)
    stds = sub.std(axis=0)
    stds[stds < 1e-6] = 1.0
    return KGStats(means=means.astype(np.float32), stds=stds.astype(np.float32))


def row_kg_features(index: KGEmbeddingIndex, drugs: list[str], stats: KGStats | None = None) -> tuple[np.ndarray, np.ndarray]:
    vec, mask_f = index.pooled(drugs)
    if stats is not None and mask_f > 0.5:
        vec = (vec - stats.means) / stats.stds
    mask = np.full(KG_EMB_DIM, mask_f, dtype=np.float32)
    return vec.astype(np.float32), mask
