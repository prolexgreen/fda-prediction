"""Chronological merge of TOP + CTO trial corpora with era pooling."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fda_predictor.data.cto_dataset import build_cto_human_frame, load_cto_processed, save_cto_processed
from fda_predictor.data.stock_features import StockFeatureStats, attach_stock_features, fit_stock_normalizer
from fda_predictor.data.tabular_features import TabularStats, attach_tabular_features
from fda_predictor.data.top_dataset import TOPSplits, assert_temporal_order, load_top_splits, nct_number
from fda_predictor.utils.paths import (  # noqa: F401 — MERGED used below
    MERGED_PROCESSED_PARQUET,
    PROCESSED_DATA_DIR,
    ensure_dirs,
)

APPROVAL_LABELS_PARQUET = PROCESSED_DATA_DIR / "approval_labels.parquet"
STAGE7_FEATURES_PARQUET = PROCESSED_DATA_DIR / "stage7_features.parquet"
STAGE8_KG_FEATURES_PARQUET = PROCESSED_DATA_DIR / "stage8_kg_pca.parquet"


def _join_stage8_kg_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Left-join TxGNN-derived KG embedding features (PCA-64 + mask).

    Set FDA_DISABLE_KG=1 to skip the join (v14 ablation: DAPT-only, no KG).
    The downstream tabular builder then feeds zeros + mask=0 for the KG block.
    """
    if os.environ.get("FDA_DISABLE_KG", "") == "1":
        return frame
    if not STAGE8_KG_FEATURES_PARQUET.exists():
        return frame
    side = pd.read_parquet(STAGE8_KG_FEATURES_PARQUET)
    keep = [c for c in ("nctid", "kg_pca", "kg_pca_mask") if c in side.columns]
    side = side[keep].drop_duplicates(subset="nctid")
    drop_cols = [c for c in side.columns if c != "nctid" and c in frame.columns]
    out = frame.drop(columns=drop_cols) if drop_cols else frame
    return out.merge(side, on="nctid", how="left")


def _join_stage7_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Left-join the stage-7 sidecar (chem/modality/mechanism/sponsor)."""
    if not STAGE7_FEATURES_PARQUET.exists():
        return frame
    side = pd.read_parquet(STAGE7_FEATURES_PARQUET)
    drop_cols = [c for c in side.columns if c != "nctid" and c in frame.columns]
    out = frame.drop(columns=drop_cols) if drop_cols else frame
    return out.merge(side, on="nctid", how="left")


@dataclass
class MergedSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stock_stats: StockFeatureStats
    tabular_stats: TabularStats | None = None

    def __iter__(self):
        return iter((self.train, self.val, self.test))


def _join_approval_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Left-join the approval-label sidecar if present; keep success `label` intact."""
    if not APPROVAL_LABELS_PARQUET.exists():
        return frame
    side = pd.read_parquet(APPROVAL_LABELS_PARQUET)
    keep = [
        c
        for c in (
            "nctid",
            "approval_label",
            "previously_approved",
            "n_drugs",
            "n_resolved",
            "n_prior_drug_approvals",
            "earliest_post_start_approval",
        )
        if c in side.columns
    ]
    side = side[keep].drop_duplicates(subset="nctid", keep="last")
    drop_cols = [c for c in side.columns if c != "nctid" and c in frame.columns]
    out = frame.drop(columns=drop_cols) if drop_cols else frame
    return out.merge(side, on="nctid", how="left")


def _parse_chronology(row) -> float:
    """Comparable ordering key within each data_source (not across sources)."""
    source = row.get("data_source") or row.get("source") or "TOP"
    if source == "TOP":
        if "nctid" in row and row["nctid"]:
            return float(nct_number(str(row["nctid"])))
        return 0.0
    d = row.get("chronology_date") or row.get("start_date") or row.get("completion_date")
    if d is None or (isinstance(d, float) and np.isnan(d)):
        if "nctid" in row and row["nctid"]:
            return float(nct_number(str(row["nctid"])))
        return 0.0
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return float(nct_number(str(row["nctid"])))
    return ts.timestamp()


def _top_frame_with_meta(top_splits: TOPSplits) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def enrich(part: pd.DataFrame) -> pd.DataFrame:
        df = part.copy()
        df["source"] = "TOP"
        df["data_source"] = "TOP"
        df["chronology_date"] = df["nctid"].map(lambda x: nct_number(str(x)))
        df["molecule_mask"] = (df["n_smiles"] > 0).astype(int)
        df["ticker"] = None
        df["stock_mask"] = 0
        df["stock_feats"] = [np.zeros(7, dtype=np.float32).tolist()] * len(df)
        return df

    return enrich(top_splits.train), enrich(top_splits.val), enrich(top_splits.test)


def _chronological_split_frame(
    df: pd.DataFrame,
    test_fraction: float,
    val_fraction_of_trainval: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["_chrono"] = work.apply(_parse_chronology, axis=1)
    work = work.sort_values("_chrono", kind="mergesort").drop_duplicates(subset="nctid", keep="first")
    n_total = len(work)
    n_test = int(round(n_total * test_fraction))
    n_trainval = n_total - n_test
    n_val = int(round(n_trainval * val_fraction_of_trainval))
    train = work.iloc[: n_trainval - n_val].drop(columns="_chrono").reset_index(drop=True)
    val = work.iloc[n_trainval - n_val : n_trainval].drop(columns="_chrono").reset_index(drop=True)
    test = work.iloc[n_trainval:].drop(columns="_chrono").reset_index(drop=True)
    return train, val, test


def assert_merged_temporal_order(splits: MergedSplits) -> None:
    """Per-source monotonic chronology + era boundaries (units differ across sources)."""
    for source in ("TOP", "CTO"):
        tr_sub = splits.train[splits.train["data_source"] == source]
        va_sub = splits.val[splits.val["data_source"] == source]
        te_sub = splits.test[splits.test["data_source"] == source]
        for name, part in (("train", tr_sub), ("val", va_sub), ("test", te_sub)):
            if len(part) < 2:
                continue
            chrono = part.apply(_parse_chronology, axis=1)
            assert chrono.is_monotonic_increasing, f"{source} {name} not chronological"
        if len(tr_sub) and len(va_sub):
            assert tr_sub.apply(_parse_chronology, axis=1).max() <= va_sub.apply(
                _parse_chronology, axis=1
            ).min(), f"{source} LOOKAHEAD: train era > val era"
        if len(va_sub) and len(te_sub):
            assert va_sub.apply(_parse_chronology, axis=1).max() <= te_sub.apply(
                _parse_chronology, axis=1
            ).min(), f"{source} LOOKAHEAD: val era > test era"

    # Global nctid disjointness across pooled splits
    seen: set[str] = set()
    for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        ids = part["nctid"].astype(str)
        assert ids.is_unique, f"{name} contains duplicate nctid rows"
        overlap = seen.intersection(set(ids))
        assert not overlap, f"split leakage into {name}: {sorted(overlap)[:5]}"
        seen.update(ids)


def load_merged_splits(
    val_fraction_of_trainval: float = 0.15,
    test_fraction: float = 0.2,
    min_criteria_chars: int = 20,
    rebuild_cto: bool = False,
    fetch_ctgov: bool = True,
    resolve_smiles: bool = True,
) -> MergedSplits:
    ensure_dirs()
    top = load_top_splits(
        val_fraction_of_trainval=val_fraction_of_trainval,
        min_criteria_chars=min_criteria_chars,
        split_mode="chronological",
    )
    top_tr, top_va, top_te = _top_frame_with_meta(top)

    if rebuild_cto:
        cto, _ = build_cto_human_frame(
            min_criteria_chars=min_criteria_chars,
            fetch_ctgov=fetch_ctgov,
            resolve_smiles=resolve_smiles,
        )
        save_cto_processed(cto)
    else:
        try:
            cto = load_cto_processed()
        except Exception:
            cto, _ = build_cto_human_frame(
                min_criteria_chars=min_criteria_chars,
                fetch_ctgov=fetch_ctgov,
                resolve_smiles=resolve_smiles,
            )
            save_cto_processed(cto)

    cto = cto.copy()
    cto["data_source"] = "CTO"
    cto_unique = cto.drop_duplicates(subset="nctid")
    top_nct_set = set(pd.concat([top_tr, top_va, top_te])["nctid"].astype(str))
    cto_unique = cto_unique[~cto_unique["nctid"].astype(str).isin(top_nct_set)].reset_index(drop=True)
    cto_tr, cto_va, cto_te = _chronological_split_frame(
        cto_unique, test_fraction, val_fraction_of_trainval
    )

    train = pd.concat([top_tr, cto_tr], ignore_index=True).drop_duplicates(subset="nctid", keep="first")
    val = pd.concat([top_va, cto_va], ignore_index=True).drop_duplicates(subset="nctid", keep="first")
    test = pd.concat([top_te, cto_te], ignore_index=True).drop_duplicates(subset="nctid", keep="first")

    train = _join_approval_labels(train)
    val = _join_approval_labels(val)
    test = _join_approval_labels(test)

    # Stage-7 sidecar must land BEFORE attach_tabular_features so the
    # tabular encoder sees the full layout; stage-8 KG block added after.
    train = _join_stage7_features(train)
    val = _join_stage7_features(val)
    test = _join_stage7_features(test)
    train = _join_stage8_kg_features(train)
    val = _join_stage8_kg_features(val)
    test = _join_stage8_kg_features(test)

    train = attach_stock_features(train, fit_stats=True)
    stats = fit_stock_normalizer(train)
    val = attach_stock_features(val, stats=stats)
    test = attach_stock_features(test, stats=stats)

    train, tab_stats = attach_tabular_features(train, fit_stats=True)
    val, _ = attach_tabular_features(val, stats=tab_stats)
    test, _ = attach_tabular_features(test, stats=tab_stats)

    splits = MergedSplits(
        train=train, val=val, test=test, stock_stats=stats, tabular_stats=tab_stats
    )
    assert_merged_temporal_order(splits)

    MERGED_PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    # chronology_date mixes TOP's numeric NCT keys with CTO date strings;
    # stringify ALL identifier/date-ish columns so pyarrow gets consistent
    # column types regardless of which sources contributed.
    pooled = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ]
    )
    for col in (
        "chronology_date",
        "start_date",
        "completion_date",
        "ticker",
        "overall_status",
        "sponsor",
    ):
        if col in pooled.columns:
            pooled[col] = pooled[col].astype("string").fillna("")
    pooled.to_parquet(MERGED_PROCESSED_PARQUET, index=False)
    return splits


def split_summary(splits: MergedSplits) -> dict:
    out = {}
    for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        out[name] = {
            "n": len(part),
            "pos_frac": float(part["label"].mean()) if len(part) else 0.0,
            "top_n": int((part["data_source"] == "TOP").sum()),
            "cto_n": int((part["data_source"] == "CTO").sum()),
            "stock_mask_frac": float(part["stock_mask"].mean()) if "stock_mask" in part.columns else 0.0,
            "mol_mask_frac": float(part["molecule_mask"].mean()) if "molecule_mask" in part.columns else 0.0,
            "phase3_n": int((part["phase_index"] == 2).sum()) if "phase_index" in part.columns else 0,
        }
    return out
