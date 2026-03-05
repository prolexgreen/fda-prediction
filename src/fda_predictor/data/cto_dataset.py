"""CTO benchmark ingestion: human labels + tickers joined with CTGov metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fda_predictor.data.preprocessing import (
    CanonicalizationReport,
    canonicalize_molecule_list,
    parse_phase,
)
from fda_predictor.data.pubchem_smiles import PubChemReport, resolve_drug_list
from fda_predictor.inference.ctgov_client import CTGovClient, cache_studies_json
from fda_predictor.utils.paths import CTO_PROCESSED_PARQUET, CTO_RAW_DIR, ensure_dirs

CTO_HF_REPO = "chufangao/CTO"


@dataclass
class CTOBuildReport:
    n_human_labels: int = 0
    n_with_ticker: int = 0
    n_ctgov_fetched: int = 0
    n_after_criteria_filter: int = 0
    n_with_smiles: int = 0
    pubchem: PubChemReport | None = None
    canon: CanonicalizationReport | None = None


def _normalize_nct_col(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("nct_id", "nctid", "NCTId", "nctId"):
        if col in df.columns:
            out = df.copy()
            out["nctid"] = out[col].astype(str).str.strip().str.upper()
            out["nctid"] = out["nctid"].apply(
                lambda x: x if x.startswith("NCT") else f"NCT{''.join(c for c in x if c.isdigit())}"
            )
            return out
    raise KeyError(f"No NCT column in frame: {list(df.columns)}")


def _label_col(df: pd.DataFrame) -> str:
    for col in ("labels", "label", "Y", "outcome"):
        if col in df.columns:
            return col
    raise KeyError(f"No label column in frame: {list(df.columns)}")


def download_cto_raw(force: bool = False) -> dict[str, Path]:
    """Download CTO human_labels and stocks_and_amendments to data/raw/cto/."""
    ensure_dirs()
    CTO_RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install datasets: uv sync") from exc

    for config, out_name in (
        ("human_labels", "human_labels.parquet"),
        ("stocks_and_amendments", "stocks_and_tickers.parquet"),
        ("phase3_CTO_preds", "phase3_cto_preds.parquet"),
    ):
        dest = CTO_RAW_DIR / out_name
        if dest.exists() and not force:
            paths[config] = dest
            continue
        print(f"Downloading CTO config={config} ...")
        ds = load_dataset(CTO_HF_REPO, config, split="test", trust_remote_code=False)
        frame = ds.to_pandas()
        frame.to_parquet(dest, index=False)
        paths[config] = dest
        print(f"  -> {dest.name} ({len(frame)} rows)")
    return paths


def _load_ticker_frame() -> pd.DataFrame:
    path = CTO_RAW_DIR / "stocks_and_tickers.parquet"
    if not path.exists():
        download_cto_raw()
    tick = pd.read_parquet(path)
    tick = _normalize_nct_col(tick)
    ticker_col = next(
        (c for c in tick.columns if "ticker" in c.lower() or c.lower() in ("symbol", "stock")),
        None,
    )
    if ticker_col is None:
        # common CTO column names
        for c in ("Ticker", "ticker_symbol", "stock_ticker"):
            if c in tick.columns:
                ticker_col = c
                break
    if ticker_col is None:
        tick["ticker"] = None
    else:
        tick = tick.rename(columns={ticker_col: "ticker"})
        tick["ticker"] = tick["ticker"].astype(str).str.strip().str.upper()
        tick.loc[tick["ticker"].isin(("", "NAN", "NONE", "NA")), "ticker"] = None
    keep = ["nctid", "ticker"]
    for extra in ("amendments", "num_amendments", "slope", "Slope"):
        if extra in tick.columns:
            keep.append(extra)
    return tick[keep].drop_duplicates(subset="nctid", keep="first")


def _load_human_labels() -> pd.DataFrame:
    path = CTO_RAW_DIR / "human_labels.parquet"
    if not path.exists():
        download_cto_raw()
    human = pd.read_parquet(path)
    human = _normalize_nct_col(human)
    lcol = _label_col(human)
    human = human.rename(columns={lcol: "label"})
    human["label"] = pd.to_numeric(human["label"], errors="coerce")
    human = human.dropna(subset=["label", "nctid"])
    human["label"] = human["label"].astype(int)
    return human


def build_cto_human_frame(
    min_criteria_chars: int = 20,
    fetch_ctgov: bool = True,
    resolve_smiles: bool = True,
    max_drugs_pubchem: int = 3,
    ctgov_cache_path: Path | None = None,
    max_trials: int | None = None,
) -> tuple[pd.DataFrame, CTOBuildReport]:
    """Join human labels + tickers + CTGov + optional PubChem SMILES."""
    ensure_dirs()
    report = CTOBuildReport()
    human = _load_human_labels()
    if max_trials is not None:
        human = human.head(max_trials).copy()
    report.n_human_labels = len(human)
    tickers = _load_ticker_frame()
    merged = human.merge(tickers, on="nctid", how="left")

    # Pre-fill chronology + phase from CTTI human_labels (before CTGov enrichment).
    if "study_first_submitted_date" in merged.columns:
        merged["chronology_date"] = merged["study_first_submitted_date"]
    if "phase" in merged.columns and "phase_index" not in merged.columns:
        merged["phase_index"] = merged["phase"].apply(parse_phase)
    for col_src, col_dst in (
        ("start_date", "start_date"),
        ("completion_date", "completion_date"),
        ("overall_status", "overall_status"),
    ):
        if col_src in merged.columns:
            merged[col_dst] = merged[col_src]

    report.n_with_ticker = int(merged["ticker"].notna().sum())

    cache_path = ctgov_cache_path or (CTO_RAW_DIR / "ctgov_studies_cache.json")
    nct_list = merged["nctid"].tolist()

    if fetch_ctgov:
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            from fda_predictor.inference.ctgov_client import _parse_study

            records = {}
            for nct, study in raw.items():
                rec = _parse_study(study)
                if rec:
                    records[nct] = rec
            missing = [n for n in nct_list if n not in records]
            if missing:
                client = CTGovClient()
                records.update(client.fetch_by_nct_ids(missing))
                cache_studies_json(cache_path, records)
        else:
            client = CTGovClient()
            records = client.fetch_by_nct_ids(nct_list)
            cache_studies_json(cache_path, records)
        report.n_ctgov_fetched = len(records)

        meta_rows = []
        for nct in nct_list:
            rec = records.get(nct)
            if not rec:
                continue
            meta_rows.append(
                {
                    "nctid": nct,
                    "criteria": rec.criteria,
                    "phase": rec.phase_raw,
                    "phase_index": rec.phase_index,
                    "drugs": rec.drugs,
                    "sponsor": rec.sponsor,
                    "start_date": rec.start_date,
                    "completion_date": rec.completion_date,
                    "chronology_date": rec.study_first_submitted_date or rec.start_date,
                    "overall_status": rec.overall_status,
                }
            )
        meta = pd.DataFrame(meta_rows)
        # Drop our pre-filled placeholders first so CTGov values land under the
        # canonical names (no _x/_y suffix collision, which broke date-based
        # stock feature computation downstream).
        placeholder_cols = [
            c
            for c in ("phase", "phase_index", "start_date", "completion_date", "chronology_date", "overall_status")
            if c in merged.columns
        ]
        merged = merged.drop(columns=placeholder_cols)
        merged = merged.merge(meta, on="nctid", how="left")
    else:
        merged["criteria"] = merged.get("criteria", "")
        merged["phase_index"] = merged.get("phase", "").apply(parse_phase) if "phase" in merged.columns else 4
        merged["drugs"] = merged.get("drugs", [[]] * len(merged))
        merged["chronology_date"] = merged.get("study_first_submitted_date", merged.get("completion_date"))

    ok_crit = merged["criteria"].fillna("").astype(str).str.len() >= min_criteria_chars
    merged = merged[ok_crit].reset_index(drop=True)
    report.n_after_criteria_filter = len(merged)

    canon_report = CanonicalizationReport()
    pubchem_report = PubChemReport()
    report.pubchem = pubchem_report
    report.canon = canon_report

    smiles_lists: list[list[str]] = []
    mol_masks: list[int] = []
    n_rows = len(merged)

    for i, (_, row) in enumerate(merged.iterrows()):
        drugs = row.get("drugs") or []
        if isinstance(drugs, np.ndarray):
            drugs = [str(d) for d in drugs.tolist()]
        elif not isinstance(drugs, list):
            drugs = [str(drugs)] if drugs else []
        drugs = [d for d in drugs if str(d).strip()]
        canon: list[str] = []
        if resolve_smiles and drugs:
            try:
                resolved = resolve_drug_list(
                    drugs,
                    report=pubchem_report,
                    max_drugs=max_drugs_pubchem,
                    split_combos=True,
                    enhanced=True,
                )
                canon = canonicalize_molecule_list(resolved, canon_report)
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the ingest
                print(f"[pubchem] row {i} ({row.get('nctid')}) failed: {exc!r}", flush=True)
                pubchem_report.n_invalid += 1
        smiles_lists.append(canon)
        mol_masks.append(1 if len(canon) > 0 else 0)
        if (i + 1) % 500 == 0:
            hit = pubchem_report.n_hits
            q = max(1, pubchem_report.n_queries)
            print(
                f"[pubchem] {i + 1}/{n_rows} rows | queries={pubchem_report.n_queries} "
                f"hits={hit} ({hit / q:.0%}) | smiles_rows={sum(mol_masks)}",
                flush=True,
            )

    merged["smiles_canonical"] = smiles_lists
    merged["molecule_mask"] = mol_masks
    merged["n_smiles"] = merged["smiles_canonical"].apply(len)
    report.n_with_smiles = int((merged["molecule_mask"] == 1).sum())

    merged["source"] = "CTO"
    merged["data_source"] = "CTO"
    if "chronology_date" not in merged.columns:
        merged["chronology_date"] = merged.get("completion_date")

    return merged, report


def save_cto_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or CTO_PROCESSED_PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_cto_processed(path: Path | None = None) -> pd.DataFrame:
    path = path or CTO_PROCESSED_PARQUET
    if not path.exists():
        frame, _ = build_cto_human_frame()
        save_cto_processed(frame, path)
    return pd.read_parquet(path)
