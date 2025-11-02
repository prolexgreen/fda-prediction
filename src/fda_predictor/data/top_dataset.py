"""Loading and splitting the TOP Phase III clinical-trial outcome benchmark.

Data source and split policy:
- Primary source: the TOP benchmark authors' predefined Phase III
  train/valid/test CSVs from the HINT repository (futianfan/
  clinical-trial-outcome-prediction), pinned to a commit SHA. This is the
  original TOP dataset with intact identifiers AND the authors' official
  chronological partition.
- Why not TDC's TrialOutcome class: installed PyTDC 0.4.1 only exposes
  packaged phase1/phase2/phase3 variants whose schema lacks `nctid`
  entirely, and its get_split() offers random/cold/combination splits
  only -- a random split would silently inject lookahead bias.
- Anti-lookahead guard: asserts disjointness by nctid and non-decreasing
  numeric NCT across train -> val -> test boundaries (NCT numbers are
  assigned monotonically over time, so this is a sound chronological proxy).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

import pandas as pd

from fda_predictor.data.preprocessing import (
    CanonicalizationReport,
    canonicalize_molecule_list,
    parse_phase,
)
from fda_predictor.utils.paths import RAW_DATA_DIR

DATASET_NAME = "TOP-PhaseIII"  # benchmark label (authors' predefined splits)
HINT_REPO_SHA = "8dc0497f23fdb84e2905da7655924a91e6e79798"  # pin for reproducibility
HINT_RAW_BASE = (
    "https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/"
    f"{HINT_REPO_SHA}/data"
)
SPLIT_FILES = {
    "train": "phase_III_train.csv",
    "val": "phase_III_valid.csv",
    "test": "phase_III_test.csv",
}
LIST_FIELDS = ("drugs", "smiless")
DROP_FIELDS = ("diseases", "icdcodes")  # removed from the revised spec; ignored entirely


@dataclass
class TOPSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    canon_report: CanonicalizationReport
    dropped_no_valid_smiles: int = 0
    dropped_short_or_missing_criteria: int = 0

    def __iter__(self):
        return iter((self.train, self.val, self.test))


def _parse_list_field(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return parsed if isinstance(parsed, list) else [s]
            except (ValueError, SyntaxError):
                return [s] if s else []
        return [s] if s else []
    return []


def nct_number(nctid: str) -> int:
    digits = "".join(ch for ch in str(nctid) if ch.isdigit())
    if not digits:
        raise ValueError(f"NCT id without numeric portion: {nctid!r}")
    return int(digits)


def _fetch_hint_splits() -> dict[str, pd.DataFrame]:
    """Download (once) and load the benchmark authors' predefined splits.

    Why not TDC's TrialOutcome: installed PyTDC 0.4.1 exposes only packaged
    phase1/phase2/phase3 variants whose schema lacks `nctid` entirely, and
    its get_split() supports random/cold/combination only -- no temporal
    split, which would make the anti-lookahead guard unverifiable. The
    original TOP benchmark repo provides the same data with intact
    identifiers AND the authors' predefined Phase III split files.
    """
    import requests

    frames: dict[str, pd.DataFrame] = {}
    for split, fname in SPLIT_FILES.items():
        local = RAW_DATA_DIR / fname
        if not local.exists():
            url = f"{HINT_RAW_BASE}/{fname}"
            print(f"Downloading {url}")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            local.write_bytes(resp.content)
        else:
            print(f"Found local copy: {local.name}")
        frames[split] = pd.read_csv(local)
    return frames


# Backwards-compatible alias for earlier call sites.
fetch_hint_splits = _fetch_hint_splits


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float,
    val_fraction_of_trainval: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three contiguous chronological blocks: oldest -> train, middle -> val,
    newest -> test.

    Contiguity is what the anti-lookahead guard asserts: max(train NCT) <=
    min(val NCT) <= max(val NCT) <= min(test NCT). Validation must sit in
    its own future era so model selection reflects out-of-time performance.
    `seed` is accepted for API stability; the split itself is deterministic.
    """
    ordered = df.assign(_nct=df["nctid"].map(nct_number)).sort_values("_nct", kind="mergesort")
    n_total = len(ordered)
    n_test = int(round(n_total * test_fraction))
    n_trainval = n_total - n_test
    n_val = int(round(n_trainval * val_fraction_of_trainval))

    train = ordered.iloc[: n_trainval - n_val].drop(columns="_nct").reset_index(drop=True)
    val = ordered.iloc[n_trainval - n_val : n_trainval].drop(columns="_nct").reset_index(drop=True)
    test = ordered.iloc[n_trainval:].drop(columns="_nct").reset_index(drop=True)
    return train, val, test


def assert_temporal_order(splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Guard: splits are disjoint by nctid and numerically non-decreasing.

    NCT numbers grow monotonically with registration time, so
    max(train_num) <= min(val_num) <= max(val_num) <= min(test_num)
    is a sound chronological proxy in the absence of explicit dates.
    """
    names = ("train", "val", "test")

    seen: set[str] = set()
    for name, part in zip(names, splits):
        ids = part["nctid"].astype(str)
        assert ids.is_unique, f"{name} contains duplicate nctid rows"
        overlap = seen.intersection(set(ids))
        assert not overlap, f"split leakage between earlier splits and {name}: {sorted(overlap)[:5]}"
        seen.update(ids)

    tr = splits[0]["nctid"].map(nct_number)
    va = splits[1]["nctid"].map(nct_number)
    te = splits[2]["nctid"].map(nct_number)
    assert tr.max() <= va.min(), f"LOOKAHEAD BIAS: max(train_nct={tr.max()}) > min(val_nct={va.min()})"
    assert va.max() <= te.min(), f"LOOKAHEAD BIAS: max(val_nct={va.max()}) > min(test_nct={te.min()})"


def _hash_val_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    def is_val(nctid: str) -> bool:
        digest = hashlib.sha256(f"{seed}:{nctid}".encode()).digest()
        return (int.from_bytes(digest[:8], "big") / 2**64) < val_fraction

    mask = df["nctid"].astype(str).apply(is_val)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def load_top_splits(
    val_fraction_of_trainval: float = 0.15,
    seed: int = 42,
    min_criteria_chars: int = 20,
    test_fraction: float = 0.2,
    split_mode: str = "chronological",
) -> TOPSplits:
    """Load Phase III trials and produce guarded train/val/test frames.

    split_mode="chronological" (default): pool all trials, order by numeric
    NCT ID, hold out the newest `test_fraction` as test, and hash-carve
    validation from the oldest remaining block -- then assert the guard.

    split_mode="benchmark": return the authors' predefined phase_III_* files
    untouched. NOTE: those files are NOT chronological (train contains
    trials newer than much of test; validation is era-identical to train),
    so the anti-lookahead guard is NOT asserted in this mode. Use only for
    leaderboard-comparable ablations, never for headline results.
    """
    raw = _fetch_hint_splits()

    report = CanonicalizationReport()
    processed: list[pd.DataFrame] = []
    dropped_no_smiles = 0
    dropped_criteria = 0

    for name in ("train", "val", "test"):
        part = raw[name].copy()
        for col in LIST_FIELDS:
            part[col] = part[col].apply(_parse_list_field)
        for col in DROP_FIELDS:
            if col in part.columns:
                part = part.drop(columns=col)

        part["smiles_canonical"] = part["smiless"].apply(
            lambda lst: canonicalize_molecule_list(lst, report)
        )
        part["n_smiles"] = part["smiles_canonical"].str.len()
        part["phase_index"] = part["phase"].apply(parse_phase)

        ok_criteria = part["criteria"].notna() & (part["criteria"].str.len() >= min_criteria_chars)
        dropped_no_smiles += int((part["n_smiles"] == 0).sum())
        dropped_criteria += int((~ok_criteria).sum())
        processed.append(part[ok_criteria & (part["n_smiles"] > 0)].reset_index(drop=True))

    if split_mode == "chronological":
        pooled = pd.concat(processed, ignore_index=True)
        pooled = pooled.drop_duplicates(subset="nctid", keep="first").reset_index(drop=True)
        train, val, test = chronological_split(pooled, test_fraction, val_fraction_of_trainval, seed)
    elif split_mode == "benchmark":
        import warnings

        warnings.warn(
            "split_mode='benchmark': the predefined files are not chronological; "
            "the anti-lookahead guard is skipped in this mode.",
            stacklevel=2,
        )
        train, val, test = processed
    else:
        raise ValueError(f"unknown split_mode: {split_mode!r}")

    splits = TOPSplits(
        train=train,
        val=val,
        test=test,
        canon_report=report,
        dropped_no_valid_smiles=dropped_no_smiles,
        dropped_short_or_missing_criteria=dropped_criteria,
    )
    if split_mode == "chronological":
        assert_temporal_order((splits.train, splits.val, splits.test))
    return splits


def phase_coverage(splits: TOPSplits) -> dict:
    """Per-split counts of exact-phase vs UNK (combo/missing) trials."""
    names = ("train", "val", "test")
    parts = (splits.train, splits.val, splits.test)
    out: dict[str, dict] = {}
    for name, part in zip(names, parts):
        idx = part["phase_index"]
        out[name] = {
            "phase_1": int((idx == 0).sum()),
            "phase_2": int((idx == 1).sum()),
            "phase_3": int((idx == 2).sum()),
            "phase_4": int((idx == 3).sum()),
            "unk_combo_or_missing": int((idx == 4).sum()),
        }
    return out


def split_stats(splits: TOPSplits) -> dict:
    stats: dict[str, dict] = {}
    for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        n_mol = part["n_smiles"]
        stats[name] = {
            "n_trials": len(part),
            "n_pos": int(part["label"].sum()),
            "pos_fraction": float(part["label"].mean()) if len(part) else 0.0,
            "molecules_per_trial_mean": float(n_mol.mean()) if len(n_mol) else 0.0,
            "molecules_per_trial_max": int(n_mol.max()) if len(n_mol) else 0,
            "min_nct": int(part["nctid"].map(nct_number).min()),
            "max_nct": int(part["nctid"].map(nct_number).max()),
        }
    return stats
