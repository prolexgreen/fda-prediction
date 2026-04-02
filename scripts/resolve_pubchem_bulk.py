"""Offline bulk resolution of drug names -> canonical SMILES.

PubChem PUG REST 503s under load; the FTP bulk files are unaffected.

Pass 1: CID-Synonym-filtered.gz  -> candidate name -> CID (wanted names only)
Pass 2: CID-SMILES.gz            -> CID -> isomeric SMILES (matched CIDs only)

Resolves every wanted cache key: cached-null drug names, every raw name in
merged_trials.drugs, their combo-split parts, and normalized forms.
Writes results into the existing per-name JSON cache, so downstream code
(pubchem_smiles.resolve_drug_smiles / cached_smiles / backfill) is unchanged.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.data.preprocessing import canonicalize_smiles  # noqa: E402
from fda_predictor.data.pubchem_smiles import (  # noqa: E402
    normalize_drug_name,
    split_combination,
    _cache_path,
)
from fda_predictor.utils.paths import MERGED_PROCESSED_PARQUET, PUBCHEM_CACHE_DIR  # noqa: E402

BULK_DIR = Path(r"D:\Models\pubchem_bulk")
SYN_GZ = BULK_DIR / "CID-Synonym-filtered.gz"
SMILES_GZ = BULK_DIR / "CID-SMILES.gz"


def iter_tsv_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            idx = line.find("\t")
            if idx <= 0:
                continue
            yield line[:idx], line[idx + 1 :].rstrip("\n")


def build_wanted() -> dict[str, set[str]]:
    """cache key string -> set of lowercased lookup candidate names."""
    wanted: dict[str, set[str]] = defaultdict(set)

    def add(raw: str):
        raw = str(raw).strip()
        if not raw:
            return
        cands = {raw.lower(), normalize_drug_name(raw)}
        for part in split_combination(raw):
            cands.add(part.strip().lower())
            cands.add(normalize_drug_name(part))
        cands.discard("")
        # Emit cache entries for the raw key, normalized key, and each part —
        # the site:cache lookup order in the patch loop is (raw, normalized).
        wanted[raw].update(cands)
        norm = normalize_drug_name(raw)
        if norm and norm != raw:
            wanted[norm].update(cands)
        for part in split_combination(raw):
            p = part.strip().lower()
            if p and p != raw.lower():
                wanted[p].update(cands)

    for f in sorted(PUBCHEM_CACHE_DIR.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("smiles") is None and payload.get("drug"):
            add(str(payload["drug"]))

    m = pd.read_parquet(MERGED_PROCESSED_PARQUET, columns=["drugs"])
    for raw in m["drugs"].tolist():
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = [raw]
        else:
            try:
                raw = list(raw)
            except TypeError:
                continue
        for d in raw:
            add(d)
    return wanted


def main() -> int:
    if not SYN_GZ.exists() or not SMILES_GZ.exists():
        print("bulk files missing; run download_pubchem_bulk.py first")
        return 1

    wanted = build_wanted()
    cand_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, cands in wanted.items():
        for c in cands:
            cand_to_keys[c].append(key)
    print(f"wanted cache keys: {len(wanted)} | lookup candidates: {len(cand_to_keys)}", flush=True)

    # pass 1: name -> cid
    name_to_cid: dict[str, int] = {}
    n_lines = 0
    for cid_s, syn in iter_tsv_gz(SYN_GZ):
        n_lines += 1
        key = syn.strip().lower()
        if key in cand_to_keys and key not in name_to_cid:
            try:
                name_to_cid[key] = int(cid_s)
            except ValueError:
                pass
        if n_lines % 10_000_000 == 0:
            print(f"  synonyms: {n_lines/1e6:.0f}M lines, matched {len(name_to_cid)}", flush=True)
    print(f"pass1 done: {n_lines} lines, {len(name_to_cid)} candidates matched", flush=True)

    # pass 2: cid -> smiles
    needed_cids = set(name_to_cid.values())
    cid_to_smiles: dict[int, str] = {}
    n_lines = 0
    for cid_s, smi in iter_tsv_gz(SMILES_GZ):
        n_lines += 1
        try:
            cid = int(cid_s)
        except ValueError:
            continue
        if cid in needed_cids:
            cid_to_smiles[cid] = smi
        if n_lines % 10_000_000 == 0:
            print(f"  smiles: {n_lines/1e6:.0f}M lines, resolved {len(cid_to_smiles)}", flush=True)
    print(f"pass2 done: {n_lines} lines, {len(cid_to_smiles)} smiles resolved", flush=True)

    # write cache entries
    written = 0
    resolved_keys: set[str] = set()
    for cand, cid in name_to_cid.items():
        smi = cid_to_smiles.get(cid)
        if not smi:
            continue
        canon = canonicalize_smiles(smi)
        if not canon:
            continue
        for key in cand_to_keys[cand]:
            if key in resolved_keys:
                continue
            p = _cache_path(key)
            if p.exists():
                try:
                    existing = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = None
                if existing and existing.get("smiles"):
                    resolved_keys.add(key)
                    continue  # never clobber a real hit
            p.write_text(
                json.dumps({"drug": key, "smiles": canon, "source": "pubchem_bulk"}),
                encoding="utf-8",
            )
            resolved_keys.add(key)
            written += 1

    print(f"cache entries written/upgraded: {written}", flush=True)

    cache_hits = sum(
        1
        for f in PUBCHEM_CACHE_DIR.glob("*.json")
        if json.loads(f.read_text(encoding="utf-8")).get("smiles")
    )
    print(f"pubchem cache now: {cache_hits} hits", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
