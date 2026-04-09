"""Backfill SMILES coverage: re-resolve cached-null drug names with the
enhanced lookup ladder (normalization, combo splitting, synonym fallback),
then patch data/processed/merged_trials.parquet in place.

Prints before/after molecule coverage (overall, per split, Phase III).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.pubchem_smiles import (  # noqa: E402
    PubChemReport,
    cached_smiles,
    resolve_drug_list,
    resolve_drug_smiles,
)
from fda_predictor.utils.paths import (  # noqa: E402
    MERGED_PROCESSED_PARQUET,
    PUBCHEM_CACHE_DIR,
    ensure_dirs,
)


def _iter_null_drug_names(cache_dir: Path) -> list[str]:
    out: list[str] = []
    for f in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("smiles") is None and payload.get("drug") and not payload.get("enhanced_miss"):
            out.append(str(payload["drug"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-patch-merged", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the network retry pass; patch merged parquet from cache only",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ensure_dirs()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    max_drugs = int(config.get("cto", {}).get("max_drugs_pubchem", 5))

    recovered: list[tuple[str, str]] = []
    n_retried = 0
    if args.offline:
        print("offline mode: skipping network retry pass", flush=True)
    else:
        nulls = _iter_null_drug_names(PUBCHEM_CACHE_DIR)
        if args.limit:
            nulls = nulls[: args.limit]
        n_retried = len(nulls)
        print(f"cached-null entries to re-attempt: {len(nulls)} (workers={args.workers})", flush=True)

        def _one(name: str):
            # enhanced=True: reads null cache, runs normalization/synonym ladder,
            # upgrades cache on success.
            return name, resolve_drug_smiles(name, report=None, enhanced=True, delay_s=0.15)

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_one, n) for n in nulls]
            for fut in as_completed(futures):
                done += 1
                name, smi = fut.result()
                if smi:
                    recovered.append((name, smi))
                if done % 500 == 0:
                    print(f"  {done}/{len(nulls)} retried, recovered={len(recovered)}", flush=True)

        print(f"\nrecovered {len(recovered)}/{n_retried} null entries", flush=True)
        for name, smi in recovered[:20]:
            safe = name.encode("ascii", errors="replace").decode("ascii")
            print(f"  + {safe!r} -> {smi[:60]}", flush=True)

    if args.no_patch_merged:
        return 0

    # ---- patch source + merged parquets ---------------------------------
    # merged_trials.parquet is REGENERATED from CTO source on every
    # load_merged_splits() call, so the CTO parquet must be patched first;
    # merged is patched (or rebuilt) after.
    from fda_predictor.utils.paths import CTO_PROCESSED_PARQUET

    def _patch_frame(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, float, int]:
        before = float(df["molecule_mask"].mean())
        needs = df[df["molecule_mask"] == 0]
        n_up = 0
        for idx, row in needs.iterrows():
            drugs = row.get("drugs")
            if drugs is None:
                continue
            if isinstance(drugs, str):
                drugs = [drugs]
            elif not isinstance(drugs, list):
                try:
                    drugs = [str(d) for d in list(drugs)]
                except TypeError:
                    continue
            drugs = [str(d) for d in drugs if str(d).strip() and str(d).lower() not in ("nan", "none")]
            if not drugs:
                continue
            smiles: list[str] = []
            seen: set[str] = set()
            from fda_predictor.data.pubchem_smiles import (
                normalize_drug_name,
                split_combination,
            )

            for drug in drugs:
                for part in split_combination(drug):
                    for cand in (part, normalize_drug_name(part)):
                        smi = cached_smiles(cand)
                        if smi and smi not in seen:
                            seen.add(smi)
                            smiles.append(smi)
                            break
                    if len(smiles) >= max_drugs:
                        break
            if smiles:
                df.at[idx, "smiles_canonical"] = smiles
                df.at[idx, "molecule_mask"] = 1
                df.at[idx, "n_smiles"] = len(smiles)
                n_up += 1
        after = float(df["molecule_mask"].mean())
        print(f"{name}: mask 0->1 on {n_up} rows | coverage {before:.3f} -> {after:.3f}", flush=True)
        return df, after, n_up

    if CTO_PROCESSED_PARQUET.exists():
        print("\nPatching CTO source parquet (cache-only) ...", flush=True)
        cto = pd.read_parquet(CTO_PROCESSED_PARQUET)
        cto, _, _ = _patch_frame(cto, "cto_human")
        cto.to_parquet(CTO_PROCESSED_PARQUET, index=False)

    print(f"\nPatching {MERGED_PROCESSED_PARQUET} ...", flush=True)
    merged = pd.read_parquet(MERGED_PROCESSED_PARQUET)
    before = float(merged["molecule_mask"].mean())
    before_p3 = float(merged.loc[merged["phase_index"] == 2, "molecule_mask"].mean())
    print(f"before: molecule_mask=1 overall={before:.3f} phase3={before_p3:.3f}", flush=True)

    merged, after, n_patched_rows = _patch_frame(merged, "merged_trials")
    after_p3 = float(merged.loc[merged["phase_index"] == 2, "molecule_mask"].mean())
    merged.to_parquet(MERGED_PROCESSED_PARQUET, index=False)

    split_cov = {
        s: float(merged.loc[merged["split"] == s, "molecule_mask"].mean())
        for s in ("train", "val", "test")
        if (merged["split"] == s).any()
    }
    summary = {
        "nulls_retried": n_retried,
        "nulls_recovered": len(recovered),
        "rows_patched": n_patched_rows,
        "coverage": {
            "before_overall": before,
            "after_overall": after,
            "before_phase3": before_p3,
            "after_phase3": after_p3,
            "by_split_after": split_cov,
        },
    }
    out_path = PUBCHEM_CACHE_DIR.parent.parent / "processed" / "smiles_backfill_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nafter:  molecule_mask=1 overall={after:.3f} phase3={after_p3:.3f}", flush=True)
    print(f"rows upgraded 0->1 molecule: {n_patched_rows}", flush=True)
    print(f"report: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
