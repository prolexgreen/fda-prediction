"""Build Stage-7 feature sidecar keyed by nctid.

Sources:
- Chemistry descriptors (rdkit) from resolved canonical SMILES
- Modality one-hots from drug names
- ChEMBL mechanism/target match from sqlite dump
- Sponsor track record from Drugs@FDA bulk dump

Writes data/processed/stage7_features.parquet. All design-time information:
leakage-safe by construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.data.chem_features import (  # noqa: E402
    ChemblMechanismIndex,
    DESCRIPTOR_NAMES,
    MODALITY_NAMES,
    aggregate_descriptors,
    classify_modality,
    trial_mechanism_features,
)
from fda_predictor.data.sponsor_features import (  # noqa: E402
    load_drugsfda_records,
    normalize_sponsor,
    trial_sponsor_features,
)
from fda_predictor.utils.paths import (  # noqa: E402
    MERGED_PROCESSED_PARQUET,
    STAGE7_FEATURES_PARQUET,
    ensure_dirs,
)

CHEMBL_DB = next(
    Path(r"D:\Models\chembl").rglob("*.db"),
    None,
)

MECH_NAMES = tuple(
    ["mech_known"]
    + [f"mech_{b}" for b in ("enzyme", "gpcr", "ion_channel", "kinase", "nuclear_receptor", "other_protein")]
    + [f"act_{b}" for b in ("inhibitor", "agonist", "antagonist", "other")]
    + ["mech_competitors_log1p"]
)
SPONSOR_NAMES = ("sponsor_prior_approvals_log1p", "sponsor_has_prior", "sponsor_prior_trials_log1p")

FEATURE_NAMES = DESCRIPTOR_NAMES + MODALITY_NAMES + MECH_NAMES + SPONSOR_NAMES
MASK_NAMES = tuple(f"mask_{n}" for n in FEATURE_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def _parse_smiles(raw) -> list[str]:
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
    return []


def _parse_drugs(raw) -> list[str]:
    return _parse_smiles(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(MERGED_PROCESSED_PARQUET))
    parser.add_argument("--out", default=str(STAGE7_FEATURES_PARQUET))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit).copy()
    n = len(df)
    print(f"rows: {n}", flush=True)
    print(f"chembl db: {CHEMBL_DB}", flush=True)

    print("loading ChEMBL mechanism index ...", flush=True)
    mech_index = ChemblMechanismIndex(db_path=CHEMBL_DB) if CHEMBL_DB else None

    print("loading Drugs@FDA bulk dump ...", flush=True)
    try:
        drugsfda = load_drugsfda_records()
        print(f"  drugsfda approvals rows: {len(drugsfda)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: drugsfda load failed ({e!r}); sponsor features will be masked", flush=True)
        drugsfda = None

    print("computing sponsor features ...", flush=True)
    sp_feats, sp_masks = trial_sponsor_features(df, drugsfda)

    print("computing chemistry + mechanism features ...", flush=True)
    rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 2000 == 0:
            print(f"  {i}/{n}", flush=True)
        smiles = _parse_smiles(row.get("smiles_canonical"))
        drugs = _parse_drugs(row.get("drugs"))
        chem_feats, chem_mask = aggregate_descriptors(smiles)
        mod_vec = classify_modality(drugs, has_smiles=bool(smiles))
        mech_feats, mech_mask = trial_mechanism_features(mech_index, drugs)
        feats = np.concatenate([chem_feats, mod_vec, mech_feats, sp_feats[i]])
        masks = np.concatenate(
            [
                chem_mask,
                np.ones(6, dtype=np.float32),  # modality always determinable
                mech_mask,
                sp_masks[i],
            ]
        )
        rows.append(
            {
                "nctid": row["nctid"],
                "stage7_feats": feats.tolist(),
                "stage7_mask": masks.tolist(),
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out.to_parquet(out_path, index=False)

    # Coverage report
    chem_cov = float(np.mean([1.0 if any(m) else 0.0 for m in out["stage7_mask"].apply(lambda m: m[:8])]))
    mech_cov = float(np.mean([m[8 + 6] for m in out["stage7_mask"]]))
    sp_cov = float(np.mean([m[8 + 6 + 12] for m in out["stage7_mask"]]))
    print(f"\nfeature coverage: chemistry={chem_cov:.1%} mechanism={mech_cov:.1%} sponsor={sp_cov:.1%}", flush=True)
    print(f"wrote {out_path} ({len(out)} rows, {N_FEATURES} features)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
