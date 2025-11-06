"""Design-time tabular features for TriStreamNet (leakage-safe).

Uses only information known at trial start / design time:
  enrollment, number_of_arms, has_dmc, source_class, molecule_mask,
  n_prior_drug_approvals (from approval_labels / Drugs@FDA pre-start).

Explicitly EXCLUDED (post-hoc / leaky): why_stopped, overall_status,
completion_year, results dates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fixed feature layout (order matters for the encoder).
# Indices 0-6: legacy design features (stage <= 6).
# Indices 7+: stage-7 additions (chemistry descriptors, modality, mechanism,
# sponsor track record) — joined from data/processed/stage7_features.parquet
# via merge_datasets._join_stage7_features().
LEGACY_FEATURE_NAMES = (
    "enrollment_log1p",
    "number_of_arms",
    "has_dmc",
    "source_class_industry",
    "molecule_present",
    "n_prior_drug_approvals",
    "is_fda_regulated_drug",
)
STAGE7_FEATURE_NAMES = (
    # rdkit descriptors (masked when no resolved molecule)
    "chem_mw_mean", "chem_logp_mean", "chem_tpsa_mean", "chem_rotb_mean",
    "chem_hbd_mean", "chem_hba_mean", "chem_fsp3_mean", "chem_arom_frac_mean",
    # modality one-hots (always present)
    "mod_small_molecule", "mod_mab", "mod_biologic_other", "mod_cell_gene",
    "mod_vaccine", "mod_unknown",
    # mechanism block (ChEMBL; masked when no drug mapped)
    "mech_known", "mech_enzyme", "mech_gpcr", "mech_ion_channel",
    "mech_kinase", "mech_nuclear_receptor", "mech_other_protein",
    "act_inhibitor", "act_agonist", "act_antagonist", "act_other",
    "mech_competitors_log1p",
    # sponsor block (masked when sponsor unknown/unmatched)
    "sponsor_prior_approvals_log1p", "sponsor_has_prior",
    "sponsor_prior_trials_log1p",
)
# Stage-8 block: TxGNN drug-node embeddings, PCA-projected to 64 dims (fit on
# train split only to prevent leakage); masked where no drug maps onto KG.
STAGE8_FEATURE_NAMES = tuple(f"kg_{i}" for i in range(64))

TABULAR_FEATURE_NAMES = LEGACY_FEATURE_NAMES + STAGE7_FEATURE_NAMES + STAGE8_FEATURE_NAMES
N_TABULAR_FEATURES = len(TABULAR_FEATURE_NAMES)  # 100 = 7 legacy + 29 stage7 + 64 stage8(kg pca)
N_LEGACY_FEATURES = len(LEGACY_FEATURE_NAMES)
N_STAGE7_FEATURES = len(STAGE7_FEATURE_NAMES)
N_STAGE8_FEATURES = len(STAGE8_FEATURE_NAMES)


@dataclass
class TabularStats:
    """Train-split means / stds for continuous columns (mask-aware)."""

    means: np.ndarray
    stds: np.ndarray
    feature_names: tuple[str, ...] = TABULAR_FEATURE_NAMES

    def to_dict(self) -> dict:
        return {
            "means": self.means.astype(float).tolist(),
            "stds": self.stds.astype(float).tolist(),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TabularStats":
        return cls(
            means=np.asarray(d["means"], dtype=np.float32),
            stds=np.asarray(d["stds"], dtype=np.float32),
            feature_names=tuple(d.get("feature_names") or TABULAR_FEATURE_NAMES),
        )


def _to_float(val, default: float = np.nan) -> float:
    if val is None:
        return default
    if isinstance(val, (bool, np.bool_)):
        return float(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        if isinstance(val, float) and np.isnan(val):
            return default
        return float(val)
    s = str(val).strip().lower()
    if s in ("", "nan", "none", "null", "nat"):
        return default
    if s in ("t", "true", "yes", "y", "1"):
        return 1.0
    if s in ("f", "false", "no", "n", "0"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return default


def _source_class_industry(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().upper()
    if not s or s in ("NAN", "NONE", "NULL", ""):
        return np.nan
    return 1.0 if "INDUSTRY" in s else 0.0


def row_tabular_raw(row: dict | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_feats[N], present_mask[N]) without normalization."""
    get = row.get if isinstance(row, dict) else lambda k, default=None: row[k] if k in row.index else default

    enrollment = _to_float(get("enrollment"))
    enrollment_log = np.log1p(max(enrollment, 0.0)) if np.isfinite(enrollment) else np.nan

    arms = _to_float(get("number_of_arms"))
    dmc = _to_float(get("has_dmc"))
    industry = _source_class_industry(get("source_class"))
    mol = _to_float(get("molecule_mask"), default=0.0)
    if not np.isfinite(mol):
        mol = 0.0
    prior = _to_float(get("n_prior_drug_approvals"), default=0.0)
    if not np.isfinite(prior):
        prior = 0.0
    fda_reg = _to_float(get("is_fda_regulated_drug"))

    legacy = [enrollment_log, arms, dmc, industry, mol, prior, fda_reg]
    legacy_mask = list(np.isfinite(np.array(legacy, dtype=np.float64)).astype(float))
    legacy_mask[4] = 1.0  # molecule_present default meaningful
    legacy_mask[5] = 1.0  # n_prior_drug_approvals default meaningful
    legacy = np.nan_to_num(np.array(legacy, dtype=np.float32), nan=0.0)

    # Stage-7 block: sidecar columns stage7_feats / stage7_mask (lists of
    # floats), zero-filled + mask-off when the join missed (older rows).
    s7_feats_raw = get("stage7_feats")
    s7_mask_raw = get("stage7_mask")

    def _as_vec(raw, size):
        if raw is None:
            return np.zeros(size, dtype=np.float32)
        try:
            arr = np.asarray(list(raw), dtype=np.float32)
        except (TypeError, ValueError):
            return np.zeros(size, dtype=np.float32)
        if arr.shape != (size,):
            return np.zeros(size, dtype=np.float32)
        return arr

    s7_feats = _as_vec(s7_feats_raw, len(STAGE7_FEATURE_NAMES))
    s7_mask = _as_vec(s7_mask_raw, len(STAGE7_FEATURE_NAMES))

    # Stage-8 block: sidecar columns kg_pca (list[64]) / kg_pca_mask (scalar
    # 0/1 in the sidecar; broadcast to all 64 dims).
    s8_feats = _as_vec(get("kg_pca"), len(STAGE8_FEATURE_NAMES))
    kg_mask_raw = get("kg_pca_mask")
    try:
        kg_mask_f = float(kg_mask_raw)
    except (TypeError, ValueError):
        kg_mask_f = 0.0
    s8_mask = np.full(len(STAGE8_FEATURE_NAMES), kg_mask_f, dtype=np.float32)

    feats = np.concatenate([legacy, s7_feats, s8_feats])
    mask = np.concatenate([np.asarray(legacy_mask, dtype=np.float32), s7_mask, s8_mask])
    return feats.astype(np.float32), mask.astype(np.float32)


def fit_tabular_stats(frame: pd.DataFrame) -> TabularStats:
    mats = []
    masks = []
    for _, row in frame.iterrows():
        f, m = row_tabular_raw(row)
        mats.append(f)
        masks.append(m)
    X = np.stack(mats)
    M = np.stack(masks)
    means = np.zeros(N_TABULAR_FEATURES, dtype=np.float32)
    stds = np.ones(N_TABULAR_FEATURES, dtype=np.float32)
    for j in range(N_TABULAR_FEATURES):
        present = M[:, j] > 0.5
        if present.any():
            means[j] = float(X[present, j].mean())
            s = float(X[present, j].std())
            stds[j] = s if s > 1e-6 else 1.0
    return TabularStats(means=means, stds=stds)


def normalize_tabular(feats: np.ndarray, mask: np.ndarray, stats: TabularStats | None) -> np.ndarray:
    """Z-score continuous cols using train stats; missing stays 0 with mask."""
    out = feats.astype(np.float32).copy()
    if stats is None:
        return out
    # Continuous columns: legacy enrollment_log(0), arms(1) + stage-7
    # descriptor block (indices 7..14 = mw/logp/tpsa/rotb/hbd/hba/fsp3/arom).
    # Stage-8 KG PCA block (indices 36..99) is NOT re-normalized — PCA column
    # ordering carries variance structure we want to keep.
    continuous = (0, 1, 7, 8, 9, 10, 11, 12, 13, 14)
    competitors_idx = TABULAR_FEATURE_NAMES.index("mech_competitors_log1p")
    continuous_eff = tuple(j for j in continuous + (competitors_idx,) if j < len(out))
    for j in continuous_eff:
        present = mask[j] > 0.5
        if present:
            out[j] = ((out[j] - stats.means[j]) / stats.stds[j]) * mask[j]
        else:
            out[j] = 0.0
    return out


def attach_tabular_features(
    frame: pd.DataFrame,
    stats: TabularStats | None = None,
    fit_stats: bool = False,
) -> tuple[pd.DataFrame, TabularStats | None]:
    """Add tabular_feats (list[float]) + tabular_mask columns."""
    out = frame.copy()
    fitted = stats
    if fit_stats:
        fitted = fit_tabular_stats(out)
    feats_col = []
    mask_col = []
    for _, row in out.iterrows():
        raw, mask = row_tabular_raw(row)
        feats_col.append(normalize_tabular(raw, mask, fitted).tolist())
        mask_col.append(mask.tolist())
    out["tabular_feats"] = feats_col
    out["tabular_mask"] = mask_col
    return out, fitted
