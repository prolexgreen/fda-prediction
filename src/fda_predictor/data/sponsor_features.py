"""Sponsor track-record features from the Drugs@FDA bulk dump.

Leakage-safe by construction: only submissions with dates strictly before the
trial's own start_date are counted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fda_predictor.utils.paths import OPENFDA_BULK_DIR

_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|corp(?:oration)?|llc|ltd|limited|plc|s\.?a\.?|gmbh|ag|"
    r"pharmaceuticals?|pharma(?:ceutical)?|laboratories|labs|intl|international|"
    r"biotech(?:nology)?|therapeutics|company|co|holdings?)\b",
    re.IGNORECASE,
)


def normalize_sponsor(name: str | None) -> str:
    """Loose canonical form: lowercase, drop corporate suffixes, collapse spaces."""
    if not name:
        return ""
    s = str(name).strip().lower()
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())  # collapse
    return s.strip()


def _parse_date(raw: str | int | None) -> pd.Timestamp | None:
    if raw in (None, ""):
        return None
    s = str(raw).strip()
    ts = (
        pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if len(s) == 8 and s.isdigit()
        else pd.to_datetime(s, errors="coerce")
    )
    return None if ts is None or pd.isna(ts) else ts


def load_drugsfda_records(bulk_dir: Path | None = None) -> pd.DataFrame:
    """Parse every drug-drugsfda-*.json under the bulk dir into one frame.

    Columns: application_number, sponsor_name, approval_date (Timestamp),
    submission_type.
    """
    bulk_dir = bulk_dir or OPENFDA_BULK_DIR
    rows: list[dict] = []
    for path in sorted(bulk_dir.rglob("*.json")):
        if path.name.startswith("._"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for res in payload.get("results", []):
            app_no = res.get("application_number")
            sponsor = res.get("sponsor_name")
            products_approvals: list[dict] = []
            for sub in res.get("submissions", []) or []:
                status = str(sub.get("submission_status", "")).upper()
                # AP = approved (full) only; TA = tentative (not an approval grant).
                if status != "AP":
                    continue
                # ORIG = original marketing application approvals only
                stype = str(sub.get("submission_type") or "").upper()
                if stype != "ORIG":
                    continue
                date = sub.get("submission_status_date")
                stype = sub.get("submission_type") or sub.get("review_category") or ""
                ts = _parse_date(date)
                if ts is None:
                    continue
                rows.append(
                    {
                        "application_number": str(app_no),
                        "sponsor_name": sponsor,
                        "submission_type": str(stype).upper(),
                        "approval_date": ts,
                    }
                )
    df = pd.DataFrame(rows)
    return df


def sponsor_prior_stats(
    trials: pd.DataFrame,
    drugsfda: pd.DataFrame,
    max_window_years: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial (prior_approvals_log1p, sponsor_has_prior, sponsor_prior_trials_log1p, masks).

    Trials input must have: sponsor, start_date (Timestamp), nctid.
    """
    # 1) approvals per sponsor over time
    drugsfda = drugsfda.copy()
    drugsfda["sponsor_norm"] = drugsfda["sponsor_name"].map(normalize_sponsor)
    apa = (
        drugsfda[drugsfda["sponsor_norm"] != ""]
        .sort_values("approval_date")
        .groupby("sponsor_norm")["approval_date"]
        .apply(list)
        .to_dict()
    )

    # trial-set sponsor history for prior_trials (leakage-safe: earlier starts only)
    trials = trials.copy()
    trials["sponsor_norm"] = trials["sponsor"].map(normalize_sponsor)
    start_dt = pd.to_datetime(trials["start_date"], errors="coerce")
    # fall back to chronology_date where it's a real date (CTO); TOP rows carry
    # NCT-number proxies there, which never parse to real datetimes upstream
    # of this coalesce because they're numeric strings.
    chrono_raw = trials["chronology_date"] if "chronology_date" in trials.columns else None
    if chrono_raw is not None:
        chrono_dt = pd.to_datetime(chrono_raw, errors="coerce")
        start_dt = start_dt.fillna(
            chrono_dt.where(chrono_dt > pd.Timestamp("1980-01-01"))
        )
    trials["start_dt"] = start_dt
    prior_trial_dates: dict[str, list[pd.Timestamp]] = {}
    p3 = trials[pd.to_numeric(trials.get("phase_index", np.nan), errors="coerce") == 2]
    for sponsor, part in p3.groupby("sponsor_norm"):
        prior_trial_dates[sponsor] = sorted(part["start_dt"].dropna().tolist())

    n = len(trials)
    prior_app = np.zeros(n, dtype=np.float32)
    has_prior = np.zeros(n, dtype=np.float32)
    prior_trials = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)

    for i, row in enumerate(trials.itertuples(index=False)):
        sponsor_norm = getattr(row, "sponsor_norm", "")
        start_dt = getattr(row, "start_dt", pd.NaT)
        if not sponsor_norm or pd.isna(start_dt):
            continue
        mask[i] = 1.0
        dates = apa.get(sponsor_norm, [])
        if dates:
            # unique approvals before start
            count = 0
            for d in dates:
                if d < start_dt:
                    count += 1
                else:
                    break
            prior_app[i] = float(np.log1p(count))
            has_prior[i] = 1.0 if count > 0 else 0.0
        tdates = prior_trial_dates.get(sponsor_norm, [])
        if tdates:
            prior_trials[i] = float(np.log1p(sum(1 for d in tdates if d < start_dt)))

    return prior_app, has_prior, prior_trials, mask


def trial_sponsor_features(
    trials: pd.DataFrame,
    drugsfda: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (feats[3], mask[3]) per trial; rows aligned with trials order.

    Layout: [prior_approvals_log1p, has_prior_approval, prior_trials_log1p]
    Mask: [has_sponsor, has_sponsor, has_sponsor]
    """
    if drugsfda is None or drugsfda.empty:
        return np.zeros((len(trials), 3), dtype=np.float32), np.zeros((len(trials), 3), dtype=np.float32)

    prior_app, has_prior, prior_trials, mask = sponsor_prior_stats(trials, drugsfda)
    feats = np.stack([prior_app, has_prior, prior_trials], axis=1)
    masks = np.repeat(mask[:, None], 3, axis=1)
    return feats.astype(np.float32), masks.astype(np.float32)
