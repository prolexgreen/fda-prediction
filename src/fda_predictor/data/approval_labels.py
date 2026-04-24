"""FDA approval labels from Drugs@FDA ORIG submission dates.

For each trial, look up active-ingredient / brand approvals for its drugs
and compare the earliest ORIG+APPROVED date against the trial start date:

- approval_label = 1  if any drug has an ORIG approval AFTER trial start
- approval_label = 0  if drugs resolve and none have a post-start ORIG approval,
                      and at least one is NOT previously approved (novel failure)
- previously_approved = True when every resolvable drug was approved BEFORE start
  (ambiguous indication/relabeling cases; excluded from approval-target metrics)
- approval_label = NaN when no drug name resolves in Drugs@FDA

All network calls go through openfda_client (disk-cached). Features derived
here that use only pre-start approvals are leakage-safe for training.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from fda_predictor.inference.openfda_client import (
    ApprovalInfo,
    check_drug_approvals,
    fetch_drug_application,
    parse_approval_info,
)


def _parse_drugs(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                raw = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return [s] if s else []
        else:
            return [s] if s else []
    if isinstance(raw, (list, tuple)):
        return [str(d).strip() for d in raw if str(d).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _parse_date(value) -> datetime | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "nat", ""):
        return None
    # openFDA dates are often YYYYMMDD
    if len(s) >= 8 and s[:8].isdigit() and "-" not in s[:10]:
        try:
            return datetime.strptime(s[:8], "%Y%m%d")
        except ValueError:
            pass
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def earliest_orig_approval_date(payload: dict | None) -> datetime | None:
    """Earliest ORIG/ORIGINAL + APPROVED submission date in a Drugs@FDA payload."""
    if not payload:
        return None
    dates: list[datetime] = []
    for res in payload.get("results") or []:
        for sub in res.get("submissions") or []:
            stype = str(sub.get("submission_type") or "").upper()
            status = str(sub.get("submission_status") or "").upper()
            if stype not in ("ORIG", "ORIGINAL") or status != "APPROVED":
                continue
            for key in ("submission_status_date", "approval_date"):
                d = _parse_date(sub.get(key))
                if d is not None:
                    dates.append(d)
                    break
    return min(dates) if dates else None


def resolve_drug_approval(
    drug_name: str,
    fetch_fn: Callable[..., dict | None] | None = None,
) -> tuple[ApprovalInfo, datetime | None]:
    """Return (ApprovalInfo, earliest ORIG approval datetime) for one drug."""
    fetch = fetch_fn or fetch_drug_application
    payload = fetch(drug_name, search_field="active_ingredient", use_cache=True)
    if payload is None or not (payload.get("results") or []):
        payload = fetch(drug_name, search_field="brand_name", use_cache=True)
    info = parse_approval_info(payload, query=str(drug_name).strip())
    orig = earliest_orig_approval_date(payload)
    # Fall back to any parsed first_approval_date if ORIG parsing missed
    if orig is None and info.first_approval_date:
        orig = _parse_date(info.first_approval_date)
    return info, orig


@dataclass
class TrialApprovalLabel:
    nctid: str
    approval_label: float | None  # 0/1 or None (unknown)
    previously_approved: bool
    n_drugs: int
    n_resolved: int
    n_prior_drug_approvals: int  # drugs approved BEFORE trial start
    earliest_post_start_approval: str | None = None
    drug_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nctid": self.nctid,
            "approval_label": self.approval_label,
            "previously_approved": self.previously_approved,
            "n_drugs": self.n_drugs,
            "n_resolved": self.n_resolved,
            "n_prior_drug_approvals": self.n_prior_drug_approvals,
            "earliest_post_start_approval": self.earliest_post_start_approval,
        }


def label_trial_approval(
    nctid: str,
    drugs,
    start_date,
    fetch_fn: Callable[..., dict | None] | None = None,
    max_drugs: int = 5,
) -> TrialApprovalLabel:
    """Compute approval label for one trial (offline-friendly with fetch_fn)."""
    drug_list = _parse_drugs(drugs)[:max_drugs]
    start = _parse_date(start_date)
    details: list[dict] = []
    resolved = 0
    prior = 0
    post_start_dates: list[datetime] = []

    for name in drug_list:
        info, orig = resolve_drug_approval(name, fetch_fn=fetch_fn)
        entry = {
            "drug": name,
            "resolved": bool(info.has_prior_approval or orig is not None),
            "orig_approval_date": orig.strftime("%Y-%m-%d") if orig else None,
            "relative_to_start": None,
        }
        if info.has_prior_approval or orig is not None:
            resolved += 1
        if start is not None and orig is not None:
            if orig < start:
                prior += 1
                entry["relative_to_start"] = "before"
            elif orig > start:
                post_start_dates.append(orig)
                entry["relative_to_start"] = "after"
            else:
                entry["relative_to_start"] = "same_day"
        details.append(entry)

    previously_approved = bool(resolved > 0 and prior == resolved and not post_start_dates)

    if resolved == 0:
        label: float | None = None
    elif post_start_dates:
        label = 1.0
    elif previously_approved:
        # Ambiguous: already-approved drugs (new indication / combo). Exclude.
        label = None
    else:
        # Resolved drugs exist, none approved after start, not all previously approved.
        label = 0.0

    earliest_post = min(post_start_dates).strftime("%Y-%m-%d") if post_start_dates else None
    return TrialApprovalLabel(
        nctid=str(nctid),
        approval_label=label,
        previously_approved=previously_approved,
        n_drugs=len(drug_list),
        n_resolved=resolved,
        n_prior_drug_approvals=prior,
        earliest_post_start_approval=earliest_post,
        drug_details=details,
    )


def attach_approval_labels(
    frame: pd.DataFrame,
    fetch_fn: Callable[..., dict | None] | None = None,
    max_drugs: int = 5,
    date_col_candidates: tuple[str, ...] = (
        "start_date",
        "study_first_submitted_date",
        "chronology_date",
    ),
    progress_every: int = 200,
    workers: int = 8,
) -> pd.DataFrame:
    """Add approval_label (+ helpers) columns; preserves existing success `label`."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out = frame.copy()
    date_col = next((c for c in date_col_candidates if c in out.columns), None)

    unique_drugs: list[str] = []
    seen: set[str] = set()
    for raw in out["drugs"].tolist() if "drugs" in out.columns else []:
        for d in _parse_drugs(raw)[:max_drugs]:
            key = d.lower()
            if key not in seen:
                seen.add(key)
                unique_drugs.append(d)
    print(f"  unique drugs to resolve: {len(unique_drugs)} (workers={workers})", flush=True)

    drug_cache: dict[str, tuple] = {}

    def _one(name: str):
        return name, resolve_drug_approval(name, fetch_fn=fetch_fn)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_one, name) for name in unique_drugs]
        for fut in as_completed(futures):
            name, pair = fut.result()
            drug_cache[name.lower()] = pair
            done += 1
            if progress_every and done % max(progress_every, 50) == 0:
                print(f"  drug resolve: {done}/{len(unique_drugs)}", flush=True)

    rows: list[dict] = []
    n = len(out)
    for i, row in enumerate(out.itertuples(index=False), start=1):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(out.columns, row))
        nctid = row_dict.get("nctid") or row_dict.get("nct_id") or ""
        start = row_dict.get(date_col) if date_col else None
        drug_list = _parse_drugs(row_dict.get("drugs"))[:max_drugs]
        start_dt = _parse_date(start)
        details: list[dict] = []
        resolved = 0
        prior = 0
        post_start_dates: list = []
        for name in drug_list:
            info, orig = drug_cache.get(name.lower()) or resolve_drug_approval(name, fetch_fn=fetch_fn)
            entry = {
                "drug": name,
                "resolved": bool(info.has_prior_approval or orig is not None),
                "orig_approval_date": orig.strftime("%Y-%m-%d") if orig else None,
                "relative_to_start": None,
            }
            if info.has_prior_approval or orig is not None:
                resolved += 1
            if start_dt is not None and orig is not None:
                if orig < start_dt:
                    prior += 1
                    entry["relative_to_start"] = "before"
                elif orig > start_dt:
                    post_start_dates.append(orig)
                    entry["relative_to_start"] = "after"
                else:
                    entry["relative_to_start"] = "same_day"
            details.append(entry)
        previously_approved = bool(resolved > 0 and prior == resolved and not post_start_dates)
        if resolved == 0:
            label: float | None = None
        elif post_start_dates:
            label = 1.0
        elif previously_approved:
            label = None
        else:
            label = 0.0
        earliest_post = min(post_start_dates).strftime("%Y-%m-%d") if post_start_dates else None
        rows.append(
            TrialApprovalLabel(
                nctid=str(nctid),
                approval_label=label,
                previously_approved=previously_approved,
                n_drugs=len(drug_list),
                n_resolved=resolved,
                n_prior_drug_approvals=prior,
                earliest_post_start_approval=earliest_post,
                drug_details=details,
            ).to_dict()
        )
        if progress_every and i % progress_every == 0:
            print(f"  approval labels: {i}/{n}", flush=True)

    side = pd.DataFrame(rows)
    drop_cols = [c for c in side.columns if c != "nctid" and c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    if "nctid" in out.columns:
        out = out.merge(side, on="nctid", how="left")
    else:
        for col in side.columns:
            if col == "nctid":
                continue
            out[col] = side[col].to_numpy()
    return out


def approval_label_coverage(frame: pd.DataFrame) -> dict:
    """Summary stats for the approval-label gate."""
    if "approval_label" not in frame.columns:
        return {"n": len(frame), "labeled": 0, "coverage": 0.0}
    lab = pd.to_numeric(frame["approval_label"], errors="coerce")
    labeled = lab.notna()
    prev = frame["previously_approved"].fillna(False).astype(bool) if "previously_approved" in frame.columns else False
    out = {
        "n": int(len(frame)),
        "labeled": int(labeled.sum()),
        "coverage": float(labeled.mean()) if len(frame) else 0.0,
        "pos_frac_labeled": float(lab[labeled].mean()) if labeled.any() else 0.0,
        "previously_approved": int(prev.sum()) if isinstance(prev, pd.Series) else 0,
        "unresolved": int((~labeled & ~prev).sum()) if isinstance(prev, pd.Series) else int((~labeled).sum()),
    }
    if "phase_index" in frame.columns:
        out["by_phase"] = {}
        for idx, name in {0: "I", 1: "II", 2: "III", 3: "IV", 4: "UNK"}.items():
            m = frame["phase_index"] == idx
            if not m.any():
                continue
            sub = lab[m]
            out["by_phase"][name] = {
                "n": int(m.sum()),
                "labeled": int(sub.notna().sum()),
                "coverage": float(sub.notna().mean()),
                "pos_frac_labeled": float(sub.dropna().mean()) if sub.notna().any() else 0.0,
            }
    if "split" in frame.columns:
        out["by_split"] = {}
        for split, part in frame.groupby("split"):
            sub = pd.to_numeric(part["approval_label"], errors="coerce")
            out["by_split"][str(split)] = {
                "n": int(len(part)),
                "labeled": int(sub.notna().sum()),
                "coverage": float(sub.notna().mean()) if len(part) else 0.0,
                "pos_frac_labeled": float(sub.dropna().mean()) if sub.notna().any() else 0.0,
            }
    return out


# Convenience re-export for scripts that already import check_drug_approvals
__all__ = [
    "TrialApprovalLabel",
    "attach_approval_labels",
    "approval_label_coverage",
    "earliest_orig_approval_date",
    "label_trial_approval",
    "resolve_drug_approval",
    "check_drug_approvals",
]
