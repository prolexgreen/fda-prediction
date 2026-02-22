"""openFDA Drugs@FDA lookup: prior approvals + first approval date.

Queries https://api.fda.gov/drug/drugsfda.json by active ingredient or
brand name. JSON responses are cached on disk (same pattern as
pubchem_smiles.py). Fully failure-tolerant: network errors, 404s, and
malformed payloads return None / empty results so the live pipeline and
tests work offline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests

OPENFDA_BASE = "https://api.fda.gov/drug/drugsfda.json"
REQUEST_TIMEOUT_S = 15.0
MAX_ATTEMPTS = 2  # 1 try + at most 1 retry


def _cache_dir():
    from pathlib import Path

    from fda_predictor.utils.paths import RAW_DATA_DIR

    return Path(RAW_DATA_DIR) / "openfda_cache"


def _cache_path(query: str):
    safe = "".join(c if c.isalnum() else "_" for c in query.lower())[:120]
    return _cache_dir() / f"{safe}.json"


@dataclass
class ApprovalInfo:
    query: str
    has_prior_approval: bool
    first_approval_date: str | None = None
    application_numbers: list[str] | None = None
    brand_names: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "has_prior_approval": self.has_prior_approval,
            "first_approval_date": self.first_approval_date,
            "application_numbers": self.application_numbers or [],
            "brand_names": self.brand_names or [],
        }


def _search_expr(name: str, search_field: str) -> str:
    field = "openfda.generic_name" if search_field == "active_ingredient" else "openfda.brand_name"
    return f'{field}:"{name}"'


def fetch_drug_application(
    drug_name: str,
    search_field: str = "active_ingredient",
    limit: int = 25,
    timeout_s: float = REQUEST_TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
    use_cache: bool = True,
) -> dict | None:
    """Return the openFDA result payload, or None when unavailable.

    Cache stores either {"results": [...]} on success or {"results": [],
    "not_found": true} for definitive 404 no-hits, so offline reruns are
    stable. Transient failures return None without poisoning the cache.
    """
    name = str(drug_name).strip()
    if not name:
        return None
    cache = _cache_path(f"{search_field}:{name}")
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # corrupt cache entry -> refetch once below

    url = (
        f"{OPENFDA_BASE}?search={requests.utils.quote(_search_expr(name, search_field))}"
        f"&limit={min(int(limit), 100)}"
    )
    payload: dict | None = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            if resp.status_code == 404:
                payload = {"results": [], "not_found": True}
                break
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError):
            if attempt >= int(max_attempts):
                payload = None
                break
            time.sleep(0.5)

    if payload is None:
        return None  # transient failure: do not cache

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass
    return payload


def parse_approval_info(payload: dict | None, query: str = "") -> ApprovalInfo:
    """Extract prior-approval facts from an openFDA Drugs@FDA payload."""
    if not payload:
        return ApprovalInfo(query=query, has_prior_approval=False)
    results = payload.get("results") or []
    if not results:
        return ApprovalInfo(query=query, has_prior_approval=False)

    apps: list[str] = []
    brands: list[str] = []
    dates: list[str] = []
    for res in results:
        num = res.get("application_number")
        if num:
            apps.append(str(num))
        openfda = res.get("openfda") or {}
        for b in openfda.get("brand_name") or []:
            if b not in brands:
                brands.append(str(b))
        # submissions carry approval-event dates (YYYYMMDD strings)
        for sub in res.get("submissions") or []:
            stype = str(sub.get("submission_type") or "").upper()
            status = str(sub.get("submission_status") or "").upper()
            for key in ("submission_status_date", "approval_date"):
                d = sub.get(key)
                if isinstance(d, str) and len(d) >= 8 and d[:8].isdigit():
                    if stype in ("ORIG", "ORIGINAL") and status == "APPROVED":
                        dates.insert(0, d[:8])
                    else:
                        dates.append(d[:8])

    first_date = min(dates) if dates else None
    return ApprovalInfo(
        query=query,
        has_prior_approval=True,
        first_approval_date=first_date,
        application_numbers=apps[:10],
        brand_names=brands[:10],
    )


def check_drug_approvals(
    drug_name: str,
    search_field: str = "active_ingredient",
    timeout_s: float = REQUEST_TIMEOUT_S,
    use_cache: bool = True,
) -> ApprovalInfo:
    """One-call convenience API used by the live report."""
    payload = fetch_drug_application(
        drug_name,
        search_field=search_field,
        timeout_s=timeout_s,
        use_cache=use_cache,
    )
    return parse_approval_info(payload, query=str(drug_name).strip())
