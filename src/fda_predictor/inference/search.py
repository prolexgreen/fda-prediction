"""ClinicalTrials.gov API v2 study search (live pipeline).

Thin wrapper over the same /api/v2/studies endpoint the batch client
uses, adding query-based search + bounded pagination. Every HTTP call
passes an explicit short timeout and performs at most one retry; any
failure degrades to an empty result instead of hanging.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from fda_predictor.inference.ctgov_client import CTGOV_BASE, _parse_study

DEFAULT_LIMIT = 20
MAX_PAGE_SIZE = 100
REQUEST_TIMEOUT_S = 15.0
MAX_ATTEMPTS = 2  # 1 try + at most 1 retry


def search_studies_by_intervention(
    drug: str,
    limit: int = DEFAULT_LIMIT,
    overall_status: str | None = None,
    page_size: int = 20,
    timeout_s: float = REQUEST_TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
) -> list[dict[str, Any]]:
    """Search CTGov by intervention name; returns raw study dicts (<= limit).

    Network errors and HTTP failures return [] after the bounded retries;
    pagination stops as soon as `limit` studies are collected.
    """
    if not str(drug).strip():
        return []
    params: dict[str, Any] = {
        "query.intr": str(drug).strip(),
        "pageSize": min(int(page_size), MAX_PAGE_SIZE),
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,OverallStatus,Phase,EligibilityCriteria,"
            "LeadSponsorName,StartDate,PrimaryCompletionDate,CompletionDate,"
            "InterventionName"
        ),
    }
    if overall_status:
        params["filter.overallStatus"] = overall_status.upper().replace(" ", "_")

    out: list[dict[str, Any]] = []
    session = requests.Session()
    url = CTGOV_BASE
    while len(out) < int(limit):
        if params is not None:  # None after switching to an encoded nextPageUrl
            remaining = int(limit) - len(out)
            params["pageSize"] = min(params["pageSize"], remaining, MAX_PAGE_SIZE)
        payload: dict[str, Any] | None = None
        for attempt in range(1, int(max_attempts) + 1):
            try:
                resp = session.get(url, params=params, timeout=timeout_s)
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError):
                if attempt >= int(max_attempts):
                    payload = None
                    break
                time.sleep(0.5)
        if not payload:
            break  # degrade gracefully: partial results only

        studies = payload.get("studies") or []
        out.extend(studies)
        next_url = (payload.get("nextPageUrl") or "").strip()
        if not next_url or not studies:
            break
        url, params = next_url, None  # nextPageUrl already encodes the query

    return out[: int(limit)]


def search_trials(
    drug: str,
    limit: int = DEFAULT_LIMIT,
    overall_status: str | None = None,
) -> list:
    """Search and parse into StudyRecord objects via the shared parser."""
    records = []
    for study in search_studies_by_intervention(
        drug, limit=limit, overall_status=overall_status
    ):
        rec = _parse_study(study)
        if rec is not None:
            records.append(rec)
    return records
