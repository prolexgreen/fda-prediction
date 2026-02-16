"""ClinicalTrials.gov API v2 client for batched study metadata fetch.

Shared by CTO ingestion (Task 4.5) and the live Phase 2 pipeline. Text
fields are unescaped via preprocessing.clean_criteria_text for parity
with the TOP training corpus.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from fda_predictor.data.preprocessing import clean_criteria_text, parse_phase

CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_FIELDS = (
    "NCTId,"
    "BriefTitle,"
    "OverallStatus,"
    "Phase,"
    "EligibilityCriteria,"
    "LeadSponsorName,"
    "StartDate,"
    "PrimaryCompletionDate,"
    "CompletionDate,"
    "StudyFirstSubmitDate,"
    "StudyFirstPostDate,"
    "InterventionName,"
    "InterventionType,"
    "Condition"
)


@dataclass
class StudyRecord:
    nct_id: str
    criteria: str
    phase_raw: str
    phase_index: int
    sponsor: str
    drugs: list[str]
    start_date: str | None
    completion_date: str | None
    study_first_submitted_date: str | None
    overall_status: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _normalize_nct(nct_id: str) -> str:
    s = str(nct_id).strip().upper()
    if not s.startswith("NCT"):
        digits = "".join(ch for ch in s if ch.isdigit())
        s = f"NCT{digits}" if digits else s
    return s


def _extract_date(module: dict[str, Any] | None, key: str) -> str | None:
    if not module:
        return None
    block = module.get(key)
    if isinstance(block, dict):
        return block.get("date") or block.get("value")
    if isinstance(block, str):
        return block
    return None


def _parse_study(study: dict[str, Any]) -> StudyRecord | None:
    proto = study.get("protocolSection") or {}
    ident = proto.get("identificationModule") or {}
    nct_id = ident.get("nctId")
    if not nct_id:
        return None

    status_mod = proto.get("statusModule") or {}
    design = proto.get("designModule") or {}
    sponsor_mod = proto.get("sponsorCollaboratorsModule") or {}
    arms = proto.get("armsInterventionsModule") or {}
    eligibility = proto.get("eligibilityModule") or {}

    phases = design.get("phases") or []
    phase_raw = phases[0] if phases else ""
    if isinstance(phase_raw, str):
        phase_raw = phase_raw.replace("PHASE", "Phase ").replace("_", " ").strip()

    interventions = arms.get("interventions") or []
    drugs: list[str] = []
    for intr in interventions:
        if not isinstance(intr, dict):
            continue
        name = intr.get("name")
        itype = (intr.get("type") or "").lower()
        if name and ("drug" in itype or itype in ("biological", "combination product", "")):
            drugs.append(str(name))

    criteria_raw = eligibility.get("eligibilityCriteria") or ""
    criteria = clean_criteria_text(str(criteria_raw))

    lead = sponsor_mod.get("leadSponsor") or {}
    sponsor = str(lead.get("name") or "")

    start_date = _extract_date(status_mod, "startDateStruct") or _extract_date(
        status_mod, "studyFirstSubmitDate"
    )
    completion_date = _extract_date(status_mod, "completionDateStruct") or _extract_date(
        status_mod, "primaryCompletionDateStruct"
    )
    study_first_submitted = _extract_date(status_mod, "studyFirstSubmitDate")

    return StudyRecord(
        nct_id=_normalize_nct(nct_id),
        criteria=criteria,
        phase_raw=str(phase_raw),
        phase_index=parse_phase(str(phase_raw)),
        sponsor=sponsor,
        drugs=drugs,
        start_date=start_date,
        completion_date=completion_date,
        study_first_submitted_date=study_first_submitted or start_date,
        overall_status=str(status_mod.get("overallStatus") or ""),
        raw=study,
    )


class CTGovClient:
    def __init__(
        self,
        fields: str = DEFAULT_FIELDS,
        page_size: int = 100,
        request_delay_s: float = 0.35,
        timeout_s: float = 60.0,
    ):
        self.fields = fields
        self.page_size = min(page_size, 100)
        self.request_delay_s = request_delay_s
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def fetch_by_nct_ids(self, nct_ids: list[str]) -> dict[str, StudyRecord]:
        """Fetch studies in batches; returns map nct_id -> StudyRecord."""
        unique = [_normalize_nct(n) for n in nct_ids if n]
        unique = list(dict.fromkeys(unique))
        out: dict[str, StudyRecord] = {}
        chunk = 50  # API filter.ids practical batch size
        for i in range(0, len(unique), chunk):
            batch = unique[i : i + chunk]
            params = {
                "filter.ids": ",".join(batch),
                "pageSize": self.page_size,
                "fields": self.fields,
                "format": "json",
            }
            resp = self.session.get(CTGOV_BASE, params=params, timeout=self.timeout_s)
            resp.raise_for_status()
            payload = resp.json()
            for study in payload.get("studies") or []:
                rec = _parse_study(study)
                if rec:
                    out[rec.nct_id] = rec
            time.sleep(self.request_delay_s)
        return out

    def fetch_one(self, nct_id: str) -> StudyRecord | None:
        return self.fetch_by_nct_ids([nct_id]).get(_normalize_nct(nct_id))


def studies_to_frame_rows(records: dict[str, StudyRecord]) -> list[dict[str, Any]]:
    rows = []
    for nct, rec in records.items():
        rows.append(
            {
                "nctid": nct,
                "criteria": rec.criteria,
                "phase": rec.phase_raw,
                "phase_index": rec.phase_index,
                "sponsor": rec.sponsor,
                "drugs": rec.drugs,
                "start_date": rec.start_date,
                "completion_date": rec.completion_date,
                "chronology_date": rec.study_first_submitted_date or rec.start_date,
                "overall_status": rec.overall_status,
            }
        )
    return rows


def cache_studies_json(path, records: dict[str, StudyRecord]) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.raw for k, v in records.items()}
    p.write_text(json.dumps(serializable, indent=0), encoding="utf-8")
