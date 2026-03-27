"""Drug-name -> canonical SMILES via PubChem PUG REST with on-disk cache."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

from fda_predictor.data.preprocessing import canonicalize_smiles
from fda_predictor.utils.paths import PUBCHEM_CACHE_DIR

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# --- name normalization (for backfill of cached misses) --------------------

_DOSE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:mg\s*/\s*kg|mcg\s*/\s*kg|iu\s*/\s*kg|mg\s*/\s*ml|mcg\s*/\s*ml|mg\s*/\s*m\^?2|"
    r"ml\s*/\s*kg|"
    r"mg|mcg|µg|ug|g|kg|ml|iu|units?|%)"
    r"(?:\s*/\s*(?:day|dose|kg|m\^?2|hour|hr|week|ml))?"
    r"(?=\s|$|[,;:()])",
    re.IGNORECASE,
)
_CELLS_DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*x\s*10\^?\d*\s*cells\s*/\s*\w+\b", re.IGNORECASE)
_PAREN_RE = re.compile(r"\([^)]*\)")
_FORM_WORDS = re.compile(
    r"\b(?:tablets?|capsules?|caps?|solution|solutions|gel|gels|gel[- ]forming|cream|creams|"
    r"ointment|ointments?|patch|patches|spray|sprays|rinse|drops?|eye\s+drops?|suspension|"
    r"suspensions|lotion|powder|sachets?|inhaler|inhalation|aerosol| foam\b|"
    r"ophthalmic|topical|nasal|oral(?:/nasal)?|intravitreal|intravenous|subcutaneous|"
    r"injection(?:s)?|infusion|epidural|paravertebral|anaesthesia|anesthesia|"
    r"implant(?:able)?|suppositor(?:y|ies)|enema|elixir|syrup|sustained[- ]release|"
    r"extended[- ]release|delayed[- ]release|placebo|formulation(?:s)?|vehicle)\b",
    re.IGNORECASE,
)
_SPLIT_WORDS_RE = re.compile(r"\s+(?:or|and)\s+", re.IGNORECASE)
_JUNK_RE = re.compile(r"[\s\-–—,;:./+_()]+")


def normalize_drug_name(name: str) -> str:
    """Strip doses, parentheticals, formulation words from a free-text drug name."""
    s = str(name)
    s = _PAREN_RE.sub(" ", s)
    s = _CELLS_DOSE_RE.sub(" ", s)
    s = _DOSE_RE.sub(" ", s)
    s = _FORM_WORDS.sub(" ", s)
    s = _JUNK_RE.sub(" ", s)
    return " ".join(s.split()).strip(" -,").lower()


def split_combination(name: str) -> list[str]:
    """Split combo interventions ("A + B", "A/B", "A or B") into parts.

    Conservative: parts must normalize to len>=3 and are deduped in order.
    """
    s = str(name).strip()
    parts = re.split(r"\s*\+\s*|\s*/\s*", s)
    flat: list[str] = []
    for p in parts:
        flat.extend(_SPLIT_WORDS_RE.split(p))
    out: list[str] = []
    seen: set[str] = set()
    for p in flat:
        p = p.strip()
        if len(normalize_drug_name(p)) < 3:
            continue
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    # Alternatives phrased as "10 mg X or 20 mg X" collapse onto X — the split
    # above keeps both; normalize dedupes at resolve time via canonical SMILES.
    return out


def _fetch_synonyms(name: str, timeout: float = 30.0) -> list[str]:
    """PubChem synonym list for a name; [] when unresolvable."""
    url = f"{PUG_BASE}/compound/name/{requests.utils.quote(name)}/synonyms/JSON"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    info = resp.json().get("InformationList", {}).get("Information") or []
    if not info:
        return []
    return [str(s) for s in info[0].get("Synonym") or []]


@dataclass
class PubChemReport:
    n_queries: int = 0
    n_hits: int = 0
    n_miss: int = 0
    n_invalid: int = 0

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_queries if self.n_queries else 0.0


def _cache_path(drug_name: str):
    safe = "".join(c if c.isalnum() else "_" for c in drug_name.lower())[:120]
    return PUBCHEM_CACHE_DIR / f"{safe}.json"


def _fetch_smiles_from_pubchem(name: str, timeout: float = 30.0) -> str | None:
    """Resolve a drug name to an isomeric SMILES string.

    Returns None for a definitive miss (HTTP 404). Raises requests.RequestException
    on transient failures so callers can avoid caching them.
    """
    url = f"{PUG_BASE}/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES/JSON"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 404:
        return None
    if resp.status_code == 405:
        # PUG returns 405 for names that resolve to no computable property;
        # treat as definitive miss.
        return None
    resp.raise_for_status()
    props = resp.json().get("PropertyTable", {}).get("Properties") or []
    if not props:
        return None
    # PubChem renamed response keys (2025 API migration): requesting
    # IsomericSMILES now returns {"SMILES": ...}. Check all known keys.
    entry = props[0]
    return (
        entry.get("IsomericSMILES")
        or entry.get("SMILES")
        or entry.get("ConnectivitySMILES")
        or entry.get("CanonicalSMILES")
    )


def _enhanced_lookup(name: str, delay_s: float) -> str | None:
    """Retry ladder for cached misses: normalized name -> synonyms -> first-synonym SMILES."""
    norm = normalize_drug_name(name)
    candidates: list[str] = []
    for c in (norm, name):
        c = c.strip()
        if c and c.lower() not in [x.lower() for x in candidates]:
            candidates.append(c)
    for cand in candidates:
        if cand.lower() != name.lower():
            raw = _fetch_smiles_from_pubchem(cand)
            time.sleep(delay_s)
            if raw:
                return raw
        # Synonym fallback when the direct/property lookup 404s.
        syns = _fetch_synonyms(cand)
        time.sleep(delay_s)
        for syn in syns[:3]:
            raw = _fetch_smiles_from_pubchem(syn)
            time.sleep(delay_s)
            if raw:
                return raw
    return None


def cached_smiles(name: str) -> str | None:
    """Pure cache read (no network). None when not cached or cached miss."""
    p = _cache_path(name.strip())
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("smiles")


def resolve_drug_smiles(
    drug_name: str,
    report: PubChemReport | None = None,
    delay_s: float = 0.25,
    use_cache: bool = True,
    enhanced: bool = False,
    enhance_null_cache: bool = True,
) -> str | None:
    name = str(drug_name).strip()
    if not name:
        return None
    if report is not None:
        report.n_queries += 1

    cache = _cache_path(name)
    if use_cache and cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        smi = payload.get("smiles")
        if smi:
            if report is not None:
                report.n_hits += 1
            return smi
        # Cached definitive miss: enhanced mode re-attempts through the
        # normalization/combo/synonym ladder and upgrades the cache on a hit.
        if enhanced:
            try:
                raw = _enhanced_lookup(name, delay_s)
            except requests.RequestException:
                if report is not None:
                    report.n_miss += 1
                return None
            canon = canonicalize_smiles(raw) if raw else None
            if canon and enhance_null_cache:
                cache.write_text(
                    json.dumps({"drug": name, "smiles": canon, "recovered": True}),
                    encoding="utf-8",
                )
            elif canon is None and enhance_null_cache:
                # Full ladder exhausted: mark as enhanced-miss so future
                # backfills skip it instead of burning network quota forever.
                cache.write_text(
                    json.dumps({"drug": name, "smiles": None, "enhanced_miss": True}),
                    encoding="utf-8",
                )
            if report is not None:
                if canon:
                    report.n_hits += 1
                else:
                    report.n_miss += 1
            return canon
        if report is not None:
            report.n_miss += 1
        return None

    PUBCHEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        raw = _fetch_smiles_from_pubchem(name)
        time.sleep(delay_s)
        if raw is None and enhanced:
            raw = _enhanced_lookup(name, delay_s)
    except requests.RequestException:
        # Transient network/API failure: do NOT write a null cache entry,
        # otherwise the miss gets poisoned permanently (704 null files happened).
        if report is not None:
            report.n_miss += 1
        return None

    canon = canonicalize_smiles(raw) if raw else None
    if raw is None or canon is not None:
        cache.write_text(json.dumps({"drug": name, "smiles": canon}), encoding="utf-8")

    if report is not None:
        if canon:
            report.n_hits += 1
        else:
            report.n_miss += 1
    return canon


def resolve_drug_list(
    drugs: list[str],
    report: PubChemReport | None = None,
    max_drugs: int = 5,
    split_combos: bool = False,
    enhanced: bool = False,
) -> list[str]:
    """Resolve up to max_drugs names; dedupe canonical SMILES in order.

    split_combos=True expands "A + B"/"A/B"/"A or B" interventions into their
    constituent parts before lookup. Output is capped at max_drugs unique
    SMILES; lookups stop once the cap is hit (or after 2x max_drugs attempts).
    """
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    for drug in drugs:
        parts = split_combination(drug) if split_combos else [str(drug)]
        for part in parts:
            if len(out) >= max_drugs or attempts >= 2 * max_drugs:
                return out
            attempts += 1
            smi = resolve_drug_smiles(part, report=report, enhanced=enhanced)
            if smi and smi not in seen:
                seen.add(smi)
                out.append(smi)
    return out
