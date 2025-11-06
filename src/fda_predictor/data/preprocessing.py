"""Shared preprocessing: SMILES canonicalization and criteria text cleaning.

`clean_criteria_text` must be the single source of truth for text
normalization so Phase 1 (TDC) and Phase 2 (ClinicalTrials.gov live data)
feed the protocol encoder identically formatted strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger

# TDC rows contain some deliberately malformed / placeholder SMILES;
# silence rdkit's per-molecule stderr spam and count drops instead.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

_MARKDOWN_ESCAPES = {
    r"\>": ">",
    r"\<": "<",
    r"\*": "*",
    r"\_": "_",
    r"\-": "-",
    r"\+": "+",
    r"\#": "#",
    r"\!": "!",
}

_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class CanonicalizationReport:
    n_input: int = 0
    n_invalid: int = 0
    n_duplicates_removed: int = 0
    invalid_examples: list[str] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        kept = self.n_input - self.n_invalid
        return kept / self.n_input if self.n_input else 0.0

    def summary(self) -> str:
        return (
            f"SMILES canonicalization: {self.n_input} input | {self.n_invalid} invalid "
            f"({(1 - self.validity_rate) * 100:.2f}%) | "
            f"{self.n_duplicates_removed} intra-trial duplicates removed"
        )


def canonicalize_smiles(smiles: str) -> str | None:
    """Return rdkit-canonical SMILES, or None if unparseable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canonicalize_molecule_list(raw_list: list, report: CanonicalizationReport) -> list[str]:
    """Canonicalize one trial's SMILES list: drop invalids, dedupe, keep order.

    Deterministic per the approved plan: first-listed order after
    canonicalization is preserved.
    """
    report.n_input += len(raw_list)
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_list:
        canon = canonicalize_smiles(str(raw))
        if canon is None:
            report.n_invalid += 1
            if len(report.invalid_examples) < 10:
                report.invalid_examples.append(str(raw)[:120])
            continue
        if canon in seen:
            report.n_duplicates_removed += 1
            continue
        seen.add(canon)
        out.append(canon)
    return out


# --- Trial phase parsing ----------------------------------------------------
# Exact phases map to 0..3 (I, II, III, IV); combined designations like
# "phase 2/phase 3", other formats, and missing values map to UNK=4 so we
# never relabel a combo trial as a pure single-phase trial.
_PHASE_MAP = {"i": 0, "ii": 1, "iii": 2, "iv": 3}
PHASE_UNK_INDEX = 4
NUM_PHASE_CATEGORIES = 5


def parse_phase(value) -> int:
    """Map raw trial phase strings to embedding indices 0-4 (4 = UNK)."""
    if not isinstance(value, str):
        return PHASE_UNK_INDEX
    s = value.strip().lower().replace("phase", "").replace("-", "").strip()
    if s in _PHASE_MAP:
        return _PHASE_MAP[s]
    if s.isdigit() and int(s) in (1, 2, 3, 4):
        return int(s) - 1
    return PHASE_UNK_INDEX


def clean_criteria_text(text: str) -> str:
    """Normalize eligibility-criteria text.

    Unescapes ClinicalTrials.gov markdown artifacts (\\>, \\*, ...),
    collapses runs of spaces/tabs and 3+ newlines, strips outer whitespace.
    """
    if not isinstance(text, str):
        return ""
    for esc, char in _MARKDOWN_ESCAPES.items():
        text = text.replace(esc, char)
    # any remaining backslash before a punctuation char is an escaping artifact
    text = re.sub(r"\\(?=[^\w\s])", "", text)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()
