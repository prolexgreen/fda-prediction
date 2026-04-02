"""Stage-7 chemistry features: rdkit descriptors, drug modality, ChEMBL mechanism.

All inputs are known at trial design time or earlier (drug identity), so every
feature here is leakage-safe by construction.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

# ---------------------------------------------------------------- descriptors

DESCRIPTOR_NAMES = (
    "chem_mw_mean",
    "chem_logp_mean",
    "chem_tpsa_mean",
    "chem_rotb_mean",
    "chem_hbd_mean",
    "chem_hba_mean",
    "chem_fsp3_mean",
    "chem_arom_frac_mean",
)


def smiles_descriptors(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotb": float(Lipinski.NumRotatableBonds(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)) if n_atoms else 0.0,
        "arom_frac": float(n_arom / n_atoms) if n_atoms else 0.0,
    }


def aggregate_descriptors(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean descriptor vector + mask (present flag set per-dimension)."""
    vals = [smiles_descriptors(s) for s in smiles_list]
    vals = [v for v in vals if v is not None]
    keys = ("mw", "logp", "tpsa", "rotb", "hbd", "hba", "fsp3", "arom_frac")
    if not vals:
        return np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)
    arr = np.array([[v[k] for k in keys] for v in vals], dtype=np.float64).mean(axis=0)
    return arr.astype(np.float32), np.ones(8, dtype=np.float32)


# ------------------------------------------------------------------- modality

_MAB_RE = re.compile(r"(?:umab|ximab|zumab|mumab|omab|tumab)\b|monoclonal", re.IGNORECASE)
_BIOLOGIC_OTHER_RE = re.compile(
    r"\b(?:insulin|cytokine|interferon|hormone|enzyme|growth\s+factor|"
    r"erythropoietin|filgrastim|etanercept|fusion\s+protein|recombinant)\b",
    re.IGNORECASE,
)
_CELL_GENE_RE = re.compile(
    r"\b(?:car[- ]?t(?:19)?(?:\s|$)|cart[\s-]?cell|cell\s+therap|gene\s+therap|"
    r"adoptive\s+cell|stem\s+cell|tcr[\s-]?t|oncolytic)\b",
    re.IGNORECASE,
)
_VACCINE_RE = re.compile(r"\bvaccine\b|\btoxoid\b|\bdtap\b|\bmmr\b", re.IGNORECASE)

MODALITY_NAMES = (
    "mod_small_molecule",
    "mod_mab",
    "mod_biologic_other",
    "mod_cell_gene",
    "mod_vaccine",
    "mod_unknown",
)


def classify_modality(drug_names: list[str], has_smiles: bool) -> np.ndarray:
    """6-dim one-hot-ish vector; multi-class possible for combo trials.

    any-hit wins; a trial with SMILES and no keyword hit is a small molecule.
    """
    vec = np.zeros(6, dtype=np.float32)
    joined = " ".join(drug_names).lower()
    if _VACCINE_RE.search(joined):
        vec[4] = 1.0
    if _CELL_GENE_RE.search(joined):
        vec[3] = 1.0
    if _MAB_RE.search(joined):
        vec[1] = 1.0
    if _BIOLOGIC_OTHER_RE.search(joined):
        vec[2] = 1.0
    if has_smiles and not vec[1:5].any():
        vec[0] = 1.0
    if not vec.any():
        vec[5] = 1.0
    return vec


# ------------------------------------------------------------------ ChEMBL

TARGET_BUCKETS = ("enzyme", "gpcr", "ion_channel", "kinase", "nuclear_receptor", "other_protein")
ACTION_BUCKETS = ("inhibitor", "agonist", "antagonist", "other")


@dataclass
class MechRecord:
    tid: int
    target_type: str
    pref_name: str
    action_type: str
    max_phase: int


class ChemblMechanismIndex:
    """name -> mechanism rows, built once from chembl sqlite.

    Falls back to keyword heuristics for target bucketing (cheap and robust).
    """

    _SQL = (
        "SELECT m.synonyms, md.pref_name AS drug_name, t.tid, t.target_type, "
        "       t.pref_name AS target_name, dm.action_type, md.max_phase "
        "FROM molecule_synonyms m "
        "JOIN molecule_dictionary md ON md.molregno = m.molregno "
        "JOIN drug_mechanism dm ON dm.molregno = md.molregno "
        "JOIN target_dictionary t ON t.tid = dm.tid"
    )

    # keyword heuristics for target buckets (from pref_name / target_type)
    _BUCKETS: dict[str, tuple] = {
        "kinase": (re.compile(r"kinase", re.IGNORECASE),),
        "gpcr": (
            re.compile(r"G-protein coupled receptor|GPCR|class A receptor|opsin", re.IGNORECASE),
        ),
        "ion_channel": (re.compile(r"channel", re.IGNORECASE),),
        "nuclear_receptor": (re.compile(r"nuclear|glucocorticoid|estrogen|androgen|thyroid", re.IGNORECASE),),
        "enzyme": (
            re.compile(
                r"ase\b|dehydrogenase|reductase|transferase|hydrolase|synthase|"
                r"phosphatase|protease|oxidase|lyase|isomerase|ligase",
                re.IGNORECASE
            ),
        ),
    }

    def __init__(self, db_path: str | Path | None = None):
        self._by_name: dict[str, list[MechRecord]] = {}
        self._competitors: dict[int, int] = {}  # tid -> max_phase=4 drug count
        if db_path is not None:
            self._build(str(db_path))

    def _build(self, db_path: str) -> None:
        con = sqlite3.connect(db_path)
        try:
            for synonyms, drug_name, tid, ttype, tname, action, max_phase in con.execute(self._SQL):
                rec = MechRecord(
                    tid=int(tid),
                    target_type=str(ttype or ""),
                    pref_name=str(tname or ""),
                    action_type=str(action or "").upper(),
                    max_phase=int(max_phase if max_phase is not None else -1),
                )
                for key in {str(synonyms).lower(), str(drug_name).lower()}:
                    if key:
                        self._by_name.setdefault(key, []).append(rec)
                if rec.max_phase == 4:
                    self._competitors[rec.tid] = self._competitors.get(rec.tid, 0) + 1
        finally:
            con.close()

    def lookup(self, name: str) -> list[MechRecord]:
        return self._by_name.get(str(name).lower(), [])

    def classify_target(self, target_name: str) -> str:
        for bucket, patterns in self._BUCKETS.items():
            if any(p.search(target_name) for p in patterns):
                return bucket
        return "other_protein"

    @staticmethod
    def classify_action(action_type: str) -> str:
        a = action_type.upper()
        if "INHIBITOR" in a or "BLOCKER" in a:
            return "inhibitor"
        if "AGONIST" in a:
            return "agonist"
        if "ANTAGONIST" in a:
            return "antagonist"
        return "other"

    def competitors_for_target(self, tid: int) -> int:
        return self._competitors.get(tid, 0)


def trial_mechanism_features(
    index: ChemblMechanismIndex | None,
    drug_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (feats, mask) for the mechanism block.

    feats: [mech_known, enzyme, gpcr, ion_channel, kinase, nuclear_receptor,
            other_protein, inhibitor, agonist, antagonist, other_action,
            target_competitors_log1p]
    mask: all 1 if any drug mapped to a mechanism row, else all 0.
    """
    feats = np.zeros(12, dtype=np.float32)
    if index is None or not drug_names:
        return feats, np.zeros(12, dtype=np.float32)

    recs: list[MechRecord] = []
    for name in drug_names:
        recs.extend(index.lookup(name))
    if not recs:
        return feats, np.zeros(12, dtype=np.float32)

    feats[0] = 1.0
    targets, actions, max_comp = set(), set(), 0
    for r in recs:
        targets.add(index.classify_target(r.pref_name))
        actions.add(index.classify_action(r.action_type))
        max_comp = max(max_comp, index.competitors_for_target(r.tid))
    t_off = dict(zip(TARGET_BUCKETS, range(1, 7)))
    a_off = dict(zip(ACTION_BUCKETS, range(7, 11)))
    for t in targets:
        feats[t_off[t]] = 1.0
    for a in actions:
        feats[a_off[a]] = 1.0
    feats[11] = float(np.log1p(max_comp))
    return feats, np.ones(12, dtype=np.float32)
