"""PyTorch datasets for TOP-only and merged TOP+CTO corpora."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fda_predictor.data.preprocessing import clean_criteria_text
from fda_predictor.data.stock_features import N_STOCK_FEATURES
from fda_predictor.data.tabular_features import N_TABULAR_FEATURES
from fda_predictor.data.tokenizers import (
    EncoderSpec,
    encode_smiles_list,
    encode_text,
    get_tokenizer,
)

PHASE_UNK_INDEX = 4
PLACEHOLDER_SMILES = "C"  # dummy when molecule_mask=0 (zeroed in model forward)
APPROVAL_LABEL_UNSET = torch.tensor(-1.0)  # sentinel for unlabeled approval rows


class TOPTrialDataset(Dataset):
    """Backward-compatible TOP-only dataset (no stock features)."""

    def __init__(
        self,
        frame: pd.DataFrame,
        chemberta_spec: EncoderSpec,
        molformer_spec: EncoderSpec,
        clinicalbert_spec: EncoderSpec,
        max_molecules_per_trial: int = 5,
    ):
        self.frame = frame.reset_index(drop=True)
        self.tok_a = get_tokenizer(chemberta_spec)
        self.tok_b = get_tokenizer(molformer_spec)
        self.tok_c = get_tokenizer(clinicalbert_spec)
        self.max_len_a = chemberta_spec.max_length
        self.max_len_b = molformer_spec.max_length
        self.max_len_c = clinicalbert_spec.max_length
        self.max_molecules = max_molecules_per_trial

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        raw_smiles = row["smiles_canonical"]
        if isinstance(raw_smiles, str):
            import ast

            try:
                smiles = ast.literal_eval(raw_smiles)
            except (ValueError, SyntaxError):
                smiles = [raw_smiles] if raw_smiles else []
        elif isinstance(raw_smiles, (list, tuple, np.ndarray)):
            smiles = [str(s) for s in raw_smiles]
        else:
            smiles = []
        smiles = smiles[: self.max_molecules]
        mol_mask = 1
        if not smiles:
            smiles = [PLACEHOLDER_SMILES]
            mol_mask = 0
        elif "molecule_mask" in self.frame.columns:
            mol_mask = int(row.get("molecule_mask", 1))

        criteria = clean_criteria_text(row["criteria"]) or "[NO CRITERIA REPORTED]"

        phase_val = PHASE_UNK_INDEX
        if "phase_index" in self.frame.columns and pd.notna(row.get("phase_index")):
            phase_val = int(row["phase_index"])

        item = {
            "nctid": str(row["nctid"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "mol_input_a": encode_smiles_list(self.tok_a, smiles, self.max_len_a),
            "mol_input_b": encode_smiles_list(self.tok_b, smiles, self.max_len_b),
            "crit_input": encode_text(self.tok_c, criteria, self.max_len_c),
            "n_molecules": len(smiles),
            "phase_index": torch.tensor(phase_val, dtype=torch.long),
            "molecule_mask": torch.tensor(float(mol_mask), dtype=torch.float32),
            "stock_feats": torch.zeros(N_STOCK_FEATURES, dtype=torch.float32),
            "stock_mask": torch.tensor(0.0, dtype=torch.float32),
            "tabular_feats": torch.zeros(N_TABULAR_FEATURES, dtype=torch.float32),
            "tabular_mask": torch.zeros(N_TABULAR_FEATURES, dtype=torch.float32),
            # Sparse approval target: sentinel -1 + mask 0 when unlabeled so
            # the dual-head loss can skip these rows.
            "approval_label": APPROVAL_LABEL_UNSET.clone(),
            "approval_mask": torch.tensor(0.0, dtype=torch.float32),
        }
        if "data_source" in self.frame.columns:
            item["data_source"] = str(row["data_source"])
        if "approval_label" in self.frame.columns and pd.notna(row.get("approval_label")):
            prev = row.get("previously_approved", False)
            if isinstance(prev, float) and np.isnan(prev):
                prev = False
            if not bool(prev):
                item["approval_label"] = torch.tensor(
                    float(row["approval_label"]), dtype=torch.float32
                )
                item["approval_mask"] = torch.tensor(1.0, dtype=torch.float32)
        return item


class MergedTrialDataset(TOPTrialDataset):
    """TOP+CTO trials with stock + tabular feature vectors."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        row = self.frame.iloc[idx]
        if "stock_feats" in self.frame.columns:
            feats = row["stock_feats"]
            if isinstance(feats, list):
                arr = np.array(feats, dtype=np.float32)
            else:
                arr = np.array(list(feats), dtype=np.float32)
            item["stock_feats"] = torch.tensor(np.nan_to_num(arr, nan=0.0), dtype=torch.float32)
        if "stock_mask" in self.frame.columns:
            item["stock_mask"] = torch.tensor(float(row["stock_mask"]), dtype=torch.float32)
        if "tabular_feats" in self.frame.columns:
            tfeats = row["tabular_feats"]
            if isinstance(tfeats, list):
                tarr = np.array(tfeats, dtype=np.float32)
            else:
                tarr = np.array(list(tfeats), dtype=np.float32)
            item["tabular_feats"] = torch.tensor(
                np.nan_to_num(tarr, nan=0.0), dtype=torch.float32
            )
        if "tabular_mask" in self.frame.columns:
            tmask = row["tabular_mask"]
            if isinstance(tmask, list):
                marr = np.array(tmask, dtype=np.float32)
            else:
                marr = np.array(list(tmask), dtype=np.float32)
            item["tabular_mask"] = torch.tensor(
                np.nan_to_num(marr, nan=0.0), dtype=torch.float32
            )
        return item


def _pad_stack(seqs: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in seqs]
    masks = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in seqs]
    input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return input_ids, attention_mask


class TrialCollator:
    def __init__(self, tok_a_pad_id: int, tok_b_pad_id: int, tok_c_pad_id: int):
        self.pad_a = tok_a_pad_id
        self.pad_b = tok_b_pad_id
        self.pad_c = tok_c_pad_id

    @classmethod
    def from_dataset(cls, ds: TOPTrialDataset) -> "TrialCollator":
        return cls(
            tok_a_pad_id=ds.tok_a.pad_token_id,
            tok_b_pad_id=ds.tok_b.pad_token_id,
            tok_c_pad_id=ds.tok_c.pad_token_id,
        )

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        flat_a: list[dict] = []
        flat_b: list[dict] = []
        group_index: list[int] = []

        for i, sample in enumerate(batch):
            mols_a = sample["mol_input_a"]
            mols_b = sample["mol_input_b"]
            assert len(mols_a) == len(mols_b) == sample["n_molecules"] >= 1
            flat_a.extend(mols_a)
            flat_b.extend(mols_b)
            group_index.extend([i] * sample["n_molecules"])

        ids_a, mask_a = _pad_stack(flat_a, self.pad_a)
        ids_b, mask_b = _pad_stack(flat_b, self.pad_b)
        ids_c, mask_c = _pad_stack([s["crit_input"] for s in batch], self.pad_c)

        out = {
            "nctid": [s["nctid"] for s in batch],
            "label": torch.stack([s["label"] for s in batch]),
            "mol_input_a": {"input_ids": ids_a, "attention_mask": mask_a},
            "mol_input_b": {"input_ids": ids_b, "attention_mask": mask_b},
            "group_index": torch.tensor(group_index, dtype=torch.long),
            "crit_input": {"input_ids": ids_c, "attention_mask": mask_c},
            "phase_index": torch.stack([s["phase_index"] for s in batch]),
            "molecule_mask": torch.stack([s["molecule_mask"] for s in batch]),
            "stock_feats": torch.stack([s["stock_feats"] for s in batch]),
            "stock_mask": torch.stack([s["stock_mask"] for s in batch]),
            "tabular_feats": torch.stack([s["tabular_feats"] for s in batch]),
            "tabular_mask": torch.stack([s["tabular_mask"] for s in batch]),
            "approval_label": torch.stack([s["approval_label"] for s in batch]),
            "approval_mask": torch.stack([s["approval_mask"] for s in batch]),
        }
        if "data_source" in batch[0]:
            out["data_source"] = [s["data_source"] for s in batch]
        return out


def build_collate_fn(ds: TOPTrialDataset):
    return TrialCollator.from_dataset(ds)
