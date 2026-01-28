"""Backtest driver: temporal out-of-sample evaluation + baselines.

Protocol:
- Score the untouched TEST era once with the trained checkpoint.
- Tune the decision threshold on VAL predictions ONLY, then apply it
  frozen to test (val -> test transfer).
- Report overall metrics, the Phase-3 slice (the core use case), and a
  per-phase breakdown.
- Baselines establish floors: a constant majority predictor and logistic
  regression on cheap classical features (RDKit descriptors + TF-IDF
  criteria text), both trained on the TRAIN split only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fda_predictor.data.preprocessing import clean_criteria_text
from fda_predictor.models.multimodal_net import TriStreamNet
from fda_predictor.training.metrics import (
    classification_report,
    f1_at_threshold,
    tune_threshold_cost,
    tune_threshold_for_precision,
    tune_threshold_on_val,
)

PHASE_PURE3_INDEX = 2
PHASE_LABELS = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "UNK(combo/missing)"}


@dataclass
class ScoredSplit:
    nctids: list[str]
    y_true: np.ndarray
    scores: np.ndarray
    phase_index: np.ndarray
    data_source: np.ndarray | None = None
    approval_true: np.ndarray | None = None  # sentinel -1 where unlabeled
    approval_mask: np.ndarray | None = None
    approval_scores: np.ndarray | None = None

    def slice_mask(self, phase_idx: int | None = None) -> np.ndarray:
        if phase_idx is None:
            return np.ones(len(self.y_true), dtype=bool)
        return self.phase_index == phase_idx

    def source_mask(self, source: str) -> np.ndarray:
        if self.data_source is None:
            return np.zeros(len(self.y_true), dtype=bool)
        return self.data_source == source

    def approval_labeled_mask(self) -> np.ndarray:
        if self.approval_mask is None or self.approval_true is None:
            return np.zeros(len(self.y_true), dtype=bool)
        return (self.approval_mask > 0) & (self.approval_true >= 0)


def filter_compatible_state_dict(
    model: TriStreamNet, state: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop checkpoint tensors whose shapes differ from the live model."""
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != tensor.shape:
            skipped.append(key)
            continue
        filtered[key] = tensor
    return filtered, skipped


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device) -> tuple[TriStreamNet, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    layout = int(payload.get("checkpoint_layout", 1))
    expected = TriStreamNet.CHECKPOINT_LAYOUT
    supported = (3, 4, 5, expected)
    if layout not in supported:
        raise ValueError(
            f"checkpoint layout {layout} incompatible with model layout {expected}; "
            f"retrain with the current architecture (layout {expected})."
        )
    # Layout-3 checkpoints predate the tabular stream; build a matching fusion dim.
    force_layout = 3 if layout == 3 else None
    net = TriStreamNet.from_config(config, force_layout=force_layout)
    compatible, skipped = filter_compatible_state_dict(net, payload["model_state"])
    missing, unexpected = net.load_state_dict(compatible, strict=False)
    if missing or unexpected or skipped:
        print(
            f"checkpoint load notes: missing={len(missing)} unexpected={len(unexpected)} "
            f"skipped_shape={len(skipped)} (layout {layout} -> model layout {expected})"
        )
    net.to(device)
    net.eval()
    return net, payload


@torch.no_grad()
def score_split(net: TriStreamNet, dataset, collate_fn, device: torch.device, batch_size: int = 32) -> ScoredSplit:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    use_amp = device.type == "cuda"
    # fp16 autocast overflows to NaN in MoLFormer's rotary attention (verified
    # empirically); eval must match training precision (bf16) or run fp32.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    nctids: list[str] = []
    ys, ps, phs, srcs = [], [], [], []
    aps: list[np.ndarray] = []
    ays: list[np.ndarray] = []
    ams: list[np.ndarray] = []
    for batch in loader:
        nctids.extend(batch["nctid"])
        ys.append(batch["label"].numpy())
        phs.append(batch["phase_index"].numpy())
        if "data_source" in batch:
            srcs.extend(batch["data_source"])
        mol_a = {k: v.to(device, non_blocking=True) for k, v in batch["mol_input_a"].items()}
        mol_b = {k: v.to(device, non_blocking=True) for k, v in batch["mol_input_b"].items()}
        crit = {k: v.to(device, non_blocking=True) for k, v in batch["crit_input"].items()}
        group = batch["group_index"].to(device, non_blocking=True)
        phase = batch["phase_index"].to(device, non_blocking=True)
        mol_mask = batch.get("molecule_mask")
        if mol_mask is not None:
            mol_mask = mol_mask.to(device, non_blocking=True)
        stock_feats = batch.get("stock_feats")
        if stock_feats is not None:
            stock_feats = stock_feats.to(device, non_blocking=True)
        stock_mask = batch.get("stock_mask")
        if stock_mask is not None:
            stock_mask = stock_mask.to(device, non_blocking=True)
        tabular_feats = batch.get("tabular_feats")
        if tabular_feats is not None:
            tabular_feats = tabular_feats.to(device, non_blocking=True)
        tabular_mask = batch.get("tabular_mask")
        if tabular_mask is not None:
            tabular_mask = tabular_mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            succ_logits, appr_logits = net.forward_with_approval(
                mol_input_a=mol_a,
                mol_input_b=mol_b,
                group_index=group,
                batch_size=len(batch["label"]),
                crit_input=crit,
                phase_index=phase,
                stock_feats=stock_feats,
                stock_mask=stock_mask,
                molecule_mask=mol_mask,
                tabular_feats=tabular_feats,
                tabular_mask=tabular_mask,
            )
        ps.append(torch.sigmoid(succ_logits.float()).cpu().numpy().ravel())
        if appr_logits is not None:
            aps.append(torch.sigmoid(appr_logits.float()).cpu().numpy().ravel())
        else:
            aps.append(np.full(len(batch["label"]), np.nan))
        if "approval_label" in batch:
            ays.append(batch["approval_label"].float().numpy().ravel())
            ams.append(batch["approval_mask"].float().numpy().ravel())
        else:
            n_b = len(batch["label"])
            ays.append(np.full(n_b, -1.0))
            ams.append(np.zeros(n_b))
    scores = np.concatenate(ps)
    if not np.all(np.isfinite(scores)):
        n_bad = int((~np.isfinite(scores)).sum())
        raise FloatingPointError(
            f"score_split produced {n_bad}/{len(scores)} non-finite probabilities; "
            "refusing to mask with defaults -- fix the forward pass dtype/inputs."
        )
    return ScoredSplit(
        nctids=nctids,
        y_true=np.concatenate(ys),
        scores=scores,
        phase_index=np.concatenate(phs),
        data_source=np.array(srcs) if srcs else None,
        approval_true=np.concatenate(ays),
        approval_mask=np.concatenate(ams),
        approval_scores=np.concatenate(aps),
    )


# ------------------------------------------------------------------ baselines


def majority_baseline(train_pos_frac: float, split: ScoredSplit) -> dict:
    """Constant-score predictor: AUPRC degenerates to the evaluated set's
    positive rate; F1 reported for the always-positive rule."""
    y = split.y_true
    pos_rate = float(y.mean()) if len(y) else 0.0
    const_scores = np.full_like(y, train_pos_frac, dtype=float)
    rep = classification_report(y, const_scores, threshold=train_pos_frac)
    return {
        "kind": "majority_constant",
        "auprc": pos_rate,  # trivial floor for AUPRC on this split
        "f1_always_positive": f1_at_threshold(y, const_scores, threshold=0.5),
        "report_at_train_rate": rep,
    }


def _rdkit_descriptors(smiles_list: list[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    feats = np.zeros((len(smiles_list), 6), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feats[i] = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ]
    return feats


class ClassicalBaseline:
    """TF-IDF(criteria) + RDKit descriptors -> LogisticRegression.

    Fit on the TRAIN dataframe only. `score(frame, ids)` produces
    probabilities aligned with the requested trial order.
    """

    def __init__(self, train_df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import StandardScaler

        self.desc_scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=20000, sublinear_tf=True, stop_words="english")
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")

        train_desc = self._desc(train_df)
        train_text = train_df["criteria"].apply(clean_criteria_text).tolist()
        self.desc_scaler.fit(train_desc)
        Xd = self.desc_scaler.transform(train_desc)
        Xt = self.tfidf.fit_transform(train_text)
        from scipy.sparse import hstack, csr_matrix

        X = hstack([csr_matrix(Xd), Xt]).tocsr()
        self.clf.fit(X, train_df["label"].to_numpy())

    @staticmethod
    def _desc(df: pd.DataFrame) -> np.ndarray:
        def first_smiles(raw):
            if isinstance(raw, str):
                import ast

                try:
                    raw = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    return raw
            if isinstance(raw, (list, tuple)) and len(raw):
                return raw[0]
            return "C"

        first = df["smiles_canonical"].apply(first_smiles)
        return _rdkit_descriptors(first.tolist())

    def score(self, df: pd.DataFrame, ids: list[str] | None = None) -> np.ndarray:
        from scipy.sparse import hstack, csr_matrix

        if ids is not None:
            df = df.set_index("nctid").loc[ids].reset_index()
        desc = self._desc(df)
        text = df["criteria"].apply(clean_criteria_text).tolist()
        Xd = self.desc_scaler.transform(desc)
        Xt = self.tfidf.transform(text)
        X = hstack([csr_matrix(Xd), Xt]).tocsr()
        return self.clf.predict_proba(X)[:, 1]


class ClassicalBaselineWithStock(ClassicalBaseline):
    """Extends classical baseline with pre-event stock feature vectors."""

    def __init__(self, train_df: pd.DataFrame, include_stock: bool = True):
        from scipy.sparse import csr_matrix, hstack
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.include_stock = include_stock
        self.desc_scaler = StandardScaler()
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf = TfidfVectorizer(max_features=20000, sublinear_tf=True, stop_words="english")
        self.stock_scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")

        train_desc = self._desc(train_df)
        train_text = train_df["criteria"].apply(clean_criteria_text).tolist()
        self.desc_scaler.fit(train_desc)
        Xd = self.desc_scaler.transform(train_desc)
        Xt = self.tfidf.fit_transform(train_text)
        parts = [csr_matrix(Xd), Xt]
        if include_stock and "stock_feats" in train_df.columns:
            stock_mat = np.stack(
                [np.array(x, dtype=np.float64) for x in train_df["stock_feats"].tolist()]
            )
            mask = train_df["stock_mask"].fillna(0).astype(float).values.reshape(-1, 1)
            stock_mat = stock_mat * mask
            self.stock_scaler.fit(stock_mat)
            parts.append(csr_matrix(self.stock_scaler.transform(stock_mat)))
        X = hstack(parts).tocsr()
        self.clf.fit(X, train_df["label"].to_numpy())

    def score(self, df: pd.DataFrame, ids: list[str] | None = None) -> np.ndarray:
        from scipy.sparse import csr_matrix, hstack

        if ids is not None:
            df = df.set_index("nctid").loc[ids].reset_index()
        desc = self._desc(df)
        text = df["criteria"].apply(clean_criteria_text).tolist()
        Xd = self.desc_scaler.transform(desc)
        Xt = self.tfidf.transform(text)
        parts = [csr_matrix(Xd), Xt]
        if self.include_stock and "stock_feats" in df.columns:
            stock_mat = np.stack([np.array(x, dtype=np.float64) for x in df["stock_feats"].tolist()])
            mask = df["stock_mask"].fillna(0).astype(float).values.reshape(-1, 1)
            stock_mat = stock_mat * mask
            parts.append(csr_matrix(self.stock_scaler.transform(stock_mat)))
        X = hstack(parts).tocsr()
        return self.clf.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ protocol


def dump_scored_split(split: ScoredSplit, path: Path) -> Path:
    """Write per-trial scores for offline threshold re-tuning / error analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {
        "nctid": split.nctids,
        "y_true": split.y_true.astype(float),
        "score": split.scores.astype(float),
        "phase_index": split.phase_index.astype(int),
    }
    if split.data_source is not None:
        rows["data_source"] = split.data_source.astype(str)
    if split.approval_true is not None:
        rows["approval_true"] = split.approval_true.astype(float)
        rows["approval_mask"] = split.approval_mask.astype(float)
        rows["approval_score"] = split.approval_scores.astype(float)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_scored_split(path: Path) -> ScoredSplit:
    """Inverse of dump_scored_split."""
    df = pd.read_csv(path)
    src = df["data_source"].to_numpy() if "data_source" in df.columns else None
    app_true = df["approval_true"].to_numpy(dtype=float) if "approval_true" in df.columns else None
    app_mask = df["approval_mask"].to_numpy(dtype=float) if "approval_mask" in df.columns else None
    app_score = df["approval_score"].to_numpy(dtype=float) if "approval_score" in df.columns else None
    return ScoredSplit(
        nctids=df["nctid"].astype(str).tolist(),
        y_true=df["y_true"].to_numpy(dtype=float),
        scores=df["score"].to_numpy(dtype=float),
        phase_index=df["phase_index"].to_numpy(dtype=int),
        data_source=src,
        approval_true=app_true,
        approval_mask=app_mask,
        approval_scores=app_score,
    )


def evaluate_with_threshold_transfer(
    val: ScoredSplit,
    test: ScoredSplit,
    grid: np.ndarray | None = None,
    objective: str = "f1",
    cost_fp: float = 1.0,
    cost_fn: float = 2.0,
    min_precision: float = 0.80,
) -> dict:
    """Tune a global threshold on VAL only; apply frozen to TEST.

    objective: 'f1' | 'cost' | 'precision'
    """
    if objective == "cost":
        thr, val_obj = tune_threshold_cost(
            val.y_true, val.scores, cost_fp=cost_fp, cost_fn=cost_fn, grid=grid
        )
    elif objective == "precision":
        thr, val_obj = tune_threshold_for_precision(
            val.y_true, val.scores, min_precision=min_precision, grid=grid
        )
    else:
        thr, val_obj = tune_threshold_on_val(val.y_true, val.scores, grid=grid)
    val_rep = classification_report(val.y_true, val.scores, threshold=thr)
    test_rep = classification_report(test.y_true, test.scores, threshold=thr)
    return {
        "threshold": float(thr),
        "objective": objective,
        "val_objective": float(val_obj),
        "val_report": val_rep,
        "test_report": test_rep,
    }


def evaluate_per_phase_threshold_transfer(
    val: ScoredSplit,
    test: ScoredSplit,
    grid: np.ndarray | None = None,
    objective: str = "cost",
    cost_fp: float = 1.0,
    cost_fn: float = 2.0,
    min_precision: float = 0.80,
    min_val_n: int = 20,
) -> dict:
    """Tune one threshold per phase on VAL; apply frozen to matching TEST slices."""
    global_proto = evaluate_with_threshold_transfer(
        val,
        test,
        grid=grid,
        objective=objective,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        min_precision=min_precision,
    )
    global_thr = float(global_proto["threshold"])
    thresholds: dict[str, float] = {}
    val_reports: dict[str, dict] = {}
    test_reports: dict[str, dict] = {}

    phases = sorted(
        set(int(x) for x in np.unique(val.phase_index))
        | set(int(x) for x in np.unique(test.phase_index))
    )
    for idx in phases:
        key = PHASE_LABELS.get(idx, str(idx))
        vmask = val.slice_mask(idx)
        tmask = test.slice_mask(idx)
        if int(vmask.sum()) >= min_val_n and len(np.unique(val.y_true[vmask])) >= 2:
            if objective == "cost":
                thr, _ = tune_threshold_cost(
                    val.y_true[vmask],
                    val.scores[vmask],
                    cost_fp=cost_fp,
                    cost_fn=cost_fn,
                    grid=grid,
                )
            elif objective == "precision":
                thr, _ = tune_threshold_for_precision(
                    val.y_true[vmask],
                    val.scores[vmask],
                    min_precision=min_precision,
                    grid=grid,
                )
            else:
                thr, _ = tune_threshold_on_val(val.y_true[vmask], val.scores[vmask], grid=grid)
        else:
            thr = global_thr
        thresholds[key] = float(thr)
        if vmask.any():
            val_reports[key] = {
                "n_trials": int(vmask.sum()),
                **classification_report(val.y_true[vmask], val.scores[vmask], thr),
            }
            if val_reports[key]["roc_auc"] is None:
                val_reports[key].pop("roc_auc")
        if tmask.any():
            test_reports[key] = {
                "n_trials": int(tmask.sum()),
                **classification_report(test.y_true[tmask], test.scores[tmask], thr),
            }
            if test_reports[key]["roc_auc"] is None:
                test_reports[key].pop("roc_auc")

    return {
        "objective": objective,
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "min_precision": float(min_precision),
        "global_threshold": global_thr,
        "global": global_proto,
        "thresholds": thresholds,
        "val": val_reports,
        "test": test_reports,
    }


def per_phase_breakdown(split: ScoredSplit, threshold: float | dict[str, float]) -> dict:
    """Per-phase metrics. `threshold` may be a float or a phase-name -> thr map."""
    out: dict[str, dict] = {}
    for idx in sorted(np.unique(split.phase_index)):
        mask = split.slice_mask(int(idx))
        key = PHASE_LABELS.get(int(idx), str(idx))
        if isinstance(threshold, dict):
            thr = float(threshold.get(key, threshold.get("default", 0.5)))
        else:
            thr = float(threshold)
        out[key] = {
            "n_trials": int(mask.sum()),
            **classification_report(split.y_true[mask], split.scores[mask], thr),
        }
        if out[key]["roc_auc"] is None:
            out[key].pop("roc_auc")
    return out
