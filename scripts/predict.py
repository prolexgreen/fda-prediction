"""Task 5 live inference CLI: FDA approval-success probability for trials
currently registered on ClinicalTrials.gov.

Examples:
    .\\.venv\\Scripts\\python.exe scripts\\predict.py NCT06688490 --device cpu
    .\\.venv\\Scripts\\python.exe scripts\\predict.py --nct-file ncts.txt --out report.json
    .\\.venv\\Scripts\\python.exe scripts\\predict.py --drug semaglutide --limit 10
    .\\.venv\\Scripts\\python.exe scripts\\predict.py NCT06688490 --ticker MRK

Inputs go through the SAME preprocessing path as training
(MergedTrialDataset / TrialCollator / specs_from_config). Probabilities
are raw sigmoids over logits - uncalibrated until Task 6 calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.utils.paths import CHECKPOINTS_DIR, ensure_dirs  # noqa: E402

MODEL_VERSION_NOTE = (
    "TriStreamNet multimodal classifier (ChemBERTa + MoLFormer on SMILES, "
    "Bio_ClinicalBERT on eligibility criteria, phase embedding, StockEncoder, "
    "optional tabular design-time features); probabilities use the calibration "
    "method stored next to the checkpoint (Platt / isotonic / raw) and "
    "per-phase decision thresholds from backtest when available."
)

DEFAULT_CHECKPOINT_GLOB = "stage2_clinicalbert*.pt"
DEFAULT_THRESHOLDS_GLOB = "artifacts/runs/*/thresholds.json"


def load_calibration(checkpoint: Path) -> dict | None:
    """Sibling *_calibration.json written by scripts/calibrate.py, if any."""
    calib = checkpoint.with_name(checkpoint.stem + "_calibration.json")
    if not calib.exists():
        return None
    try:
        payload = json.loads(calib.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    method = str(payload.get("method") or "platt").lower()
    if method == "none":
        return {"method": "none", "source": calib.name}

    if method == "isotonic":
        iso = payload.get("isotonic") or {}
        xs = iso.get("x_thresholds")
        ys = iso.get("y_thresholds")
        if not xs or not ys or len(xs) != len(ys):
            return None
        return {
            "method": "isotonic",
            "x_thresholds": [float(x) for x in xs],
            "y_thresholds": [float(y) for y in ys],
            "source": calib.name,
        }

    # default / platt
    params = payload.get("platt") or {}
    try:
        scale = float(params["logit_scale"])
        intercept = float(params["intercept"])
    except (KeyError, TypeError, ValueError):
        return None
    if scale <= 0:
        return None
    return {
        "method": "platt",
        "logit_scale": scale,
        "intercept": intercept,
        "source": calib.name,
    }


def apply_calibration(probs: np.ndarray, calib: dict) -> np.ndarray:
    """Apply stored calibration (platt / isotonic / none) to raw probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    method = str(calib.get("method") or "platt").lower()
    if method in ("none", "raw"):
        return p
    if method == "isotonic":
        xs = np.asarray(calib["x_thresholds"], dtype=float)
        ys = np.asarray(calib["y_thresholds"], dtype=float)
        return np.interp(p, xs, ys).clip(0.0, 1.0)
    # platt
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(calib["logit_scale"] * z + calib["intercept"])))


def load_thresholds(path: str | Path | None = None) -> dict | None:
    """Load per-phase thresholds JSON from an explicit path or newest run."""
    if path:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    runs = root / "artifacts" / "runs"
    if not runs.exists():
        return None
    candidates = sorted(runs.glob("*/thresholds.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    try:
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def decision_for_phase(prob: float, phase_raw: str, thresholds: dict | None) -> dict:
    """Map probability -> approve/reject/uncertain using per-phase thresholds."""
    phase_key = "UNK(combo/missing)"
    pr = (phase_raw or "").upper().replace("PHASE", "").strip()
    if pr in ("1", "I"):
        phase_key = "I"
    elif pr in ("2", "II"):
        phase_key = "II"
    elif pr in ("3", "III"):
        phase_key = "III"
    elif pr in ("4", "IV"):
        phase_key = "IV"

    if not thresholds:
        return {"decision": "score_only", "threshold": None, "phase_key": phase_key}

    per_phase = thresholds.get("per_phase") or {}
    thr = per_phase.get(phase_key, thresholds.get("global_threshold"))
    if thr is None:
        return {"decision": "score_only", "threshold": None, "phase_key": phase_key}
    thr = float(thr)
    # Abstention band: within 5% of the threshold relative to [0,1]
    band = float(thresholds.get("abstain_band", 0.05))
    if abs(float(prob) - thr) < band:
        decision = "uncertain"
    elif float(prob) >= thr:
        decision = "approve"
    else:
        decision = "reject"
    return {"decision": decision, "threshold": thr, "phase_key": phase_key, "abstain_band": band}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict FDA approval success for live ClinicalTrials.gov trials"
    )
    p.add_argument("nct_ids", nargs="*", help="one or more NCT IDs (e.g. NCT06688490)")
    p.add_argument("--nct-file", default=None, help="text file with one NCT ID per line")
    p.add_argument("--drug", default=None, help="search ClinicalTrials.gov by intervention name")
    p.add_argument("--limit", type=int, default=20, help="max studies returned by --drug search")
    p.add_argument(
        "--overall-status",
        default=None,
        help="optional CTGov status filter for --drug search (e.g. RECRUITING)",
    )
    p.add_argument(
        "--ticker",
        default=None,
        help="sponsor ticker for pre-event stock features (omitted => stock_mask=0)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=".pt checkpoint (default: newest stage1_merged_best* under artifacts/checkpoints)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device (default: cpu so smoke runs never touch the training GPU)",
    )
    p.add_argument("--out", default=None, help="write JSON report to this path")
    p.add_argument(
        "--openfda",
        dest="openfda",
        action="store_true",
        default=True,
        help="check openFDA for prior approvals of each trial's drugs (default: on)",
    )
    p.add_argument(
        "--no-openfda",
        dest="openfda",
        action="store_false",
        help="skip openFDA lookups entirely (fully offline)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--thresholds",
        default=None,
        help="path to thresholds.json from a backtest run (default: newest under artifacts/runs)",
    )
    args = p.parse_args(argv)
    if not args.nct_ids and not args.nct_file and not args.drug:
        p.error("provide NCT IDs, --nct-file, or --drug")
    return args


def read_nct_file(path: str | Path) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def collect_nct_ids(args: argparse.Namespace) -> tuple[list[str], list[dict] | None]:
    """Return (nct_ids, searched_studies); studies set only when --drug used."""
    if args.drug:
        from fda_predictor.inference.search import search_trials

        records = search_trials(
            args.drug, limit=max(int(args.limit), 1), overall_status=args.overall_status
        )
        print(f"--drug '{args.drug}' matched {len(records)} study/studies (limit {args.limit})")
        return [r.nct_id for r in records], [r.raw for r in records]
    if args.nct_file:
        return read_nct_file(args.nct_file), None
    return list(args.nct_ids), None


def find_default_checkpoint(checkpoints_dir: Path | None = None) -> Path | None:
    base = Path(checkpoints_dir) if checkpoints_dir else CHECKPOINTS_DIR
    if not base.exists():
        return None
    candidates = [p for p in base.glob(DEFAULT_CHECKPOINT_GLOB) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint(cli_value: str | None, checkpoints_dir: Path | None = None) -> Path:
    base = checkpoints_dir or CHECKPOINTS_DIR
    if cli_value:
        ckpt = Path(cli_value)
        if not ckpt.exists() and not ckpt.is_absolute():
            # allow bare filenames like "stage1_phase_best.pt" from CHECKPOINTS_DIR
            alt = base / ckpt.name
            if alt.exists():
                return alt
        return ckpt
    found = find_default_checkpoint(checkpoints_dir)
    if found is None:
        raise FileNotFoundError(
            f"No checkpoint matching '{DEFAULT_CHECKPOINT_GLOB}' under "
            f"{checkpoints_dir or CHECKPOINTS_DIR}. Train first (scripts/train.py) "
            "or pass --checkpoint explicitly."
        )
    return found


def trial_frame_row(rec, ticker: str | None, max_drugs: int) -> dict:
    """StudyRecord -> dataframe row matching MergedTrialDataset expectations."""
    from fda_predictor.data.pubchem_smiles import PubChemReport, resolve_drug_list
    from fda_predictor.data.stock_features import (
        N_STOCK_FEATURES,
        compute_pre_event_features,
    )
    from fda_predictor.data.tabular_features import row_tabular_raw

    smiles: list[str] = []
    if rec.drugs:
        try:
            smiles = resolve_drug_list(rec.drugs, report=PubChemReport(), max_drugs=max_drugs)
        except Exception:  # noqa: BLE001 - offline-safe degradation
            smiles = []

    stock_feats = np.zeros(N_STOCK_FEATURES, dtype=np.float32)
    stock_mask = 0
    if ticker:
        try:
            arr, mask = compute_pre_event_features(
                ticker, rec.completion_date or rec.start_date, rec.start_date
            )
            stock_feats = np.asarray(arr, dtype=np.float32)
            stock_mask = int(mask)
        except Exception:  # noqa: BLE001 - yfinance/network failure degrades to mask=0
            stock_feats, stock_mask = np.zeros(N_STOCK_FEATURES, dtype=np.float32), 0

    # Live CTGov records lack enrollment/arms/dmc; molecule_present still informative.
    tab_raw, tab_mask = row_tabular_raw(
        {
            "enrollment": None,
            "number_of_arms": None,
            "has_dmc": None,
            "source_class": None,
            "molecule_mask": int(bool(smiles)),
            "n_prior_drug_approvals": 0,
            "is_fda_regulated_drug": None,
        }
    )

    return {
        "nctid": rec.nct_id,
        "label": 0.0,  # placeholder; unused at inference
        "criteria": rec.criteria or "[NO CRITERIA REPORTED]",
        "smiles_canonical": smiles,
        "molecule_mask": int(bool(smiles)),
        "phase_index": int(rec.phase_index),
        "stock_feats": stock_feats.tolist(),
        "stock_mask": int(stock_mask),
        "tabular_feats": tab_raw.tolist(),
        "tabular_mask": tab_mask.tolist(),
        "ticker": ticker,
        "data_source": "LIVE",
        "chronology_date": rec.study_first_submitted_date or rec.start_date,
        "start_date": rec.start_date,
        "completion_date": rec.completion_date,
    }


def _load_net(checkpoint: Path, config: dict, device):
    import torch

    from fda_predictor.models.multimodal_net import TriStreamNet

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    layout = payload.get("checkpoint_layout")
    expected = TriStreamNet.CHECKPOINT_LAYOUT
    if layout not in (3, 4, 5, expected):
        raise ValueError(
            f"checkpoint_layout={layout} but this build expects {expected}; refusing to "
            "score with a stale architecture. Retrain or pick another --checkpoint."
        )
    net = TriStreamNet.from_config(config, force_layout=3 if int(payload.get("checkpoint_layout") or 0) == 3 else None)
    net.load_state_dict(payload["model_state"], strict=False)
    net.to(device)
    net.eval()
    return net, payload


def score_records(net, frame_rows: list[dict], config: dict, device, batch_size: int = 4):
    """Forward through the same collator path as training; sigmoid at inference.

    Returns {nctid: {"success": p, "approval": p_or_None}}.
    """
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from fda_predictor.data.datasets import MergedTrialDataset, build_collate_fn
    from fda_predictor.data.tokenizers import specs_from_config

    specs = specs_from_config(config)
    ds = MergedTrialDataset(
        pd.DataFrame(frame_rows),
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    collate = build_collate_fn(ds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    out: dict[str, dict] = {}
    with torch.no_grad():
        for batch in loader:
            mol_a = {k: v.to(device) for k, v in batch["mol_input_a"].items()}
            mol_b = {k: v.to(device) for k, v in batch["mol_input_b"].items()}
            crit = {k: v.to(device) for k, v in batch["crit_input"].items()}
            succ_logits, appr_logits = net.forward_with_approval(
                mol_input_a=mol_a,
                mol_input_b=mol_b,
                group_index=batch["group_index"].to(device),
                batch_size=len(batch["label"]),
                crit_input=crit,
                phase_index=batch["phase_index"].to(device),
                stock_feats=batch["stock_feats"].to(device),
                stock_mask=batch["stock_mask"].to(device),
                molecule_mask=batch["molecule_mask"].to(device),
                tabular_feats=batch["tabular_feats"].to(device),
                tabular_mask=batch["tabular_mask"].to(device),
            )
            probs = torch.sigmoid(succ_logits.float()).cpu().numpy().ravel()
            appr_probs = (
                torch.sigmoid(appr_logits.float()).cpu().numpy().ravel()
                if appr_logits is not None
                else np.full(len(batch["label"]), np.nan)
            )
            all_p = np.concatenate([probs, appr_probs[np.isfinite(appr_probs)]])
            if not np.all(np.isfinite(all_p)):
                raise FloatingPointError("non-finite probabilities in score_records")
            for nctid, p_s, p_a in zip(batch["nctid"], probs, appr_probs):
                out[str(nctid)] = {
                    "success": float(p_s),
                    "approval": float(p_a) if np.isfinite(p_a) else None,
                }
    return out


def _openfda_for_drugs(drug_names: list[str]) -> dict:
    from fda_predictor.inference.openfda_client import check_drug_approvals

    out = {}
    for name in drug_names[:3]:
        try:
            out[name] = check_drug_approvals(name, timeout_s=10.0).to_dict()
        except Exception:  # noqa: BLE001 - openFDA must never break the run
            continue
    return out


def build_report(
    records: dict,
    probabilities: dict,
    frame_rows_by_nct: dict,
    checkpoint: Path,
    layout,
    device_str: str,
    openfda_enabled: bool,
    calibration: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    trials = []
    for nct_id, rec in records.items():
        row = frame_rows_by_nct[nct_id]
        pair = probabilities.get(rec.nct_id) or {}
        if isinstance(pair, dict):
            prob = float(pair.get("success", float("nan")))
            approval_prob = pair.get("approval")
            approval_prob = float(approval_prob) if approval_prob is not None else None
        else:  # legacy float mapping from older callers
            prob = float(pair)
            approval_prob = None

        # Decision prefers approval-head thresholds when available.
        appr_thresholds = (thresholds or {}).get("approval_per_phase")
        if approval_prob is not None and appr_thresholds:
            decision = decision_for_phase(approval_prob, rec.phase_raw, {"per_phase": appr_thresholds, "global_threshold": (thresholds or {}).get("approval_global")})
            decision["basis"] = "approval_head"
        else:
            decision = decision_for_phase(prob, rec.phase_raw, thresholds)
            decision["basis"] = "success_head"
        entry = {
            "nct_id": rec.nct_id,
            "drugs": rec.drugs,
            "phase": rec.phase_raw or "UNKNOWN",
            "sponsor": rec.sponsor,
            "start_date": rec.start_date,
            "completion_date": rec.completion_date,
            "overall_status": rec.overall_status,
            "success_probability": round(prob, 4),
            "approval_probability": (
                round(approval_prob, 4) if approval_prob is not None else None
            ),
            "decision": decision.get("decision"),
            "decision_threshold": decision.get("threshold"),
            "decision_phase_key": decision.get("phase_key"),
            "decision_basis": decision.get("basis"),
            "smiles_resolved_count": len(row["smiles_canonical"]),
            "molecule_mask": int(row["molecule_mask"]),
            "stock_mask": int(row["stock_mask"]),
            "ticker": row["ticker"],
        }
        if openfda_enabled and rec.drugs:
            entry["openfda_prior_approvals"] = _openfda_for_drugs(rec.drugs)
        trials.append(entry)

    if calibration and calibration.get("method") not in (None, "none"):
        method = calibration["method"]
        params = (
            {"logit_scale": calibration["logit_scale"], "intercept": calibration["intercept"]}
            if method == "platt"
            else {
                "n_knots": len(calibration.get("x_thresholds") or []),
            }
        )
        calibration_block = {
            "method": method,
            "fit_on": "chronological validation split",
            "params": params,
            "source_file": calibration.get("source"),
        }
    else:
        calibration_block = {
            "method": "none",
            "note": "raw sigmoid; no *_calibration.json next to checkpoint",
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {
            "filename": Path(checkpoint).name,
            "path": str(Path(checkpoint).resolve()),
            "checkpoint_layout": layout,
        },
        "calibration": calibration_block,
        "thresholds": thresholds,
        "model_version_note": MODEL_VERSION_NOTE,
        "device": device_str,
        "n_trials": len(trials),
        "trials": trials,
    }


def print_report(report: dict) -> None:
    ckpt = report["checkpoint"]["filename"]
    print(f"\ncheckpoint: {ckpt} (layout {report['checkpoint']['checkpoint_layout']})")
    print(f"device: {report['device']} | trials scored: {report['n_trials']}")
    print("-" * 100)
    for t in report["trials"]:
        drugs = ", ".join(t["drugs"][:3]) if t["drugs"] else "(none listed)"
        decision = t.get("decision") or "score_only"
        thr = t.get("decision_threshold")
        thr_s = f" thr={thr:.3f}" if isinstance(thr, (int, float)) else ""
        print(
            f"{t['nct_id']}  phase={t['phase']:<14} P(approval)={t['success_probability']:.4f} "
            f"-> {decision}{thr_s}"
        )
        print(f"   drugs:   {drugs}")
        print(f"   sponsor: {t['sponsor'] or '(unknown)'} | status: {t['overall_status']}")
        ofda = t.get("openfda_prior_approvals") or {}
        if ofda:
            bits = [
                f"{name}: prior={info.get('has_prior_approval')} first={info.get('first_approval_date')}"
                for name, info in list(ofda.items())[:2]
            ]
            print(f"   openFDA: {'; '.join(bits)}")
        print(f"   window:  {t['start_date']} -> {t['completion_date']} | masks(mol/stock)={t['molecule_mask']}/{t['stock_mask']}")
        print("-" * 100)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dirs()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())

    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Loading checkpoint: {checkpoint}")

    nct_ids, _searched = collect_nct_ids(args)
    if not nct_ids:
        print("ERROR: no NCT IDs resolved from arguments", file=sys.stderr)
        return 1

    from fda_predictor.inference.ctgov_client import CTGovClient
    from fda_predictor.utils.cuda_utils import resolve_device

    device = resolve_device(args.device)
    client = CTGovClient()
    records = client.fetch_by_nct_ids(nct_ids)
    missing = [n for n in dict.fromkeys(nct_ids) if n not in records]
    for m in missing:
        print(f"WARN: {m} not found on ClinicalTrials.gov; skipping", file=sys.stderr)
    if not records:
        print("ERROR: no trials fetched; nothing to score", file=sys.stderr)
        return 1

    max_drugs = int(config.get("cto", {}).get("max_drugs_pubchem", 3))
    frame_rows_by_nct = {
        nct_id: trial_frame_row(rec, args.ticker, max_drugs)
        for nct_id, rec in records.items()
    }
    frame_rows = [frame_rows_by_nct[k] for k in records.keys()]

    try:
        net, payload = _load_net(checkpoint, config, device)
    except Exception as exc:  # noqa: BLE001 - fail with a clear message, no traceback dump
        print(f"ERROR loading checkpoint: {exc}", file=sys.stderr)
        return 1

    scored_pairs = score_records(net, frame_rows, config, device, batch_size=args.batch_size)

    calibration = load_calibration(checkpoint)
    probabilities: dict[str, float] = {}
    for k, pair in scored_pairs.items():
        p_s = np.array([pair["success"]])
        if calibration and calibration.get("method") not in (None, "none"):
            p_s = apply_calibration(p_s, calibration)
        probabilities[k] = float(p_s[0])

    # Approval-head probabilities use the approval block of the same
    # calibration file when present (isotonic or Platt).
    approval_calibration = (calibration or {}).get("approval")
    for k, pair in scored_pairs.items():
        if pair.get("approval") is None:
            continue
        p_a = np.array([pair["approval"]])
        if approval_calibration and approval_calibration.get("method") not in (None, "none"):
            p_a = apply_calibration(
                p_a,
                {
                    "method": approval_calibration["method"],
                    "logit_scale": approval_calibration.get("platt", {}).get("logit_scale", 1.0),
                    "intercept": approval_calibration.get("platt", {}).get("intercept", 0.0),
                    "x_thresholds": approval_calibration.get("isotonic", {}).get("x_thresholds"),
                    "y_thresholds": approval_calibration.get("isotonic", {}).get("y_thresholds"),
                },
            )
        scored_pairs[k]["approval"] = float(p_a[0])

    if calibration and calibration.get("method") not in (None, "none"):
        print(f"Applied {calibration['method']} calibration from {calibration['source']}")

    thresholds = load_thresholds(args.thresholds)
    if thresholds:
        print(
            f"Using thresholds (objective={thresholds.get('objective')}) "
            f"per_phase={thresholds.get('per_phase')}"
        )

    report = build_report(
        records=records,
        probabilities=scored_pairs,
        frame_rows_by_nct=frame_rows_by_nct,
        checkpoint=checkpoint,
        layout=payload.get("checkpoint_layout"),
        device_str=str(device),
        openfda_enabled=bool(args.openfda),
        calibration=calibration,
        thresholds=thresholds,
    )
    print_report(report)

    scored = [t["success_probability"] for t in report["trials"]]
    if scored and all(abs(p - 0.5) < 0.02 for p in scored):
        print(
            "NOTE: all probabilities ~0.5 - expected while the parallel retrain is "
            "incomplete; scores are not meaningful yet.",
            file=sys.stderr,
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
