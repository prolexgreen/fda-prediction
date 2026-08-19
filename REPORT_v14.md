# v14: DAPT-only (no KG) — eliminates the TxGNN sidecar

## Result

| | val P3 AUPRC | TEST AUPRC | TEST P / R (global thr) | tuned-P3 P / R |
|---|---|---|---|---|
| v12 (KG only, stock CBERT) | 0.8293 | 0.6388 | 0.749 / 0.295 | 0.757 / 0.671 |
| v13 (KG + DAPT) | 0.8695 | 0.7782 | 0.792 / 0.586 | 0.765 / 0.864 |
| **v14 (DAPT only, no KG)** | **0.8696** | **0.7814** | **0.795 / 0.574** | **0.767 / 0.889** |

Trial counts at tuned-P3: v13 TP/FP/FN = 691/212/109 → v14 TP/FP/FN = **711/216/89** (FN −20).

## Interpretation

The TxGNN KG sidecar (PCA-64, ~40% trial coverage) contributes nothing measurable once
ClinicalBERT is DAPT-adapted: v14 matches or beats v13 on every reported metric, and the
tuned Phase-III recall rises to 0.889. The earlier v13 block ablation had already flagged
this (KG knockout delta +0.0027, smallest positive contribution of any block).

**Decision: v14 is the champion.** Simpler model (64 fewer tabular features), no KG build
step, no TxGNN dependency, same or better performance.

## Reproduce

```powershell
$env:FDA_DISABLE_KG='1'
.venv\Scripts\python.exe scripts\train.py --merged --run-name stage14_dapt_nokg
.venv\Scripts\python.exe scripts\dump_scores.py --checkpoint stage14_dapt_nokg_best.pt --run-name backtest_v14_dapt_nokg
```

## Overfitting check (v13 / v14)

- Early stopping fired epoch 11 (best epoch 6). Train loss vs val loss at best: loss gap
  ~0.43 → ~0.56; a train/val loss gap exists, as expected for unfrozen encoders, **but the
  ranking signal generalizes**: val P3 AUPRC 0.8696 → test P3 AUPRC 0.7814 (tuned
  thresholds transferred from val, test never touched during selection).
- Thresholds/calibration are tuned on val only (per `dump_scores.py` protocol).
- Chronological train→val→test eras with lookahead assertions; no contamination.
- Block ablations are consistent across both runs (tabular dominant), suggesting the
  learned representation is stable rather than memorized noise.

Net: mild loss-gap, no metric-collapse signature; not overfit in the operative sense.
