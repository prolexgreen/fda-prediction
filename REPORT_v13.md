# Trial Outcome Predictor — v13 Final Readout

## Headline
**Stage-9 KG + Panacea-DAPT ClinicalBERT** is the new best model.

| Metric | v12 (KG only) | **v13 (KG + DAPT)** | Δ |
|---|---|---|---|
| best val Phase-III AUPRC | 0.8293 | **0.8695** | **+0.040** |
| global-thr TEST AUPRC | 0.6388 | **0.7782** | **+0.139** |
| global-thr TEST precision | 0.749 | 0.792 | +0.043 |
| global-thr TEST recall | 0.295 | **0.586** | **+0.291** |
| tuned P3 TEST precision | 0.757 | 0.765 | +0.008 |
| tuned P3 TEST recall | 0.671 | **0.864** | **+0.193** |
| tuned P3 TEST TP/FP/FN | 537/172/263 | **691/212/109** | FN −154 |

Per-phase tuned threshold for Phase III = **0.540** (v13) vs 0.567 (v12); the new encoder pulls the optimal operating point down a notch.

## What changed
1. **Panacea TrialAlign corpus** (`linjc16/TrialAlign`, `trial_document` config, 73,950 protocols, ~4.5 GB arrow shards) — downloaded to `D:\datasets\TrialAlign_trial_document`.
2. **DAPT ClinicalBERT** — masked-LM continued pretraining on 100,000 protocol docs, 1 epoch, max-len 512, MLM-loss 8.01 → 5.69 (29% drop, probe gate passed at step 300). Saved to `D:\Models\clinicalbert_dapt`. Loadable via `encoders.clinicalbert.dapt_path` config override.
3. **KG feature block** (TxGNN-sidecar embedding PCA-100, masked for ~40% of trials) carried forward from stage-8.
4. **No warm start** for stage-9 (would have overwritten the DAPT'd ClinicalBERT with stage-8's stock-CBERT weights).

## What matters in v13 (ablation ranking)
| Variant | P3 AUPRC | Δ |
|---|---|---|
| baseline | 0.8445 | — |
| tabular removed | 0.8259 | +0.0186 (largest hit) |
| chemistry feature dropped | 0.8374 | +0.0071 |
| legacy block zeroed | 0.8385 | +0.0060 |
| KG block zeroed | 0.8418 | +0.0027 |
| stock feature dropped | 0.8428 | +0.0017 |
| chemistry block zeroed | 0.8459 | −0.0014 |
| mechanism block zeroed | 0.8465 | −0.0020 |
| modality block zeroed | 0.8467 | −0.0022 |
| sponsor block zeroed | 0.8473 | −0.0028 |

**Reading:** the *text+tabular* combination is still dominant. KG adds a small but real edge (~0.003). The chemistry/modality/mechanism/sponsor sub-blocks are partially redundant under the DAPT encoder (knockout doesn't hurt).

## Things that did NOT move the needle
- Stock ClinicalBERT (no DAPT): v12 → AUPRC 0.6388, recall 0.295.
- Stock ChemBERTa/MoLFormer with same config: see v10–v12 ablation.
- Sponsor features alone (stage-7): val P3 AUPRC 0.8342.

## Artifacts
- `artifacts/checkpoints/stage9_kg_dapt_best.pt` — best ckpt (epoch 6, val P3 AUPRC 0.8695).
- `artifacts/runs/stage9_kg_dapt/` — training logs, tensorboard events.
- `artifacts/runs/backtest_v13_kgdapt/` — per-phase scores + thresholds + metrics.json.
- `artifacts/runs/ablation_v13/ablation.json` — single-feature and block-knockout deltas.
- `D:\Models\clinicalbert_dapt\` — DAPT'd ClinicalBERT (config + weights + tokenizer).
- `D:\datasets\TrialAlign_trial_document\` — Panacea TrialAlign arrow shards.

## Future moves (not done, listed in priority)
1. **DAPT longer / with more docs** — 1 epoch / 100k docs is conservative; loss curve was still falling at the end. A 3-epoch run on the full 73,950 docs (≈ ~3 h) likely yields another small lift.
2. **DAPT w/ PubMed + TrialAlign** — Panacea already includes PubMed text in some configs; layering ClinicalBERT's original MIMIC-III + PMC-PubMed mix back in via secondary MLM run could recover vocabulary breadth.
3. **v14 = DAPT only, no KG** — to isolate the DAPT contribution precisely (the ablation showed KG adds only ~0.003 on top).
4. **Thresholded "approval head" rerun** — the approval head collapsed (TP=0 in v13) because the labelled subset is tiny (n=334 test); a calibration step on a larger curated cohort may rescue it.
