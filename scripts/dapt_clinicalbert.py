"""Domain-adaptive pretraining (masked LM) of Bio_ClinicalBERT on the Panacea
TrialAlign corpus (73k trial-protocol documents).

Subsampling is the default: full corpus DAPT costs multiple GPU-days. The abort
gate per plan is: if train MLM loss does not fall >=10% within the probe window,
stop and report (proceed without DAPT).

Output: D:\\Models\\clinicalbert_dapt (AutoModel + tokenizer, loadable by
TriStreamNet via encoders.clinicalbert.name override).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.utils.cuda_utils import amp_dtype, resolve_device, seed_everything  # noqa: E402
from fda_predictor.utils.paths import FDA_MODELS_ROOT  # noqa: E402

TRIALALIGN_DIR = Path(r"D:\datasets\TrialAlign_trial_document\trial_document")
OUT_DIR = FDA_MODELS_ROOT / "clinicalbert_dapt"
CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"
CLINICALBERT_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
MODELS_CACHE = FDA_MODELS_ROOT / "hub"


class MLMDataset(Dataset):
    """Torch dataset of clinical trial protocol texts for masked LM."""

    def __init__(self, texts: list[str], tokenizer, max_len: int = 512):
        self.tok = tokenizer
        self.texts = texts
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def stream_trialalign_texts(limit: int) -> list[str]:
    import pyarrow.ipc as ipc

    texts: list[str] = []
    for shard in sorted(TRIALALIGN_DIR.glob("data-*.arrow")):
        with open(shard, "rb") as f:
            tbl = ipc.open_stream(f).read_all()
        df = tbl.to_pandas()
        texts.extend(df["text"].astype(str).tolist())
        if len(texts) >= limit:
            texts = texts[:limit]
            break
    return texts


def mask_tokens(batch_ids, tok, mlm_prob: float = 0.15):
    labels = batch_ids.clone()
    probs = torch.rand_like(batch_ids, dtype=torch.float32)
    padded = batch_ids.eq(tok.pad_token_id)
    masked = (probs < mlm_prob) & (~padded)
    labels[~masked] = -100
    probs_inner = torch.rand_like(batch_ids, dtype=torch.float32)
    masked_mask = masked.clone()
    # 80/10/10 rule
    batch_ids[masked_mask & (probs_inner < 0.8)] = tok.mask_token_id
    random_words = torch.randint_like(batch_ids, low=0, high=tok.vocab_size)
    batch_ids[masked_mask & (probs_inner >= 0.8) & (probs_inner < 0.9)] = random_words[
        masked_mask & (probs_inner >= 0.8) & (probs_inner < 0.9)
    ]
    return batch_ids, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--probe-steps", type=int, default=300, help="abort gate: loss must fall >=10% after this many steps")
    parser.add_argument("--probe-min-drop", type=float, default=0.10)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    seed_everything(42)
    device = resolve_device("cuda")
    if device.type != "cuda":
        print("DAPT needs cuda; aborting", flush=True)
        return 1

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print("loading TrialAlign texts ...", flush=True)
    texts = stream_trialalign_texts(args.max_docs)
    print(f"loaded {len(texts)} documents", flush=True)

    tok = AutoTokenizer.from_pretrained(
        CLINICALBERT, revision=CLINICALBERT_REVISION, cache_dir=str(MODELS_CACHE)
    )
    model = AutoModelForMaskedLM.from_pretrained(
        CLINICALBERT, revision=CLINICALBERT_REVISION, cache_dir=str(MODELS_CACHE)
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    ds = MLMDataset(texts, tok, max_len=args.max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(loader) / args.accum) * args.epochs
    warmup = max(10, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=warmup / total_steps
    )

    amp_dt, use_scaler = amp_dtype(device, "bfloat16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    loss_history: list[float] = []
    t0 = time.time()
    global_step = 0
    running_loss = 0.0
    probe_base: float | None = None
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            ids = batch["input_ids"]
            am = batch["attention_mask"]
            if not torch.is_tensor(ids):
                ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)
            if not torch.is_tensor(am):
                am = torch.as_tensor(np.asarray(am), dtype=torch.long)
            ids = ids.to(device, non_blocking=True)
            am = am.to(device, non_blocking=True)
            ids, labels = mask_tokens(ids, tok)
            with torch.autocast(device_type="cuda", dtype=amp_dt, enabled=amp_dt is not None):
                out = model(input_ids=ids, attention_mask=am, labels=labels)
                loss = out.loss / args.accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % args.accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                running_loss += float(loss.item()) * args.accum
                if global_step % 25 == 0:
                    avg = running_loss / 25
                    loss_history.append(avg)
                    running_loss = 0.0
                    if probe_base is None:
                        probe_base = avg
                    print(
                        f"step {global_step}/{total_steps} | mlm_loss {avg:.4f} | "
                        f"elapsed {time.time()-t0:.0f}s",
                        flush=True,
                    )
                if global_step == args.probe_steps and probe_base is not None and loss_history:
                    cur = loss_history[-1]
                    drop = (probe_base - cur) / max(probe_base, 1e-8)
                    print(f"probe@step{args.probe_steps}: base={probe_base:.4f} now={cur:.4f} drop={drop:.1%}", flush=True)
                    if drop < args.probe_min_drop:
                        print(
                            "DAPT ABORT GATE: MLM loss did not fall enough in probe window.",
                            flush=True,
                        )
                        return 2
        print(f"epoch {epoch+1} done | elapsed {time.time()-t0:.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved DAPT clinicalbert -> {out}", flush=True)
    print(f"final loss history tail: {loss_history[-5:]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
