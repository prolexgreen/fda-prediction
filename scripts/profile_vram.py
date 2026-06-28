"""VRAM/throughput profiler for the pipeline config's current freeze state.

Profiles peak CUDA memory and steps/sec at the requested batch sizes using
the real pinned backbones and real TOP data. When config unfreezes encoders
(stage-5), this is the OOM gate: run before the full training run.
No checkpointing, no TB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.datasets import TOPTrialDataset, build_collate_fn  # noqa: E402
from fda_predictor.data.tokenizers import specs_from_config  # noqa: E402
from fda_predictor.data.top_dataset import load_top_splits  # noqa: E402
from fda_predictor.models.multimodal_net import TriStreamNet  # noqa: E402
from fda_predictor.training.losses import build_loss  # noqa: E402
from fda_predictor.utils.cuda_utils import amp_dtype, resolve_device, seed_everything  # noqa: E402
from fda_predictor.utils.paths import ensure_dirs  # noqa: E402


def profile_batch_size(net, criterion, batches, device, amp, n_steps: int) -> dict:
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp[1]) if amp[1] else None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for i in range(n_steps):
        batch = batches[i % len(batches)]
        with torch.autocast(device_type="cuda", dtype=amp[0], enabled=amp[0] is not None):
            logits = net(
                mol_input_a=batch["mol_input_a"],
                mol_input_b=batch["mol_input_b"],
                group_index=batch["group_index"],
                batch_size=len(batch["label"]),
                crit_input=batch["crit_input"],
            )
            loss = criterion(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    return {
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "steps_per_sec": n_steps / elapsed,
        "sec_per_step": elapsed / n_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(42)
    device = resolve_device("cuda")
    if device.type != "cuda":
        print("CUDA unavailable -- nothing to profile")
        return 1

    config = yaml.safe_load(Path("configs/config.yaml").read_text())
    net = TriStreamNet.from_config(config)
    net.to(device)
    criterion, _ = build_loss(config, [1, 0] * 16)

    specs = specs_from_config(config)
    common = dict(
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    splits = load_top_splits()
    ds = TOPTrialDataset(splits.train.head(128), **common)
    collate = build_collate_fn(ds)
    amp = amp_dtype(device, config["compute"].get("amp_dtype", "bfloat16"))
    print(f"device: {torch.cuda.get_device_name(0)} | amp: {amp[0]} (scaler={amp[1]})")

    rows = []
    for bs in args.batch_sizes:
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
        batches = []
        for b in loader:
            batches.append({
                "mol_input_a": {k: v.to(device) for k, v in b["mol_input_a"].items()},
                "mol_input_b": {k: v.to(device) for k, v in b["mol_input_b"].items()},
                "crit_input": {k: v.to(device) for k, v in b["crit_input"].items()},
                "group_index": b["group_index"].to(device),
                "label": b["label"].to(device),
            })
            if len(batches) >= 4:
                break
        stats = profile_batch_size(net, criterion, batches, device, amp, args.steps)
        stats["batch_size"] = bs
        rows.append(stats)
        print(f"batch {bs:>3}: peak_alloc {stats['peak_allocated_mb']:>7.0f} MB | "
              f"peak_reserved {stats['peak_reserved_mb']:>7.0f} MB | "
              f"{stats['steps_per_sec']:.2f} steps/s ({stats['sec_per_step']*1000:.0f} ms/step)")

    print("\n== VRAM/throughput profile ==")
    for r in rows:
        print(f"batch {r['batch_size']:>3}: {r['peak_allocated_mb']:.0f} MB alloc / "
              f"{r['peak_reserved_mb']:.0f} MB reserved | {r['steps_per_sec']:.2f} steps/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
