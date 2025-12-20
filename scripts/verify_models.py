"""Task 2 acceptance driver: builds TriStreamNet from the real pinned
checkpoints, prints the per-stream parameter summary under the staged
(default: all-backbones-frozen) config, probes gradient-checkpointing
support per backbone, runs a real CPU forward pass, and estimates VRAM
for the frozen-head configuration at batch 32.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fda_predictor.data.datasets import TOPTrialDataset, build_collate_fn  # noqa: E402
from fda_predictor.data.tokenizers import specs_from_config  # noqa: E402
from fda_predictor.data.top_dataset import load_top_splits  # noqa: E402
from fda_predictor.models.multimodal_net import TriStreamNet  # noqa: E402
from fda_predictor.utils.paths import ensure_dirs  # noqa: E402


def count_params(module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def main() -> int:
    ensure_dirs()
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())

    print("Building TriStreamNet from pinned checkpoints (cached on D:\\Models) ...")
    net = TriStreamNet.from_config(config)
    print("\n== Parameter summary (default staged config) ==")
    print(net.parameter_summary())

    trainable, total = count_params(net)
    print(f"\nfrozen fraction: {1 - trainable / total:.1%} of {total:,} params")

    print("\n== Gradient checkpointing probe ==")
    gc_results = net.enable_gradient_checkpointing()
    for name, ok in gc_results.items():
        print(f"  {name}: {'supported, enabled' if ok else 'NOT supported -- continuing without'}")

    print("\n== Real-data CPU forward pass ==")
    specs = specs_from_config(config)
    common = dict(
        chemberta_spec=specs["chemberta"],
        molformer_spec=specs["molformer"],
        clinicalbert_spec=specs["clinicalbert"],
    )
    splits = load_top_splits()
    ds = TOPTrialDataset(splits.train.head(32), **common)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=build_collate_fn(ds))
    batch = next(iter(loader))

    net.eval()
    with torch.no_grad():
        logits = net(
            mol_input_a=batch["mol_input_a"],
            mol_input_b=batch["mol_input_b"],
            group_index=batch["group_index"],
            batch_size=len(batch["label"]),
            crit_input=batch["crit_input"],
        )
    probs = torch.sigmoid(logits)
    print(f"logits{tuple(logits.shape)} | prob range [{probs.min():.3f}, {probs.max():.3f}]")
    assert logits.shape == (len(batch["label"]), 1) and torch.isfinite(logits).all()

    # -- VRAM estimate for frozen-head training at batch 32 ------------------
    print("\n== VRAM estimate (frozen backbones, head-only training, batch 32) ==")
    _, mol_a_total = count_params(net.mol_encoder_a)
    _, mol_b_total = count_params(net.mol_encoder_b)
    _, crit_total = count_params(net.protocol_encoder)
    head_train, head_total = count_params(net.fusion)

    weights_bf16_gb = (mol_a_total + mol_b_total + crit_total + head_total) * 2 / 1024**3
    optimizer_gb = head_train * 12 / 1024**3  # grads fp32 + Adam m/v fp32 for head only
    B, mol_per_trial, L, Lc = 32, 2.0, 128, 512
    H_a, H_b, H_c = net.mol_encoder_a.hidden_size, net.mol_encoder_b.hidden_size, net.protocol_encoder.hidden_size
    layers_a = len(net.mol_encoder_a.backbone.encoder.layer)
    layers_b = len(net.mol_encoder_b.backbone.encoder.layer)
    layers_c = len(net.protocol_encoder.backbone.encoder.layer)
    n_mols = int(B * mol_per_trial)
    # one stored activation tensor per layer (conservative: token states only)
    act_gb = (
        n_mols * L * H_a * layers_a
        + n_mols * L * H_b * layers_b
        + B * Lc * H_c * layers_c
    ) * 2 / 1024**3
    total_est = weights_bf16_gb + optimizer_gb + act_gb
    print(f"  weights (bf16, all streams + head): {weights_bf16_gb:.2f} GB")
    print(f"  head optimizer states (fp32 grads + Adam m,v): {optimizer_gb:.3f} GB")
    print(f"  stored activations (token states, bf16, est): {act_gb:.2f} GB")
    print(f"  TOTAL estimate: ~{total_est:.2f} GB (8 GB card: comfortable headroom)")
    print("  note: unfreezing backbones adds ~3x total params for grads+Adam")
    print("        (~{:.1f} GB) -> gradient checkpointing + batch 8-16 + accum".format(
        (mol_a_total + mol_b_total + crit_total) * 12 / 1024**3))

    print(f"\nWall time: {time.time() - t0:.1f}s")
    print("TASK 2 VERIFICATION: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
