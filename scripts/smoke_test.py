"""Task 0 smoke test: verifies CUDA, bf16, library imports, and local loading
of all three transformer checkpoints from D:\Models (including a MoLFormer
forward pass through its Hub-hosted remote code), then prints the hidden
sizes and the resulting fusion input dimension."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.utils.paths import HUB_CACHE  # noqa: E402  (env setup must run first)


def main() -> int:
    t0 = time.time()
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, info: str = "") -> None:
        results.append((name, ok, info))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {info}" if info else ""))

    # --- 1. Environment -------------------------------------------------
    import os

    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    record(
        "hf_cache_points_to_D_Models",
        str(HUB_CACHE).lower().startswith("d:\\models"),
        str(HUB_CACHE),
    )

    # --- 2. Core imports -------------------------------------------------
    try:
        import torch
        import transformers
        import sklearn
        import rdkit
        from rdkit import Chem
        import tdc
        import einops
        import yaml
        import requests
        from importlib.metadata import version as _pkg_version

        record(
            "core_imports",
            True,
            f"torch {torch.__version__} | transformers {transformers.__version__} "
            f"| sklearn {sklearn.__version__} | rdkit {rdkit.__version__} "
            f"| tdc {_pkg_version('PyTDC')}",
        )
    except Exception as e:
        record("core_imports", False, repr(e))
        return finish(results, t0)

    # --- 3. CUDA / bf16 ---------------------------------------------------
    cuda_ok = torch.cuda.is_available()
    dev = torch.device("cuda:0") if cuda_ok else torch.device("cpu")
    bf16_ok = cuda_ok and torch.cuda.is_bf16_supported()
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        cap = f"{props.major}.{props.minor}"
        vram_gb = props.total_memory / 1024**3
        record(
            "cuda_available",
            True,
            f"{props.name} | CC {cap} | {vram_gb:.1f} GB | bf16={'yes' if bf16_ok else 'no'}",
        )
    else:
        record("cuda_available", False, "torch.cuda.is_available() is False")

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    ENCODERS = {
        "chemberta": dict(
            name="DeepChem/ChemBERTa-77M-MTR",
            revision="66b895cab8adebea0cb59a8effa66b2020f204ca",
            trust_remote_code=False,
        ),
        "molformer": dict(
            name="ibm-research/MoLFormer-XL-both-10pct",
            revision="361063d0ad524ef77cf39b08469f6be770dc550f",
            trust_remote_code=True,
        ),
        "clinicalbert": dict(
            name="emilyalsentzer/Bio_ClinicalBERT",
            revision="d5892b39a4adaed74b92212a44081509db72f87b",
            trust_remote_code=False,
        ),
    }
    ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"

    hidden_sizes: dict[str, int] = {}

    # --- 4. Checkpoint download -> verify on disk -> load LOCALLY --------
    for key, spec in ENCODERS.items():
        try:
            local_dir = snapshot_download(
                repo_id=spec["name"], revision=spec["revision"]
            )
            on_disk = Path(local_dir).is_relative_to(Path(os.environ["HF_HUB_CACHE"]))
            size_mb = sum(f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()) / 1024**2

            tok = AutoTokenizer.from_pretrained(
                local_dir, local_files_only=True, trust_remote_code=spec["trust_remote_code"]
            )
            cfg = AutoConfig.from_pretrained(
                local_dir, local_files_only=True, trust_remote_code=spec["trust_remote_code"]
            )
            model = AutoModel.from_pretrained(
                local_dir, local_files_only=True, trust_remote_code=spec["trust_remote_code"]
            )
            hidden_sizes[key] = cfg.hidden_size

            cache_label = "inside D:\\Models" if on_disk else "OUTSIDE expected cache!"
            info = (
                f"hidden_size={cfg.hidden_size} | {size_mb:.0f} MB on disk ({cache_label})"
            )

            if key == "molformer":
                inputs = tok(ASPIRIN_SMILES, return_tensors="pt").to(dev)
                model = model.to(dev).eval()
                with torch.no_grad():
                    out = model(**inputs)
                hs = out.last_hidden_state
                assert hs.shape[:2] == tuple(inputs["input_ids"].shape), hs.shape
                info += f" | forward on {dev}: last_hidden_state{tuple(hs.shape)}"
            elif key == "chemberta":
                canon = Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN_SMILES))
                n = len(tok(canon)["input_ids"])
                info += f" | rdkit canonical roundtrip ok ({n} tokens)"

            record(f"checkpoint:{key}", on_disk, info)
        except Exception as e:
            traceback.print_exc()
            record(f"checkpoint:{key}", False, repr(e))

    # --- 5. Fusion dimension ----------------------------------------------
    if len(hidden_sizes) == 3:
        fusion_dim = sum(hidden_sizes.values())
        expected = hidden_sizes["chemberta"] + hidden_sizes["molformer"] + hidden_sizes["clinicalbert"]
        record(
            "fusion_input_dim",
            fusion_dim == expected,
            f"{hidden_sizes['chemberta']} + {hidden_sizes['molformer']} + "
            f"{hidden_sizes['clinicalbert']} = {fusion_dim}",
        )

    return finish(results, t0)


def finish(results: list[tuple[str, bool, str]], t0: float) -> int:
    wall = time.time() - t0
    failed = [n for n, ok, _ in results if not ok]
    print("\n==== SMOKE TEST SUMMARY ====")
    for n, ok, info in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"Wall time: {wall:.1f}s")
    print("RESULT:", "ALL CHECKS PASSED" if not failed else f"FAILED: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
