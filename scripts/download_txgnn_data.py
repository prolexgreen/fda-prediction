"""Download TxGNN knowledge-graph CSVs from Harvard Dataverse + ckpt."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "txgnn"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "kg.csv": "https://dataverse.harvard.edu/api/access/datafile/7144484",
    "node.csv": "https://dataverse.harvard.edu/api/access/datafile/7144482",
    "edges.csv": "https://dataverse.harvard.edu/api/access/datafile/7144483",
}
CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj&export=download"
)
CKPT_DIR = Path(r"D:\Models\txgnn")


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest.name} (exists)", flush=True)
        return
    print(f"downloading {dest.name} ...", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            n = 0
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                n += len(chunk)
                if n % (100 << 20) < (1 << 20):
                    print(f"  {dest.name}: {n >> 20} MB", flush=True)
    print(f"done {dest.name} ({dest.stat().st_size >> 20} MB)", flush=True)


def main() -> int:
    for name, url in FILES.items():
        download(url, OUT / name)

    # Checkpoint from Google Drive (can rate-limit; handle large-file token page).
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / "model_ckpt"
    if not ckpt.exists():
        ckpt.mkdir(parents=True, exist_ok=True)
    print(f"checking Google Drive checkpoint -> {ckpt}", flush=True)
    try:
        session = requests.Session()
        resp = session.get(CHECKPOINT_URL, stream=True, timeout=120)
        # Google Drive sometimes needs a confirm token for larger files.
        token = None
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                token = v
        if token:
            resp = session.get(CHECKPOINT_URL + f"&confirm={token}", stream=True, timeout=120)
        resp.raise_for_status()
        dest = ckpt / "model.pt"
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"checkpoint: {dest.stat().st_size >> 20} MB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"CHECKPOINT FAILED: {e!r}. will fall back to PPMI/SVD spectral embeddings.", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
