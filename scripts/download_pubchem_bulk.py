"""Download PubChem bulk files (synonyms + CID->SMILES) from the FTP mirror.

These let us resolve drug names offline when the REST API rate-limits/503s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

BULK_DIR = Path(r"D:\Models\pubchem_bulk")
FILES = (
    "CID-Synonym-filtered.gz",
    "CID-SMILES.gz",
)
BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/"


def download(name: str) -> None:
    dest = BULK_DIR / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    head = requests.head(BASE + name, timeout=30)
    total = int(head.headers.get("Content-Length", 0))
    if dest.exists() and total and dest.stat().st_size == total:
        print(f"skip {name} (already complete)", flush=True)
        return
    print(f"downloading {name} ({total / 1e6:.0f} MB) ...", flush=True)
    got = 0
    with requests.get(BASE + name, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got % (200 << 20) < (1 << 20):
                    print(f"  {name}: {got >> 20} MB", flush=True)
    tmp.replace(dest)
    print(f"done {name}", flush=True)


def main() -> int:
    BULK_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
