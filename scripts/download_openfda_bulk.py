"""Download the openFDA Drugs@FDA bulk JSON dump for sponsor features."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.utils.paths import OPENFDA_BULK_DIR, ensure_dirs  # noqa: E402

DOWNLOAD_MANIFEST = "https://api.fda.gov/download.json"


def find_drugsfda_url() -> str:
    r = requests.get(DOWNLOAD_MANIFEST, timeout=60)
    r.raise_for_status()
    payload = r.json()
    parts = payload["results"]["drug"]["drugsfda"]["partitions"]
    # Normally a single file; loop defensively.
    return [p["file"] for p in parts]


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            n = 0
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                n += len(chunk)
                if n % (50 << 20) < (1 << 20):
                    print(f"  {n >> 20} MB ...", flush=True)
    tmp.replace(dest)


def main() -> int:
    ensure_dirs()
    OPENFDA_BULK_DIR.mkdir(parents=True, exist_ok=True)
    urls = find_drugsfda_url()
    print(f"drugsfda partitions: {len(urls)}", flush=True)
    for url in urls:
        dest = OPENFDA_BULK_DIR / Path(url).name
        if dest.exists():
            print(f"skip {dest.name} (exists)", flush=True)
        else:
            print(f"downloading {url}", flush=True)
            download(url, dest)
        # unzip in place
        json_dest = OPENFDA_BULK_DIR / dest.stem.replace(".json", "") / "_unpacked"
        unpacked = OPENFDA_BULK_DIR / dest.stem
        if unpacked.exists():
            print(f"skip unpack {unpacked.name}", flush=True)
        else:
            print(f"unpacking {dest.name} ...", flush=True)
            with zipfile.ZipFile(dest) as z:
                z.extractall(OPENFDA_BULK_DIR)
            unpacked = OPENFDA_BULK_DIR / dest.stem
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
