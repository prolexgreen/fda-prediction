"""Download ChEMBL SQLite dump (latest release) into D:\\Models\\chembl."""

from __future__ import annotations

import re
import sys
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_predictor.utils.paths import CHEMBL_DIR, ensure_dirs  # noqa: E402

FTP_LATEST = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"


def main() -> int:
    ensure_dirs()
    CHEMBL_DIR.mkdir(parents=True, exist_ok=True)

    listing = requests.get(FTP_LATEST, timeout=60).text
    match = re.findall(r'(chembl_\d+_sqlite\.tar\.gz)', listing)
    if not match:
        print("no sqlite tarball found in listing", file=sys.stderr)
        return 1
    name = sorted(set(match))[-1]
    url = FTP_LATEST + name
    dest = CHEMBL_DIR / name

    if not dest.exists():
        head = requests.head(url, timeout=60)
        total = int(head.headers.get("Content-Length", 0))
        print(f"downloading {name} ({total / 1e9:.1f} GB) ...", flush=True)
        tmp = dest.with_suffix(".part")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                n = 0
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk)
                    n += len(chunk)
                    if n % (500 << 20) < (1 << 22):
                        print(f"  {n >> 30} GiB", flush=True)
        tmp.replace(dest)
    else:
        print(f"skip {name} (exists)", flush=True)

    # Extract SQLite db file
    with tarfile.open(dest, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".db")]
        if not members:
            print("no .db member found in tarball", file=sys.stderr)
            return 1
        member = members[0]
        out = CHEMBL_DIR / member.name
        if out.exists() and out.stat().st_size > 0:
            print(f"skip extract {out.name} (exists)", flush=True)
        else:
            print(f"extracting {member.name} ...", flush=True)
            tar.extract(member, CHEMBL_DIR)
            # tar may place it in a subdir like chembl_37/chembl_37_sqlite/xxx.db
            inner = CHEMBL_DIR / member.name
            if inner.parent != CHEMBL_DIR and inner.exists():
                inner.replace(out)
    db_files = list(CHEMBL_DIR.rglob("*.db"))
    print(f"chembl db: {[str(p) for p in db_files]}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
