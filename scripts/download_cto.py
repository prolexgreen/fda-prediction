"""Download and build CTO human-label join table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.cto_dataset import (  # noqa: E402
    build_cto_human_frame,
    download_cto_raw,
    save_cto_processed,
)
from fda_predictor.utils.paths import ensure_dirs


def main() -> int:
    p = argparse.ArgumentParser(description="Download CTO configs and build joined parquet")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-ctgov", action="store_true")
    p.add_argument("--skip-pubchem", action="store_true")
    p.add_argument("--max-trials", type=int, default=None, help="limit CTO rows for dev builds")
    args = p.parse_args()
    ensure_dirs()

    if args.force_download:
        download_cto_raw(force=True)

    frame, report = build_cto_human_frame(
        fetch_ctgov=not args.skip_ctgov,
        resolve_smiles=not args.skip_pubchem,
        max_trials=args.max_trials,
    )
    path = save_cto_processed(frame)
    print(f"Saved {len(frame)} trials -> {path}")
    print(f"human_labels={report.n_human_labels} tickers={report.n_with_ticker}")
    print(f"ctgov_fetched={report.n_ctgov_fetched} with_smiles={report.n_with_smiles}")
    if report.pubchem:
        print(f"pubchem hit_rate={report.pubchem.hit_rate:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
