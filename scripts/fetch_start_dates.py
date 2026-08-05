"""STUB - not yet executed (per approved plan).

Enrichment hook for stronger temporal verification: TOP's csv has no date
column, so Task 1 guards chronology via monotonic NCT numbers. This script,
when run later, batch-fetches real trial start dates from the
ClinicalTrials.gov API v2 so the guard can be upgraded to explicit dates.

Intended usage (DO NOT run during Task 1):
    uv run python scripts/fetch_start_dates.py --input data/raw/splits.parquet \
        --output data/raw/nct_start_dates.csv [--limit 500]

Design notes:
- Batch endpoint: GET https://clinicaltrials.gov/api/v2/studies
    params: filter.ids=<comma-separated up to ~50 NCT IDs>,
            fields=NCTId,StartDate, pageSize=1000, pageToken=<from response>
- Follow nextPageToken until exhausted; be polite (sleep between pages).
- Parse protocolSection.statusModule.startDateStruct.date (ISO-ish string);
  write nctid,start_date CSV.
- Rate limits are generous but not unlimited; cache results and resume.
"""

from __future__ import annotations

import argparse
import sys

API_BASE = "https://clinicaltrials.gov/api/v2/studies"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="parquet/csv with an 'nctid' column")
    parser.add_argument("--output", required=True, help="CSV path for nctid,start_date pairs")
    parser.add_argument("--limit", type=int, default=None, help="fetch only first N ids (testing)")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    print(
        "[STUB] ClinicalTrials.gov start-date enrichment is intentionally not "
        "implemented in Task 1. The NCT-monotonicity guard covers chronology for "
        "now; wire this script in before Phase 2."
    )
    print(f"Would read {args.input}, fetch batches of {args.batch_size} from {API_BASE}")
    if args.limit:
        print(f"(test mode: only {args.limit} ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
