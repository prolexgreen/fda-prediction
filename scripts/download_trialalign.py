"""Download the Panacea TrialAlign corpus (linjc16/TrialAlign, trial_document config).

The FDA_MODELS_ROOT / HUB_CACHE env override is set by fda_predictor.utils.paths
on import, so this lands in D:\\Models\\hub\\datasets--linjc16--TrialAlign.
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download


def main() -> int:
    print("downloading linjc16/TrialAlign (trial_document) -> HF cache (D:\\Models) ...", flush=True)
    path = snapshot_download(
        repo_id="linjc16/TrialAlign",
        repo_type="dataset",
        allow_patterns=["trial_document/*"],
    )
    print(f"done: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
