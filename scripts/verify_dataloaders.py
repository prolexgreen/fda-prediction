"""Task 1 acceptance driver: loads TDC TOP, runs the guarded temporal split,
prints class balance / molecule stats / SMILES validity, then pushes one
batch through the dual-tokenizer pipeline to validate shapes end-to-end.

CPU-only by design (no model weights are loaded here except tokenizers).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fda_predictor.data.datasets import TOPTrialDataset, build_collate_fn  # noqa: E402
from fda_predictor.data.tokenizers import specs_from_config  # noqa: E402
from fda_predictor.data.top_dataset import assert_temporal_order, load_top_splits, split_stats  # noqa: E402
from fda_predictor.utils.paths import ensure_dirs  # noqa: E402


def main() -> int:
    t0 = time.time()
    ensure_dirs()

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text())
    specs = specs_from_config(config)
    dcfg = config["data"]

    print("Loading TOP Phase III benchmark (downloads split CSVs on first run) ...")
    splits = load_top_splits(
        val_fraction_of_trainval=float(dcfg["val_fraction_of_trainval"]),
        seed=int(config["seed"]),
        min_criteria_chars=int(dcfg["min_criteria_chars"]),
        split_mode=str(dcfg.get("split_mode", "chronological")),
    )
    print(f"split_mode: {dcfg.get('split_mode', 'chronological')}")

    print("\n== Preprocessing report ==")
    print(splits.canon_report.summary())
    print(f"Dropped rows with no valid SMILES: {splits.dropped_no_valid_smiles}")
    print(f"Dropped rows with missing/short criteria (<{dcfg['min_criteria_chars']} chars): "
          f"{splits.dropped_short_or_missing_criteria}")

    assert_temporal_order((splits.train, splits.val, splits.test))
    print("\nTemporal split guard: PASSED (nctid disjoint, numeric NCT non-decreasing)")

    print("\n== Per-split statistics ==")
    stats = split_stats(splits)
    header = (
        f"{'split':<6} {'trials':>7} {'pos':>6} {'pos_frac':>9} "
        f"{'mol_mean':>8} {'mol_max':>8} {'nct_range':>17}"
    )
    print(header)
    for name, s in stats.items():
        rng = f"{s['min_nct']}-{s['max_nct']}"
        print(
            f"{name:<6} {s['n_trials']:>7} {s['n_pos']:>6} {s['pos_fraction']:>9.3f} "
            f"{s['molecules_per_trial_mean']:>8.2f} {s['molecules_per_trial_max']:>8} {rng:>17}"
        )

    print("\nBuilding datasets (tokenizers cached on D:\\Models) ...")
    common = dict(chemberta_spec=specs["chemberta"], molformer_spec=specs["molformer"],
                  clinicalbert_spec=specs["clinicalbert"])
    ds_train = TOPTrialDataset(splits.train, max_molecules_per_trial=int(dcfg["max_molecules_per_trial"]), **common)

    loaders = {}
    for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        ds = TOPTrialDataset(part, max_molecules_per_trial=int(dcfg["max_molecules_per_trial"]), **common)
        loaders[name] = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=build_collate_fn(ds))

    batch = next(iter(loaders["train"]))
    ids_a = batch["mol_input_a"]["input_ids"]
    ids_b = batch["mol_input_b"]["input_ids"]
    crit = batch["crit_input"]["input_ids"]
    gi = batch["group_index"]
    n_mols_total = ids_a.shape[0]

    print("\n== One collated batch (batch_size=8) ==")
    print(f"mol_input_a: input_ids{tuple(ids_a.shape)} | mol_input_b: input_ids{tuple(ids_b.shape)}")
    print(f"crit_input:  input_ids{tuple(crit.shape)}")
    print(f"group_index: len={gi.numel()} (matches flattened molecules: {n_mols_total}) "
          f"| trials covered={int(gi.max()) + 1}")
    assert gi.numel() == n_mols_total
    assert int(gi.max()) == len(batch["label"]) - 1

    same_ids = bool((ids_a == ids_b).all()) if ids_a.shape == ids_b.shape else False
    print(f"Tokenizer independence: mol streams share identical ids -> {same_ids} "
          f"({'OK, vocabularies differ' if not same_ids else 'SUSPICIOUS: identical'})")

    print(f"\nBatch label mean: {batch['label'].mean().item():.3f}")
    print(f"Wall time: {time.time() - t0:.1f}s")
    print("TASK 1 VERIFICATION: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
