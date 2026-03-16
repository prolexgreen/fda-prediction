import pandas as pd

cto = pd.read_parquet("data/processed/cto_human.parquet")
print(f"rows: {len(cto)}")
print(f"tickers: {cto['ticker'].notna().sum()}")
n_smiles = int((cto["n_smiles"] > 0).sum())
print(f"rows with SMILES: {n_smiles} ({n_smiles / len(cto):.0%})")
if "completion_date" in cto.columns:
    print(f"completion_date notna: {int(cto['completion_date'].notna().sum())}")
suffix = [c for c in cto.columns if c.endswith("_x") or c.endswith("_y")]
print(f"suffix collision cols remaining: {suffix}")
