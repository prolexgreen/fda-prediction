"""Print a compact comparison table of all backtest_v* runs."""
import json, glob, os, sys

ROOT = r"C:\Users\prole\OneDrive\Desktop\Code\FDA"
os.chdir(ROOT)

rows = []
for run in sorted(glob.glob("artifacts/runs/backtest_v*")):
    if not os.path.isdir(run):
        continue
    p = run + "/metrics.json"
    if not os.path.exists(p):
        continue
    try:
        m = json.load(open(p))
        model = m.get("model", {}).get("test", {})
        tuned = m.get("per_phase_tuned", {}).get("test", {}).get("III", {})
        rows.append(
            (
                os.path.basename(run),
                model.get("auprc", 0.0),
                model.get("precision", 0.0),
                model.get("recall", 0.0),
                model.get("threshold", 0.0),
                model.get("tp", 0),
                model.get("fp", 0),
                model.get("fn", 0),
                tuned.get("precision", 0.0),
                tuned.get("recall", 0.0),
            )
        )
    except Exception as e:
        print(run, "err", e, file=sys.stderr)

print(f"{'run':28s} | {'AUPRC':>6} {'P':>6} {'R':>6} {'thr':>6} | {'TP':>4} {'FP':>4} {'FN':>4} | tunedP3 {'P':>6} {'R':>6}")
for r in rows:
    print(f"{r[0]:28s} | {r[1]:6.4f} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6.3f} | {r[5]:>4} {r[6]:>4} {r[7]:>4} | {r[8]:6.3f} {r[9]:6.3f}")
