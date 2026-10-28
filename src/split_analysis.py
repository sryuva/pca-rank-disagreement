"""Beyond-the-gap results split by whether stability selection can retain the signal.
PC4-5: a candidate rank (4 or 5) that retains the signal exists for both rules.
PC6  : stability selection (k <= 5) cannot retain it; only CV (k = 6) can."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
import pandas as pd
df = pd.read_csv(RES / "results.csv")
b = df[df.signal_pc >= 4].copy()
b["group"] = b.signal_pc.map(lambda j: "PC4-5" if j <= 5 else "PC6")
b["stability_retains_signal"] = b.k_stability >= b.signal_pc
out = b.groupby(["group", "separation", "n"]).agg(
    disagreement=("disagree", "mean"), mean_delta=("delta_mse", "mean"),
    stability_retains=("stability_retains_signal", "mean")).round(3)
out.to_csv(RES / "beyond_gap_split.csv")
s = b[(b.separation == "large") & (b.n >= 60)]
for g, x in list(s.groupby("group")) + [("all", s)]:
    print(f"{g:6s} runs={len(x):5d}  P(delta>0.5)={(x.delta_mse>0.5).mean():.4f}  "
          f"P(delta>0.9)={(x.delta_mse>0.9).mean():.4f}  mean delta={x.delta_mse.mean():.3f}")
