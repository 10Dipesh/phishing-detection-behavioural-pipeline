"""
Do real response times look lognormal? Compares skewness before and after a
log transform and plots both views. Uses the cleaned Zenodo simulation data.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "real_zenodo_clean.csv")

COLUMNS = {
    "time_to_click_sec": "Time to click (from delivery)",
    "time_to_report_sec": "Time to report (from delivery)",
}

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
print(f"{'measure':32s} {'campaign':8s} {'n':>5s} {'skew raw':>9s} {'skew log':>9s}")

for row, (col, label) in enumerate(COLUMNS.items()):
    for campaign in ["sim1", "sim2"]:
        values = df.loc[df["campaign"] == campaign, col].dropna()
        values = values[values > 0]  # log needs positive values
        print(f"{label:32s} {campaign:8s} {len(values):5d} "
              f"{skew(values):9.2f} {skew(np.log(values)):9.2f}")

        cap = values.quantile(0.99)  # trim the extreme tail so the raw plot is readable
        axes[row, 0].hist(values[values <= cap] / 60, bins=40, alpha=0.5, label=campaign)
        axes[row, 1].hist(np.log10(values), bins=40, alpha=0.5, label=campaign)

    axes[row, 0].set_title(f"{label}: raw minutes (99th percentile cap)")
    axes[row, 1].set_title(f"{label}: log10(seconds)")
    axes[row, 0].set_xlabel("Minutes")
    axes[row, 1].set_xlabel("log10(seconds)")
    axes[row, 0].legend()

fig.tight_layout()
fig.savefig(RESULTS_DIR / "time_shape_check.png", dpi=200)
print(f"\nSaved plot to {RESULTS_DIR / 'time_shape_check.png'}")