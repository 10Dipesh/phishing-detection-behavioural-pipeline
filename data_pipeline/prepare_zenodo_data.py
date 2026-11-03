from pathlib import Path

import numpy as np
import pandas as pd

# Paths are built from this file's location, so the run folder doesn't matter
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "study_data_merged.xlsx"
OUT_PATH = ROOT / "data" / "real_zenodo_clean.csv"


def to_datetime(value):
    """Handles real datetimes, Excel serial numbers, and dd/mm/yyyy text."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    try:
        # Excel serial: days since 1899-12-30
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    except (ValueError, TypeError):
        return pd.to_datetime(value, dayfirst=True, errors="coerce")


df = pd.read_excel(DATA_PATH).replace("--", np.nan)

rows = []
for sim in ["sim1", "sim2"]:
    delivered = df[f"{sim}_SuccessfullyDeliveredEmail_TimeStamp"].map(to_datetime)
    clicked = df[f"{sim}_EmailLinkClicked_TimeStamp"].map(to_datetime)
    reported = df[f"{sim}_Phishing_Reported_On"].map(to_datetime)

    part = pd.DataFrame({
        "campaign": sim,
        "clicked": clicked.notna().astype(int),
        "time_to_click_sec": (clicked - delivered).dt.total_seconds(),
        "reported": reported.notna().astype(int),
        "time_to_report_sec": (reported - delivered).dt.total_seconds(),
    })
    part = part[delivered.notna()]  # only people who actually received the email
    rows.append(part)

out = pd.concat(rows, ignore_index=True)

# Some event timestamps are dated months BEFORE the email was delivered
# (stale records merged into the file). Drop them as inconsistent.
bad = (out.time_to_click_sec < 0) | (out.time_to_report_sec < 0)
print(f"Dropping {bad.sum()} rows with timestamps before delivery")
out = out[~bad].reset_index(drop=True)

print(f"Rows: {len(out)}")
print(out.groupby("campaign")[["clicked", "reported"]].agg(["sum", "mean"]).round(3))
print()
print("Median seconds from delivery:")
print(out.groupby("campaign")[["time_to_click_sec", "time_to_report_sec"]].median().round(0))
print()
print("Negative times (should be 0):",
      int((out.time_to_click_sec < 0).sum()), "click,",
      int((out.time_to_report_sec < 0).sum()), "report")

out.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")