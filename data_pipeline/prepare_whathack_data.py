from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "real_whathack_events.csv"
OUT_PATH = ROOT / "data" / "real_whathack_pairs.csv"

df = pd.read_csv(DATA_PATH)
df["ts"] = pd.to_datetime(df["action_timestamp"], format="%d/%m/%Y %H:%M:%S")
df["row"] = np.arange(len(df))  # original order, to break timestamp ties
df = df.sort_values(["user_id", "email_id", "ts", "row"])

# The user's FIRST decision (Allow/Block) on each email
decisions = df[df["email_decision"].astype(str) != "0"]
first = (
    decisions.groupby(["user_id", "email_id"])
    .agg(
        decision=("email_decision", "first"),
        decision_ts=("ts", "first"),
        decision_row=("row", "first"),
        email_type=("email_type", "first"),
    )
    .reset_index()
)

# Keep only events up to and including the decision, so no feature
# can contain anything that happened after the user decided.
m = df.merge(first, on=["user_id", "email_id"], suffixes=("", "_d"))
m = m[m["row"] <= m["decision_row"]].copy()
m["is_hover"] = m["event_type"] == "LinkHovering"
m["is_sender_gaze"] = m["event_detail_str"] == "FromText"

pairs = (
    m.groupby(["user_id", "email_id"])
    .agg(
        hover_count=("is_hover", "sum"),
        sender_checked=("is_sender_gaze", "max"),
        first_ts=("ts", "min"),
        decision_ts=("decision_ts", "first"),
        decision=("decision", "first"),
        email_type=("email_type", "first"),
    )
    .reset_index()
)
pairs["sender_checked"] = pairs["sender_checked"].astype(int)
pairs["decision_time_sec"] = (pairs["decision_ts"] - pairs["first_ts"]).dt.total_seconds()

# Only phishing-type emails get a label: Allow on these = fell for it
pairs = pairs[pairs["email_type"].isin(["masquerading", "unSafe"])].copy()
pairs["fell_for_it"] = (pairs["decision"] == "Allow").astype(int)

out = pairs[["user_id", "email_type", "hover_count", "sender_checked",
             "decision_time_sec", "fell_for_it"]]

print(f"Rows: {len(out)}  Users: {out['user_id'].nunique()}")
print(f"Fell for it: {out['fell_for_it'].sum()} ({out['fell_for_it'].mean():.1%})")
print()
print(out.groupby("fell_for_it")[["hover_count", "sender_checked", "decision_time_sec"]]
      .agg(["mean", "median"]).round(2))

out.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")