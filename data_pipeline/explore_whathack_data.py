from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "real_whathack_events.csv"

df = pd.read_csv(DATA_PATH)

# Timestamps are day/month/year
df["ts"] = pd.to_datetime(df["action_timestamp"], format="%d/%m/%Y %H:%M:%S")

print(f"Rows: {len(df)}")
print(f"Users: {df['user_id'].nunique()}")
print(f"Emails: {df['email_id'].nunique()}")
print(f"User-email pairs: {df.groupby(['user_id', 'email_id']).ngroups}")
print()

print("Event types:")
print(df["event_type"].value_counts().to_string())
print()

# One decision (Allow/Block) per user-email pair. 0 means "not a decision row".
decisions = df[df["email_decision"].astype(str) != "0"]
last_decision = (
    decisions.groupby(["user_id", "email_id"])
    .agg(decision=("email_decision", "last"), email_type=("email_type", "first"))
    .reset_index()
)

print("Email type vs user decision:")
print(pd.crosstab(last_decision["email_type"], last_decision["decision"], margins=True))