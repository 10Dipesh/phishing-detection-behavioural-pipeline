"""
RQ2: how does the behavioural model degrade when features are missing at
prediction time? Uses the EXPORTED model, so results describe the model the
dashboard will actually use. Missing features are filled with the training
median (the simplest realistic fallback).
"""
import itertools
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
LEAKY_COLUMNS = ["user_id", "_profile_score_debug_only", "time_to_report_sec"]

model = joblib.load(ROOT / "models" / "behaviour_mlp.joblib")
scaler = joblib.load(ROOT / "models" / "behaviour_scaler.joblib")
config = joblib.load(ROOT / "models" / "behaviour_config.joblib")
threshold = config["threshold"]
names = config["feature_names"]

# Recreate exactly the same split used for training
df = pd.read_csv(ROOT / "data" / "behavioural_dataset_clean.csv")
y = df["clicked_link"].values
X = df.drop(columns=LEAKY_COLUMNS + ["clicked_link"])[names]
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
X_train, _, _, _ = train_test_split(
    X_tv, y_tv, test_size=0.2, stratify=y_tv, random_state=RANDOM_SEED
)
train_median = X_train.median()


def evaluate(missing):
    """F1 and AUC on the test set with the listed features replaced by the training median."""
    data = X_test.copy()
    for col in missing:
        data[col] = train_median[col]
    probs = model.predict_proba(scaler.transform(data))[:, 1]
    return f1_score(y_test, probs >= threshold), roc_auc_score(y_test, probs)


full_f1, full_auc = evaluate([])
print(f"All features present: F1 {full_f1:.3f}  AUC {full_auc:.3f}")
print(f"Reference: always predicting 'click' gives F1 {f1_score(y_test, [1] * len(y_test)):.3f}")
print()

# 1) One feature missing at a time
single_rows = []
for col in names:
    f1, auc = evaluate([col])
    single_rows.append({"missing": col, "f1": f1, "auc": auc,
                        "f1_change": f1 - full_f1, "auc_change": auc - full_auc})
single = pd.DataFrame(single_rows).sort_values("auc_change")
print("One feature missing (sorted by AUC drop):")
print(single.round(3).to_string(index=False))
print()

# 2) k features missing, over every possible combination
summary_rows = []
for k in range(0, len(names) + 1):
    results = [evaluate(list(c)) for c in itertools.combinations(names, k)]
    f1s = [r[0] for r in results]
    aucs = [r[1] for r in results]
    summary_rows.append({
        "n_missing": k, "combinations": len(results),
        "f1_mean": sum(f1s) / len(f1s), "f1_min": min(f1s),
        "auc_mean": sum(aucs) / len(aucs), "auc_min": min(aucs), "auc_max": max(aucs),
    })
summary = pd.DataFrame(summary_rows)
print("Any k features missing (all combinations):")
print(summary.round(3).to_string(index=False))

single.to_csv(RESULTS_DIR / "rq2_single_feature_missing.csv", index=False)
summary.to_csv(RESULTS_DIR / "rq2_k_features_missing.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(summary["n_missing"], summary["auc_mean"], marker="o", label="Mean AUC")
ax.fill_between(summary["n_missing"], summary["auc_min"], summary["auc_max"], alpha=0.2, label="Min to max")
ax.axhline(0.5, linestyle="--", color="grey", label="Chance (0.5)")
ax.set_xlabel("Number of behavioural features missing")
ax.set_ylabel("ROC-AUC on test set")
ax.set_title("Behavioural model under missing features")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "rq2_missing_features.png", dpi=200)
print(f"\nSaved tables and plot to {RESULTS_DIR}")