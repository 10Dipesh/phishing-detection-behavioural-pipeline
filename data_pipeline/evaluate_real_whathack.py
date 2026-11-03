from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
real = pd.read_csv("data/real_whathack_pairs.csv")
synth = pd.read_csv("data/behavioural_dataset_clean.csv")

REAL_FEATURES = ["hover_count", "sender_checked", "decision_time_sec"]
X = real[REAL_FEATURES].values
y = real["fell_for_it"].values
groups = real["user_id"].values

# ---- A: real-only baseline, split by USER so nobody is in train and test ----
aucs = []
for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    ).fit(X[train_idx], y[train_idx])
    probs = model.predict_proba(X[test_idx])[:, 1]
    aucs.append(roc_auc_score(y[test_idx], probs))

print("A) Real-only baseline (5-fold, split by user)")
print("   AUC per fold:", np.round(aucs, 2))
print(f"   AUC mean {np.mean(aucs):.3f}  sd {np.std(aucs):.3f}")
print()

# ---- B: train on synthetic (same 3 features), test on all real pairs ----
synth_X = synth[["hover_count", "sender_checked", "time_to_first_click_sec"]].values
synth_y = synth["clicked_link"].values

scaler = StandardScaler().fit(synth_X)
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    early_stopping=True,
    random_state=42,
    max_iter=500,
).fit(scaler.transform(synth_X), synth_y)

probs = mlp.predict_proba(scaler.transform(X))[:, 1]
pred = probs >= 0.35  # the threshold tuned on synthetic validation data

print("B) Synthetic-trained model tested on real data")
print(f"   AUC {roc_auc_score(y, probs):.3f}  (0.5 = chance)")
print(f"   F1 {f1_score(y, pred):.3f}  precision {precision_score(y, pred):.3f}"
      f"  recall {recall_score(y, pred):.3f}")