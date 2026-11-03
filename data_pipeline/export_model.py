from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "behavioural_dataset_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
# Columns that must never be used as features (leakage / identifiers)
LEAKY_COLUMNS = ["user_id", "_profile_score_debug_only", "time_to_report_sec"]

df = pd.read_csv(DATA_PATH)
y = df["clicked_link"].values
X = df.drop(columns=LEAKY_COLUMNS + ["clicked_link"])
feature_names = list(X.columns)

# 64 / 16 / 20 split, stratified
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.2, stratify=y_tv, random_state=RANDOM_SEED
)

scaler = StandardScaler().fit(X_train)  # fit on training data only

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    early_stopping=True,
    random_state=RANDOM_SEED,
    max_iter=500,
).fit(scaler.transform(X_train), y_train)

# Choose the threshold on the VALIDATION set, never the test set
val_probs = model.predict_proba(scaler.transform(X_val))[:, 1]
grid = np.arange(0.10, 0.91, 0.05)
best_threshold = round(float(grid[int(np.argmax([f1_score(y_val, val_probs >= t) for t in grid]))]), 2)
# Report test performance as a check
test_probs = model.predict_proba(scaler.transform(X_test))[:, 1]
pred = test_probs >= best_threshold
print(f"Threshold chosen on validation: {best_threshold:.2f}")
print(f"Test accuracy:  {accuracy_score(y_test, pred):.3f}")
print(f"Test precision: {precision_score(y_test, pred):.3f}")
print(f"Test recall:    {recall_score(y_test, pred):.3f}")
print(f"Test F1:        {f1_score(y_test, pred):.3f}")

# Save everything another person needs to use the model
joblib.dump(model, MODEL_DIR / "behaviour_mlp.joblib")
joblib.dump(scaler, MODEL_DIR / "behaviour_scaler.joblib")
joblib.dump(
    {"threshold": best_threshold, "feature_names": feature_names},
    MODEL_DIR / "behaviour_config.joblib",
)
print(f"Saved model, scaler and config to {MODEL_DIR}")