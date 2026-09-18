import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

CLEAN_DATA_PATH = "data/behavioural_dataset_clean.csv"
MODEL_PATH = "data/behaviour_encoder_mlp.joblib"
SCALER_PATH = "data/behaviour_encoder_scaler.joblib"

RANDOM_SEED = 42

# The 3 columns that must NEVER be used as training features:
#   user_id                    - just an identifier, not a signal
#   _profile_score_debug_only  - the hidden ground truth everything was
#                                 generated from - using it is direct leakage
#   time_to_report_sec         - only non-missing when clicked_link == 0,
#                                 so its missingness pattern alone reveals
#                                 the label - a subtler but equally real
#                                 leakage trap
LEAKY_COLUMNS = ["user_id", "_profile_score_debug_only", "time_to_report_sec"]
LABEL_COLUMN = "clicked_link"

# Step 1: Load data and split into features (X) and label (y)
def load_features_and_label(path: str):
    df = pd.read_csv(path)
 
    feature_cols = [
        col for col in df.columns
        if col not in LEAKY_COLUMNS and col != LABEL_COLUMN
    ]
 
    X = df[feature_cols]
    y = df[LABEL_COLUMN]
 
    print(f"Loaded {len(df)} rows")
    print(f"Training features ({len(feature_cols)}): {feature_cols}")
    print(f"Excluded as leakage: {LEAKY_COLUMNS}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
 
    return X, y

# Step 2: Train/test split and feature scaling
def prepare_train_val_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=RANDOM_SEED, stratify=y_temp
    )
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
 
    print(f"Train set: {len(X_train)} rows, click rate {y_train.mean():.3f}")
    print(f"Validation set: {len(X_val)} rows, click rate {y_val.mean():.3f}")
    print(f"Test set: {len(X_test)} rows, click rate {y_test.mean():.3f}")
 
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

# Step 3: Train the MLP
def train_mlp(X_train, y_train) -> MLPClassifier:
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
 
    mlp.fit(X_train, y_train)
 
    print(f"Training complete after {mlp.n_iter_} iterations")
    print(f"Stopped early: {mlp.n_iter_ < 500}")
 
    return mlp

# # Step 4: Evaluate on the test set
# def evaluate_model(model: MLPClassifier, X_test, y_test) -> dict:
#     y_pred = model.predict(X_test)
 
#     metrics = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred),
#         "recall": recall_score(y_test, y_pred),
#         "f1": f1_score(y_test, y_pred),
#     }
 
#     print("Test set performance (behavioural-only baseline):")
#     for name, value in metrics.items():
#         print(f"  {name.capitalize()}: {value:.3f}")
 
#     cm = confusion_matrix(y_test, y_pred)
#     print()
#     print("Confusion matrix:")
#     print(f"                Predicted: No-click   Predicted: Click")
#     print(f"Actual: No-click      {cm[0][0]:>6}              {cm[0][1]:>6}")
#     print(f"Actual: Click         {cm[1][0]:>6}              {cm[1][1]:>6}")

    #Tune the classification threshold on the VALIDATION set
def find_best_threshold(model: MLPClassifier, X_val, y_val) -> float:
    probs = model.predict_proba(X_val)[:, 1]
 
    best_f1 = -1
    best_threshold = 0.5
    print("Threshold search on validation set:")
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            marker = "  <- best so far"
        print(f"  threshold={t:.2f}  F1={f1:.3f}{marker}")
 
    print(f"Chosen threshold: {best_threshold:.2f}")
    return best_threshold


 #Final evaluation on the TEST set, using the tuned threshold
def evaluate_model(model: MLPClassifier, X_test, y_test, threshold: float = 0.5) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
 
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
 
    print(f"Test set performance (behavioural-only baseline, threshold={threshold:.2f}):")
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.3f}")
 
    cm = confusion_matrix(y_test, y_pred)
    print()
    print("Confusion matrix:")
    print(f"                Predicted: No-click   Predicted: Click")
    print(f"Actual: No-click      {cm[0][0]:>6}              {cm[0][1]:>6}")
    print(f"Actual: Click         {cm[1][0]:>6}              {cm[1][1]:>6}")
 
    return metrics
 
if __name__ == "__main__":
    X, y = load_features_and_label(CLEAN_DATA_PATH)
    print()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_train_val_test(X, y)
    print()
    model = train_mlp(X_train, y_train)
    print()
    print("--- BASELINE (default 0.5 threshold) ---")
    baseline_metrics = evaluate_model(model, X_test, y_test, threshold=0.5)
    print()
    print("--- TUNING THRESHOLD ---")
    best_threshold = find_best_threshold(model, X_val, y_val)
    print()
    print("--- FINAL (tuned threshold) ---")
    tuned_metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)

 

 