from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_model = joblib.load(MODEL_DIR / "behaviour_mlp.joblib")
_scaler = joblib.load(MODEL_DIR / "behaviour_scaler.joblib")
_config = joblib.load(MODEL_DIR / "behaviour_config.joblib")

FEATURE_NAMES = _config["feature_names"]
THRESHOLD = _config["threshold"]


def predict_susceptibility(features: dict) -> dict:
    """
    features: one user's behavioural features, using the names in FEATURE_NAMES.
    Returns the click probability and a 0/1 label (1 = likely to click).
    """
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([features])[FEATURE_NAMES]
    probability = float(_model.predict_proba(_scaler.transform(row))[0, 1])
    return {
        "click_probability": round(probability, 4),
        "likely_to_click": int(probability >= THRESHOLD),
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    example = {
        "time_to_first_click_sec": 12.0,
        "hover_time_sec": 1.5,
        "hover_count": 1,
        "mean_mouse_speed_px_s": 400.0,
        "slow_movement_ratio": 0.2,
        "reopen_count": 0,
        "sender_checked": 0,
    }
    print(predict_susceptibility(example))