"""
SmartStock – Forecasting Service
Loads the XGBRegressor and exposes a predict() function.
"""

import logging
import os
import numpy as np
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_model = None


def _load_model():
    global _model
    if _model is None:
        path = os.path.join(_MODELS_DIR, "forecasting_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Forecasting model not found at {path}. "
                "Please run `python train_models.py` first."
            )
        _model = joblib.load(path)
        logger.info("Forecasting model loaded from %s", path)
    return _model


def predict_sales(feature_df: pd.DataFrame) -> float:
    """
    Run the XGBRegressor on the pre-processed feature DataFrame.

    Returns the predicted sales value rounded up to the nearest 10.
    Sales are clamped to a minimum of 0 — negative predictions are not
    physically meaningful and indicate an out-of-distribution input.
    """
    model = _load_model()
    raw   = float(model.predict(feature_df)[0])

    logger.info("Raw prediction: %.2f", raw)

    # Clamp: sales cannot be negative
    raw = max(raw, 0.0)

    # Round up to nearest 10 (matches training post-processing)
    return float(np.ceil(raw / 10) * 10) if raw > 0 else 0.0