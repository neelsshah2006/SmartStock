"""
SmartStock – Halt Decision Service
Loads the XGBClassifier and produces a halt decision with an explanation.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
print(_MODELS_DIR)
_model = None


def _load_model():
    global _model
    if _model is None:
        path = os.path.join(_MODELS_DIR, "halt_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Halt model not found at {path}. "
                "Please run `python train_models.py` first."
            )
        _model = joblib.load(path)
        logger.info("Halt model loaded from %s", path)
    return _model


def predict_halt(
    feature_df: pd.DataFrame,
    avg_7:      float,
    predicted:  float,
    std_7:      float,
    volatility: float,
) -> tuple[str, str, float]:
    """
    Returns (decision, reason, confidence_score).

    decision   : "Automated" | "Human Review"
    reason     : plain-English explanation
    confidence : probability of the predicted class (0-1)
    """
    model   = _load_model()
    label   = int(model.predict(feature_df)[0])
    prob    = float(model.predict_proba(feature_df)[0][label])

    decision = "Human Review" if label == 1 else "Automated"
    reason   = _build_reason(
        label      = label,
        predicted  = predicted,
        avg_7      = avg_7,
        std_7      = std_7,
        volatility = volatility,
        confidence = prob,
    )
    return decision, reason, prob


# ── Reason generator ─────────────────────────────────────────────────────────

def _build_reason(
    label:      int,
    predicted:  float,
    avg_7:      float,
    std_7:      float,
    volatility: float,
    confidence: float,
) -> str:
    pct_diff = abs(predicted - avg_7) / (avg_7 + 1) * 100

    if label == 0:
        if volatility < 0.15:
            return (
                f"Prediction ({predicted:,.0f}) is within the stable 7-day range "
                f"(avg {avg_7:,.0f}, σ {std_7:,.0f}). Low volatility – safe to automate."
            )
        return (
            f"Prediction ({predicted:,.0f}) deviates {pct_diff:.1f}% from 7-day avg "
            f"but the model is confident ({confidence:.0%}). Proceeding automatically."
        )
    else:
        if pct_diff > 30:
            return (
                f"Prediction ({predicted:,.0f}) is {pct_diff:.1f}% above/below the "
                f"7-day average ({avg_7:,.0f}). Unusual demand pattern detected – "
                "escalating for human review."
            )
        if volatility > 0.4:
            return (
                f"High demand volatility (ratio {volatility:.2f}) detected for this "
                "store. Forecast may be unreliable – sending for human review."
            )
        return (
            "Abnormal feature pattern detected by the halt model. "
            f"Confidence of automation is low ({confidence:.0%}) – human review recommended."
        )
