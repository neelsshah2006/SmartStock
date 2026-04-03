"""
SmartStock – Encoder Utilities
Provides safe label encoding with fallback for unseen categories.
"""

import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Mapping from raw string values to encoded ints (built from training data).
# If the encoder artifact doesn't exist yet (first run before training),
# these simple hard-coded maps ensure the app still works for demo data.
_FALLBACK_MAPS: dict[str, dict[str, int]] = {
    "StateHoliday": {"0": 0, "a": 1, "b": 2, "c": 3},
    "StoreType":    {"a": 0, "b": 1, "c": 2, "d": 3},
    "Assortment":   {"a": 0, "b": 1, "c": 2},
}


def encode_value(
    col: str,
    value: str,
    encoders: dict[str, LabelEncoder] | None,
) -> int:
    """
    Encode a single categorical value.

    Falls back to the hard-coded map if the trained encoder is not available
    or if the value is unseen (handles demo mode gracefully).
    """
    if encoders and col in encoders:
        le: LabelEncoder = encoders[col]
        if value in le.classes_:
            return int(le.transform([value])[0])
        else:
            logger.warning("Unseen category '%s' for column '%s' – using 0", value, col)
            return 0

    # Fallback
    fb = _FALLBACK_MAPS.get(col, {})
    if value in fb:
        return fb[value]
    logger.warning("No encoder for '%s=%s' – using 0", col, value)
    return 0
