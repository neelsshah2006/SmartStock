"""
SmartStock – Input Validators
Extra validation logic beyond Pydantic's field-level checks.
"""

from datetime import date as Date
from app.schemas.schemas import SaleRecordInput


def validate_sale_record(record: SaleRecordInput) -> list[str]:
    """
    Returns a list of warning strings (empty == all clear).
    These are soft warnings logged/returned to the caller, not hard errors.
    """
    warnings: list[str] = []

    if record.competition_distance > 100_000:
        warnings.append(
            f"competition_distance={record.competition_distance} looks unusually large."
        )
    if record.sale_date > Date.today():
        warnings.append(
            "sale_date is in the future – this record will be used as a prediction input."
        )

    return warnings
