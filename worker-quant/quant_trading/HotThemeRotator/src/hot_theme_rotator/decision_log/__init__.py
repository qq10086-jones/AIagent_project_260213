"""Decision log subsystem — §8.6 mandatory feedback log persistence.

Single `PredictionRecord` schema covering attribution and opportunity prediction
paths. JSONL writer persists to `reports/predictions/`. See ADR-0003 for design
rationale.
"""
from hot_theme_rotator.decision_log.schema import (
    ALLOWED_OUTCOME_STATUSES,
    ALLOWED_SCORE_STATUSES,
    OutcomeRecord,
    OutcomeRecordValidationError,
    PredictionRecord,
    PredictionRecordValidationError,
    compute_outcome_id,
    compute_prediction_id,
)

__all__ = [
    "ALLOWED_OUTCOME_STATUSES",
    "ALLOWED_SCORE_STATUSES",
    "OutcomeRecord",
    "OutcomeRecordValidationError",
    "PredictionRecord",
    "PredictionRecordValidationError",
    "compute_outcome_id",
    "compute_prediction_id",
]
