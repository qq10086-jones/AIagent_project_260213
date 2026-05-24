"""JSONL persistence for `PredictionRecord` (§8.6 / §10 gate 3) and
`OutcomeRecord` (§10 gate 4).

One JSONL file per trade date at:
- `reports/predictions/{trade_date}.jsonl`
- `reports/outcomes/{trade_date}.jsonl`

Append-only, fail-closed on duplicate id and malformed JSONL, deterministic
read order.

F7 — concurrency assumption: this writer assumes a SINGLE-WRITER deployment
(one scanner per host). The duplicate-id guard is a read-then-append sequence
and is NOT atomic across processes. If you ever run two scanners with
overlapping inputs against the same `base_dir` they can both bypass the guard
and produce duplicate lines. Mitigations for future scale: a sibling `.lock`
file (`fcntl` on POSIX, `msvcrt` on Windows), SQLite, or atomic write+rename
with post-write dedup. Until then, run one writer at a time.

This module is research-only persistence. It does not call any execution path.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

from hot_theme_rotator.decision_log.schema import (
    OutcomeRecord,
    OutcomeRecordValidationError,
    PredictionRecord,
    PredictionRecordValidationError,
)


DEFAULT_REPORTS_SUBDIR = Path("reports") / "predictions"
DEFAULT_OUTCOMES_SUBDIR = Path("reports") / "outcomes"


class DecisionLogStorageError(RuntimeError):
    """Raised when the JSONL store cannot be safely read or appended."""


def _validate_trade_date(trade_date: str) -> str:
    """F6 — `trade_date` must be a non-empty ISO date (YYYY-MM-DD).

    Path helpers used to accept arbitrary strings, so a value like
    `'../../etc/passwd'` or `'2026/05/23'` would resolve to an unsafe filename.
    """
    raw = str(trade_date).strip()
    if not raw:
        raise DecisionLogStorageError("trade_date must be non-empty")
    try:
        date.fromisoformat(raw)
    except ValueError as exc:
        raise DecisionLogStorageError(
            f"trade_date must be ISO date (YYYY-MM-DD): {trade_date!r}"
        ) from exc
    return raw


def predictions_path(
    *,
    trade_date: str,
    base_dir: Path | str,
) -> Path:
    """Return the JSONL file path for one trade date under base_dir."""
    validated = _validate_trade_date(trade_date)
    base = Path(base_dir)
    return base / DEFAULT_REPORTS_SUBDIR / f"{validated}.jsonl"


def outcomes_path(
    *,
    trade_date: str,
    base_dir: Path | str,
) -> Path:
    """Return the outcomes JSONL file path for one trade date (§10 gate 4)."""
    validated = _validate_trade_date(trade_date)
    base = Path(base_dir)
    return base / DEFAULT_OUTCOMES_SUBDIR / f"{validated}.jsonl"


def append_prediction(
    record: PredictionRecord,
    *,
    base_dir: Path | str,
) -> Path:
    """Append one validated PredictionRecord. Fail closed on duplicate id."""
    if not isinstance(record, PredictionRecord):
        raise DecisionLogStorageError(
            "append_prediction requires a PredictionRecord instance"
        )
    target = predictions_path(trade_date=record.trade_date, base_dir=base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = read_predictions(trade_date=record.trade_date, base_dir=base_dir)
    existing_ids = {entry.prediction_id for entry in existing}
    if record.prediction_id in existing_ids:
        raise DecisionLogStorageError(
            f"prediction_id {record.prediction_id!r} already present in {target}"
        )

    line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
    with target.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    return target


def append_predictions(
    records: Iterable[PredictionRecord],
    *,
    base_dir: Path | str,
) -> tuple[Path, ...]:
    """Append multiple records. Halts on first failure (no partial rollback)."""
    written: list[Path] = []
    for record in records:
        written.append(append_prediction(record, base_dir=base_dir))
    return tuple(written)


def read_predictions(
    *,
    trade_date: str,
    base_dir: Path | str,
) -> tuple[PredictionRecord, ...]:
    """Read all records for one trade date. Empty tuple if file is missing."""
    target = predictions_path(trade_date=trade_date, base_dir=base_dir)
    if not target.exists():
        return ()
    records: list[PredictionRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DecisionLogStorageError(
                    f"{target}:{line_no} is not valid JSON: {exc}"
                ) from exc
            try:
                records.append(PredictionRecord.from_dict(payload))
            except PredictionRecordValidationError as exc:
                raise DecisionLogStorageError(
                    f"{target}:{line_no} failed schema validation: {exc}"
                ) from exc
    return tuple(records)


# ============================================================================
# Outcomes (§10 gate 4) — mirror of predictions API but for OutcomeRecord
# ============================================================================


def append_outcome(
    record: OutcomeRecord,
    *,
    base_dir: Path | str,
) -> Path:
    """Append one validated OutcomeRecord. Fail closed on duplicate id."""
    if not isinstance(record, OutcomeRecord):
        raise DecisionLogStorageError(
            "append_outcome requires an OutcomeRecord instance"
        )
    target = outcomes_path(trade_date=record.trade_date, base_dir=base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = read_outcomes(trade_date=record.trade_date, base_dir=base_dir)
    existing_ids = {entry.outcome_id for entry in existing}
    if record.outcome_id in existing_ids:
        raise DecisionLogStorageError(
            f"outcome_id {record.outcome_id!r} already present in {target}"
        )

    line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
    with target.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    return target


def append_outcomes(
    records: Iterable[OutcomeRecord],
    *,
    base_dir: Path | str,
) -> tuple[Path, ...]:
    """Append multiple outcomes. Halts on first failure (no partial rollback)."""
    written: list[Path] = []
    for record in records:
        written.append(append_outcome(record, base_dir=base_dir))
    return tuple(written)


def read_outcomes(
    *,
    trade_date: str,
    base_dir: Path | str,
) -> tuple[OutcomeRecord, ...]:
    """Read all outcomes for one trade date. Empty tuple if file missing."""
    target = outcomes_path(trade_date=trade_date, base_dir=base_dir)
    if not target.exists():
        return ()
    records: list[OutcomeRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DecisionLogStorageError(
                    f"{target}:{line_no} is not valid JSON: {exc}"
                ) from exc
            try:
                records.append(OutcomeRecord.from_dict(payload))
            except OutcomeRecordValidationError as exc:
                raise DecisionLogStorageError(
                    f"{target}:{line_no} failed schema validation: {exc}"
                ) from exc
    return tuple(records)
