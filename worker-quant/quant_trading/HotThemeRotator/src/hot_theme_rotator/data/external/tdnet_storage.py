"""JSONL persistence for `TdnetDisclosure` (P10-14 Cycle 1).

One JSONL file per trade date at `reports/tdnet/{trade_date}.jsonl`.
Append-only, fail-closed on duplicate disclosure_id and malformed JSONL.

Mirrors the P9-01 `decision_log/jsonl_writer.py` single-writer pattern. The
duplicate-id guard is a read-then-append sequence and is NOT atomic across
processes. Single-writer-per-host deployment assumed. If scaling beyond that,
add a sibling `.lock` file or migrate to SQLite.

Research-only persistence. Does not call any execution path (Rule 3).
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from hot_theme_rotator.data.external.tdnet_schema import (
    TdnetDisclosure,
    TdnetDisclosureValidationError,
)


DEFAULT_TDNET_SUBDIR = Path("reports") / "tdnet"


class TdnetStorageError(RuntimeError):
    """Raised when the TDnet JSONL store cannot be safely read or appended."""


def _validate_trade_date(trade_date: str) -> str:
    """`trade_date` must be a non-empty ISO date (YYYY-MM-DD).

    Same hardening as P9-01 F6: refuse `'../../etc/passwd'` or `'2026/05/25'`.
    """
    raw = str(trade_date).strip()
    if not raw:
        raise TdnetStorageError("trade_date must be non-empty")
    try:
        date.fromisoformat(raw)
    except ValueError as exc:
        raise TdnetStorageError(
            f"trade_date must be ISO date (YYYY-MM-DD): {trade_date!r}"
        ) from exc
    return raw


def _trade_date_from_published_ts(published_ts: str) -> str:
    """Extract YYYY-MM-DD from a TdnetDisclosure.published_ts."""
    try:
        return datetime.fromisoformat(published_ts).date().isoformat()
    except ValueError as exc:
        raise TdnetStorageError(
            f"cannot derive trade_date from non-ISO published_ts {published_ts!r}"
        ) from exc


def disclosures_path(
    *,
    trade_date: str,
    base_dir: Path | str,
) -> Path:
    """Return the JSONL file path for one trade date under base_dir."""
    validated = _validate_trade_date(trade_date)
    base = Path(base_dir)
    return base / DEFAULT_TDNET_SUBDIR / f"{validated}.jsonl"


def append_disclosure(
    record: TdnetDisclosure,
    *,
    base_dir: Path | str,
) -> Path:
    """Append one validated TdnetDisclosure. Fail closed on duplicate id."""
    if not isinstance(record, TdnetDisclosure):
        raise TdnetStorageError(
            "append_disclosure requires a TdnetDisclosure instance"
        )
    trade_date = _trade_date_from_published_ts(record.published_ts)
    target = disclosures_path(trade_date=trade_date, base_dir=base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = read_disclosures(trade_date=trade_date, base_dir=base_dir)
    existing_ids = {entry.disclosure_id for entry in existing}
    if record.disclosure_id in existing_ids:
        raise TdnetStorageError(
            f"disclosure_id {record.disclosure_id!r} already present in {target}"
        )

    line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
    with target.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    return target


def append_disclosures(
    records: Iterable[TdnetDisclosure],
    *,
    base_dir: Path | str,
) -> tuple[Path, ...]:
    """Append multiple records. Halts on first failure (no partial rollback)."""
    written: list[Path] = []
    for record in records:
        written.append(append_disclosure(record, base_dir=base_dir))
    return tuple(written)


def read_disclosures_for_symbol(
    *,
    symbol: str,
    base_dir: Path | str,
    max_days_back: int = 7,
    asof_date: str | None = None,
) -> tuple[TdnetDisclosure, ...]:
    """P2 — Read recent disclosures filtered by symbol.

    Scans the past ``max_days_back`` daily TDnet files (or all of them if
    ``asof_date`` is None and the directory has fewer days). Returns matching
    records sorted by ``published_ts`` descending.

    This is a STRUCTURAL UI helper (Rule 11.4 chart-tooltip clarification
    territory) — disclosure text is shown as factual data in the UI, NOT
    concatenated into LLM prompts. LLM brief input is unchanged.
    """
    import os
    from datetime import datetime as _dt, timedelta as _td

    base = Path(base_dir)
    storage_dir = base / DEFAULT_TDNET_SUBDIR
    if not storage_dir.exists():
        return ()

    if asof_date is not None:
        try:
            cutoff = _dt.fromisoformat(asof_date) - _td(days=max_days_back)
        except ValueError:
            cutoff = None
    else:
        cutoff = None

    results: list[TdnetDisclosure] = []
    for path in sorted(storage_dir.glob("*.jsonl"), reverse=True):
        try:
            file_date = _dt.fromisoformat(path.stem)
        except ValueError:
            continue
        if cutoff and file_date < cutoff:
            break
        try:
            for rec in read_disclosures(trade_date=path.stem, base_dir=base):
                if rec.ticker == symbol:
                    results.append(rec)
        except TdnetStorageError:
            continue
    # Sort by published_ts desc
    results.sort(key=lambda r: r.published_ts, reverse=True)
    return tuple(results)


def read_disclosures(
    *,
    trade_date: str,
    base_dir: Path | str,
) -> tuple[TdnetDisclosure, ...]:
    """Read all disclosures for one trade date. Empty tuple if file missing."""
    target = disclosures_path(trade_date=trade_date, base_dir=base_dir)
    if not target.exists():
        return ()
    records: list[TdnetDisclosure] = []
    with target.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TdnetStorageError(
                    f"{target}:{line_no} is not valid JSON: {exc}"
                ) from exc
            try:
                records.append(TdnetDisclosure.from_dict(payload))
            except TdnetDisclosureValidationError as exc:
                raise TdnetStorageError(
                    f"{target}:{line_no} failed schema validation: {exc}"
                ) from exc
    return tuple(records)
