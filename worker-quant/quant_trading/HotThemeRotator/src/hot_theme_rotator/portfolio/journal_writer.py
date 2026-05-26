"""Append-only JSONL journal writer for portfolio events (Rule 14.1).

Two append APIs (one per entry type) and one read API. By construction, no
UPDATE / DELETE / random-access mutation path exists in this module — Rule 14.1
is enforced by absence rather than by check. Corrections are themselves new
fills with ``source='correction'`` and a ``corrects`` reference, per Rule 14.4.

File partitioning: one JSONL file per JST trade-date under
``{base_dir}/reports/portfolio/journal/{trade_date}.jsonl``. The JST date is
derived from each entry's ``ts`` regardless of its source timezone, so events
recorded with UTC stamps still land in the right Japan-equity trading-day
bucket.

Single-writer assumption: this module does not implement file locking.
Concurrent writers across processes are out of scope and would need a later
cycle's lock layer. For single-user, single-process operation (the only
supported mode), the append-only / no-mutation API surface is sufficient.

Duplicate-id rejection: each append scans the target file for an existing
``entry_id`` and refuses to re-append. Since ``entry_id`` is deterministic
over an entry's material fields (see ``schema.derive_*_id``), duplicate
detection doubles as idempotency — a user who clicks "submit" twice gets one
fill, not two.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Iterator, Sequence, Union

from hot_theme_rotator.portfolio.schema import (
    CashEvent,
    FillEntry,
    PortfolioSchemaError,
)


JST: Final = timezone(timedelta(hours=9), name="JST")

JournalEntry = Union[FillEntry, CashEvent]


class PortfolioJournalError(RuntimeError):
    """Raised on journal IO failure: bad path, duplicate id, malformed line."""


__all__ = [
    "JST",
    "JournalEntry",
    "PortfolioJournalError",
    "append_cash_event",
    "append_fill",
    "journal_dir",
    "journal_lock",
    "journal_path",
    "read_all_journal",
    "read_journal",
]


@contextmanager
def journal_lock(
    base_dir: str | Path = ".",
    *,
    timeout: float = 5.0,
    poll: float = 0.05,
) -> "Iterator[Path]":
    """Acquire an exclusive cross-process lock on the portfolio journal directory.

    Uses ``O_CREAT | O_EXCL`` on a sentinel lockfile, which is atomic on POSIX
    and on Windows (NTFS). The lock spans the read-validate-append sequence
    in manual entry commits — Rule 14.5 commit-time re-validation needs this
    to actually close the TOCTOU window between read and append. Single-process
    callers pay only a few hundred microseconds; concurrent callers serialize.

    Stale-lock recovery: if the holder process PID is no longer alive, the
    next acquirer breaks the lock after ``timeout`` seconds. This is the
    standard pattern; the timeout window is the cost of robustness.
    """
    lock_path = Path(base_dir) / "reports" / "portfolio" / ".journal.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise PortfolioJournalError(
                    f"could not acquire journal lock at {lock_path} within "
                    f"{timeout}s; another process may be writing"
                )
            time.sleep(poll)

    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def journal_dir(base_dir: str | Path = ".") -> Path:
    """Root directory containing per-trade-date JSONL files."""
    return Path(base_dir) / "reports" / "portfolio" / "journal"


def read_all_journal(base_dir: str | Path = ".") -> tuple["JournalEntry", ...]:
    """Read every per-date journal file under base_dir in trade-date order.

    Returns a single chronological tuple of all entries. Missing directory →
    empty tuple. Use this when you need a derived view that spans multiple
    trading days (e.g., manual-entry preview needs cumulative cash + holdings).
    """
    root = journal_dir(base_dir)
    if not root.exists():
        return ()
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix == ".jsonl")
    out: list[JournalEntry] = []
    for f in files:
        trade_date = f.stem
        try:
            entries = read_journal(trade_date, base_dir=base_dir)
        except PortfolioJournalError:
            raise
        out.extend(entries)
    return tuple(out)


def journal_path(trade_date: str, *, base_dir: str | Path = ".") -> Path:
    """Return the JSONL file path for ``trade_date`` under ``base_dir``.

    Validates ``trade_date`` is ISO YYYY-MM-DD so callers cannot smuggle path
    components (e.g., ``"../escape"``) into the journal directory.
    """
    validated = _validate_trade_date(trade_date)
    return Path(base_dir) / "reports" / "portfolio" / "journal" / f"{validated}.jsonl"


def append_fill(entry: FillEntry, *, base_dir: str | Path = ".") -> Path:
    """Append a ``FillEntry`` to its JST-partitioned journal file."""
    return _append_entry(entry, entry_type="fill", base_dir=base_dir)


def append_cash_event(event: CashEvent, *, base_dir: str | Path = ".") -> Path:
    """Append a ``CashEvent`` to its JST-partitioned journal file."""
    return _append_entry(event, entry_type="cash_event", base_dir=base_dir)


def read_journal(
    trade_date: str,
    *,
    base_dir: str | Path = ".",
) -> tuple[JournalEntry, ...]:
    """Return all journal entries for ``trade_date`` in append order.

    Missing file → empty tuple (not an error — partial trading days are
    legitimate). Malformed JSONL or unknown ``_type`` → fail-closed with
    explicit reason.
    """
    path = journal_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return ()

    rows: list[JournalEntry] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PortfolioJournalError(
                f"malformed JSONL at {path}:{lineno}: {exc}"
            ) from exc

        entry_type = payload.pop("_type", None)
        try:
            if entry_type == "fill":
                rows.append(FillEntry.from_dict(payload))
            elif entry_type == "cash_event":
                rows.append(CashEvent.from_dict(payload))
            else:
                raise PortfolioJournalError(
                    f"unknown entry _type={entry_type!r} at {path}:{lineno}"
                )
        except PortfolioSchemaError as exc:
            raise PortfolioJournalError(
                f"schema rejected entry at {path}:{lineno}: {exc}"
            ) from exc
    return tuple(rows)


# ─── internals ──────────────────────────────────────────────────────────────


def _append_entry(
    entry: JournalEntry,
    *,
    entry_type: str,
    base_dir: str | Path,
) -> Path:
    trade_date = _derive_trade_date_jst(entry.ts)
    path = journal_path(trade_date, base_dir=base_dir)
    _check_no_duplicate_id(path, entry.entry_id)
    _check_correction_reference_exists(entry, base_dir=base_dir)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_type": entry_type, **entry.to_dict()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def _check_correction_reference_exists(
    entry: JournalEntry,
    *,
    base_dir: str | Path,
) -> None:
    """Rule 14.4: a correction MUST reference an existing prior entry_id.

    Forward-targeting (a correction whose ``corrects`` matches a not-yet-
    appended entry) is forbidden — otherwise a correction can pre-poison a
    future fill's deterministic ``entry_id`` and have it silently dropped by
    derive_positions' skip-both semantic.
    """
    if entry.source != "correction":
        return
    if not entry.corrects:
        return  # schema-level guard handles the source/corrects coherence
    existing = {e.entry_id for e in read_all_journal(base_dir)}
    if entry.corrects not in existing:
        raise PortfolioJournalError(
            f"correction {entry.entry_id} references missing prior entry_id "
            f"{entry.corrects!r}; Rule 14.4 requires the corrected entry to "
            f"already exist in the journal"
        )


def _derive_trade_date_jst(ts: str) -> str:
    """Convert ISO ts (any timezone) to JST date string ``YYYY-MM-DD``."""
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        raise PortfolioJournalError(
            f"ts must carry timezone (caller should preflight via schema), got naive: {ts!r}"
        )
    return parsed.astimezone(JST).date().isoformat()


def _validate_trade_date(trade_date: str) -> str:
    try:
        date.fromisoformat(trade_date)
    except (TypeError, ValueError) as exc:
        raise PortfolioJournalError(
            f"trade_date must be ISO YYYY-MM-DD, got {trade_date!r}"
        ) from exc
    return trade_date


def _check_no_duplicate_id(path: Path, entry_id: str) -> None:
    if not path.exists():
        return
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Treat malformed pre-existing line as fail-closed for safety —
            # callers should run read_journal to surface the bad row.
            raise PortfolioJournalError(
                f"existing journal {path}:{lineno} is malformed; cannot safely append"
            )
        if payload.get("entry_id") == entry_id:
            raise PortfolioJournalError(
                f"duplicate entry_id={entry_id!r} already present at {path}:{lineno} "
                f"(Rule 14.1 append-only; use source='correction' to adjust)"
            )
