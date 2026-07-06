"""Server-side watchlist (Rule 14.9 user_state).

Mutable JSON file at ``reports/user_state/watchlist.json``. Schema:

    {
      "watchlist": [{"symbol": "6768.T", "added_ts": "...", "note": ""}, ...],
      "updated_ts": "ISO 8601 + tz",
      "schema_version": 1
    }

Rule 14.9 hard contract:
- file is overwritten atomically (tempfile + os.replace)
- only add/remove handlers may mutate
- never feeds calibration (Rule 14.6 exclusion)
- never touches portfolio journal
- max 100 entries
- symbol gate: 4-digit + ``.T``
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _compute_writer_token(entries: tuple, updated_ts: str) -> str:
    """Deterministic fingerprint over (entries, updated_ts).

    H5 fix for Rule 14.9.3 — load_watchlist re-computes the token from
    disk contents and compares against the persisted writer_token. Mismatch
    means an unauthorized writer modified the file; we fail-closed by
    treating the watchlist as empty (rather than honoring the tampered list).
    """
    payload = json.dumps({
        "entries": [
            {"symbol": e.symbol, "added_ts": e.added_ts, "note": e.note}
            for e in entries
        ],
        "updated_ts": updated_ts,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "WatchlistEntry",
    "WatchlistError",
    "WatchlistState",
    "add_to_watchlist",
    "default_watchlist_path",
    "load_watchlist",
    "remove_from_watchlist",
]


MAX_WATCHLIST_SIZE = 100
SCHEMA_VERSION = 1
SYMBOL_PATTERN = re.compile(r"^\d{4}\.T$")


class WatchlistError(ValueError):
    """Raised when watchlist operation cannot proceed."""


@dataclass(frozen=True)
class WatchlistEntry:
    symbol: str
    added_ts: str
    note: str = ""

    def __post_init__(self) -> None:
        if not SYMBOL_PATTERN.match(self.symbol):
            raise WatchlistError(
                f"symbol must match \\d{{4}}.T, got {self.symbol!r}"
            )
        # ensure added_ts parses as ISO 8601 with tz
        _parse_tz_iso(self.added_ts, "added_ts")
        if len(self.note) > 200:
            raise WatchlistError(
                f"note too long ({len(self.note)} chars, max 200)"
            )


@dataclass(frozen=True)
class WatchlistState:
    entries: tuple[WatchlistEntry, ...] = field(default_factory=tuple)
    updated_ts: str = ""
    schema_version: int = SCHEMA_VERSION
    writer_token: str = ""  # H5 fix — Rule 14.9.3 tamper detection

    @property
    def size(self) -> int:
        return len(self.entries)

    def has(self, symbol: str) -> bool:
        return any(e.symbol == symbol for e in self.entries)

    def to_dict(self) -> dict:
        return {
            "watchlist": [
                {"symbol": e.symbol, "added_ts": e.added_ts, "note": e.note}
                for e in self.entries
            ],
            "updated_ts": self.updated_ts,
            "schema_version": self.schema_version,
            "writer_token": self.writer_token,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WatchlistState":
        if not isinstance(d, dict):
            raise WatchlistError("watchlist file root must be a dict")
        raw = d.get("watchlist", [])
        if not isinstance(raw, list):
            raise WatchlistError("'watchlist' must be a list")
        entries = tuple(
            WatchlistEntry(
                symbol=str(r["symbol"]),
                added_ts=str(r["added_ts"]),
                note=str(r.get("note", "")),
            )
            for r in raw
        )
        return cls(
            entries=entries,
            updated_ts=str(d.get("updated_ts", "")),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            writer_token=str(d.get("writer_token", "")),
        )


def default_watchlist_path(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        here = Path(__file__).resolve()
        # parents: user_state, hot_theme_rotator, src, HTR_root
        base = here.parents[3]
    else:
        base = Path(base_dir)
    return base / "reports" / "user_state" / "watchlist.json"


def load_watchlist(*, base_dir: str | Path | None = None) -> WatchlistState:
    """Read the watchlist file; return empty state if file is missing.

    Malformed JSON raises WatchlistError. Missing file is treated as
    'no entries yet' (return WatchlistState() with no entries).

    H5 fix — Rule 14.9.3 tamper detection: re-compute the writer_token from
    on-disk values and compare against the persisted token. Mismatch means
    an external writer mutated the file (or the file pre-dates H5). In that
    case fail-closed by returning an empty WatchlistState — never honor a
    tampered list.
    """
    path = default_watchlist_path(base_dir)
    if not path.exists():
        return WatchlistState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WatchlistError(f"watchlist at {path} not valid JSON: {exc}") from exc
    state = WatchlistState.from_dict(payload)
    expected = _compute_writer_token(state.entries, state.updated_ts)
    if expected != state.writer_token:
        # Tamper detected — fail-closed (return empty). Don't raise to avoid
        # blocking the whole dashboard; the API surface will simply show an
        # empty watchlist until the user re-adds entries through the API.
        return WatchlistState()
    return state


def _atomic_write(path: Path, payload: dict) -> None:
    """Write payload to path via tempfile + os.replace (atomic on POSIX + Windows NTFS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=".json.tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def add_to_watchlist(
    symbol: str,
    *,
    note: str = "",
    now_iso: Optional[str] = None,
    base_dir: str | Path | None = None,
) -> WatchlistState:
    """Add a symbol to the watchlist. Returns the new state.

    Idempotent: if the symbol is already present, returns the existing state
    unchanged (no duplicate, no error). This matches the user-state mutation
    model where the user wants 'this should be in the watchlist'.

    Raises WatchlistError on invalid symbol or size limit.
    """
    state = load_watchlist(base_dir=base_dir)
    if state.has(symbol):
        return state
    if state.size >= MAX_WATCHLIST_SIZE:
        raise WatchlistError(
            f"watchlist full: {state.size}/{MAX_WATCHLIST_SIZE}; "
            f"remove an entry before adding {symbol}"
        )
    ts = now_iso or datetime.now(tz=timezone.utc).isoformat()
    entry = WatchlistEntry(symbol=symbol, added_ts=ts, note=note)
    new_entries = state.entries + (entry,)
    token = _compute_writer_token(new_entries, ts)
    new_state = WatchlistState(
        entries=new_entries,
        updated_ts=ts,
        schema_version=SCHEMA_VERSION,
        writer_token=token,
    )
    _atomic_write(default_watchlist_path(base_dir), new_state.to_dict())
    return new_state


def remove_from_watchlist(
    symbol: str,
    *,
    now_iso: Optional[str] = None,
    base_dir: str | Path | None = None,
) -> WatchlistState:
    """Remove a symbol from the watchlist. Returns the new state.

    Idempotent: if the symbol is not present, returns the existing state
    unchanged.

    Rule 12.4 anti-evasion note: cooling-off windows are tracked from the
    *first* add_ts. If a user removes-then-re-adds rapidly, the new
    added_ts resets cooling-off; this is acceptable surface for now (the
    discipline layer enforces actual alert suppression).
    """
    state = load_watchlist(base_dir=base_dir)
    if not state.has(symbol):
        return state
    new_entries = tuple(e for e in state.entries if e.symbol != symbol)
    ts = now_iso or datetime.now(tz=timezone.utc).isoformat()
    token = _compute_writer_token(new_entries, ts)
    new_state = WatchlistState(
        entries=new_entries,
        updated_ts=ts,
        schema_version=SCHEMA_VERSION,
        writer_token=token,
    )
    _atomic_write(default_watchlist_path(base_dir), new_state.to_dict())
    return new_state


def _parse_tz_iso(value: str, field_name: str) -> datetime:
    """Parse ISO 8601 with timezone; raise if missing tz."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WatchlistError(f"{field_name}={value!r} not valid ISO 8601") from exc
    if dt.tzinfo is None:
        raise WatchlistError(f"{field_name}={value!r} must be timezone-aware")
    return dt
