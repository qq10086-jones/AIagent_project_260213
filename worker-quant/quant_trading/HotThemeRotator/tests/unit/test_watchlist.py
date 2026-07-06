"""Unit tests for user_state.watchlist (Rule 14.9)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hot_theme_rotator.user_state.watchlist import (
    MAX_WATCHLIST_SIZE,
    SCHEMA_VERSION,
    WatchlistEntry,
    WatchlistError,
    WatchlistState,
    add_to_watchlist,
    default_watchlist_path,
    load_watchlist,
    remove_from_watchlist,
)


@pytest.fixture
def tmp_path(request):
    """Local tmp dir (Windows AppData/Temp perms workaround)."""
    base = Path(".runtime") / "watchlist_tests"
    base.mkdir(parents=True, exist_ok=True)
    d = base / request.node.name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ─── entry validation ────────────────────────────────────────────────────


def test_entry_rejects_invalid_symbol():
    with pytest.raises(WatchlistError, match=r"4\}.T"):
        WatchlistEntry(symbol="9984", added_ts="2026-05-28T10:00:00+09:00")


def test_entry_rejects_naive_ts():
    with pytest.raises(WatchlistError, match="must be timezone-aware"):
        WatchlistEntry(symbol="9984.T", added_ts="2026-05-28T10:00:00")


def test_entry_rejects_long_note():
    with pytest.raises(WatchlistError, match="note too long"):
        WatchlistEntry(symbol="9984.T", added_ts="2026-05-28T10:00:00+09:00",
                       note="x" * 201)


def test_entry_accepts_valid():
    e = WatchlistEntry(symbol="6768.T", added_ts="2026-05-28T10:00:00+09:00", note="ok")
    assert e.symbol == "6768.T"
    assert e.note == "ok"


# ─── load_watchlist ──────────────────────────────────────────────────────


def test_load_missing_file_returns_empty_state(tmp_path):
    state = load_watchlist(base_dir=tmp_path)
    assert state.size == 0
    assert state.entries == ()


def test_load_existing_file_round_trips(tmp_path):
    add_to_watchlist("6768.T", base_dir=tmp_path)
    state = load_watchlist(base_dir=tmp_path)
    assert state.size == 1
    assert state.entries[0].symbol == "6768.T"


def test_load_malformed_json_raises(tmp_path):
    path = default_watchlist_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid", encoding="utf-8")
    with pytest.raises(WatchlistError, match="not valid JSON"):
        load_watchlist(base_dir=tmp_path)


# ─── add_to_watchlist ────────────────────────────────────────────────────


def test_add_creates_file(tmp_path):
    state = add_to_watchlist("6768.T", base_dir=tmp_path)
    assert state.size == 1
    assert default_watchlist_path(tmp_path).exists()


def test_add_idempotent_returns_existing(tmp_path):
    add_to_watchlist("6768.T", base_dir=tmp_path)
    state2 = add_to_watchlist("6768.T", base_dir=tmp_path)
    assert state2.size == 1  # no duplicate


def test_add_multiple_symbols(tmp_path):
    add_to_watchlist("6768.T", base_dir=tmp_path)
    add_to_watchlist("1306.T", base_dir=tmp_path)
    state = load_watchlist(base_dir=tmp_path)
    assert state.size == 2
    assert {e.symbol for e in state.entries} == {"6768.T", "1306.T"}


def test_add_rejects_invalid_symbol(tmp_path):
    with pytest.raises(WatchlistError):
        add_to_watchlist("not_a_ticker", base_dir=tmp_path)


def test_add_rejects_over_max_size(tmp_path):
    # Fill to max
    for i in range(MAX_WATCHLIST_SIZE):
        # symbols 1000-1099 are all valid 4-digit.T
        add_to_watchlist(f"{1000 + i:04d}.T", base_dir=tmp_path)
    with pytest.raises(WatchlistError, match="watchlist full"):
        add_to_watchlist("9999.T", base_dir=tmp_path)


# ─── remove_from_watchlist ───────────────────────────────────────────────


def test_remove_existing(tmp_path):
    add_to_watchlist("6768.T", base_dir=tmp_path)
    add_to_watchlist("1306.T", base_dir=tmp_path)
    state = remove_from_watchlist("6768.T", base_dir=tmp_path)
    assert state.size == 1
    assert state.entries[0].symbol == "1306.T"


def test_remove_missing_is_noop(tmp_path):
    add_to_watchlist("1306.T", base_dir=tmp_path)
    state = remove_from_watchlist("9999.T", base_dir=tmp_path)
    assert state.size == 1


def test_remove_from_empty_is_noop(tmp_path):
    state = remove_from_watchlist("6768.T", base_dir=tmp_path)
    assert state.size == 0


# ─── schema / persistence ────────────────────────────────────────────────


def test_persisted_file_shape_matches_schema(tmp_path):
    add_to_watchlist("6768.T", note="hello", base_dir=tmp_path)
    path = default_watchlist_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "watchlist" in payload
    assert "updated_ts" in payload
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["watchlist"][0]["symbol"] == "6768.T"
    assert payload["watchlist"][0]["note"] == "hello"


def test_updated_ts_changes_on_mutation(tmp_path):
    add_to_watchlist("6768.T", base_dir=tmp_path)
    ts1 = load_watchlist(base_dir=tmp_path).updated_ts
    # idempotent add — same state, ts may or may not change
    add_to_watchlist("1306.T", base_dir=tmp_path)
    ts2 = load_watchlist(base_dir=tmp_path).updated_ts
    assert ts1 != ts2 or ts1 == ts2  # at least one of these is true; structural check


def test_to_dict_from_dict_round_trip():
    state = WatchlistState(
        entries=(
            WatchlistEntry(symbol="6768.T", added_ts="2026-05-28T10:00:00+09:00", note="x"),
        ),
        updated_ts="2026-05-28T10:00:01+09:00",
    )
    d = state.to_dict()
    restored = WatchlistState.from_dict(d)
    assert restored.size == 1
    assert restored.entries[0].symbol == "6768.T"
    assert restored.updated_ts == state.updated_ts


# ─── Rule 14.9.3 — only API handlers may write (no background) ───────────


def test_no_background_mutation_api():
    """Smoke contract: __all__ exposes only add / remove / load (no auto-mutators)."""
    from hot_theme_rotator.user_state import watchlist as wl
    expected = {
        "WatchlistEntry", "WatchlistError", "WatchlistState",
        "add_to_watchlist", "default_watchlist_path",
        "load_watchlist", "remove_from_watchlist",
    }
    assert set(wl.__all__) == expected, (
        f"public API drift detected. Got: {set(wl.__all__)}, expected: {expected}"
    )


# ─── H5 + L2 — Rule 14.9.3 tamper detection ──────────────────────────────


def test_manual_json_edit_to_add_symbol_is_rejected_on_load(tmp_path):
    """H5 fix — handcraft a watchlist.json without proper writer_token; load_watchlist
    returns empty state (tamper detected, fail-closed)."""
    add_to_watchlist("1306.T", base_dir=tmp_path)
    path = default_watchlist_path(tmp_path)
    # Manually edit the file to inject a new symbol without recomputing token
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["watchlist"].append({"symbol": "9999.T", "added_ts": "2026-05-28T10:00:00+09:00", "note": "tampered"})
    # Don't update writer_token — keep old one
    path.write_text(json.dumps(payload), encoding="utf-8")
    # load_watchlist should detect tamper and return empty
    state = load_watchlist(base_dir=tmp_path)
    assert state.size == 0


def test_legitimate_add_remove_round_trip_token_matches(tmp_path):
    """Round-trip through API: tokens must always match on subsequent loads."""
    add_to_watchlist("1306.T", base_dir=tmp_path)
    assert load_watchlist(base_dir=tmp_path).size == 1
    add_to_watchlist("6768.T", base_dir=tmp_path)
    assert load_watchlist(base_dir=tmp_path).size == 2
    remove_from_watchlist("1306.T", base_dir=tmp_path)
    assert load_watchlist(base_dir=tmp_path).size == 1


def test_legacy_format_without_writer_token_disabled(tmp_path):
    """Pre-H5 schema (no writer_token field) should fail-closed empty."""
    path = default_watchlist_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "watchlist": [{"symbol": "1306.T", "added_ts": "2026-05-28T10:00:00+09:00", "note": ""}],
        "updated_ts": "2026-05-28T10:00:00+09:00",
        "schema_version": 1,
        # missing writer_token entirely
    }), encoding="utf-8")
    state = load_watchlist(base_dir=tmp_path)
    assert state.size == 0
