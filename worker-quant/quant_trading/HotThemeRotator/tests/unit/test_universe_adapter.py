"""Tests for universe_adapter (P8-15 / ADR-0005)."""
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.universe_adapter import (  # noqa: E402
    ScreenedTicker,
    ScreenerSnapshot,
    UniverseAdapterError,
    default_selected_tickers_path,
    load_screener_snapshot,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


_REAL_PAYLOAD = {
    "asof": "2026-04-27",
    "version": "screener_v2",
    "count": 2,
    "symbols": ["6768.T", "5074.T"],
    "details": [
        {
            "symbol": "6768.T",
            "score": 0.506,
            "close": 801.0,
            "adv": 497960965.0,
            "vol": 0.0267,
            "hard_fail": False,
            "reason": "",
            "mom_20": 0.4190,
            "mom_60": 0.4086,
            "sharpe_20": 0.9521,
            "adv_rank": 0.802,
            "fundamental_score": 0.9,
        },
        {
            "symbol": "5074.T",
            "score": 0.487,
            "close": 1245.0,
            "adv": 320000000.0,
            "vol": 0.031,
            "hard_fail": False,
            "reason": "",
        },
    ],
}


def test_load_real_screener_payload(tmp_path):
    p = tmp_path / "selected_tickers.json"
    _write(p, _REAL_PAYLOAD)
    snapshot = load_screener_snapshot(p)
    assert isinstance(snapshot, ScreenerSnapshot)
    assert snapshot.asof == "2026-04-27"
    assert snapshot.version == "screener_v2"
    assert snapshot.count == 2
    assert len(snapshot.tickers) == 2
    assert snapshot.tickers[0].symbol == "6768.T"
    assert snapshot.tickers[0].score == 0.506
    assert snapshot.tickers[0].close == 801.0
    assert snapshot.tickers[0].fundamental_score == 0.9


def test_tickers_preserve_input_order(tmp_path):
    p = tmp_path / "selected_tickers.json"
    _write(p, _REAL_PAYLOAD)
    snapshot = load_screener_snapshot(p)
    assert [t.symbol for t in snapshot.tickers] == ["6768.T", "5074.T"]


def test_optional_fields_default_to_zero(tmp_path):
    """`details` row with only required keys still loads cleanly."""
    p = tmp_path / "selected_tickers.json"
    _write(p, {
        **_REAL_PAYLOAD,
        "details": [{
            "symbol": "9999.T", "score": 0.5, "close": 100.0, "adv": 1e8, "vol": 0.02,
        }],
    })
    snapshot = load_screener_snapshot(p)
    assert snapshot.tickers[0].mom_20 == 0.0
    assert snapshot.tickers[0].sharpe_20 == 0.0
    assert snapshot.tickers[0].fundamental_score == 0.0


def test_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(UniverseAdapterError, match="not found"):
        load_screener_snapshot(tmp_path / "nope.json")


def test_fails_closed_on_invalid_json(tmp_path):
    p = tmp_path / "selected_tickers.json"
    p.write_text("{not valid", encoding="utf-8")
    with pytest.raises(UniverseAdapterError, match="not valid JSON"):
        load_screener_snapshot(p)


def test_fails_closed_on_missing_top_keys(tmp_path):
    p = tmp_path / "selected_tickers.json"
    _write(p, {"asof": "2026-04-27"})  # missing 'symbols' and 'details'
    with pytest.raises(UniverseAdapterError, match="missing required keys"):
        load_screener_snapshot(p)


def test_fails_closed_on_missing_detail_keys(tmp_path):
    p = tmp_path / "selected_tickers.json"
    _write(p, {
        **_REAL_PAYLOAD,
        "details": [{"symbol": "6768.T", "score": 0.5}],  # missing close, adv, vol
    })
    with pytest.raises(UniverseAdapterError, match="missing required keys"):
        load_screener_snapshot(p)


def test_empty_details_is_valid(tmp_path):
    p = tmp_path / "selected_tickers.json"
    _write(p, {**_REAL_PAYLOAD, "details": []})
    snapshot = load_screener_snapshot(p)
    assert snapshot.tickers == ()


def test_default_path_resolves_to_sibling_project_optimized():
    p = default_selected_tickers_path()
    assert p.name == "selected_tickers.json"
    assert "Project_optimized" in str(p)
