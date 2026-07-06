"""P12-03 — the dashboard candidate panel must prefer the freshest HTR-native
screener snapshot over the sibling file, which goes stale (Rule 11.9 freshness)."""
from __future__ import annotations

import json

import api.serializers as ser


def _write(p, asof, symbols=("X.T",)):
    p.write_text(json.dumps({"asof": asof, "symbols": list(symbols)}), encoding="utf-8")


def test_prefers_htr_native_snapshot_when_fresher(tmp_path, monkeypatch):
    htr = tmp_path / "screener"
    htr.mkdir()
    _write(htr / "selected_tickers_2026-05-29.json", "2026-05-29")
    _write(htr / "selected_tickers_2026-06-01.json", "2026-06-01")   # newest
    sibling = tmp_path / "selected_tickers.json"
    _write(sibling, "2026-05-27")   # stale sibling
    monkeypatch.setattr(ser, "_htr_screener_dir", lambda: htr)
    monkeypatch.setattr(ser, "default_selected_tickers_path", lambda: sibling)

    chosen = ser._freshest_selected_tickers_path()
    assert chosen == htr / "selected_tickers_2026-06-01.json"


def test_falls_back_to_sibling_when_no_htr_snapshot(tmp_path, monkeypatch):
    htr = tmp_path / "screener"
    htr.mkdir()
    sibling = tmp_path / "selected_tickers.json"
    _write(sibling, "2026-05-27")
    monkeypatch.setattr(ser, "_htr_screener_dir", lambda: htr)
    monkeypatch.setattr(ser, "default_selected_tickers_path", lambda: sibling)

    assert ser._freshest_selected_tickers_path() == sibling


def test_keeps_sibling_if_it_is_actually_fresher(tmp_path, monkeypatch):
    # defensive: a stale HTR snapshot must not override a fresher sibling
    htr = tmp_path / "screener"
    htr.mkdir()
    _write(htr / "selected_tickers_2026-05-20.json", "2026-05-20")
    sibling = tmp_path / "selected_tickers.json"
    _write(sibling, "2026-05-27")
    monkeypatch.setattr(ser, "_htr_screener_dir", lambda: htr)
    monkeypatch.setattr(ser, "default_selected_tickers_path", lambda: sibling)

    assert ser._freshest_selected_tickers_path() == sibling
