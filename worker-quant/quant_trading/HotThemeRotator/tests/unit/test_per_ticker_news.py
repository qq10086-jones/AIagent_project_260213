"""Tests for per-ticker news (P2) — structural UI data, NOT in LLM grounding."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hot_theme_rotator.data.external.tdnet_schema import (
    TdnetDisclosure, compute_disclosure_id,
)
from hot_theme_rotator.data.external.tdnet_storage import (
    append_disclosure, read_disclosures_for_symbol,
)


@pytest.fixture
def tmp_path(request):
    base = Path(".runtime") / "p2_news_tests"
    base.mkdir(parents=True, exist_ok=True)
    d = base / request.node.name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_disclosure(ticker: str, *, title: str, published_ts: str,
                     category: str = "earnings"):
    return TdnetDisclosure(
        disclosure_id=compute_disclosure_id(ticker, published_ts, title),
        ticker=ticker,
        published_ts=published_ts,
        collected_ts=published_ts,
        title=title,
        category=category,
        url=f"https://example.com/{ticker}",
    )


def test_read_for_symbol_empty_dir_returns_empty(tmp_path):
    out = read_disclosures_for_symbol(symbol="6768.T", base_dir=tmp_path)
    assert out == ()


def test_read_for_symbol_filters_by_ticker(tmp_path):
    append_disclosure(
        _make_disclosure("6768.T", title="決算", published_ts="2026-05-27T15:00:00+09:00"),
        base_dir=tmp_path,
    )
    append_disclosure(
        _make_disclosure("1306.T", title="その他", published_ts="2026-05-27T16:00:00+09:00"),
        base_dir=tmp_path,
    )
    out_6768 = read_disclosures_for_symbol(symbol="6768.T", base_dir=tmp_path)
    out_1306 = read_disclosures_for_symbol(symbol="1306.T", base_dir=tmp_path)
    assert len(out_6768) == 1
    assert len(out_1306) == 1
    assert out_6768[0].ticker == "6768.T"
    assert out_1306[0].ticker == "1306.T"


def test_read_for_symbol_sorts_descending_by_published_ts(tmp_path):
    append_disclosure(
        _make_disclosure("6768.T", title="A", published_ts="2026-05-25T10:00:00+09:00"),
        base_dir=tmp_path,
    )
    append_disclosure(
        _make_disclosure("6768.T", title="B", published_ts="2026-05-27T10:00:00+09:00"),
        base_dir=tmp_path,
    )
    append_disclosure(
        _make_disclosure("6768.T", title="C", published_ts="2026-05-26T10:00:00+09:00"),
        base_dir=tmp_path,
    )
    out = read_disclosures_for_symbol(symbol="6768.T", base_dir=tmp_path)
    titles = [r.title for r in out]
    assert titles == ["B", "C", "A"]


def test_read_for_symbol_respects_asof_window(tmp_path):
    """asof_date + max_days_back limits the scan window."""
    append_disclosure(
        _make_disclosure("6768.T", title="recent", published_ts="2026-05-27T10:00:00+09:00"),
        base_dir=tmp_path,
    )
    # Disclosure from 10 days before asof — should be excluded with max_days_back=7
    append_disclosure(
        _make_disclosure("6768.T", title="old", published_ts="2026-05-17T10:00:00+09:00"),
        base_dir=tmp_path,
    )
    out = read_disclosures_for_symbol(
        symbol="6768.T", base_dir=tmp_path,
        asof_date="2026-05-27", max_days_back=7,
    )
    titles = [r.title for r in out]
    assert "recent" in titles
    assert "old" not in titles


# ─── API integration ─────────────────────────────────────────────────────


def test_profile_includes_recent_disclosures_field(monkeypatch):
    """/profile must surface recent_disclosures (Rule 11.4 structural data)."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/symbol/1306.T/profile")
    assert resp.status_code == 200
    payload = resp.json()
    assert "recent_disclosures" in payload
    assert isinstance(payload["recent_disclosures"], list)


# ─── Rule 8.3.1 / 11.4 — disclosures NOT in LLM prompt ──────────────────


def test_per_ticker_brief_does_not_consume_disclosures():
    """Per-ticker brief schema does NOT have a `disclosures` / `news_body` field,
    confirming structural news data is kept out of LLM grounding."""
    from hot_theme_rotator.llm.per_ticker_brief import PerTickerBriefInput
    fields = PerTickerBriefInput.__dataclass_fields__
    # Forward-compat fields are: news (Optional), factors, fundamentals
    # The `news` field accepts tuple of generic Mappings, but the prompt
    # builder does NOT auto-concatenate disclosure bodies. P2 design: news
    # remains opt-in via user_context, not auto-merged.
    # This test pins down that no field named `disclosures` / `news_body` exists.
    forbidden_fields = {"disclosures", "news_body", "tdnet_full_text"}
    actual_fields = set(fields.keys())
    assert actual_fields.isdisjoint(forbidden_fields), (
        f"P2 contract violation — LLM input must not auto-consume disclosures: "
        f"unexpected fields {actual_fields & forbidden_fields}"
    )
