"""Tests for P10-17 silent watchlist intelligence queue."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.watchlist_intelligence.silent_queue import (  # noqa: E402
    SilentWatchlistEvent,
    append_silent_event,
    read_silent_events,
    silent_queue_path,
)


def test_silent_event_roundtrip_writes_jsonl_without_notifier(tmp_path):
    event = SilentWatchlistEvent(
        event_id="evt-1",
        trade_date="2026-05-26",
        symbol="6779.T",
        event_type="tdnet_disclosure",
        severity="info",
        reason="TDnet disclosure matched watchlist",
        created_ts="2026-05-26T09:00:00+09:00",
        source="tdnet",
        push_allowed=False,
        study_only=True,
    )

    path = append_silent_event(event, base_dir=tmp_path)
    rows = read_silent_events("2026-05-26", base_dir=tmp_path)

    assert path == tmp_path / "reports" / "observability" / "silent_queue" / "2026-05-26.jsonl"
    assert rows == (event,)
    assert rows[0].push_allowed is False


def test_silent_queue_path_rejects_non_iso_trade_date(tmp_path):
    with pytest.raises(ValueError, match="trade_date must be ISO"):
        silent_queue_path("2026/05/26", base_dir=tmp_path)


def test_silent_event_rejects_push_allowed_true():
    with pytest.raises(ValueError, match="silent queue cannot contain push_allowed=True"):
        SilentWatchlistEvent(
            event_id="evt-2",
            trade_date="2026-05-26",
            symbol="6779.T",
            event_type="quote_uncertain",
            severity="warning",
            reason="consensus unavailable",
            created_ts="2026-05-26T09:00:00+09:00",
            source="price_health",
            push_allowed=True,
        )


def test_silent_event_rejects_bad_symbol():
    with pytest.raises(ValueError, match="symbol must end with '.T'"):
        SilentWatchlistEvent(
            event_id="evt-3",
            trade_date="2026-05-26",
            symbol="6779",
            event_type="quote_unavailable",
            severity="warning",
            reason="missing quote",
            created_ts="2026-05-26T09:00:00+09:00",
            source="price_health",
        )
