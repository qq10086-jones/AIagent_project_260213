"""Tests for P10-17 watchlist monitor wiring."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.watchlist_intelligence.monitor import run_watchlist_monitor  # noqa: E402
from hot_theme_rotator.watchlist_intelligence.silent_queue import read_silent_events  # noqa: E402


def test_monitor_detects_and_persists_silent_events(tmp_path):
    events = run_watchlist_monitor(
        watchlist=("6779.T",),
        trade_date="2026-05-26",
        created_ts="2026-05-26T09:00:00+09:00",
        price_health_rows=(),
        tdnet_disclosures=(
            {
                "ticker": "6779.T",
                "title": "業績予想の修正",
                "published_ts": "2026-05-26T08:30:00+09:00",
            },
        ),
        base_dir=tmp_path,
    )

    stored = read_silent_events("2026-05-26", base_dir=tmp_path)
    assert len(events) == 2
    assert stored == events
    assert all(event.push_allowed is False for event in stored)
