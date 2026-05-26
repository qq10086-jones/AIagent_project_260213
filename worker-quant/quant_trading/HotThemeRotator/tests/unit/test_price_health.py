"""Tests for P10-19 price source health checks."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.health import (  # noqa: E402
    PriceSourceHealth,
    price_health_report_path,
    read_price_health_report,
    run_price_source_health_checks,
    write_price_health_report,
)
from hot_theme_rotator.data.external.realtime_price.schema import (  # noqa: E402
    PriceQuote,
)


def _quote(source="yahoo_japan", *, inferred=False):
    return PriceQuote(
        symbol="6779.T",
        price=3015.0,
        source=source,
        data_ts="2026-05-26T09:05:00+09:00",
        wall_ts="2026-05-26T09:06:00+09:00",
        data_ts_inferred=inferred,
    )


def test_health_check_records_success_with_quote_metadata():
    checks = run_price_source_health_checks(
        "6779.T",
        [("yahoo_japan", lambda symbol: _quote("yahoo_japan"))],
        checked_ts="2026-05-26T09:07:00+09:00",
    )

    assert checks == (
        PriceSourceHealth(
            source="yahoo_japan",
            symbol="6779.T",
            ok=True,
            checked_ts="2026-05-26T09:07:00+09:00",
            price=3015.0,
            data_ts="2026-05-26T09:05:00+09:00",
            wall_ts="2026-05-26T09:06:00+09:00",
            data_ts_inferred=False,
            price_uncertain=False,
            fail_reason=None,
        ),
    )


def test_health_check_records_failure_without_raising():
    def failing_fetcher(symbol):
        raise RuntimeError("robots blocked")

    checks = run_price_source_health_checks(
        "6779.T",
        [("yahoo_japan", failing_fetcher)],
        checked_ts="2026-05-26T09:07:00+09:00",
    )

    assert len(checks) == 1
    assert checks[0].source == "yahoo_japan"
    assert checks[0].ok is False
    assert checks[0].price is None
    assert checks[0].fail_reason == "robots blocked"


def test_health_check_marks_inferred_timestamps_as_caveated():
    checks = run_price_source_health_checks(
        "6779.T",
        [("kabutan", lambda symbol: _quote("kabutan", inferred=True))],
        checked_ts="2026-05-26T09:07:00+09:00",
    )

    assert checks[0].ok is True
    assert checks[0].data_ts_inferred is True
    assert checks[0].freshness_caveat == "data_ts_inferred"


def test_health_check_rejects_mismatched_source_name():
    checks = run_price_source_health_checks(
        "6779.T",
        [("kabutan", lambda symbol: _quote("yahoo_japan"))],
        checked_ts="2026-05-26T09:07:00+09:00",
    )

    assert checks[0].ok is False
    assert "source mismatch" in (checks[0].fail_reason or "")


def test_write_and_read_price_health_report(tmp_path):
    rows = run_price_source_health_checks(
        "6779.T",
        [("yahoo_japan", lambda symbol: _quote("yahoo_japan"))],
        checked_ts="2026-05-26T09:07:00+09:00",
    )

    path = write_price_health_report(
        rows,
        trade_date="2026-05-26",
        base_dir=tmp_path,
    )

    assert path == tmp_path / "reports" / "observability" / "price_health" / "2026-05-26.json"
    loaded = read_price_health_report("2026-05-26", base_dir=tmp_path)
    assert loaded == rows


def test_price_health_report_path_rejects_non_iso_date(tmp_path):
    try:
        price_health_report_path("2026/05/26", base_dir=tmp_path)
    except ValueError as exc:
        assert "trade_date must be ISO date" in str(exc)
    else:
        raise AssertionError("expected ValueError")
