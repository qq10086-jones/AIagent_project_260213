import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reporting.paper_review import (  # noqa: E402
    PaperTradeRecord,
    PaperReviewSummary,
    summarize_paper_trades,
)


def test_paper_trade_record_computes_realized_return_when_exit_is_present():
    record = PaperTradeRecord.from_dict(
        {
            "signal_id": "sig-1",
            "symbol": "8035.T",
            "theme_id": "ai_semi",
            "entry_ts": "2026-05-19T10:00:00+09:00",
            "entry_price": 45_000,
            "exit_ts": "2026-05-20T10:00:00+09:00",
            "exit_price": 46_350,
            "exit_reason": "TAKE_PROFIT",
        }
    )

    assert record.realized_return == 0.03
    assert record.is_closed is True


def test_paper_trade_record_keeps_open_trade_without_return():
    record = PaperTradeRecord.from_dict(
        {
            "signal_id": "sig-2",
            "symbol": "7203.T",
            "theme_id": "auto_export",
            "entry_ts": "2026-05-19T10:00:00+09:00",
            "entry_price": 3100,
            "exit_ts": None,
            "exit_price": None,
            "exit_reason": None,
        }
    )

    assert record.realized_return is None
    assert record.is_closed is False


def test_summarize_paper_trades_reports_win_rate_average_win_loss_and_worst_loss():
    records = [
        PaperTradeRecord.from_dict(
            {
                "signal_id": "win-1",
                "symbol": "8035.T",
                "theme_id": "ai_semi",
                "entry_ts": "2026-05-19T10:00:00+09:00",
                "entry_price": 100,
                "exit_ts": "2026-05-19T14:00:00+09:00",
                "exit_price": 103,
                "exit_reason": "TAKE_PROFIT",
            }
        ),
        PaperTradeRecord.from_dict(
            {
                "signal_id": "win-2",
                "symbol": "7203.T",
                "theme_id": "auto_export",
                "entry_ts": "2026-05-19T10:00:00+09:00",
                "entry_price": 100,
                "exit_ts": "2026-05-20T10:00:00+09:00",
                "exit_price": 102,
                "exit_reason": "TAKE_PROFIT",
            }
        ),
        PaperTradeRecord.from_dict(
            {
                "signal_id": "loss-1",
                "symbol": "6501.T",
                "theme_id": "robotics",
                "entry_ts": "2026-05-19T10:00:00+09:00",
                "entry_price": 100,
                "exit_ts": "2026-05-20T10:00:00+09:00",
                "exit_price": 96,
                "exit_reason": "STOP_LOSS",
            }
        ),
    ]

    summary = summarize_paper_trades(records)

    assert isinstance(summary, PaperReviewSummary)
    assert summary.closed_trades == 3
    assert summary.open_trades == 0
    assert summary.win_rate == 2 / 3
    assert summary.average_win == 0.025
    assert summary.average_loss == -0.04
    assert summary.max_single_loss == -0.04


def test_summarize_paper_trades_ignores_open_trades_in_performance_stats():
    closed = PaperTradeRecord.from_dict(
        {
            "signal_id": "closed",
            "symbol": "8035.T",
            "theme_id": "ai_semi",
            "entry_ts": "2026-05-19T10:00:00+09:00",
            "entry_price": 100,
            "exit_ts": "2026-05-19T14:00:00+09:00",
            "exit_price": 105,
            "exit_reason": "TAKE_PROFIT",
        }
    )
    open_trade = PaperTradeRecord.from_dict(
        {
            "signal_id": "open",
            "symbol": "7203.T",
            "theme_id": "auto_export",
            "entry_ts": "2026-05-19T10:00:00+09:00",
            "entry_price": 100,
            "exit_ts": None,
            "exit_price": None,
            "exit_reason": None,
        }
    )

    summary = summarize_paper_trades([closed, open_trade])

    assert summary.closed_trades == 1
    assert summary.open_trades == 1
    assert summary.win_rate == 1.0
    assert summary.average_win == 0.05

