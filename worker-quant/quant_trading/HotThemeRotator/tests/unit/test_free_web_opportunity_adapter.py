import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import NewsItem  # noqa: E402
from hot_theme_rotator.data.free_web_opportunity_adapter import (  # noqa: E402
    FreeWebContextSnapshot,
    FreeWebDataValidationError,
    FreeWebOpportunityAdapter,
    FreeWebQuote,
    RefreshSchedule,
    YFinanceQuoteClient,
)


class FakeQuoteClient:
    def __init__(self, quotes):
        self.quotes = quotes

    def fetch_quotes(self, symbols):
        return [self.quotes[symbol] for symbol in symbols if symbol in self.quotes]


class FakeNewsClient:
    def __init__(self, news):
        self.news = news

    def fetch_news(self, symbols, since_ts, until_ts):
        return list(self.news)


class FakeContextClient:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def fetch_context(self, symbols):
        return {symbol: self.snapshots[symbol] for symbol in symbols if symbol in self.snapshots}


def _quote(symbol: str, close: float, available_ts: str = "2026-05-23T09:05:00+09:00"):
    return FreeWebQuote(
        symbol=symbol,
        available_ts=available_ts,
        open=close * 0.98,
        high=close * 1.03,
        low=close * 0.95,
        close=close,
        volume=1_000_000,
        previous_close=close * 0.97,
        avg_volume_20d=500_000,
    )


def _news(news_id: str, symbol: str, headline: str, available_ts: str = "2026-05-23T09:04:00+09:00"):
    return NewsItem.from_dict(
        {
            "news_id": news_id,
            "available_ts": available_ts,
            "source": "free-web",
            "headline": headline,
            "body": "",
            "symbols": [symbol],
        }
    )


def test_refresh_schedule_uses_intraday_and_post_close_frequencies():
    schedule = RefreshSchedule()

    assert schedule.interval_minutes("2026-05-23T08:30:00+09:00") == 10
    assert schedule.interval_minutes("2026-05-23T09:30:00+09:00") == 3
    assert schedule.interval_minutes("2026-05-23T12:00:00+09:00") == 15
    assert schedule.interval_minutes("2026-05-23T16:00:00+09:00") == 180
    assert schedule.interval_minutes("2026-05-23T23:30:00+09:00") == 360
    assert schedule.interval_minutes("2026-05-23T16:00:00+09:00", event_trigger=True) == 0


def test_adapter_converts_free_web_quotes_news_and_context_to_opportunity_inputs():
    adapter = FreeWebOpportunityAdapter(
        quote_client=FakeQuoteClient(
            {
                "8035.T": _quote("8035.T", 45000),
                "7203.T": _quote("7203.T", 3000),
            }
        ),
        news_client=FakeNewsClient(
            [
                _news("n1", "8035.T", "AI semiconductor demand expands"),
                _news("n2", "7203.T", "Exporters gain from weaker yen"),
            ]
        ),
        context_client=FakeContextClient(
            {
                "8035.T": FreeWebContextSnapshot(
                    symbol="8035.T",
                    available_ts="2026-05-23T09:03:00+09:00",
                    market_context_score=0.30,
                ),
                "7203.T": FreeWebContextSnapshot(
                    symbol="7203.T",
                    available_ts="2026-05-23T09:03:00+09:00",
                    market_context_score=0.10,
                ),
            }
        ),
    )

    inputs = adapter.build_opportunity_inputs(
        symbols=["8035.T", "7203.T"],
        decision_cutoff="2026-05-23T09:10:00+09:00",
    )

    by_symbol = {item.bar.symbol: item for item in inputs}
    assert by_symbol["8035.T"].trigger_theme == "ai_semiconductor"
    assert by_symbol["8035.T"].theme_score > by_symbol["7203.T"].theme_score
    assert by_symbol["8035.T"].news_score > 0
    assert by_symbol["8035.T"].relative_strength == pytest.approx((45000 - 43650) / 43650)
    assert by_symbol["8035.T"].volume_ratio == pytest.approx(2.0)
    assert by_symbol["8035.T"].liquidity_jpy == pytest.approx(45_000_000_000)
    assert by_symbol["8035.T"].context_score == pytest.approx(0.30)


def test_adapter_rejects_quote_news_or_context_after_decision_cutoff():
    with pytest.raises(FreeWebDataValidationError, match="later than decision cutoff"):
        FreeWebOpportunityAdapter(
            quote_client=FakeQuoteClient(
                {"8035.T": _quote("8035.T", 45000, "2026-05-23T09:11:00+09:00")}
            ),
            news_client=FakeNewsClient([]),
        ).build_opportunity_inputs(
            symbols=["8035.T"],
            decision_cutoff="2026-05-23T09:10:00+09:00",
        )

    with pytest.raises(FreeWebDataValidationError, match="later than decision cutoff"):
        FreeWebOpportunityAdapter(
            quote_client=FakeQuoteClient({"8035.T": _quote("8035.T", 45000)}),
            news_client=FakeNewsClient(
                [_news("late", "8035.T", "late AI headline", "2026-05-23T09:11:00+09:00")]
            ),
        ).build_opportunity_inputs(
            symbols=["8035.T"],
            decision_cutoff="2026-05-23T09:10:00+09:00",
        )

    with pytest.raises(FreeWebDataValidationError, match="later than decision cutoff"):
        FreeWebOpportunityAdapter(
            quote_client=FakeQuoteClient({"8035.T": _quote("8035.T", 45000)}),
            news_client=FakeNewsClient([]),
            context_client=FakeContextClient(
                {
                    "8035.T": FreeWebContextSnapshot(
                        symbol="8035.T",
                        available_ts="2026-05-23T09:11:00+09:00",
                        market_context_score=0.30,
                    )
                }
            ),
        ).build_opportunity_inputs(
            symbols=["8035.T"],
            decision_cutoff="2026-05-23T09:10:00+09:00",
        )


def test_adapter_marks_missing_context_without_blocking_candidate_creation():
    adapter = FreeWebOpportunityAdapter(
        quote_client=FakeQuoteClient({"8306.T": _quote("8306.T", 1700)}),
        news_client=FakeNewsClient([_news("bank", "8306.T", "Bank shares rise with rate expectations")]),
    )

    inputs = adapter.build_opportunity_inputs(
        symbols=["8306.T"],
        decision_cutoff="2026-05-23T09:10:00+09:00",
    )

    assert len(inputs) == 1
    assert inputs[0].context_score is None
    assert inputs[0].trigger_theme == "rate_sensitive_bank"


def test_yfinance_quote_client_converts_history_to_free_web_quotes_without_network():
    class FakeSeries:
        def __init__(self, values):
            self.values = values

        @property
        def iloc(self):
            return self

        def __getitem__(self, index):
            return self.values[index]

        def tail(self, size):
            return FakeSeries(self.values[-size:])

        def mean(self):
            return sum(self.values) / len(self.values)

    class FakeHistory:
        empty = False

        def __getitem__(self, key):
            data = {
                "Open": [98.0, 99.0],
                "High": [103.0, 104.0],
                "Low": [95.0, 96.0],
                "Close": [97.0, 100.0],
                "Volume": [400_000, 1_000_000],
            }
            return FakeSeries(data[key])

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            assert period == "21d"
            assert interval == "1d"
            return FakeHistory()

    class FakeYFinance:
        def Ticker(self, symbol):
            return FakeTicker(symbol)

    client = YFinanceQuoteClient(
        yf_module=FakeYFinance(),
        clock=lambda: "2026-05-23T09:05:00+09:00",
    )

    quotes = client.fetch_quotes(["8035.T"])

    assert quotes == [
        FreeWebQuote(
            symbol="8035.T",
            available_ts="2026-05-23T09:05:00+09:00",
            open=99.0,
            high=104.0,
            low=96.0,
            close=100.0,
            volume=1_000_000.0,
            previous_close=97.0,
            avg_volume_20d=700_000.0,
        )
    ]
