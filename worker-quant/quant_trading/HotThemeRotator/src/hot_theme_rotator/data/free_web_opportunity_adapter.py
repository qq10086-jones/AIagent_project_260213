"""Free web data adapter for realtime opportunity inputs.

This adapter defines the internal boundary for free webpages, yfinance-like
clients, and public news sources. Network clients live outside the scoring
rules and only need to provide the small fetch methods used here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Protocol, Sequence

from hot_theme_rotator.common.schema import NewsItem, PriceBar
from hot_theme_rotator.opportunity.opportunity_scanner import OpportunityInput


class FreeWebDataValidationError(ValueError):
    """Raised when free-web data cannot safely feed the opportunity scanner."""


class QuoteClient(Protocol):
    def fetch_quotes(self, symbols: Sequence[str]) -> list["FreeWebQuote"]:
        """Fetch quote rows for symbols."""


class NewsClient(Protocol):
    def fetch_news(self, symbols: Sequence[str], since_ts: str, until_ts: str) -> list[NewsItem]:
        """Fetch point-in-time news rows for symbols."""


class ContextClient(Protocol):
    def fetch_context(self, symbols: Sequence[str]) -> dict[str, "FreeWebContextSnapshot"]:
        """Fetch market context rows keyed by symbol."""


@dataclass(frozen=True)
class RefreshSchedule:
    preopen_minutes: int = 10
    trading_minutes: int = 3
    lunch_minutes: int = 15
    post_close_minutes: int = 180
    overnight_minutes: int = 360

    def interval_minutes(self, asof_ts: str, *, event_trigger: bool = False) -> int:
        """Return recommended refresh interval for Japan market time."""
        if event_trigger:
            return 0
        asof = _parse_ts(asof_ts, "asof_ts").timetz().replace(tzinfo=None)
        if time(8, 0) <= asof < time(9, 0):
            return self.preopen_minutes
        if time(9, 0) <= asof < time(11, 30):
            return self.trading_minutes
        if time(11, 30) <= asof < time(12, 30):
            return self.lunch_minutes
        if time(12, 30) <= asof < time(15, 30):
            return self.trading_minutes
        if time(15, 30) <= asof < time(23, 0):
            return self.post_close_minutes
        return self.overnight_minutes


@dataclass(frozen=True)
class FreeWebQuote:
    symbol: str
    available_ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    previous_close: float
    avg_volume_20d: float

    def to_price_bar(self, asof: str) -> PriceBar:
        return PriceBar.from_dict(
            {
                "symbol": self.symbol,
                "asof": asof,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "turnover_jpy": self.close * self.volume,
            }
        )


@dataclass(frozen=True)
class FreeWebContextSnapshot:
    symbol: str
    available_ts: str
    market_context_score: float


class FreeWebOpportunityAdapter:
    """Build opportunity scanner inputs from free-web style clients."""

    def __init__(
        self,
        *,
        quote_client: QuoteClient,
        news_client: NewsClient,
        context_client: ContextClient | None = None,
        news_lookback_hours: int = 24,
    ) -> None:
        self.quote_client = quote_client
        self.news_client = news_client
        self.context_client = context_client
        self.news_lookback_hours = int(news_lookback_hours)

    def build_opportunity_inputs(
        self,
        *,
        symbols: Sequence[str],
        decision_cutoff: str,
    ) -> tuple[OpportunityInput, ...]:
        """Fetch and normalize free-web data into OpportunityInput rows."""
        cutoff = _parse_ts(decision_cutoff, "decision_cutoff")
        quotes = self.quote_client.fetch_quotes(symbols)
        news_items = self.news_client.fetch_news(symbols, since_ts="", until_ts=decision_cutoff)
        contexts = self.context_client.fetch_context(symbols) if self.context_client else {}

        quote_by_symbol = {quote.symbol: quote for quote in quotes}
        inputs: list[OpportunityInput] = []
        for symbol in symbols:
            quote = quote_by_symbol.get(symbol)
            if quote is None:
                continue
            _reject_after_cutoff(quote.available_ts, cutoff, f"{symbol} quote")
            symbol_news = _news_for_symbol(news_items, symbol)
            for item in symbol_news:
                _reject_after_cutoff(item.available_ts, cutoff, f"{symbol} news")
            context = contexts.get(symbol)
            if context is not None:
                _reject_after_cutoff(context.available_ts, cutoff, f"{symbol} context")

            theme, theme_score, news_score = _theme_from_news(symbol_news)
            bar = quote.to_price_bar(asof=decision_cutoff[:10])
            inputs.append(
                OpportunityInput(
                    bar=bar,
                    available_ts=quote.available_ts,
                    trigger_theme=theme,
                    theme_score=theme_score,
                    news_score=news_score,
                    relative_strength=_relative_strength(quote),
                    volume_ratio=_volume_ratio(quote),
                    liquidity_jpy=quote.close * quote.volume,
                    context_score=context.market_context_score if context else None,
                )
            )
        return tuple(inputs)


class YFinanceQuoteClient:
    """Quote client for yfinance-compatible modules.

    The dependency is optional at import time so tests and non-web workflows can
    use the adapter without network packages loaded.
    """

    def __init__(self, yf_module: object | None = None, clock: object | None = None) -> None:
        self.yf_module = yf_module
        self.clock = clock or _jst_now_iso

    def fetch_quotes(self, symbols: Sequence[str]) -> list[FreeWebQuote]:
        yf_module = self.yf_module or _load_yfinance()
        available_ts = self.clock() if callable(self.clock) else str(self.clock)
        quotes: list[FreeWebQuote] = []
        for symbol in symbols:
            ticker = yf_module.Ticker(symbol)
            history = ticker.history(period="21d", interval="1d")
            if getattr(history, "empty", False):
                continue
            close = float(history["Close"].iloc[-1])
            quotes.append(
                FreeWebQuote(
                    symbol=symbol,
                    available_ts=available_ts,
                    open=float(history["Open"].iloc[-1]),
                    high=float(history["High"].iloc[-1]),
                    low=float(history["Low"].iloc[-1]),
                    close=close,
                    volume=float(history["Volume"].iloc[-1]),
                    previous_close=float(history["Close"].iloc[-2])
                    if len(history["Close"].values) >= 2
                    else close,
                    avg_volume_20d=float(history["Volume"].tail(20).mean()),
                )
            )
        return quotes


def _theme_from_news(news_items: list[NewsItem]) -> tuple[str, float, float]:
    if not news_items:
        return "price_volume", 45.0, 0.0

    text = " ".join(
        f"{item.headline} {item.body}".lower()
        for item in news_items
    )
    theme_rules = [
        ("ai_semiconductor", ("ai", "semiconductor", "chip", "gpu"), 92.0, 0.85),
        ("fx_export", ("export", "weaker yen", "yen", "usd"), 72.0, 0.35),
        ("rate_sensitive_bank", ("bank", "rate", "yield", "jgb"), 76.0, 0.30),
        ("energy_commodity", ("oil", "gas", "energy", "commodity"), 72.0, 0.25),
        ("shareholder_return", ("buyback", "dividend", "repurchase"), 80.0, 0.50),
    ]
    for theme, keywords, theme_score, news_score in theme_rules:
        if any(_keyword_matches(text, keyword) for keyword in keywords):
            return theme, theme_score, news_score
    return "news_watch", 60.0, 0.15


def _keyword_matches(text: str, keyword: str) -> bool:
    if len(keyword) <= 2:
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


def _news_for_symbol(news_items: list[NewsItem], symbol: str) -> list[NewsItem]:
    market_wide = {"*", "ALL", "JP", "MARKET"}
    out: list[NewsItem] = []
    for item in news_items:
        symbols = {item_symbol.upper() for item_symbol in item.symbols}
        if symbol.upper() in symbols or symbols & market_wide:
            out.append(item)
    return out


def _relative_strength(quote: FreeWebQuote) -> float:
    if quote.previous_close <= 0:
        return 0.0
    return (quote.close - quote.previous_close) / quote.previous_close


def _volume_ratio(quote: FreeWebQuote) -> float:
    if quote.avg_volume_20d <= 0:
        return 0.0
    return quote.volume / quote.avg_volume_20d


def _reject_after_cutoff(value: str, cutoff: datetime, label: str) -> None:
    if _parse_ts(value, "available_ts") > cutoff:
        raise FreeWebDataValidationError(f"{label} available_ts is later than decision cutoff")


def _parse_ts(value: str, field: str) -> datetime:
    if not str(value).strip():
        raise FreeWebDataValidationError(f"{field} must be non-empty")
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FreeWebDataValidationError(f"{field} must be an ISO timestamp") from exc


def _load_yfinance() -> object:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise FreeWebDataValidationError("yfinance is required for YFinanceQuoteClient") from exc
    return yf


def _jst_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
