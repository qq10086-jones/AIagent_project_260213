"""End-to-end daily briefing pipeline MVP."""
from __future__ import annotations

from dataclasses import dataclass

from hot_theme_rotator.common.schema import (
    MarketTemperature,
    NewsItem,
    PriceBar,
    TradingSignal,
)
from hot_theme_rotator.leader_ranking.leader_ranker import (
    LeaderCandidateInput,
    RankedLeader,
    rank_theme_leaders,
)
from hot_theme_rotator.market_temperature.external_temperature import (
    ExternalTemperatureAdjustment,
)
from hot_theme_rotator.market_temperature.japan_temperature import (
    JapanTemperatureInput,
    compute_japan_market_temperature,
)
from hot_theme_rotator.reporting.daily_briefing import (
    DailyBriefingInput,
    render_daily_briefing_markdown,
)
from hot_theme_rotator.risk.risk_governor import (
    PortfolioExposure,
    ProposedOrder,
    RiskConfig,
    evaluate_risk,
)
from hot_theme_rotator.signal_engine.signal_engine import SignalInput, generate_signal
from hot_theme_rotator.theme_detection.theme_detector import ThemeMatch, detect_themes


@dataclass(frozen=True)
class DailyPipelineInput:
    asof: str
    current_bars: list[PriceBar]
    previous_bars: list[PriceBar]
    news_items: list[NewsItem]
    leader_candidates: list[LeaderCandidateInput]
    reference_prices: dict[str, float]
    proposed_notional_by_symbol: dict[str, float]
    portfolio_exposure: PortfolioExposure
    external_adjustment: ExternalTemperatureAdjustment
    risk_config: RiskConfig | None = None


@dataclass(frozen=True)
class DailyPipelineResult:
    market_temperature: MarketTemperature
    theme_matches: list[ThemeMatch]
    leaders: list[RankedLeader]
    signals: list[TradingSignal]
    markdown: str


def run_daily_pipeline(payload: DailyPipelineInput) -> DailyPipelineResult:
    """Run the advice-only daily pipeline and render a Markdown briefing."""
    theme_matches = detect_themes(payload.news_items)
    market_temperature = compute_japan_market_temperature(
        JapanTemperatureInput(
            asof=payload.asof,
            current_bars=payload.current_bars,
            previous_bars=payload.previous_bars,
            hot_theme_count=len({match.theme_id for match in theme_matches}),
            opening_gap_down_pct=0.0,
        )
    )
    leaders = rank_theme_leaders(payload.leader_candidates)
    signals, missing_price_symbols = _generate_signals(
        payload.asof,
        leaders,
        payload.reference_prices,
        payload.proposed_notional_by_symbol,
        payload.portfolio_exposure,
        payload.risk_config or RiskConfig(),
        market_temperature,
        payload.external_adjustment,
    )
    risk_notes = ["No automatic execution. Advice-only."]
    if missing_price_symbols:
        risk_notes.append(f"Missing reference prices: {', '.join(missing_price_symbols)}")

    markdown = render_daily_briefing_markdown(
        DailyBriefingInput(
            asof=payload.asof,
            market_temperature=market_temperature,
            theme_matches=theme_matches,
            leaders=leaders,
            signals=signals,
            risk_notes=risk_notes,
        )
    )
    return DailyPipelineResult(
        market_temperature=market_temperature,
        theme_matches=theme_matches,
        leaders=leaders,
        signals=signals,
        markdown=markdown,
    )


def _generate_signals(
    asof: str,
    leaders: list[RankedLeader],
    reference_prices: dict[str, float],
    proposed_notional_by_symbol: dict[str, float],
    portfolio_exposure: PortfolioExposure,
    risk_config: RiskConfig,
    market_temperature: MarketTemperature,
    external_adjustment: ExternalTemperatureAdjustment,
) -> tuple[list[TradingSignal], list[str]]:
    signals: list[TradingSignal] = []
    missing: list[str] = []
    for leader in leaders:
        reference_price = reference_prices.get(leader.symbol)
        if reference_price is None:
            missing.append(leader.symbol)
            continue
        signal = generate_signal(
            SignalInput(
                asof=asof,
                leader=leader,
                reference_price=reference_price,
                market_temperature=market_temperature,
                external_adjustment=external_adjustment,
            )
        )
        if signal.action == "BUY":
            risk = evaluate_risk(
                ProposedOrder(
                    symbol=leader.symbol,
                    theme_id=leader.theme_id,
                    side="BUY",
                    notional_jpy=proposed_notional_by_symbol.get(leader.symbol, 0.0),
                ),
                portfolio_exposure,
                risk_config,
            )
            if risk.allowed:
                signal = _with_reason_codes(signal, [*signal.reason_codes, *risk.reason_codes])
            else:
                signal = _replace_signal_action(
                    signal,
                    "NO_TRADE",
                    [*signal.reason_codes, *risk.reason_codes],
                )
        signals.append(signal)
    return signals, missing


def _replace_signal_action(
    signal: TradingSignal,
    action: str,
    reason_codes: list[str],
) -> TradingSignal:
    return TradingSignal.from_dict(
        {
            "asof": signal.asof,
            "symbol": signal.symbol,
            "theme_id": signal.theme_id,
            "action": action,
            "entry_score": signal.entry_score,
            "reference_price": signal.reference_price,
            "target_profit_pct": signal.target_profit_pct,
            "take_profit_prices": signal.take_profit_prices,
            "stop_loss_price": signal.stop_loss_price,
            "max_hold_days": signal.max_hold_days,
            "reason_codes": tuple(dict.fromkeys(reason_codes)),
        }
    )


def _with_reason_codes(signal: TradingSignal, reason_codes: list[str]) -> TradingSignal:
    return _replace_signal_action(signal, signal.action, reason_codes)
