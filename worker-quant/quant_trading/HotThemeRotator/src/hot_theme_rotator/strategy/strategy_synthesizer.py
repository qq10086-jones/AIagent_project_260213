"""Strategy Card synthesis — Rule 11.6 governed Strategy / Action output.

Combines:
- 7-tier price ladder (opportunity/price_ladder)
- Rule 12 alert discipline (chase / cooldown / concentration / stale)
- Catalyst calendar placeholder (P10-09 future, currently empty)
- score_status from calibration gate

Rule 11.6 hard contract — the returned ``StrategyCard``:
1. Always carries an advice-only banner string with literal "Rule 3"
2. Always carries the literal "Rule 3 — manual execution outside HTR" disclaimer
3. Risk warnings section is present (may list "no_active_warnings" when none)
4. When ``score_status="uncalibrated_research_score"``, no "建议买入" phrasing —
   only factual ladder language ("达 ladder 保守档")
5. Price target labels use "建议价位" / "目标参考" / "止损参考"
6. Catalyst block carries "结构化日历数据，非建议持有至该日"
7. Never returns broker fields (account, order_id, submit_endpoint, etc.)

This module is a pure function — no I/O, no globals. Callers
(api/symbol.py) fetch the inputs and pass them in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from hot_theme_rotator.alerts.discipline import (
    AlertDisciplineConfig,
    AlertDisciplineInput,
    evaluate_alert_discipline,
)
from hot_theme_rotator.opportunity.price_ladder import PriceLadder


__all__ = [
    "RiskWarning",
    "StrategyCard",
    "StrategySynthesisError",
    "StrategySynthesisInput",
    "synthesize_strategy_card",
]


# Rule 11.6.2 forbidden imperative-against-system tokens (M1 fix — extended).
# Any of these in user-facing text fails Rule 11.6 contract.
RULE_11_6_FORBIDDEN_TOKENS = (
    # Chinese imperatives (simplified + traditional)
    "下单", "下單", "执行交易", "執行交易", "执行下单", "執行下單",
    "自动交易", "自動交易", "自动下单", "自動下單",
    "自动买入", "自動買入", "自动卖出", "自動賣出",
    # English imperatives
    "auto-trade", "auto trade", "autotrade",
    "place order", "place an order", "place trade",
    "submit order", "submit trade",
    "execute trade", "execute order",
    "send order", "send to broker",
)

# Required disclaimer strings (Rule 11.6.3 + 11.6.7).
RULE_3_DISCLAIMER = "Rule 3 — manual execution outside HTR"
CATALYST_DISCLAIMER = "结构化日历数据，非建议持有至该日"
ADVICE_ONLY_BANNER = (
    "advice-only · 由用户在外部券商手动执行 · Rule 3"
)


class StrategySynthesisError(ValueError):
    """Raised when synthesis cannot produce a Rule 11.6-compliant card."""


@dataclass(frozen=True)
class RiskWarning:
    """A single risk warning surfaced on the Strategy Card."""

    code: str         # e.g. "chase_filter", "cooling_off", "concentration_guard", "stale_data"
    severity: str     # "info" | "warn" | "block"
    message: str      # user-facing Chinese message
    rule_ref: str     # e.g. "Rule 12.3"


@dataclass(frozen=True)
class StrategySynthesisInput:
    """All non-derived inputs the synthesizer needs."""

    ticker: str
    current_price: float
    ladder: PriceLadder
    # Optional context for risk warnings — when None, the corresponding
    # warning is just not generated (e.g. no portfolio = no concentration check).
    intraday_move_pct: Optional[float] = None
    data_ts: Optional[str] = None  # ISO 8601 of the price's asof
    now_ts: Optional[str] = None   # ISO 8601 of the current decision time
    watchlist_added_ts: Optional[str] = None
    post_trade_concentration_pct: Optional[float] = None
    daily_budget_used: int = 0
    score_status: str = "uncalibrated_research_score"

    def __post_init__(self) -> None:
        if not self.ticker.endswith(".T"):
            raise StrategySynthesisError(
                f"ticker must end with '.T', got {self.ticker!r}"
            )
        if self.current_price <= 0:
            raise StrategySynthesisError(
                f"current_price must be > 0, got {self.current_price!r}"
            )


@dataclass(frozen=True)
class StrategyCard:
    """User-facing Strategy Card payload (Rule 11.6 compliant)."""

    ticker: str
    current_price: float
    advice_only: bool
    banner: str
    rule_3_disclaimer: str
    score_status: str
    ladder_tiers: tuple[dict, ...]
    risk_warnings: tuple[RiskWarning, ...]
    catalyst_calendar: tuple[dict, ...]
    catalyst_disclaimer: str
    # Rule 11.6 contract markers — present so the contract test can assert
    # they were not stripped/overridden.
    rule_11_6_markers: dict = field(default_factory=lambda: {
        "banner_present": True,
        "rule_3_disclaimer_present": True,
        "risk_warnings_section_present": True,
        "catalyst_disclaimer_present": True,
    })

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "current_price": self.current_price,
            "advice_only": self.advice_only,
            "banner": self.banner,
            "rule_3_disclaimer": self.rule_3_disclaimer,
            "score_status": self.score_status,
            "ladder_tiers": list(self.ladder_tiers),
            "risk_warnings": [
                {
                    "code": w.code,
                    "severity": w.severity,
                    "message": w.message,
                    "rule_ref": w.rule_ref,
                }
                for w in self.risk_warnings
            ],
            "catalyst_calendar": list(self.catalyst_calendar),
            "catalyst_disclaimer": self.catalyst_disclaimer,
            "rule_11_6_markers": dict(self.rule_11_6_markers),
        }


def synthesize_strategy_card(
    payload: StrategySynthesisInput,
    *,
    config: AlertDisciplineConfig = AlertDisciplineConfig(),
) -> StrategyCard:
    """Compose ladder + risk + catalyst into a Rule 11.6-compliant card."""

    ladder = payload.ladder
    # Tiers expose label + kind + price + pct_vs_ref so the UI maps them
    # 1:1 with the existing LadderMini renderer. Labels stay factual.
    tiers = (
        {"kind": "exit_stretch", "label": "目标参考·延伸卖出", "price": ladder.stretch_exit,
         "pct_vs_ref": _pct(ladder.stretch_exit, ladder.reference_price)},
        {"kind": "exit_2", "label": "目标参考·卖出 2", "price": ladder.second_exit,
         "pct_vs_ref": _pct(ladder.second_exit, ladder.reference_price)},
        {"kind": "exit_1", "label": "目标参考·首止盈", "price": ladder.first_exit,
         "pct_vs_ref": _pct(ladder.first_exit, ladder.reference_price)},
        {"kind": "entry_aggressive", "label": "建议价位·激进档", "price": ladder.aggressive_entry,
         "pct_vs_ref": _pct(ladder.aggressive_entry, ladder.reference_price)},
        {"kind": "entry_balanced", "label": "建议价位·均衡档", "price": ladder.balanced_entry,
         "pct_vs_ref": _pct(ladder.balanced_entry, ladder.reference_price)},
        {"kind": "entry_conservative", "label": "建议价位·保守档", "price": ladder.conservative_entry,
         "pct_vs_ref": _pct(ladder.conservative_entry, ladder.reference_price)},
        {"kind": "stop", "label": "止损参考", "price": ladder.stop_price,
         "pct_vs_ref": _pct(ladder.stop_price, ladder.reference_price)},
    )

    warnings = _derive_risk_warnings(payload, config=config)

    catalyst: tuple[dict, ...] = ()  # P10-09 placeholder — empty until earnings calendar lands.

    card = StrategyCard(
        ticker=payload.ticker,
        current_price=payload.current_price,
        advice_only=True,
        banner=ADVICE_ONLY_BANNER,
        rule_3_disclaimer=RULE_3_DISCLAIMER,
        score_status=payload.score_status,
        ladder_tiers=tiers,
        risk_warnings=warnings,
        catalyst_calendar=catalyst,
        catalyst_disclaimer=CATALYST_DISCLAIMER,
    )

    _enforce_rule_11_6(card)
    return card


def _derive_risk_warnings(
    payload: StrategySynthesisInput,
    *,
    config: AlertDisciplineConfig,
) -> tuple[RiskWarning, ...]:
    """Compute Rule 12 discipline warnings for a hypothetical BUY today."""
    warnings: list[RiskWarning] = []

    if (
        payload.intraday_move_pct is None
        or payload.data_ts is None
        or payload.now_ts is None
    ):
        # Insufficient context — surface a single "context missing" info
        # warning rather than silently omit. Rule 11.6.4 requires the
        # risk_warnings section to be present and non-misleading.
        warnings.append(RiskWarning(
            code="context_missing",
            severity="info",
            message="风险检查上下文不全（缺日内涨跌或时间戳）",
            rule_ref="Rule 12",
        ))
        return tuple(warnings)

    discipline_input = AlertDisciplineInput(
        symbol=payload.ticker,
        action="BUY",
        created_ts=payload.now_ts,
        data_ts=payload.data_ts,
        intraday_move_pct=payload.intraday_move_pct,
        daily_budget_used=payload.daily_budget_used,
        watchlist_added_ts=payload.watchlist_added_ts,
        post_trade_concentration_pct=(
            payload.post_trade_concentration_pct or 0.0
        ),
    )
    decision = evaluate_alert_discipline(discipline_input, config=config)

    reason_map = {
        "budget_exhausted": (
            "warn", "今日告警预算已用完", "Rule 12.1",
        ),
        "stale_data": (
            "block", f"行情数据 stale > {config.stale_threshold_hours}h", "Rule 12.2",
        ),
        "chase_filter": (
            "block",
            f"日内已涨 {payload.intraday_move_pct:.1f}% > {config.chase_threshold_pct:.0f}%, "
            "study-only · 不建议追涨",
            "Rule 12.3",
        ),
        "cooling_off": (
            f"warn", "新加 watchlist 冷却期内（<24h）, BUY 提示静默",
            "Rule 12.4",
        ),
        "concentration_guard": (
            "block",
            f"假设 BUY 后单仓位 > {config.concentration_max_pct:.0f}% NAV",
            "Rule 12.5",
        ),
    }
    for code in decision.reasons:
        sev, msg, rule = reason_map.get(
            code, ("warn", f"discipline: {code}", "Rule 12"),
        )
        warnings.append(RiskWarning(
            code=code, severity=sev, message=msg, rule_ref=rule,
        ))

    if not warnings:
        warnings.append(RiskWarning(
            code="no_active_warnings",
            severity="info",
            message="当前无激活的风险触发",
            rule_ref="Rule 12",
        ))
    return tuple(warnings)


def _enforce_rule_11_6(card: StrategyCard) -> None:
    """Guard that the synthesized card satisfies Rule 11.6 hard requirements."""
    # 11.6.1 banner present
    if not card.banner or "Rule 3" not in card.banner:
        raise StrategySynthesisError("banner missing Rule 3 marker")
    if "advice-only" not in card.banner:
        raise StrategySynthesisError("banner missing advice-only marker")
    # 11.6.3 disclaimer literal
    if card.rule_3_disclaimer != RULE_3_DISCLAIMER:
        raise StrategySynthesisError(
            f"rule_3_disclaimer must equal {RULE_3_DISCLAIMER!r}"
        )
    # 11.6.4 risk_warnings present
    if not card.risk_warnings:
        raise StrategySynthesisError("risk_warnings must be non-empty")
    # 11.6.7 catalyst disclaimer
    if card.catalyst_disclaimer != CATALYST_DISCLAIMER:
        raise StrategySynthesisError(
            f"catalyst_disclaimer must equal {CATALYST_DISCLAIMER!r}"
        )
    # 11.6.6 price label discipline — every tier label must use approved nouns
    approved_label_prefixes = (
        "目标参考", "建议价位", "止损参考",
    )
    for tier in card.ladder_tiers:
        label = str(tier.get("label", ""))
        if not any(label.startswith(p) for p in approved_label_prefixes):
            raise StrategySynthesisError(
                f"ladder tier label must start with one of "
                f"{approved_label_prefixes}, got {label!r}"
            )
    # 11.6.5 uncalibrated language guard — when score_status is uncalibrated,
    # the card MUST NOT contain "建议买入" anywhere in user-facing text.
    if card.score_status == "uncalibrated_research_score":
        flat = " ".join(
            [card.banner, card.rule_3_disclaimer, card.catalyst_disclaimer]
            + [str(t.get("label", "")) for t in card.ladder_tiers]
            + [w.message for w in card.risk_warnings]
        )
        if "建议买入" in flat:
            raise StrategySynthesisError(
                "uncalibrated score_status forbids '建议买入' phrasing"
            )
    # 11.6.2 forbidden imperative tokens (M1 fix — use centralized list).
    flat_all = " ".join(
        [card.banner, card.rule_3_disclaimer, card.catalyst_disclaimer]
        + [str(t.get("label", "")) for t in card.ladder_tiers]
        + [w.message for w in card.risk_warnings]
    )
    flat_lower = flat_all.lower()
    for token in RULE_11_6_FORBIDDEN_TOKENS:
        # English tokens checked case-insensitive; Chinese tokens are
        # case-irrelevant but the literal compare against flat_all still works.
        token_lower = token.lower()
        if any("一" <= c <= "鿿" for c in token):
            # CJK token — exact match in raw text
            hit = token in flat_all
        else:
            # English token — match against lowercase flat
            hit = token_lower in flat_lower
        if hit:
            raise StrategySynthesisError(
                f"forbidden imperative token {token!r} in card text (Rule 11.6.2)"
            )


def _pct(price: float, ref: float) -> float:
    if ref == 0:
        return 0.0
    return round(100.0 * (float(price) - float(ref)) / float(ref), 4)
