import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.risk.risk_governor import (  # noqa: E402
    PortfolioExposure,
    RiskConfig,
    ProposedOrder,
    evaluate_risk,
)


def test_blocks_order_when_single_position_would_exceed_limit():
    result = evaluate_risk(
        ProposedOrder(symbol="8035.T", theme_id="ai_semi", side="BUY", notional_jpy=200_000),
        PortfolioExposure(
            nav_jpy=1_000_000,
            position_notional_by_symbol={"8035.T": 100_000},
            theme_notional_by_theme={"ai_semi": 100_000},
            total_long_notional=300_000,
        ),
        RiskConfig(max_position_nav_pct=0.15),
    )

    assert result.allowed is False
    assert result.action == "REDUCE"
    assert "POSITION_LIMIT_EXCEEDED" in result.reason_codes


def test_blocks_order_when_theme_exposure_would_exceed_limit():
    result = evaluate_risk(
        ProposedOrder(symbol="6857.T", theme_id="ai_semi", side="BUY", notional_jpy=150_000),
        PortfolioExposure(
            nav_jpy=1_000_000,
            position_notional_by_symbol={"8035.T": 120_000},
            theme_notional_by_theme={"ai_semi": 350_000},
            total_long_notional=500_000,
        ),
        RiskConfig(max_theme_nav_pct=0.40),
    )

    assert result.allowed is False
    assert result.action == "BLOCK"
    assert "THEME_LIMIT_EXCEEDED" in result.reason_codes


def test_blocks_order_when_total_long_exposure_would_exceed_limit():
    result = evaluate_risk(
        ProposedOrder(symbol="7203.T", theme_id="auto_export", side="BUY", notional_jpy=200_000),
        PortfolioExposure(
            nav_jpy=1_000_000,
            position_notional_by_symbol={},
            theme_notional_by_theme={},
            total_long_notional=750_000,
        ),
        RiskConfig(max_total_long_nav_pct=0.80),
    )

    assert result.allowed is False
    assert result.action == "BLOCK"
    assert "TOTAL_LONG_LIMIT_EXCEEDED" in result.reason_codes


def test_allows_order_within_limits_and_reports_remaining_capacity():
    result = evaluate_risk(
        ProposedOrder(symbol="7203.T", theme_id="auto_export", side="BUY", notional_jpy=100_000),
        PortfolioExposure(
            nav_jpy=1_000_000,
            position_notional_by_symbol={},
            theme_notional_by_theme={"auto_export": 100_000},
            total_long_notional=300_000,
        ),
        RiskConfig(),
    )

    assert result.allowed is True
    assert result.action == "ALLOW"
    assert result.remaining_position_capacity_jpy == 50_000
    assert result.remaining_theme_capacity_jpy == 200_000
    assert result.remaining_total_long_capacity_jpy == 400_000
    assert result.reason_codes == ("RISK_OK",)


def test_fails_closed_when_nav_is_not_positive():
    result = evaluate_risk(
        ProposedOrder(symbol="7203.T", theme_id="auto_export", side="BUY", notional_jpy=100_000),
        PortfolioExposure(
            nav_jpy=0,
            position_notional_by_symbol={},
            theme_notional_by_theme={},
            total_long_notional=0,
        ),
        RiskConfig(),
    )

    assert result.allowed is False
    assert result.action == "BLOCK"
    assert "INVALID_NAV" in result.reason_codes

