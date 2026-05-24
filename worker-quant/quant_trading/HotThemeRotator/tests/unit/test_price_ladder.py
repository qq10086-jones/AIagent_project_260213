import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder  # noqa: E402


def _bar() -> PriceBar:
    return PriceBar.from_dict(
        {
            "symbol": "8035.T",
            "asof": "2026-05-23",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 100.0,
            "volume": 1_000_000,
            "turnover_jpy": 100_000_000,
        }
    )


def test_build_price_ladder_outputs_three_entries_stop_and_three_exits():
    ladder = build_price_ladder(_bar())

    assert ladder.symbol == "8035.T"
    assert ladder.research_only is True
    assert ladder.method == "range_ladder_v1"
    assert ladder.aggressive_entry == 95.0
    assert ladder.balanced_entry == 90.0
    assert ladder.conservative_entry == 80.0
    assert ladder.stop_price == 70.0
    assert ladder.first_exit == 115.0
    assert ladder.second_exit == 125.0
    assert ladder.stretch_exit == 140.0
    assert ladder.stop_price < ladder.conservative_entry < ladder.balanced_entry < ladder.aggressive_entry
    assert ladder.aggressive_entry < ladder.first_exit < ladder.second_exit < ladder.stretch_exit


def test_build_price_ladder_uses_minimum_range_when_daily_range_is_too_small():
    tiny_range_bar = PriceBar.from_dict(
        {
            "symbol": "1306.T",
            "asof": "2026-05-23",
            "open": 1000.0,
            "high": 1001.0,
            "low": 999.0,
            "close": 1000.0,
            "volume": 1_000_000,
            "turnover_jpy": 1_000_000_000,
        }
    )

    ladder = build_price_ladder(tiny_range_bar)

    assert ladder.range_proxy == 20.0
    assert ladder.aggressive_entry == 995.0
    assert ladder.first_exit == 1015.0
