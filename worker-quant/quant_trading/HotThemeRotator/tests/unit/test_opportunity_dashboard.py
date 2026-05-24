import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.decision_log.schema import PredictionRecord  # noqa: E402
from hot_theme_rotator.decision_log.jsonl_writer import append_prediction  # noqa: E402
from hot_theme_rotator.opportunity.price_ladder import PriceLadder  # noqa: E402
from hot_theme_rotator.ui.opportunity_dashboard import (  # noqa: E402
    DashboardPanel,
    automation_roadmap_rows,
    build_calibration_badge,
    build_gate_progress_rows,
    build_panel_records,
    build_price_ladder_view,
    build_recent_predictions_view,
    build_retail_candidate_cards,
    build_retail_summary_metrics,
    build_sample_inputs,
    build_sample_panel,
    parse_symbols,
    refresh_interval_label,
)


def test_parse_symbols_accepts_commas_spaces_and_newlines():
    assert parse_symbols("8035.T, 7203.T\n8306.T 1306.T") == [
        "8035.T",
        "7203.T",
        "8306.T",
        "1306.T",
    ]


def test_build_sample_panel_returns_ranked_candidates_and_markdown():
    result = build_sample_panel(top_n=2)

    assert len(result.rows) == 2
    assert result.rows[0].candidate.symbol == "8035.T"
    assert "# Realtime Opportunity Candidate Panel" in result.markdown
    assert result.mode_label == "sample"


def test_build_panel_records_are_user_facing_and_include_price_ladders():
    result = build_sample_panel(top_n=1)
    records = build_panel_records(result.rows)

    assert records == [
        {
            "排名": 1,
            "股票": "8035.T",
            "触发主题": "ai_semiconductor",
            "机会分": 81.37,
            "状态": "未校准研究分",
            "激进买入": 44100.0,
            "均衡买入": 43200.0,
            "保守买入": 41400.0,
            "止损": 39600.0,
            "卖出1": 47700.0,
            "卖出2": 49500.0,
            "延伸卖出": 52200.0,
            "原因": "HOT_THEME, POSITIVE_NEWS, VOLUME_EXPANSION, LIQUID, SUPPORTIVE_CONTEXT",
            "数据缺口": "无",
        }
    ]


def test_refresh_interval_label_uses_configured_schedule():
    assert refresh_interval_label("2026-05-23T09:30:00+09:00") == "3 分钟"
    assert refresh_interval_label("2026-05-23T16:00:00+09:00") == "180 分钟"
    assert refresh_interval_label("2026-05-23T16:00:00+09:00", event_trigger=True) == "立即重算"


def test_build_retail_candidate_cards_translate_scores_reasons_and_risks():
    result = build_sample_panel(top_n=1)
    cards = build_retail_candidate_cards(result.rows)

    assert cards == [
        {
            "排名": 1,
            "股票": "8035.T",
            "主题": "AI半导体",
            "机会分": 81.37,
            "优先级": "高关注",
            "一句话判断": "高关注：等回落到均衡买入区再考虑",
            "买入区间": "41400.00 - 44100.00",
            "卖出区间": "47700.00 / 49500.00 / 52200.00",
            "止损价": "39600.00",
            "核心理由": "热门主题、正面新闻、成交放量、流动性充足、市场环境支持",
            "风险提示": "当前只是研究分，不是真实胜率；需要后续反馈校准。",
            "数据质量": "数据完整",
        }
    ]


def test_build_retail_summary_metrics_use_top_candidate_action():
    result = build_sample_panel(top_n=2)
    metrics = build_retail_summary_metrics(
        result,
        refresh_label=refresh_interval_label(result.asof),
    )

    assert metrics == {
        "候选数量": "2",
        "第一候选": "8035.T",
        "行动提示": "高关注：等回落到均衡买入区再考虑",
        "建议刷新": "3 分钟",
        "校准状态": "未校准",
    }


def test_automation_roadmap_rows_show_eight_section_10_gates():
    rows = automation_roadmap_rows()

    assert len(rows) == 8, "§10 defines exactly 8 staged gates"
    assert rows[0] == {
        "阶段": "1 候选发现",
        "当前状态": "已完成",
        "下一道门": "扩大股票池与免费新闻源",
    }
    # P9-01 is complete -> Gate 3 must show as done
    assert rows[2]["阶段"] == "3 决策日志"
    assert rows[2]["当前状态"] == "已完成"
    # Gate 8 (broker execution) is hard-blocked per §10 last paragraph
    assert rows[-1] == {
        "阶段": "8 实盘执行",
        "当前状态": "禁止中",
        "下一道门": "必须人工批准且通过 paper gate",
    }


def test_build_gate_progress_rows_aligned_with_p10_eight_gates():
    chips = build_gate_progress_rows()
    assert len(chips) == 8
    assert [chip["id"] for chip in chips] == ["1", "2", "3", "4", "5", "6", "7", "8"]
    # gate 3 is P9-01 (just completed)
    assert chips[2]["task_id"] == "P9-01"
    assert chips[2]["status"] == "done"
    # gate 6 is P9-04 human alerts
    assert chips[5]["task_id"] == "P9-04"
    assert chips[5]["status"] == "done"
    assert chips[2]["glyph"] == "✓"
    # gate 8 broker is hard-blocked, not just pending
    assert chips[-1]["task_id"] == "P9-06"
    assert chips[-1]["status"] == "blocked"
    assert chips[-1]["glyph"] == "⛔"


def test_build_price_ladder_view_orders_tiers_top_down_and_marks_deltas():
    ladder = PriceLadder(
        symbol="X",
        reference_price=1000.0,
        range_proxy=40.0,
        aggressive_entry=990.0,
        balanced_entry=980.0,
        conservative_entry=960.0,
        stop_price=940.0,
        first_exit=1030.0,
        second_exit=1050.0,
        stretch_exit=1080.0,
    )
    view = build_price_ladder_view(ladder, current_price=1000.0)

    assert [row["档位"] for row in view] == [
        "延伸卖出",
        "卖出 2",
        "卖出 1",
        "激进买入",
        "均衡买入",
        "保守买入",
        "止损",
    ]
    assert view[0]["价位"] == 1080.0
    assert view[0]["相对现价"] == pytest.approx(0.08)  # +8%
    assert view[-1]["价位"] == 940.0
    assert view[-1]["相对现价"] == pytest.approx(-0.06)  # -6%
    assert view[-1]["类型"] == "stop"


def test_build_price_ladder_view_rejects_non_positive_current_price():
    ladder = PriceLadder(
        symbol="X",
        reference_price=1000.0,
        range_proxy=40.0,
        aggressive_entry=990.0,
        balanced_entry=980.0,
        conservative_entry=960.0,
        stop_price=940.0,
        first_exit=1030.0,
        second_exit=1050.0,
        stretch_exit=1080.0,
    )
    with pytest.raises(ValueError, match="current_price"):
        build_price_ladder_view(ladder, current_price=0.0)


def test_build_calibration_badge_marks_uncalibrated_as_warning():
    badge = build_calibration_badge("uncalibrated_research_score")
    assert badge["level"] == "warning"
    assert "未校准" in badge["text"]
    assert "不是真实胜率" in badge["text"]

    insufficient = build_calibration_badge("insufficient_calibration")
    assert insufficient["level"] == "warning"
    assert "不是真实胜率" in insufficient["text"]


def test_build_calibration_badge_marks_calibrated_as_ok():
    badge = build_calibration_badge("calibrated_probability")
    assert badge["level"] == "ok"
    assert "已校准" in badge["text"]


def test_build_recent_predictions_view_returns_empty_when_log_missing(tmp_path):
    rows = build_recent_predictions_view(base_dir=tmp_path, trade_date="2026-05-23")
    assert rows == []


def test_dashboard_panel_trade_date_parses_iso_with_t_separator():
    panel = DashboardPanel(
        asof="2026-05-23T09:10:00+09:00",
        mode_label="x",
        rows=(),
        markdown="",
        data_notes=(),
    )
    assert panel.trade_date == "2026-05-23"


def test_dashboard_panel_trade_date_parses_iso_with_space_separator():
    """F10 — `'2026-05-23 09:10:00'` previously yielded the full string."""
    panel = DashboardPanel(
        asof="2026-05-23 09:10:00",
        mode_label="x",
        rows=(),
        markdown="",
        data_notes=(),
    )
    assert panel.trade_date == "2026-05-23"


def test_dashboard_panel_trade_date_accepts_date_only_asof():
    panel = DashboardPanel(
        asof="2026-05-23",
        mode_label="x",
        rows=(),
        markdown="",
        data_notes=(),
    )
    assert panel.trade_date == "2026-05-23"


def test_build_recent_predictions_view_renders_persisted_rows(tmp_path):
    record = PredictionRecord.build(
        symbol="8035.T",
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T09:10:00+09:00",
        input_snapshot_id="snap-1",
        model_version="opportunity-v0",
        score_status="uncalibrated_research_score",
        horizon_days=3,
        buy=0.81,
        sell=0.0,
        hold=0.19,
        extra={"opportunity_score": 81.37},
    )
    append_prediction(record, base_dir=tmp_path)

    rows = build_recent_predictions_view(base_dir=tmp_path, trade_date="2026-05-23")
    assert len(rows) == 1
    row = rows[0]
    assert row["符号"] == "8035.T"
    assert row["机会分"] == 81.37
    assert row["状态"] == "未校准研究分"
    assert row["模型"] == "opportunity-v0"
    assert row["prediction_id"].startswith("pred-")
    assert row["决策时刻"] == "2026-05-23T09:10:00+09:00"
