"""Data preparation helpers for the local opportunity dashboard."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hot_theme_rotator.common.schema import NewsItem
from hot_theme_rotator.data.free_web_opportunity_adapter import (
    FreeWebContextSnapshot,
    FreeWebOpportunityAdapter,
    FreeWebQuote,
    RefreshSchedule,
    YFinanceQuoteClient,
)
from hot_theme_rotator.decision_log.jsonl_writer import read_predictions
from hot_theme_rotator.opportunity.opportunity_scanner import OpportunityInput
from hot_theme_rotator.opportunity.price_ladder import PriceLadder, build_price_ladder
from hot_theme_rotator.reporting.realtime_opportunity_panel import (
    OpportunityPanelRow,
    build_realtime_opportunity_panel_markdown,
    persist_panel_predictions,
    render_realtime_opportunity_panel_markdown_v2,
)


# §10 the 8 staged gates, single source for both the chip strip and the
# automation roadmap table.
_GATE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {"id": "1", "label": "候选发现", "task_id": "P8-01", "status": "done",
     "next_gate": "扩大股票池与免费新闻源"},
    {"id": "2", "label": "阶梯生成", "task_id": "P8-02", "status": "done",
     "next_gate": "把候选与七档价位逐行落盘"},
    {"id": "3", "label": "决策日志", "task_id": "P9-01", "status": "done",
     "next_gate": "P9-02 接入 1D/3D/5D 实际结果"},
    {"id": "4", "label": "反馈回写", "task_id": "P9-02", "status": "done",
     "next_gate": "需 LegacyDailyPriceFetcher 接入真实 OHLC 才能填充 outcomes/"},
    {"id": "5", "label": "校准", "task_id": "P9-03", "status": "done",
     "next_gate": "数学层就绪；需 ≥100 配对样本流入后才能输出 calibrated_probability"},
    {"id": "6", "label": "提醒", "task_id": "P9-04", "status": "done",
     "next_gate": "校准前只提醒，不下单"},
    {"id": "7", "label": "纸面交易", "task_id": "P9-05", "status": "pending",
     "next_gate": "通过风险限制与复盘后才进入实盘"},
    {"id": "8", "label": "实盘执行", "task_id": "P9-06", "status": "blocked",
     "next_gate": "必须人工批准且通过 paper gate"},
)


_STATUS_LABELS = {
    "done": "已完成",
    "in_progress": "进行中",
    "pending": "待开发",
    "blocked": "禁止中",
}

_STATUS_GLYPHS = {
    "done": "✓",
    "in_progress": "◐",
    "pending": "○",
    "blocked": "⛔",
}


DEFAULT_ASOF = "2026-05-23T09:10:00+09:00"


@dataclass(frozen=True)
class DashboardPanel:
    asof: str
    mode_label: str
    rows: tuple[OpportunityPanelRow, ...]
    markdown: str
    data_notes: tuple[str, ...]
    markdown_v2: str = ""

    @property
    def trade_date(self) -> str:
        """Return the YYYY-MM-DD portion of `asof`, ISO-parsed.

        F10 — the previous `split("T")` implementation silently mishandled the
        ISO 8601 space separator (`"2026-05-23 09:10:00"` → full string with
        time) and any other non-T variant. Now we parse properly and only fall
        back to the raw string if `asof` is already date-only.
        """
        raw = str(self.asof or "")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            # Already a date-only string (no time component) or unparseable —
            # last-resort split, kept for backwards compatibility with callers
            # that pass YYYY-MM-DD directly.
            return raw.split("T", 1)[0].split(" ", 1)[0]


def parse_symbols(raw: str) -> list[str]:
    """Parse a user-entered symbol list from commas, whitespace, or newlines."""
    return [item.strip().upper() for item in re.split(r"[\s,]+", raw) if item.strip()]


def build_sample_inputs() -> tuple[OpportunityInput, ...]:
    """Return deterministic demo data for first-time users."""
    adapter = FreeWebOpportunityAdapter(
        quote_client=_SampleQuoteClient(),
        news_client=_SampleNewsClient(),
        context_client=_SampleContextClient(),
    )
    return adapter.build_opportunity_inputs(
        symbols=["8035.T", "7203.T"],
        decision_cutoff=DEFAULT_ASOF,
    )


def build_sample_panel(top_n: int = 10) -> DashboardPanel:
    """Build the sample candidate panel shown on first launch."""
    inputs = build_sample_inputs()
    return build_panel_from_inputs(
        asof=DEFAULT_ASOF,
        inputs=inputs,
        top_n=top_n,
        mode_label="sample",
        data_notes=("样例数据，不访问网络", "分数未校准，不代表真实胜率"),
    )


def build_yfinance_quote_panel(
    *,
    symbols: list[str],
    asof: str,
    top_n: int,
    persist_base_dir: Path | str | None = None,
) -> DashboardPanel:
    """Build a quote-only panel using yfinance-compatible data.

    If `persist_base_dir` is provided, every panel row is written to the §8.6
    decision log under that directory (P9-01 / §10 gate 3).
    """
    adapter = FreeWebOpportunityAdapter(
        quote_client=YFinanceQuoteClient(clock=lambda: asof),
        news_client=_EmptyNewsClient(),
    )
    inputs = adapter.build_opportunity_inputs(symbols=symbols, decision_cutoff=asof)
    return build_panel_from_inputs(
        asof=asof,
        inputs=inputs,
        top_n=top_n,
        mode_label="yfinance_quote_only",
        data_notes=("免费行情模式：当前只接行情，新闻和外部环境可能缺失",),
        persist_base_dir=persist_base_dir,
    )


def build_panel_from_inputs(
    *,
    asof: str,
    inputs: tuple[OpportunityInput, ...] | list[OpportunityInput],
    top_n: int,
    mode_label: str,
    data_notes: tuple[str, ...],
    persist_base_dir: Path | str | None = None,
) -> DashboardPanel:
    """Build dashboard records and Markdown from normalized opportunity inputs.

    When `persist_base_dir` is provided, panel rows are written to the §8.6
    decision log at `{persist_base_dir}/reports/predictions/{trade_date}.jsonl`.
    Sample/demo callers may leave it unset to keep tests and exploration write-free.
    """
    markdown = build_realtime_opportunity_panel_markdown(asof=asof, inputs=inputs, top_n=top_n)
    from hot_theme_rotator.opportunity.opportunity_scanner import scan_opportunities

    scan = scan_opportunities(inputs=inputs, decision_cutoff=asof, top_n=top_n)
    bar_by_symbol = {item.bar.symbol: item.bar for item in inputs}
    rows = tuple(
        OpportunityPanelRow(
            candidate=candidate,
            ladder=build_price_ladder(bar_by_symbol[candidate.symbol]),
        )
        for candidate in scan.candidates
    )
    if persist_base_dir is not None and rows:
        persist_panel_predictions(rows, base_dir=persist_base_dir)
    markdown_v2 = render_realtime_opportunity_panel_markdown_v2(asof=asof, rows=rows)
    return DashboardPanel(
        asof=asof,
        mode_label=mode_label,
        rows=rows,
        markdown=markdown,
        data_notes=data_notes,
        markdown_v2=markdown_v2,
    )


def build_panel_records(rows: tuple[OpportunityPanelRow, ...] | list[OpportunityPanelRow]) -> list[dict[str, object]]:
    """Convert panel rows into a Chinese, user-facing table shape."""
    records: list[dict[str, object]] = []
    for row in rows:
        candidate = row.candidate
        ladder = row.ladder
        records.append(
            {
                "排名": candidate.rank,
                "股票": candidate.symbol,
                "触发主题": candidate.trigger_theme,
                "机会分": candidate.opportunity_score,
                "状态": _status_label(candidate.score_status),
                "激进买入": ladder.aggressive_entry,
                "均衡买入": ladder.balanced_entry,
                "保守买入": ladder.conservative_entry,
                "止损": ladder.stop_price,
                "卖出1": ladder.first_exit,
                "卖出2": ladder.second_exit,
                "延伸卖出": ladder.stretch_exit,
                "原因": ", ".join(candidate.reason_codes) if candidate.reason_codes else "无",
                "数据缺口": ", ".join(candidate.data_gaps) if candidate.data_gaps else "无",
            }
        )
    return records


def build_retail_candidate_cards(
    rows: tuple[OpportunityPanelRow, ...] | list[OpportunityPanelRow],
) -> list[dict[str, object]]:
    """Convert panel rows into plain-language retail dashboard cards."""
    cards: list[dict[str, object]] = []
    for row in rows:
        candidate = row.candidate
        ladder = row.ladder
        priority = _priority_label(candidate.opportunity_score)
        cards.append(
            {
                "排名": candidate.rank,
                "股票": candidate.symbol,
                "主题": _theme_label(candidate.trigger_theme),
                "机会分": candidate.opportunity_score,
                "优先级": priority,
                "一句话判断": _action_text(priority),
                "买入区间": f"{ladder.conservative_entry:.2f} - {ladder.aggressive_entry:.2f}",
                "卖出区间": (
                    f"{ladder.first_exit:.2f} / {ladder.second_exit:.2f} / {ladder.stretch_exit:.2f}"
                ),
                "止损价": f"{ladder.stop_price:.2f}",
                "核心理由": _reason_summary(candidate.reason_codes),
                "风险提示": _risk_summary(candidate.score_status, candidate.data_gaps),
                "数据质量": _data_quality_label(candidate.data_gaps),
            }
        )
    return cards


def build_retail_summary_metrics(
    panel: DashboardPanel,
    *,
    refresh_label: str,
) -> dict[str, str]:
    """Build the top-line dashboard metrics for general users."""
    cards = build_retail_candidate_cards(panel.rows)
    if not cards:
        return {
            "候选数量": "0",
            "第一候选": "无候选",
            "行动提示": "等待数据",
            "建议刷新": refresh_label,
            "校准状态": "无候选",
        }

    statuses = {row.candidate.score_status for row in panel.rows}
    if statuses == {"calibrated_probability"}:
        calibration = "已校准"
    elif "insufficient_calibration" in statuses:
        calibration = "校准不足"
    else:
        calibration = "未校准"

    top = cards[0]
    return {
        "候选数量": str(len(cards)),
        "第一候选": str(top["股票"]),
        "行动提示": str(top["一句话判断"]),
        "建议刷新": refresh_label,
        "校准状态": calibration,
    }


def automation_roadmap_rows() -> list[dict[str, str]]:
    """Return the 8 §10 gates as a roadmap table (state of the world)."""
    return [
        {
            "阶段": f"{gate['id']} {gate['label']}",
            "当前状态": _STATUS_LABELS[gate["status"]],
            "下一道门": gate["next_gate"],
        }
        for gate in _GATE_DEFINITIONS
    ]


def build_gate_progress_rows() -> list[dict[str, str]]:
    """Return compact chip rows for the §10 gate progress strip."""
    return [
        {
            "id": gate["id"],
            "label": gate["label"],
            "task_id": gate["task_id"],
            "status": gate["status"],
            "status_label": _STATUS_LABELS[gate["status"]],
            "glyph": _STATUS_GLYPHS[gate["status"]],
            "next_gate": gate["next_gate"],
        }
        for gate in _GATE_DEFINITIONS
    ]


def build_price_ladder_view(
    ladder: PriceLadder,
    *,
    current_price: float,
) -> list[dict[str, object]]:
    """Return 7 ladder tiers ordered top (stretch exit) to bottom (stop).

    Each row carries the absolute price plus a signed delta vs current price
    so the dashboard can render directional ± labels without recomputing.
    """
    if float(current_price) <= 0:
        raise ValueError("current_price must be positive")
    tiers = (
        ("延伸卖出", ladder.stretch_exit, "exit_stretch"),
        ("卖出 2", ladder.second_exit, "exit_2"),
        ("卖出 1", ladder.first_exit, "exit_1"),
        ("激进买入", ladder.aggressive_entry, "entry_aggressive"),
        ("均衡买入", ladder.balanced_entry, "entry_balanced"),
        ("保守买入", ladder.conservative_entry, "entry_conservative"),
        ("止损", ladder.stop_price, "stop"),
    )
    return [
        {
            "档位": label,
            "价位": round(float(price), 2),
            "相对现价": round((float(price) - float(current_price)) / float(current_price), 4),
            "类型": kind,
        }
        for label, price, kind in tiers
    ]


def build_calibration_badge(score_status: str) -> dict[str, str]:
    """Return badge text + level for the hero card calibration pill."""
    table = {
        "uncalibrated_research_score": {
            "text": "⚠ 未校准研究分 · 不是真实胜率",
            "level": "warning",
        },
        "insufficient_calibration": {
            "text": "⚠ 校准样本不足 · 不是真实胜率",
            "level": "warning",
        },
        "calibrated_probability": {
            "text": "✓ 已校准概率",
            "level": "ok",
        },
    }
    return table.get(
        score_status,
        {"text": f"⚠ 未知校准状态：{score_status}", "level": "warning"},
    )


def build_recent_predictions_view(
    *,
    base_dir: Path | str,
    trade_date: str,
) -> list[dict[str, object]]:
    """Read §8.6 decision log for one trade date; return user-facing rows.

    Returns empty list when no JSONL exists for the date. Each row carries the
    symbol, score, calibration status label, model version, prediction id, and
    decision cutoff so the dashboard can prove the chain-of-evidence.
    """
    records = read_predictions(trade_date=trade_date, base_dir=base_dir)
    rows: list[dict[str, object]] = []
    for record in records:
        opp_score = record.extra.get("opportunity_score")
        rows.append(
            {
                "日期": record.trade_date,
                "符号": record.symbol,
                "机会分": (
                    round(float(opp_score), 2) if opp_score is not None else "—"
                ),
                "状态": _status_label(record.score_status),
                "模型": record.model_version,
                "prediction_id": record.prediction_id,
                "决策时刻": record.decision_cutoff,
            }
        )
    return rows


def refresh_interval_label(asof_ts: str, *, event_trigger: bool = False) -> str:
    """Return a user-facing refresh interval label."""
    minutes = RefreshSchedule().interval_minutes(asof_ts, event_trigger=event_trigger)
    if minutes == 0:
        return "立即重算"
    return f"{minutes} 分钟"


def _status_label(status: str) -> str:
    labels = {
        "uncalibrated_research_score": "未校准研究分",
        "insufficient_calibration": "校准不足",
        "calibrated_probability": "已校准概率",
    }
    return labels.get(status, status)


def _priority_label(score: float) -> str:
    if score >= 75:
        return "高关注"
    if score >= 60:
        return "可观察"
    return "低优先级"


def _action_text(priority: str) -> str:
    labels = {
        "高关注": "高关注：等回落到均衡买入区再考虑",
        "可观察": "可观察：只在回落到保守买入区时考虑",
        "低优先级": "低优先级：暂时只观察",
    }
    return labels.get(priority, priority)


def _theme_label(theme: str) -> str:
    labels = {
        "ai_semiconductor": "AI半导体",
        "fx_export": "出口/汇率",
        "rate_sensitive_bank": "银行/利率",
        "energy_commodity": "能源/商品",
        "shareholder_return": "股东回报",
        "price_volume": "量价异动",
        "news_watch": "新闻观察",
    }
    return labels.get(theme, theme)


def _reason_summary(reason_codes: tuple[str, ...]) -> str:
    labels = {
        "HOT_THEME": "热门主题",
        "WARM_THEME": "主题升温",
        "POSITIVE_NEWS": "正面新闻",
        "RELATIVE_STRENGTH": "相对强势",
        "VOLUME_EXPANSION": "成交放量",
        "LIQUID": "流动性充足",
        "SUPPORTIVE_CONTEXT": "市场环境支持",
        "LOW_SIGNAL": "信号偏弱",
    }
    return "、".join(labels.get(code, code) for code in reason_codes) if reason_codes else "暂无明确理由"


def _risk_summary(score_status: str, data_gaps: tuple[str, ...]) -> str:
    if data_gaps:
        return f"数据不完整：{_data_gap_summary(data_gaps)}；当前不能当胜率。"
    if score_status == "calibrated_probability":
        return "已校准概率仍需结合仓位、止损和人工确认。"
    if score_status == "insufficient_calibration":
        return "校准样本不足，当前不能当真实胜率。"
    return "当前只是研究分，不是真实胜率；需要后续反馈校准。"


def _data_quality_label(data_gaps: tuple[str, ...]) -> str:
    if not data_gaps:
        return "数据完整"
    return f"缺少：{_data_gap_summary(data_gaps)}"


def _data_gap_summary(data_gaps: tuple[str, ...]) -> str:
    labels = {
        "MISSING_CONTEXT": "市场/新闻上下文",
        "MISSING_VOLUME_RATIO": "成交量基准",
        "MISSING_LIQUIDITY": "流动性",
    }
    return "、".join(labels.get(gap, gap) for gap in data_gaps)


class _SampleQuoteClient:
    def fetch_quotes(self, symbols):
        quotes = {
            "8035.T": FreeWebQuote(
                symbol="8035.T",
                available_ts="2026-05-23T09:05:00+09:00",
                open=44100,
                high=46350,
                low=42750,
                close=45000,
                volume=1_000_000,
                previous_close=43650,
                avg_volume_20d=500_000,
            ),
            "7203.T": FreeWebQuote(
                symbol="7203.T",
                available_ts="2026-05-23T09:05:00+09:00",
                open=2940,
                high=3090,
                low=2850,
                close=3000,
                volume=1_000_000,
                previous_close=2910,
                avg_volume_20d=700_000,
            ),
        }
        return [quotes[symbol] for symbol in symbols if symbol in quotes]


class _SampleNewsClient:
    def fetch_news(self, symbols, since_ts, until_ts):
        return [
            NewsItem.from_dict(
                {
                    "news_id": "demo-ai",
                    "available_ts": "2026-05-23T09:04:00+09:00",
                    "source": "demo",
                    "headline": "AI semiconductor demand expands",
                    "body": "",
                    "symbols": ["8035.T"],
                }
            ),
            NewsItem.from_dict(
                {
                    "news_id": "demo-fx",
                    "available_ts": "2026-05-23T09:04:00+09:00",
                    "source": "demo",
                    "headline": "Exporters gain from weaker yen",
                    "body": "",
                    "symbols": ["7203.T"],
                }
            ),
        ]


class _SampleContextClient:
    def fetch_context(self, symbols):
        return {
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


class _EmptyNewsClient:
    def fetch_news(self, symbols, since_ts, until_ts):
        return []
