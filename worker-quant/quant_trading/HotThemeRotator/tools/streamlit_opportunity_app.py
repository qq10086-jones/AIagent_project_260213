"""Local Streamlit dashboard for the realtime opportunity panel."""
from __future__ import annotations

import html
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from hot_theme_rotator.ui.opportunity_dashboard import (
    automation_roadmap_rows,
    build_calibration_badge,
    build_gate_progress_rows,
    build_panel_records,
    build_price_ladder_view,
    build_recent_predictions_view,
    build_retail_candidate_cards,
    build_retail_summary_metrics,
    build_sample_panel,
    build_yfinance_quote_panel,
    parse_symbols,
    refresh_interval_label,
)


def main() -> None:
    st.set_page_config(
        page_title="HotThemeRotator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()

    st.title("今日机会中心")
    st.caption("研究模式：只生成候选、价位和风险提示，不自动下单。当前分数未校准，不是真实胜率。")
    _render_gate_progress_strip(build_gate_progress_rows())

    with st.sidebar:
        st.header("扫描设置")
        mode = st.radio(
            "数据来源",
            ["样例数据", "免费行情 yfinance"],
            index=0,
        )
        symbols_raw = st.text_area(
            "股票池",
            value="8035.T, 7203.T, 8306.T, 1306.T",
            height=82,
        )
        top_n = st.slider("显示数量", min_value=1, max_value=20, value=10)
        event_trigger = st.checkbox("重大新闻触发", value=False)
        asof = _now_jst_iso()
        refresh_label = refresh_interval_label(asof, event_trigger=event_trigger)
        st.divider()
        st.metric("建议刷新", refresh_label)
        st.caption(f"当前时间：{asof}")

    try:
        panel = _build_panel(mode=mode, symbols_raw=symbols_raw, asof=asof, top_n=top_n)
    except Exception as exc:  # pragma: no cover - Streamlit runtime path
        st.error(f"数据载入失败：{exc}")
        st.stop()

    records = build_panel_records(panel.rows)
    retail_cards = build_retail_candidate_cards(panel.rows)
    metrics = build_retail_summary_metrics(panel, refresh_label=refresh_label)

    _render_summary_metrics(metrics)

    if panel.data_notes:
        st.warning("；".join(panel.data_notes))

    if retail_cards and panel.rows:
        _render_top_candidate(
            retail_cards[0],
            top_row=panel.rows[0],
        )
    else:
        st.warning("当前没有候选。请检查股票池或数据源。")

    tab_candidates, tab_detail, tab_recent, tab_automation, tab_rules = st.tabs(
        ["机会列表", "候选详情", "最近记录 (§8.6)", "自动化路线 (§10)", "规则与原文"]
    )

    with tab_candidates:
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.warning("当前没有候选。请检查股票池或数据源。")

    with tab_detail:
        _render_retail_details(retail_cards)

    with tab_recent:
        _render_recent_predictions(panel.trade_date)

    with tab_automation:
        _render_automation_roadmap()

    with tab_rules:
        _render_rules(panel.markdown_v2 or panel.markdown)


def _build_panel(*, mode: str, symbols_raw: str, asof: str, top_n: int):
    if mode == "样例数据":
        return build_sample_panel(top_n=top_n)
    symbols = parse_symbols(symbols_raw)
    if not symbols:
        raise ValueError("股票池不能为空")
    # §8.6 / §10 gate 3: persist every yfinance prediction to the decision log.
    return build_yfinance_quote_panel(
        symbols=symbols,
        asof=asof,
        top_n=top_n,
        persist_base_dir=PROJECT_ROOT,
    )


def _render_summary_metrics(metrics: dict[str, str]) -> None:
    col_a, col_b, col_c, col_d, col_e = st.columns([0.8, 1.0, 2.0, 1.0, 1.0])
    col_a.metric("候选数量", metrics["候选数量"])
    col_b.metric("第一候选", metrics["第一候选"])
    col_c.metric("行动提示", metrics["行动提示"])
    col_d.metric("建议刷新", metrics["建议刷新"])
    col_e.metric("校准状态", metrics["校准状态"])


def _render_top_candidate(card: dict[str, object], *, top_row) -> None:
    candidate = top_row.candidate
    ladder = top_row.ladder
    badge = build_calibration_badge(candidate.score_status)
    st.markdown(
        f"""
        <section class="top-candidate">
          <div class="pill pill-{_safe(badge['level'])}">{_safe(badge['text'])}</div>
          <div class="kicker">今日最值得盯</div>
          <div class="candidate-heading">
            <span>{_safe(card["股票"])}</span>
            <strong>{_safe(card["主题"])}</strong>
            <em>{_safe(card["优先级"])}</em>
          </div>
          <div class="action-line">{_safe(card["一句话判断"])}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
    c1.metric("机会分", str(card["机会分"]))
    c2.metric("当前价", f"{candidate.reference_price:.2f}")
    c3.metric("决策时刻", candidate.decision_cutoff or "—")

    ladder_view = build_price_ladder_view(ladder, current_price=candidate.reference_price)
    _render_price_ladder(ladder_view, current_price=candidate.reference_price)

    left, right = st.columns(2)
    left.markdown(
        f"""
        <div class="explain-panel reason-panel">
          <div class="panel-title">为什么它排前面</div>
          <p>{_safe(card["核心理由"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    right.markdown(
        f"""
        <div class="explain-panel risk-panel">
          <div class="panel-title">先看清楚的风险</div>
          <p>{_safe(card["风险提示"])}</p>
          <p class="muted">{_safe(card["数据质量"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_gate_progress_strip(chips: list[dict[str, str]]) -> None:
    items = []
    for chip in chips:
        css = f"gate-chip gate-{chip['status']}"
        items.append(
            f'<div class="{css}" title="{_safe(chip["task_id"])} · {_safe(chip["next_gate"])}">'
            f'<span class="gate-id">{_safe(chip["id"])}</span>'
            f'<span class="gate-glyph">{_safe(chip["glyph"])}</span>'
            f'<span class="gate-label">{_safe(chip["label"])}</span>'
            f"</div>"
        )
    st.markdown(
        f"""
        <div class="gate-strip" title="§10 自动化八阶门槛">
          <span class="gate-strip-title">§10 八阶门槛</span>
          {''.join(items)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_price_ladder(ladder_view: list[dict[str, object]], *, current_price: float) -> None:
    rows = []
    current_inserted = False
    for row in ladder_view:
        price = float(row["价位"])
        # Insert the current-price divider after we cross down past current price
        if not current_inserted and price < current_price:
            rows.append(
                f'<div class="ladder-row ladder-current">'
                f'<span class="lr-label">► 现价</span>'
                f'<span class="lr-price">{current_price:.2f}</span>'
                f'<span class="lr-delta">—</span>'
                f'</div>'
            )
            current_inserted = True
        delta = float(row["相对现价"])
        delta_text = f"{'▲ ' if delta > 0 else ('▼ ' if delta < 0 else '· ')}{delta * 100:+.1f}%"
        kind = str(row["类型"])
        rows.append(
            f'<div class="ladder-row ladder-{kind}">'
            f'<span class="lr-label">{_safe(row["档位"])}</span>'
            f'<span class="lr-price">{price:.2f}</span>'
            f'<span class="lr-delta">{_safe(delta_text)}</span>'
            f'</div>'
        )
    if not current_inserted:
        rows.append(
            f'<div class="ladder-row ladder-current">'
            f'<span class="lr-label">► 现价</span>'
            f'<span class="lr-price">{current_price:.2f}</span>'
            f'<span class="lr-delta">—</span>'
            f'</div>'
        )
    st.markdown(
        f"""
        <div class="price-ladder">
          <div class="ladder-header">价格阶梯（七档 · 全部研究用 · 不是真实胜率）</div>
          {''.join(rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_predictions(trade_date: str) -> None:
    rows = build_recent_predictions_view(base_dir=PROJECT_ROOT, trade_date=trade_date)
    st.subheader(f"§8.6 决策日志 · {trade_date}")
    st.caption(
        "每条记录来自 `reports/predictions/{trade_date}.jsonl`，由 yfinance 模式落盘。"
        "样例模式不落盘。"
    )
    if not rows:
        st.info("当日暂无已落盘记录。切换到 yfinance 模式扫描一次即可生成。")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"共 {len(rows)} 条预测已落盘。")


def _render_retail_details(cards: list[dict[str, object]]) -> None:
    if not cards:
        st.warning("没有可展示的候选详情。")
        return
    for card in cards:
        with st.expander(
            f"{card['排名']}. {card['股票']} | {card['主题']} | {card['优先级']}",
            expanded=card["排名"] == 1,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("机会分", str(card["机会分"]))
            c2.metric("买入区间", str(card["买入区间"]))
            c3.metric("止损价", str(card["止损价"]))
            c4.metric("卖出区间", str(card["卖出区间"]))
            st.write(f"判断：{card['一句话判断']}")
            st.write(f"理由：{card['核心理由']}")
            st.warning(str(card["风险提示"]))


def _render_automation_roadmap() -> None:
    st.subheader("自动化路线")
    st.caption("当前只能做研究候选和价格阶梯；进入提醒、纸面交易、实盘前必须先完成反馈和校准。")
    st.dataframe(pd.DataFrame(automation_roadmap_rows()), width="stretch", hide_index=True)


def _render_rules(markdown: str) -> None:
    st.subheader("当前规则")
    st.write(
        "- 先搜索潜力股，再给分数和价格阶梯。\n"
        "- 所有实时输入必须带 `available_ts`，晚于决策时间的数据不能用。\n"
        "- 买入价分三档：激进、均衡、保守。\n"
        "- 卖出价分三档：卖出1、卖出2、延伸卖出。\n"
        "- 当前分数是 `uncalibrated_research_score`，不能当真实胜率。\n"
        "- 自动化必须按日志、反馈、校准、提醒、纸面交易、人工批准的顺序推进。"
    )
    with st.expander("原始 Markdown 报告", expanded=False):
        st.code(markdown, language="markdown")


def _now_jst_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f7f4; }
        .block-container { padding-top: 1.25rem; max-width: 1220px; }
        div[data-testid="stMetric"] {
            border: 1px solid #dfe4ea;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #ffffff;
        }
        .top-candidate {
            margin: 1rem 0 0.85rem 0;
            padding: 1.1rem 1.2rem;
            border: 1px solid #d8dee6;
            border-radius: 8px;
            background: #ffffff;
        }
        .kicker {
            color: #5f6b7a;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .candidate-heading {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-bottom: 0.4rem;
        }
        .candidate-heading span {
            font-size: 1.9rem;
            font-weight: 800;
            color: #111827;
        }
        .candidate-heading strong {
            color: #22543d;
            font-size: 1rem;
        }
        .candidate-heading em {
            font-style: normal;
            color: #8a4b08;
            background: #fff4d6;
            border: 1px solid #f1d28a;
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            font-size: 0.84rem;
            font-weight: 700;
        }
        .action-line {
            color: #1f2937;
            font-size: 1.05rem;
            font-weight: 650;
        }
        .explain-panel {
            min-height: 126px;
            border: 1px solid #dfe4ea;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #ffffff;
        }
        .reason-panel { border-left: 5px solid #2f855a; }
        .risk-panel { border-left: 5px solid #c2410c; }
        .panel-title {
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.35rem;
        }
        .muted {
            color: #667085;
            margin-bottom: 0;
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
            border: 1px solid transparent;
        }
        .pill-warning {
            color: #8a4b08;
            background: #fff4d6;
            border-color: #f1d28a;
        }
        .pill-ok {
            color: #1b4332;
            background: #d6f4dc;
            border-color: #8ad8a0;
        }
        .gate-strip {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.4rem 0 0.6rem 0;
            padding: 0.55rem 0.7rem;
            background: #ffffff;
            border: 1px solid #dfe4ea;
            border-radius: 8px;
        }
        .gate-strip-title {
            font-size: 0.75rem;
            font-weight: 800;
            color: #5f6b7a;
            letter-spacing: 0.05em;
            margin-right: 0.3rem;
        }
        .gate-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            border: 1px solid #d8dee6;
            background: #f5f7fa;
            font-size: 0.8rem;
            color: #374151;
        }
        .gate-chip .gate-id { font-weight: 700; color: #5f6b7a; }
        .gate-chip .gate-glyph { font-weight: 700; }
        .gate-done {
            background: #d6f4dc;
            border-color: #8ad8a0;
            color: #1b4332;
        }
        .gate-done .gate-glyph { color: #1b6b3a; }
        .gate-in_progress {
            background: #fff4d6;
            border-color: #f1d28a;
            color: #8a4b08;
        }
        .gate-pending {
            background: #f1f3f6;
            border-color: #d8dee6;
            color: #4b5563;
        }
        .gate-blocked {
            background: #fde2e1;
            border-color: #f1a5a1;
            color: #9b1c1c;
        }
        .gate-blocked .gate-glyph { color: #9b1c1c; }
        .price-ladder {
            margin: 0.65rem 0;
            border: 1px solid #dfe4ea;
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
        }
        .ladder-header {
            padding: 0.45rem 0.7rem;
            font-size: 0.8rem;
            font-weight: 700;
            color: #5f6b7a;
            background: #f7f7f4;
            border-bottom: 1px solid #dfe4ea;
        }
        .ladder-row {
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr;
            align-items: center;
            padding: 0.4rem 0.85rem;
            border-bottom: 1px solid #f1f3f6;
            font-variant-numeric: tabular-nums;
        }
        .ladder-row:last-child { border-bottom: none; }
        .ladder-row .lr-label { font-weight: 650; color: #1f2937; }
        .ladder-row .lr-price { text-align: right; font-weight: 600; }
        .ladder-row .lr-delta { text-align: right; font-size: 0.85rem; color: #5f6b7a; }
        .ladder-current {
            background: #eef2ff;
            border-top: 2px solid #6366f1;
            border-bottom: 2px solid #6366f1;
        }
        .ladder-current .lr-label { color: #4338ca; font-weight: 800; }
        .ladder-current .lr-price { color: #4338ca; font-weight: 800; }
        .ladder-exit_stretch { background: #dbeafe; }
        .ladder-exit_2 { background: #e0e8ff; }
        .ladder-exit_1 { background: #ecf1ff; }
        .ladder-exit_stretch .lr-delta,
        .ladder-exit_2 .lr-delta,
        .ladder-exit_1 .lr-delta { color: #1d4ed8; font-weight: 600; }
        .ladder-entry_aggressive { background: #f0fdf4; }
        .ladder-entry_balanced { background: #e3f8ea; }
        .ladder-entry_conservative { background: #d6f4dc; }
        .ladder-entry_aggressive .lr-delta,
        .ladder-entry_balanced .lr-delta,
        .ladder-entry_conservative .lr-delta { color: #14532d; font-weight: 600; }
        .ladder-stop {
            background: #fde2e1;
            border-top: 1px dashed #f1a5a1;
        }
        .ladder-stop .lr-label { color: #9b1c1c; font-weight: 800; }
        .ladder-stop .lr-delta { color: #9b1c1c; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe(value: object) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()
