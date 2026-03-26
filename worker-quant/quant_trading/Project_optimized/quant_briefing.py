"""
quant_briefing.py — 每日市场简报聚合脚本

用途：一次运行，生成涵盖大盘/候选池/仓位/信号的结构化报告。
      Claude 读取输出的 JSON/MD 即可完成完整分析，无需多次调用子脚本。

输出:
  reports/briefing_latest.json   — 结构化数据（供 Claude / Nexus 解析）
  reports/briefing_latest.md     — 可读报告

用法:
  python quant_briefing.py
  python quant_briefing.py --mode market        # 仅市场行情
  python quant_briefing.py --mode stock --symbols 9432.T,5401.T  # 个股深析
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

# ── 常量 ──────────────────────────────────────────────────────────────────
DB_PATH    = "japan_market.db"
REPORT_DIR = Path("reports")
JST        = pytz.timezone("Asia/Tokyo")
NIKKEI_ETF = "1570.T"   # 日经225 2倍ETF，用作大盘代理


# ── 工具函数 ──────────────────────────────────────────────────────────────

def now_jst() -> datetime:
    return datetime.now(JST)


def _safe_pct(a, b) -> Optional[float]:
    try:
        return (float(a) / float(b) - 1) * 100 if b and float(b) != 0 else None
    except Exception:
        return None


def market_session_status(now: datetime) -> dict:
    """返回当前 TSE 交易时段信息。"""
    h, m = now.hour, now.minute
    t = h * 60 + m
    if t < 9 * 60:
        status = "pre_market"
    elif t < 11 * 60 + 30:
        status = "morning"
    elif t < 12 * 60 + 30:
        status = "lunch"
    elif t < 15 * 60 + 30:
        status = "afternoon"
    else:
        status = "closed"

    close_min = 15 * 60 + 30
    mins_to_close = max(0, close_min - t)
    return {
        "time_jst": now.strftime("%H:%M"),
        "session": status,
        "mins_to_close": mins_to_close,
        "trading": status in ("morning", "afternoon"),
    }


# ── 市场数据获取 ───────────────────────────────────────────────────────────

def fetch_price_snapshot(symbols: List[str]) -> Dict[str, dict]:
    """批量拉取今日日内行情快照。"""
    try:
        import yfinance as yf
    except ImportError:
        return {}

    result: Dict[str, dict] = {}
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            df = tk.history(period="2d", interval="1m")  # 拉2天以便得到昨收
            if df.empty:
                continue
            df.index = df.index.tz_convert(JST)
            today_str = now_jst().strftime("%Y-%m-%d")
            today_df  = df[df.index.strftime("%Y-%m-%d") == today_str]
            if today_df.empty:
                today_df = df  # 回退用全部
            open_p  = float(today_df["Open"].iloc[0])
            cur     = float(today_df["Close"].iloc[-1])
            low     = float(today_df["Low"].min())
            high    = float(today_df["High"].max())
            vol     = int(today_df["Volume"].sum())
            last_t  = today_df.index[-1].strftime("%H:%M")

            # 用昨收作为涨跌基准（与券商显示一致）
            prev_df = df[df.index.strftime("%Y-%m-%d") < today_str]
            if not prev_df.empty:
                prev_close = float(prev_df["Close"].iloc[-1])
            else:
                prev_close = open_p  # 无昨收时退化到开盘价
            chg_pct = _safe_pct(cur, prev_close)

            # 5分钟趋势（最近5根K线）
            recent  = df["Close"].tail(5)
            trend   = "up" if recent.iloc[-1] > recent.iloc[0] else \
                      "down" if recent.iloc[-1] < recent.iloc[0] else "flat"

            # 异常大量检测（最大单分钟量 vs 均量）
            avg_vol = df["Volume"].mean()
            max_vol = int(df["Volume"].max())
            volume_spike = round(max_vol / avg_vol, 1) if avg_vol > 0 else 0

            result[sym] = {
                "open": open_p, "cur": cur, "low": low, "high": high,
                "prev_close": prev_close,
                "vol_total": vol, "last_time": last_t,
                "chg_pct": round(chg_pct, 2) if chg_pct is not None else None,
                "trend_5m": trend,
                "volume_spike_ratio": volume_spike,
            }
        except Exception:
            pass
    return result


def fetch_multi_day(symbols: List[str], days: int = 30) -> Dict[str, pd.Series]:
    """拉取多日收盘价，用于动量计算。"""
    try:
        import yfinance as yf
    except ImportError:
        return {}
    result = {}
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period=f"{days}d")
            if not df.empty:
                result[sym] = df["Close"].dropna()
        except Exception:
            pass
    return result


def compute_momentum_stats(closes: Dict[str, pd.Series], market_sym: str) -> Dict[str, dict]:
    """计算各标的动量指标及相对大盘超额。"""
    market = closes.get(market_sym)
    result = {}
    for sym, s in closes.items():
        if sym == market_sym or len(s) < 5:
            continue
        ret_today  = _safe_pct(s.iloc[-1], s.iloc[-2])
        ret_5d     = _safe_pct(s.iloc[-1], s.iloc[-5]) if len(s) >= 5 else None
        ret_20d    = _safe_pct(s.iloc[-1], s.iloc[-21]) if len(s) >= 21 else None
        vol_5d     = float(s.pct_change().iloc[-5:].std() * 100) if len(s) >= 5 else None

        excess = None
        if ret_today is not None and market is not None and len(market) >= 2:
            mkt_today = _safe_pct(market.iloc[-1], market.iloc[-2])
            if mkt_today is not None:
                excess = round(ret_today - mkt_today, 2)

        result[sym] = {
            "ret_today": round(ret_today, 2) if ret_today is not None else None,
            "ret_5d":    round(ret_5d, 2)    if ret_5d is not None else None,
            "ret_20d":   round(ret_20d, 2)   if ret_20d is not None else None,
            "vol_5d":    round(vol_5d, 2)    if vol_5d is not None else None,
            "excess_vs_market": excess,
        }
    return result


# ── DB 数据读取 ────────────────────────────────────────────────────────────

def read_live_state(db_path: str) -> dict:
    """读取当前仓位、挂单、账户状态。"""
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        # 仓位
        cur.execute("SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl FROM positions")
        positions = [
            {"symbol": r[0], "qty": r[1], "avg_cost": r[2],
             "market_price": r[3], "market_value": r[4], "unrealized_pnl": r[5]}
            for r in cur.fetchall()
        ]

        # 挂单（仅 proposed/open）
        cur.execute("""
            SELECT order_id, symbol, side, qty, limit_price, status, created_ts
            FROM orders WHERE status IN ('proposed','open','pending')
            ORDER BY created_ts DESC
        """)
        orders = [
            {"order_id": r[0], "symbol": r[1], "side": r[2],
             "qty": r[3], "limit_price": r[4], "status": r[5], "created_ts": r[6]}
            for r in cur.fetchall()
        ]

        # 账户快照
        cur.execute("SELECT asof, nav, cash FROM account_snapshots ORDER BY ts DESC LIMIT 1")
        snap = cur.fetchone()
        account = {"asof": snap[0], "nav": snap[1], "cash": snap[2]} if snap else {}

        # 账户状态（初始资本）
        cur.execute("SELECT starting_capital, cash_balance FROM account_state ORDER BY updated_at DESC LIMIT 1")
        state = cur.fetchone()
        if state:
            account.setdefault("starting_capital", state[0])
            account.setdefault("cash_balance", state[1])

        conn.close()
        return {"positions": positions, "orders": orders, "account": account}
    except Exception as e:
        return {"error": str(e)}


def read_latest_signals(db_path: str, top_n: int = 20) -> list:
    """读取最新信号分数。"""
    try:
        conn = sqlite3.connect(db_path)
        rows = pd.read_sql_query("""
            SELECT s.asof, s.symbol, s.score, t.name, t.sector
            FROM signals s
            LEFT JOIN tickers t ON s.symbol = t.symbol
            WHERE s.asof = (SELECT MAX(asof) FROM signals)
            ORDER BY s.score DESC
            LIMIT ?
        """, conn, params=[top_n])
        conn.close()
        return rows.to_dict(orient="records")
    except Exception:
        return []


# ── Screener 调用 ──────────────────────────────────────────────────────────

def run_screener(db_path: str, asof: str) -> dict:
    """调用 screener，返回含基本面叠加分的候选池。"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from screener import ScreenConfig, FundamentalOverlayConfig, screen
        cfg  = ScreenConfig(top_k=30, min_adv=5_000_000)
        ocfg = FundamentalOverlayConfig(enabled=True)
        result = screen(db_path, None, asof, None, cfg, write_db=False, overlay_cfg=ocfg)
        return result
    except Exception as e:
        return {"error": str(e)}


# ── 个股深析 ──────────────────────────────────────────────────────────────

def fetch_stock_deep(symbol: str) -> dict:
    """单只股票深度数据：财务指标 + 历史价格 + 新闻标题。"""
    try:
        import yfinance as yf
        tk   = yf.Ticker(symbol)
        info = tk.info or {}

        # 关键财务指标
        financials = {
            "name":              info.get("shortName", ""),
            "sector":            info.get("sector", ""),
            "market_cap_jpy":    info.get("marketCap"),
            "forward_pe":        info.get("forwardPE"),
            "pbr":               info.get("priceToBook"),
            "roe_ttm":           info.get("returnOnEquity"),
            "operating_margin":  info.get("operatingMargins"),   # 营业利润率（不受一次性扰动）
            "profit_margin_ttm": info.get("profitMargins"),       # 净利润率（参考）
            "operating_cf":      info.get("operatingCashflow"),  # 经营现金流
            "revenue_growth":    info.get("revenueGrowth"),
            "earnings_growth":   info.get("earningsGrowth"),
            "debt_to_equity":    info.get("debtToEquity"),
            "current_ratio":     info.get("currentRatio"),
            "dividend_yield":    info.get("dividendYield"),
            "trailing_eps":      info.get("trailingEps"),
            "forward_eps":       info.get("forwardEps"),
            "52w_high":          info.get("fiftyTwoWeekHigh"),
            "52w_low":           info.get("fiftyTwoWeekLow"),
        }

        # 最新季度 EPS（判断盈亏趋势）
        eps_quarters = []
        try:
            qi = tk.quarterly_income_stmt
            if not qi.empty and "Diluted EPS" in qi.index:
                eps_quarters = [
                    {"period": str(c)[:10], "diluted_eps": float(v)}
                    for c, v in qi.loc["Diluted EPS"].dropna().head(4).items()
                ]
        except Exception:
            pass

        # 近5日价格
        hist = tk.history(period="5d")
        price_history = [
            {"date": str(idx)[:10], "close": round(float(row["Close"]), 1),
             "volume": int(row["Volume"])}
            for idx, row in hist.iterrows()
        ] if not hist.empty else []

        return {
            "symbol":        symbol,
            "financials":    financials,
            "eps_quarters":  eps_quarters,
            "price_history": price_history,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ── 报告生成 ───────────────────────────────────────────────────────────────

def build_briefing(mode: str, extra_symbols: List[str]) -> dict:
    now   = now_jst()
    asof  = date.today().isoformat()
    session = market_session_status(now)

    report: dict = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M JST"),
        "asof": asof,
        "session": session,
        "mode": mode,
    }

    # ── 模式：market（市场行情 + 操作建议）────────────────────────────────
    if mode in ("market", "full"):
        # Screener
        screener_result = run_screener(DB_PATH, asof)
        candidates = screener_result.get("details", [])
        candidate_symbols = [r["symbol"] for r in candidates]

        # 价格快照
        all_syms = [NIKKEI_ETF] + candidate_symbols
        prices   = fetch_price_snapshot(all_syms)
        closes   = fetch_multi_day(all_syms, days=30)
        momentum = compute_momentum_stats(closes, NIKKEI_ETF)

        # 大盘摘要
        mkt = prices.get(NIKKEI_ETF, {})
        mkt_ret = mkt.get("chg_pct")
        report["market"] = {
            "nikkei_etf":      mkt,
            "market_ret_pct":  mkt_ret,
            "sentiment":       "weak" if (mkt_ret or 0) < -0.5 else
                               "strong" if (mkt_ret or 0) > 0.5 else "neutral",
        }

        # 候选池合并
        enriched = []
        for r in candidates:
            sym  = r["symbol"]
            px   = prices.get(sym, {})
            mom  = momentum.get(sym, {})
            enriched.append({
                "rank":              candidates.index(r) + 1,
                "symbol":            sym,
                "score_tech":        round(r.get("score", 0), 2),
                "fundamental_score": round(r.get("fundamental_score", 1.0), 2),
                "score_adjusted":    round(r.get("score_adjusted", 0), 2),
                "fundamental_note":  r.get("fundamental_note", ""),
                "price_cur":         px.get("cur"),
                "price_open":        px.get("open"),
                "chg_pct":           px.get("chg_pct"),
                "trend_5m":          px.get("trend_5m"),
                "volume_spike":      px.get("volume_spike_ratio"),
                "ret_today":         mom.get("ret_today"),
                "ret_20d":           mom.get("ret_20d"),
                "excess_vs_market":  mom.get("excess_vs_market"),
                "vol_5d":            mom.get("vol_5d"),
            })
        report["candidates"] = enriched
        report["screener_meta"] = {
            "count":              screener_result.get("count", 0),
            "hard_vetoed_count":  screener_result.get("fundamental_overlay", {}).get("hard_vetoed_count", 0),
            "downweighted_count": screener_result.get("fundamental_overlay", {}).get("downweighted_count", 0),
        }

        # 仓位 & 挂单
        live = read_live_state(DB_PATH)
        report["live_state"] = live

        # 最新信号（来自 DB）
        report["db_signals"] = read_latest_signals(DB_PATH, top_n=15)

    # ── 模式：stock（个股深析）───────────────────────────────────────────
    if mode in ("stock", "full") and extra_symbols:
        report["stock_analysis"] = [fetch_stock_deep(s) for s in extra_symbols]

    return report


def write_report(report: dict) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(exist_ok=True)
    json_path = REPORT_DIR / "briefing_latest.json"
    md_path   = REPORT_DIR / "briefing_latest.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # Markdown 摘要
    lines = [
        f"# Quant Briefing — {report['generated_at']}",
        f"\n**交易时段**: {report['session']['session']}  |  "
        f"**距收盘**: {report['session']['mins_to_close']}分钟\n",
    ]

    if "market" in report:
        mkt = report["market"]
        etf = mkt.get("nikkei_etf", {})
        lines += [
            "## 大盘",
            f"- 日经ETF: {etf.get('cur','N/A')}  "
            f"({'+' if (etf.get('chg_pct') or 0) >= 0 else ''}{etf.get('chg_pct','N/A')}%)  "
            f"情绪: **{mkt.get('sentiment','N/A')}**",
            f"- 日内区间: H {etf.get('high','N/A')} / L {etf.get('low','N/A')}\n",
        ]

    if "candidates" in report:
        lines.append("## 候选池（调整分排序）")
        lines.append(f"| 排名 | 代码 | 调整分 | 基×  | 今日% | 超额% | 20日% | 5m趋势 | 量异常 | 基本面注记 |")
        lines.append(f"|------|------|--------|------|-------|-------|-------|--------|--------|------------|")
        for r in report["candidates"]:
            spike = f"{r['volume_spike']}x" if r.get("volume_spike") and r["volume_spike"] > 3 else "—"
            lines.append(
                f"| {r['rank']} | {r['symbol']} | {r['score_adjusted']} | {r['fundamental_score']} "
                f"| {r.get('chg_pct','—')}% | {r.get('excess_vs_market','—')}% "
                f"| {r.get('ret_20d','—')}% | {r.get('trend_5m','—')} | {spike} "
                f"| {r.get('fundamental_note','') or '正常'} |"
            )
        lines.append("")

    if "live_state" in report:
        live = report["live_state"]
        lines.append("## 当前仓位 & 挂单")
        pos = live.get("positions", [])
        if pos:
            for p in pos:
                lines.append(f"- 持仓 {p['symbol']}  {p['qty']}股  均价{p['avg_cost']}  "
                              f"浮盈{p.get('unrealized_pnl','N/A')}")
        else:
            lines.append("- 当前空仓")
        orders = live.get("orders", [])
        if orders:
            for o in orders:
                lines.append(f"- 挂单 {o['symbol']} {o['side']} {o['qty']}股 @ {o['limit_price']}  [{o['status']}]")
        lines.append("")

    if "stock_analysis" in report:
        lines.append("## 个股深析")
        for s in report["stock_analysis"]:
            if "error" in s:
                lines.append(f"### {s['symbol']}  (数据获取失败: {s['error']})")
                continue
            f = s.get("financials", {})
            lines += [
                f"### {s['symbol']}  {f.get('name','')}",
                f"- 营业利润率(TTM): {f.get('operating_margin','N/A')}  "
                f"净利润率: {f.get('profit_margin_ttm','N/A')}  "
                f"OCF: {f.get('operating_cf','N/A')}",
                f"- Forward PE: {f.get('forward_pe','N/A')}  "
                f"PBR: {f.get('pbr','N/A')}  "
                f"ROE: {f.get('roe_ttm','N/A')}",
                f"- D/E: {f.get('debt_to_equity','N/A')}  "
                f"股息率: {f.get('dividend_yield','N/A')}",
                f"- 52W: {f.get('52w_low','N/A')} – {f.get('52w_high','N/A')}",
            ]
            eps = s.get("eps_quarters", [])
            if eps:
                eps_str = "  ".join(
                    f"{e['period'][:7]} EPS={e['diluted_eps']:.2f}" for e in eps
                )
                lines.append(f"- 季度EPS趋势: {eps_str}")
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


# ── 入口 ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Quant Daily Briefing")
    ap.add_argument("--mode", choices=["market", "stock", "full"], default="market",
                    help="market=行情+策略  stock=个股深析  full=全部")
    ap.add_argument("--symbols", default="",
                    help="个股分析时指定代码，逗号分隔，如 9432.T,5401.T")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 允许命令行覆盖 DB 路径
    globals()["DB_PATH"] = args.db

    extra = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.mode == "stock" and not extra:
        print("[briefing] --mode stock 需要指定 --symbols", file=sys.stderr)
        sys.exit(1)

    print(f"[briefing] 生成中... mode={args.mode}  {now_jst().strftime('%H:%M JST')}")
    report = build_briefing(args.mode, extra)
    json_p, md_p = write_report(report)

    print(f"[briefing] 完成")
    print(f"  JSON → {json_p}")
    print(f"  MD   → {md_p}")

    # 打印关键摘要到 stdout（供 Claude 快速读取）
    if "market" in report:
        mkt = report["market"]
        print(f"\n大盘: {mkt.get('nikkei_etf',{}).get('cur','N/A')} "
              f"({mkt.get('market_ret_pct','N/A')}%)  情绪: {mkt.get('sentiment')}")
        print(f"候选池: {report.get('screener_meta',{}).get('count',0)}只  "
              f"降权: {report.get('screener_meta',{}).get('downweighted_count',0)}只")
        orders = report.get("live_state", {}).get("orders", [])
        if orders:
            for o in orders:
                print(f"挂单: {o['symbol']} {o['side']} {o['qty']}股 @ {o['limit_price']} [{o['status']}]")
        else:
            print("当前: 空仓无挂单")


if __name__ == "__main__":
    main()
