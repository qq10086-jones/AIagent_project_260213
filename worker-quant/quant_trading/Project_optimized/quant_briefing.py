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
NIKKEI_ETF = "1321.T"   # 日经225 ETF（野村，1:1跟踪指数），用作大盘代理；1570.T为2倍杠杆不适合做基准


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


# ── 市场状态判断 (Regime Detection) ───────────────────────────────────────

def _detect_market_regime() -> dict:
    """
    检测当前市场状态（Regime），基于日经ETF 1570.T 近60日K线。
    vol_level 使用真实振幅(ATR/Close)：5日均值 vs 60日均值，>1.5倍判定为 HIGH。
    返回: {trend, vol_level, bias, sma5, sma20, atr_5d_pct, atr_60d_pct, ret_5d_pct, action_bias}
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(NIKKEI_ETF)
        df = tk.history(period="70d")  # 多拉几天确保60个交易日
        if df.empty or len(df) < 20:
            return {"trend": "UNKNOWN", "vol_level": "UNKNOWN", "bias": "NEUTRAL",
                    "error": "数据不足"}

        closes = df["Close"]
        highs  = df["High"]
        lows   = df["Low"]

        # ── 趋势：SMA5 vs SMA20 ──────────────────────────────────────
        sma5  = float(closes.rolling(5).mean().iloc[-1])
        sma20 = float(closes.rolling(20).mean().iloc[-1])
        cur   = float(closes.iloc[-1])
        prev5 = float(closes.iloc[-5]) if len(closes) >= 5 else cur

        if sma5 > sma20 and cur > sma20:
            trend = "UP"
        elif sma5 < sma20 and cur < sma20:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"

        # ── 波动率：ATR/Close 相对真实振幅 ─────────────────────────────
        prev_c = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_c).abs(),
            (lows  - prev_c).abs(),
        ], axis=1).max(axis=1)

        atr_ratio    = tr / closes
        atr_5d_avg   = float(atr_ratio.iloc[-5:].mean())
        atr_60d_avg  = float(atr_ratio.iloc[-60:].mean()) if len(atr_ratio) >= 60 \
                       else float(atr_ratio.mean())

        if atr_5d_avg > atr_60d_avg * 1.5:
            vol_level = "HIGH"
        elif atr_5d_avg < atr_60d_avg * 0.7:
            vol_level = "LOW"
        else:
            vol_level = "NORMAL"

        # ── 偏向：综合趋势 + 近5日收益 ───────────────────────────────
        ret_5d = (cur / prev5 - 1) * 100

        if trend == "UP" and ret_5d >= 0:
            bias = "BULLISH"
        elif trend == "DOWN" and ret_5d <= 0:
            bias = "BEARISH"
        elif trend == "DOWN" or ret_5d < -1.0:
            bias = "BEARISH"
        elif trend == "UP" or ret_5d > 1.0:
            bias = "BULLISH"
        else:
            bias = "NEUTRAL"

        # ── 操作建议文字 ─────────────────────────────────────────────
        advice_map = {
            ("UP",       "LOW"):    "趋势向上且低波动，正常仓位，正常限价",
            ("UP",       "NORMAL"): "趋势向上，正常操作",
            ("UP",       "HIGH"):   "趋势向上但波动扩大，建议减半仓位，扩大限价距离",
            ("SIDEWAYS", "LOW"):    "盘整低波动，保守操作，patient档限价",
            ("SIDEWAYS", "NORMAL"): "盘整行情，保守操作，patient档限价",
            ("SIDEWAYS", "HIGH"):   "盘整+高波动，防御姿态，暂不新开仓",
            ("DOWN",     "LOW"):    "下跌趋势，防御，检查各持仓止损距离",
            ("DOWN",     "NORMAL"): "下跌趋势，暂不新开仓，优先执行止损",
            ("DOWN",     "HIGH"):   "下跌+高波动，全面防御，立即执行止损",
        }
        action_bias = advice_map.get((trend, vol_level), "观望")

        return {
            "trend":        trend,
            "vol_level":    vol_level,
            "bias":         bias,
            "sma5":         round(sma5, 1),
            "sma20":        round(sma20, 1),
            "atr_5d_pct":   round(atr_5d_avg  * 100, 2),
            "atr_60d_pct":  round(atr_60d_avg * 100, 2),
            "ret_5d_pct":   round(ret_5d, 2),
            "action_bias":  action_bias,
        }
    except Exception as exc:
        return {"trend": "UNKNOWN", "vol_level": "UNKNOWN", "bias": "NEUTRAL",
                "error": str(exc)}


# ── 新闻交叉验证 ───────────────────────────────────────────────────────────

def _cross_validate_with_news(db_path: str, symbol: str) -> dict:
    """
    对单个候选信号查询过去48h内的【公司级别】负面新闻。
    只标注 HIGH（公司直接利空），宏观新闻(BOJ/TRADE)由 _get_macro_news 统一处理，
    不在这里重复挂到每只个股上。
    """
    result: dict = {"news_risk": "NONE", "news_notes": []}
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT sentiment_score, summary_cn
            FROM news_items
            WHERE related_tickers LIKE ?
              AND impact_category = 'COMPANY'
              AND sentiment_score < -0.5
              AND published_at >= datetime('now', '-48 hours')
            ORDER BY sentiment_score ASC
            LIMIT 5
        """, (f"%{symbol}%",)).fetchall()
        conn.close()

        company_neg = [r[1] or "" for r in rows if r[0] is not None]
        if company_neg:
            result["news_risk"]  = "HIGH"
            result["news_notes"] = company_neg[:2]
    except Exception:
        pass  # news_items 表不存在或查询失败，维持 NONE
    return result


def _get_macro_news(db_path: str) -> list[dict]:
    """
    获取过去48h内的宏观层面重要新闻（BOJ/TRADE），去重后返回。
    供报告"二、今日有效情报"统一展示一次，不挂到个股上。
    """
    items: list[dict] = []
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT DISTINCT impact_category, summary_cn, sentiment_score
            FROM news_items
            WHERE impact_category IN ('BOJ', 'TRADE')
              AND sentiment_score < -0.3
              AND published_at >= datetime('now', '-48 hours')
            ORDER BY sentiment_score ASC
            LIMIT 8
        """).fetchall()
        conn.close()
        seen: set[str] = set()
        for category, summary_cn, score in rows:
            note = summary_cn or ""
            if note not in seen:
                seen.add(note)
                items.append({"category": category, "summary": note, "score": score})
    except Exception:
        pass
    return items


# ── 持仓 ATR 止损计算 ─────────────────────────────────────────────────────

def _enrich_positions_with_stop_loss(positions: list[dict]) -> list[dict]:
    """
    为每只持仓计算 ATR 动态止损价和止损触发状态。
    ATR 止损 = avg_cost × (1 - max(ATR14日% × vol_mult, stop_floor))
    vol_mult=6.0, stop_floor=0.06（与 config.yaml 一致）
    """
    try:
        import yfinance as yf
    except ImportError:
        return positions

    VOL_MULT   = 6.0    # 与 config.yaml stop_loss_vol_mult 一致
    STOP_FLOOR = 0.06   # 6% 最低止损线
    STOP_CAP   = 0.20   # 20% 最大止损线
    TP_MULT    = 8.0    # 止盈 ATR 乘数（风险回报比 > 1）
    TP_FLOOR   = 0.08   # 8% 最低止盈线
    TP_CAP     = 0.30   # 30% 最大止盈线

    enriched = []
    for p in positions:
        sym  = p.get("symbol", "")
        cost = p.get("avg_cost") or p.get("cost_price")
        cur  = p.get("market_price") or p.get("current_price")

        stop_price   = None
        stop_pct     = None
        stop_triggered = False
        tp_price     = None
        tp_pct       = None
        tp_triggered = False
        atr_note     = ""

        if cost and cur and sym:
            try:
                df = yf.Ticker(sym).history(period="20d")
                if len(df) >= 5:
                    highs = df["High"]
                    lows  = df["Low"]
                    prev_c = df["Close"].shift(1)
                    tr = pd.concat([
                        highs - lows,
                        (highs - prev_c).abs(),
                        (lows  - prev_c).abs(),
                    ], axis=1).max(axis=1)
                    atr14 = float(tr.iloc[-14:].mean()) if len(tr) >= 14 else float(tr.mean())
                    atr_pct = atr14 / float(cost)
                    # 止损
                    stop_pct = min(STOP_CAP, max(STOP_FLOOR, atr_pct * VOL_MULT))
                    stop_price = round(float(cost) * (1 - stop_pct), 1)
                    stop_triggered = float(cur) < stop_price
                    # 止盈
                    tp_pct = min(TP_CAP, max(TP_FLOOR, atr_pct * TP_MULT))
                    tp_price = round(float(cost) * (1 + tp_pct), 1)
                    tp_triggered = float(cur) >= tp_price
                    atr_note = f"ATR14={atr14:.1f} → 止损{stop_pct*100:.1f}%/止盈{tp_pct*100:.1f}%"
            except Exception:
                pass

        updated = dict(p)
        updated["stop_loss_price"]   = stop_price
        updated["stop_loss_pct"]     = round(stop_pct * 100, 1) if stop_pct else None
        updated["stop_triggered"]    = stop_triggered
        updated["take_profit_price"] = tp_price
        updated["take_profit_pct"]   = round(tp_pct * 100, 1) if tp_pct else None
        updated["tp_triggered"]      = tp_triggered
        updated["stop_note"]         = atr_note
        # pnl_pct
        if cost and cur:
            try:
                updated["pnl_pct"] = round((float(cur) / float(cost) - 1) * 100, 2)
            except Exception:
                pass
        enriched.append(updated)
    return enriched


# ── DB 数据读取 ────────────────────────────────────────────────────────────

def read_live_state(db_path: str, strategy_id: str = "default", asof: str = None) -> dict:
    """读取当前仓位、挂单、账户状态。"""
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        # 仓位 — 取 <= asof 的最新日期
        if asof:
            latest_pos_date = cur.execute(
                "SELECT MAX(asof) FROM positions WHERE strategy_id=? AND asof<=?",
                (strategy_id, asof)
            ).fetchone()[0]
        else:
            latest_pos_date = cur.execute(
                "SELECT MAX(asof) FROM positions WHERE strategy_id=?",
                (strategy_id,)
            ).fetchone()[0]

        query_pos = "SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl FROM positions WHERE strategy_id=?"
        params_pos = [strategy_id]
        if latest_pos_date:
            query_pos += " AND asof=?"
            params_pos.append(latest_pos_date)
            
        cur.execute(query_pos, params_pos)
        positions = [
            {"symbol": r[0], "qty": r[1], "avg_cost": r[2],
             "market_price": r[3], "market_value": r[4], "unrealized_pnl": r[5]}
            for r in cur.fetchall()
        ]

        # 挂单（仅 proposed/open）
        query_ord = "SELECT order_id, symbol, side, qty, limit_price, status, created_ts FROM orders WHERE strategy_id=? AND status IN ('proposed','open','pending')"
        params_ord = [strategy_id]
        if asof:
            query_ord += " AND asof=?"
            params_ord.append(asof)
        query_ord += " ORDER BY created_ts DESC"
        
        cur.execute(query_ord, params_ord)
        orders = [
            {"order_id": r[0], "symbol": r[1], "side": r[2],
             "qty": r[3], "limit_price": r[4], "status": r[5], "created_ts": r[6]}
            for r in cur.fetchall()
        ]

        # 账户快照 — 取 <= asof 的最新
        query_snap = "SELECT asof, nav, cash FROM account_snapshots WHERE strategy_id=?"
        params_snap = [strategy_id]
        if asof:
            query_snap += " AND asof<=?"
            params_snap.append(asof)
        query_snap += " ORDER BY asof DESC, ts DESC LIMIT 1"
        
        cur.execute(query_snap, params_snap)
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

def _writeback_fundamentals(symbol: str, db_path: str, tk) -> str:
    """
    将 yfinance 季报数据写回 fundamental_snapshots + feature_daily，
    使本次深析结果在下次选股时被模型利用。
    返回写入状态说明。
    """
    import subprocess, sys
    try:
        import pandas as pd
        now_ts = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")

        def _get(df, keys):
            if df is None or df.empty: return None
            for k in keys:
                if k in df.index:
                    try: return float(df.loc[k].iloc[0])
                    except: pass
            return None

        q_income   = tk.quarterly_income_stmt
        q_balance  = tk.quarterly_balance_sheet
        q_cashflow = tk.quarterly_cashflow
        info       = tk.info or {}

        fiscal_end = str(q_income.columns[0])[:10] if not q_income.empty else now_ts[:10]

        # 日本株式の yfinance: 季報は EPS のみ収録。収益・利益・CF は
        # info (TTM) と年次 financials を優先して使う
        ann_income   = tk.financials       if hasattr(tk, "financials")       else pd.DataFrame()
        ann_cashflow = tk.cashflow         if hasattr(tk, "cashflow")         else pd.DataFrame()

        def _info_or_stmt(info_key, df, keys):
            """info dict (TTM) を第一優先、年次ステートメントを第二優先"""
            v = info.get(info_key)
            if v is not None:
                try: return float(v)
                except: pass
            return _get(df, keys)

        total_rev = _info_or_stmt("totalRevenue",  ann_income,   ["Total Revenue", "Revenue"])
        op_margin = info.get("operatingMargins")
        op_income = (total_rev * op_margin) if (total_rev and op_margin) else \
                    _get(ann_income, ["Operating Income", "EBIT"])
        net_income = _info_or_stmt("netIncomeToCommon", ann_income,
                                   ["Net Income", "Net Income Common Stockholders"])
        op_cf      = _info_or_stmt("operatingCashflow", ann_cashflow, ["Operating Cash Flow"])

        rec = {
            "symbol": symbol, "fiscal_period_end": fiscal_end,
            "published_ts": now_ts, "available_ts": now_ts,
            "source": "yfinance_deep", "currency": "JPY",
            "revenue":          total_rev,
            "operating_income": op_income,
            "net_income":       net_income,
            "eps":              info.get("trailingEps"),
            "book_value_per_share": info.get("bookValue"),
            "dividend_per_share":   (info.get("dividendRate") or 0) / 4 or None,
            "operating_cf":     op_cf,
            "free_cf":          None,
            "total_assets":     _get(q_balance, ["Total Assets"]),
            "total_equity":     _get(q_balance, ["Stockholders Equity", "Common Stock Equity",
                                                  "Total Stockholder Equity"]),
            "total_debt":       _get(q_balance, ["Total Debt"]),
            "shares_outstanding": info.get("sharesOutstanding"),
            "guidance_revenue": None, "guidance_operating_income": None,
            "guidance_eps":     info.get("forwardEps"),
        }

        import sqlite3 as _sq, csv, io
        # 借用 import_csv 的写入逻辑：写成临时 CSV 再调 subprocess
        tmp_path = f"_tmp_deep_{symbol.replace('.','_')}.csv"
        df_rec = pd.DataFrame([rec])
        df_rec.to_csv(tmp_path, index=False)
        r = subprocess.run(
            [sys.executable, "update_fundamentals.py", "--db", db_path,
             "--source", "csv", "--csv_path", tmp_path],
            capture_output=True, text=True, timeout=30
        )
        import os; os.unlink(tmp_path)
        if r.returncode == 0:
            return "✓ 季报数据已写回 DB（fundamental_snapshots + feature_daily）"
        return f"⚠️ 写回失败: {r.stderr.strip()[:80]}"
    except Exception as e:
        return f"⚠️ 写回异常: {e}"


def fetch_stock_deep(symbol: str, db_path: str = None, writeback: bool = True) -> dict:
    """
    单只股票深度数据：财务指标 + 历史价格 + EPS趋势。
    writeback=True 时将季报数据写回 fundamental_snapshots，供模型下次使用。
    """
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
            "operating_margin":  info.get("operatingMargins"),
            "profit_margin_ttm": info.get("profitMargins"),
            "operating_cf":      info.get("operatingCashflow"),
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

        # 最新4季度 EPS 趋势
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

        # 写回 DB（默认开启，使深析数据进入模型反馈环）
        wb_status = ""
        if writeback and db_path:
            wb_status = _writeback_fundamentals(symbol, db_path, tk)

        return {
            "symbol":        symbol,
            "financials":    financials,
            "eps_quarters":  eps_quarters,
            "price_history": price_history,
            "writeback_status": wb_status,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ── 报告生成 ───────────────────────────────────────────────────────────────

def _run_preflight(db_path: str) -> None:
    """
    报告生成前自动刷新数据：新闻采集 + 基本面更新。
    任何步骤失败均打印警告并继续，不中断报告生成。
    """
    import subprocess, sys
    steps = [
        {
            "name": "新闻采集",
            "cmd": [sys.executable, "news_to_db.py", "--db", db_path,
                    "--lookback_hours", "26", "--sources", "kabutan,google,boj,trade,gdelt"],
        },
        {
            "name": "基本面更新",
            "cmd": [sys.executable, "update_fundamentals.py", "--db", db_path,
                    "--source", "yfinance"],
        },
    ]
    for step in steps:
        try:
            print(f"[preflight] {step['name']}...")
            r = subprocess.run(step["cmd"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                print(f"[preflight] ⚠️ {step['name']} 返回非零 ({r.returncode})，继续")
            else:
                # 只打印最后一行有效输出，避免刷屏
                out = r.stdout.strip().splitlines()
                if out:
                    print(f"[preflight] ✓ {out[-1]}")
        except Exception as e:
            print(f"[preflight] ⚠️ {step['name']} 失败: {e}，继续")


def build_briefing(mode: str, extra_symbols: List[str], skip_preflight: bool = False, strategy_id: str = "default", asof: str = None) -> dict:
    now   = now_jst()
    if asof is None:
        asof = date.today().isoformat()
    session = market_session_status(now)

    # ── 自动预刷新（新闻 + 基本面），可用 --no-refresh 跳过 ─────────────────
    if not skip_preflight:
        _run_preflight(DB_PATH)

    report: dict = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M JST"),
        "asof": asof,
        "session": session,
        "mode": mode,
        "strategy_id": strategy_id,
    }

    # ── Regime 检测（所有模式都执行）────────────────────────────────────────
    report["regime"] = _detect_market_regime()

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
            news_val = _cross_validate_with_news(DB_PATH, sym)
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
                "news_validation":   news_val,
            })
        report["candidates"] = enriched
        # 宏观新闻（BOJ/TRADE）统一采集一次，不挂个股
        report["macro_news"] = _get_macro_news(DB_PATH)
        report["screener_meta"] = {
            "count":              screener_result.get("count", 0),
            "hard_vetoed_count":  screener_result.get("fundamental_overlay", {}).get("hard_vetoed_count", 0),
            "downweighted_count": screener_result.get("fundamental_overlay", {}).get("downweighted_count", 0),
        }

        # 仓位 & 挂单（含 ATR 止损计算）
        live = read_live_state(DB_PATH, strategy_id=strategy_id, asof=asof)
        if live.get("positions"):
            live["positions"] = _enrich_positions_with_stop_loss(live["positions"])
        report["live_state"] = live

        # 最新信号（来自 DB）
        report["db_signals"] = read_latest_signals(DB_PATH, top_n=15)

    # ── 模式：stock（个股深析）───────────────────────────────────────────
    if mode in ("stock", "full") and extra_symbols:
        report["stock_analysis"] = [fetch_stock_deep(s, db_path=DB_PATH, writeback=True) for s in extra_symbols]

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


def write_report_v2(report: dict) -> tuple[Path, Path]:
    """
    输出 v2 格式报告（6节固定结构），供 nexus / Discord 链路解析。
    输出文件：briefing_v2_latest.md / briefing_v2_latest.json
    """
    REPORT_DIR.mkdir(exist_ok=True)
    json_path = REPORT_DIR / "briefing_v2_latest.json"
    md_path   = REPORT_DIR / "briefing_v2_latest.md"

    # ── JSON：nexus schema ─────────────────────────────────────────────────
    regime    = report.get("regime", {})
    live      = report.get("live_state", {})
    positions = live.get("positions", [])
    candidates = report.get("candidates", [])
    account   = live.get("account", {})

    # 持仓对象（v2 规范字段）— 使用 _enrich_positions_with_stop_loss 计算的完整数据
    pos_v2 = []
    for p in positions:
        pnl_pct = p.get("pnl_pct")
        if pnl_pct is None:
            try:
                pnl_pct = round(p["unrealized_pnl"] / (p["avg_cost"] * p["qty"]) * 100, 2) \
                          if p.get("avg_cost") and p.get("qty") else None
            except Exception:
                pass
        action = "HOLD"
        if p.get("stop_triggered"):
            action = "STOP_LOSS"
        elif p.get("tp_triggered"):
            action = "TAKE_PROFIT"
        pos_v2.append({
            "symbol":            p.get("symbol"),
            "qty":               p.get("qty"),
            "cost_price":        p.get("avg_cost"),
            "current_price":     p.get("market_price"),
            "pnl_pct":           pnl_pct,
            "stop_loss_price":   p.get("stop_loss_price"),
            "stop_loss_pct":     p.get("stop_loss_pct"),
            "stop_triggered":    p.get("stop_triggered", False),
            "take_profit_price": p.get("take_profit_price"),
            "take_profit_pct":   p.get("take_profit_pct"),
            "tp_triggered":      p.get("tp_triggered", False),
            "stop_note":         p.get("stop_note", ""),
            "action_hint":       action,
        })

    # 操作指令：从挂单提取
    orders_v2 = []
    for o in live.get("orders", []):
        orders_v2.append({
            "symbol": o.get("symbol"),
            "action": o.get("side", "").upper(),
            "price":  o.get("limit_price"),
            "qty":    o.get("qty"),
            "reason": "来自现有挂单",
        })

    # 风险提示（宏观新闻只汇总一次，公司级负面按标的列出）
    risk_alerts = []
    if regime.get("trend") == "DOWN":
        risk_alerts.append(f"大盘下行趋势，{regime.get('action_bias','')}")
    if regime.get("vol_level") == "HIGH":
        risk_alerts.append(f"波动率异常偏高 (5日ATR {regime.get('atr_5d_pct')}% vs 60日均值 {regime.get('atr_60d_pct')}%)")
    macro_news = report.get("macro_news", [])
    if macro_news:
        cats = list({mn["category"] for mn in macro_news})
        risk_alerts.append(f"宏观负面：存在 {'/'.join(cats)} 类负面新闻，详见二节情报")
    for c in candidates:
        nv = c.get("news_validation", {})
        if nv.get("news_risk") == "HIGH":
            risk_alerts.append(f"{c['symbol']} 公司级负面新闻: {'; '.join(nv.get('news_notes', []))}")

    nexus_json = {
        "date":         report.get("asof"),
        "generated_at": report.get("generated_at"),
        "session":      report.get("session", {}),
        "regime":       regime,
        "market":       report.get("market", {}),
        "positions":    pos_v2,
        "candidates":   candidates,
        "orders":       orders_v2,
        "risk_alerts":  risk_alerts,
        "account":      account,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nexus_json, f, ensure_ascii=False, indent=2, default=str)

    # ── Markdown：固定6节结构 ──────────────────────────────────────────────
    mkt       = report.get("market", {})
    etf       = mkt.get("nikkei_etf", {})
    session   = report.get("session", {})

    # 趋势/偏向中文映射
    trend_cn  = {"UP": "上涨", "DOWN": "下跌", "SIDEWAYS": "盘整", "UNKNOWN": "未知"}
    bias_cn   = {"BULLISH": "偏多", "BEARISH": "偏空", "NEUTRAL": "中性"}
    vol_cn    = {"HIGH": "高波动", "NORMAL": "正常", "LOW": "低波动", "UNKNOWN": "未知"}

    lines = [
        f"# 每日量化简报  {report.get('generated_at', '')}",
        f"> 交易时段: **{session.get('session','N/A')}**  |  距收盘: {session.get('mins_to_close','N/A')} 分钟",
        "",
        "---",
        "",
        "## 零、隔夜跨资产信号",
        "",
    ]

    # 跨资产信号 (from DB)
    try:
        from cross_asset_signals import load_latest_cross_asset
        ca = load_latest_cross_asset(conn)
        if ca:
            sp_str = f"S&P500: {ca.get('sp500_close', 'N/A')} ({ca.get('sp500_overnight_pct', 0):+.2f}%)" if ca.get('sp500_close') else "S&P500: N/A"
            uj_str = f"USD/JPY: {ca.get('usdjpy', 'N/A')} ({ca.get('usdjpy_change_pct', 0):+.2f}%)" if ca.get('usdjpy') else "USD/JPY: N/A"
            vix_str = f"VIX: {ca.get('vix_close', 'N/A')} ({ca.get('vix_change_pct', 0):+.2f}%)" if ca.get('vix_close') else "VIX: N/A"
            nk_str = f"NK Futures: {ca.get('nk_futures', 'N/A')} ({ca.get('nk_futures_gap_pct', 0):+.2f}%)" if ca.get('nk_futures') else "NK Futures: N/A"
            score = ca.get('cross_asset_score', 0.5)
            adj = ca.get('regime_adjustment', 'neutral')
            adj_cn = {"upgrade": "偏多", "neutral": "中性", "downgrade": "偏空"}
            oil_str = f"WTI: {ca.get('crude_oil', 'N/A')} ({ca.get('crude_oil_change_pct', 0):+.2f}%)" if ca.get('crude_oil') else "WTI: N/A"
            gold_str = f"Gold: {ca.get('gold', 'N/A')} ({ca.get('gold_change_pct', 0):+.2f}%)" if ca.get('gold') else "Gold: N/A"
            copper_str = f"Copper: {ca.get('copper', 'N/A')} ({ca.get('copper_change_pct', 0):+.2f}%)" if ca.get('copper') else "Copper: N/A"
            sox_str = f"SOX: {ca.get('sox', 'N/A')} ({ca.get('sox_change_pct', 0):+.2f}%)" if ca.get('sox') else "SOX: N/A"
            lines += [
                f"- {sp_str}  {uj_str}",
                f"- {vix_str}  {nk_str}",
                f"- {oil_str}  {gold_str}",
                f"- {copper_str}  {sox_str}",
                f"- 跨资产 regime 信号: **{score:.2f}** ({adj_cn.get(adj, adj)}) → 建议 {adj.upper()}",
                "",
            ]
        else:
            lines += ["- (无跨资产数据，请运行 `python cross_asset_signals.py`)", ""]
    except Exception:
        lines += ["- (跨资产模块未就绪)", ""]

    # 宏观事件检测 (from macro_events table)
    try:
        from macro_event_detector import load_active_event
        asof_for_event = report.get("asof") or datetime.now().strftime("%Y-%m-%d")
        macro_ev = load_active_event(conn, asof_for_event)
        if macro_ev and macro_ev.get("alert_level") in ("L1", "L2"):
            level = macro_ev["alert_level"]
            level_icon = "🔴" if level == "L1" else "🟡"
            eboost = macro_ev.get("effective_boost", 0)
            decay = macro_ev.get("decay", 1.0)
            days_el = macro_ev.get("days_elapsed", 0)
            dur = macro_ev.get("duration_days", 3)
            ev_summary = macro_ev.get("event_summary") or "(规则引擎检测)"
            lines += [
                f"### 宏观事件检测 {level_icon}",
                f"- **{level}级事件** (来源日: {macro_ev.get('event_asof', 'N/A')})",
                f"- 事件: {ev_summary}",
                f"- regime boost: {macro_ev.get('original_boost', 0):+.4f} × 衰减{decay:.2f} = **{eboost:+.4f}**",
                f"- 持续: {days_el}/{dur} 天已过",
                "",
            ]
        else:
            lines += ["### 宏观事件检测", "- 无活跃宏观事件", ""]
    except Exception:
        lines += ["### 宏观事件检测", "- (宏观事件模块未就绪)", ""]

    lines += [
        "## 一、市场状态",
        f"- 日经ETF ({NIKKEI_ETF}): **{etf.get('cur','N/A')}**  "
        f"({'+' if (etf.get('chg_pct') or 0) >= 0 else ''}{etf.get('chg_pct','N/A')}%)",
        f"- 日内区间: H {etf.get('high','N/A')} / L {etf.get('low','N/A')}  |  量能异常倍数: {etf.get('volume_spike_ratio','N/A')}x",
        f"- 趋势: **{trend_cn.get(regime.get('trend',''), regime.get('trend',''))}**  "
        f"偏向: **{bias_cn.get(regime.get('bias',''), regime.get('bias',''))}**  "
        f"波动: **{vol_cn.get(regime.get('vol_level',''), regime.get('vol_level',''))}**",
        f"- SMA5={regime.get('sma5','N/A')} / SMA20={regime.get('sma20','N/A')}  "
        f"ATR5日={regime.get('atr_5d_pct','N/A')}% vs ATR60日均={regime.get('atr_60d_pct','N/A')}%",
        f"- **操作基调**: {regime.get('action_bias', '观望')}",
        "",
    ]

    # ── 二、今日有效情报 ──────────────────────────────────────────────────
    lines += ["## 二、今日有效情报", ""]
    news_shown = False
    # 宏观新闻（BOJ/TRADE）— 展示一次，不重复挂个股
    macro_news = report.get("macro_news", [])
    cat_cn = {"BOJ": "日银/货币政策", "TRADE": "贸易/关税"}
    for mn in macro_news:
        cat_label = cat_cn.get(mn["category"], mn["category"])
        lines.append(f"- [🟡 {cat_label}] {mn['summary']}")
        news_shown = True
    # 公司级负面新闻（HIGH risk，针对特定标的）
    for c in candidates:
        nv = c.get("news_validation", {})
        if nv.get("news_risk") == "HIGH":
            for note in nv.get("news_notes", []):
                lines.append(f"- [🔴 公司负面] **{c['symbol']}**: {note}")
                news_shown = True
    if not news_shown:
        lines.append("- 过去48h内无重要负面新闻（仅限数据库已采集新闻）")
    lines.append("")

    # ── 三、持仓健康度 ────────────────────────────────────────────────────
    lines += ["## 三、持仓健康度", ""]
    if pos_v2:
        lines.append("| 代码 | 持股数 | 成本价 | 当前价 | 浮盈亏% | 止损线 | 止盈线 | 状态 |")
        lines.append("|------|--------|--------|--------|---------|--------|--------|------|")
        for p in pos_v2:
            pnl_str = f"{p['pnl_pct']:+.2f}%" if p["pnl_pct"] is not None else "N/A"
            stop_str = f"¥{p['stop_loss_price']:.0f}(-{p['stop_loss_pct']}%)" if p.get('stop_loss_price') else "N/A"
            tp_str = f"¥{p['take_profit_price']:.0f}(+{p['take_profit_pct']}%)" if p.get('take_profit_price') else "N/A"
            if p.get('stop_triggered'):
                status = "⚠️止损"
            elif p.get('tp_triggered'):
                status = "🎯止盈"
            else:
                status = "HOLD"
            lines.append(
                f"| {p['symbol']} | {p['qty']:.0f} | {p['cost_price']:.1f} "
                f"| {p['current_price']:.1f} | {pnl_str} | {stop_str} | {tp_str} | {status} |"
            )
    else:
        lines.append("- 当前空仓")
    cash = account.get("cash_balance") or account.get("cash")
    nav  = account.get("nav")
    if cash or nav:
        lines.append(f"\n现金余额: ¥{cash:,.0f}" if cash else "")
        lines.append(f"组合 NAV: ¥{nav:,.0f}" if nav else "")
    lines.append("")

    # ── 四、候选信号全量 ──────────────────────────────────────────────────
    lines += [f"## 四、候选信号（共{len(candidates)}只）", ""]
    lines.append("| 排名 | 代码 | 调整分 | 基本面× | 今日% | 超额% | 20日% | 新闻风险 | 基本面注记 |")
    lines.append("|------|------|--------|---------|-------|-------|-------|---------|------------|")
    risk_icons = {"NONE": "—", "SECTOR": "🟡宏观", "HIGH": "🔴负面"}
    for r in candidates:
        nv = r.get("news_validation", {})
        nr = risk_icons.get(nv.get("news_risk", "NONE"), "—")
        lines.append(
            f"| {r['rank']} | {r['symbol']} | {r['score_adjusted']} "
            f"| {r['fundamental_score']} | {r.get('chg_pct','—')}% "
            f"| {r.get('excess_vs_market','—')}% | {r.get('ret_20d','—')}% "
            f"| {nr} | {r.get('fundamental_note','') or '正常'} |"
        )
    lines.append("")

    # ── 五、今日操作指令 ──────────────────────────────────────────────────
    lines += ["## 五、今日操作指令", ""]
    if orders_v2:
        for o in orders_v2:
            lines.append(f"- **{o['action']}** {o['symbol']}  {o['qty']}股 @ ¥{o['price']}  "
                         f"（{o['reason']}）")
    else:
        lines.append("- 当前无挂单，维持观望")
    lines.append("")

    # ── 六、风险提示 ──────────────────────────────────────────────────────
    lines += ["## 六、风险提示", ""]
    if risk_alerts:
        for alert in risk_alerts:
            lines.append(f"- ⚠️ {alert}")
    else:
        lines.append("- 暂无特殊风险提示")
    lines.append("")
    # ── 七、个股深析（仅当 --symbols 指定时出现）────────────────────────────
    stock_analysis = report.get("stock_analysis", [])
    if stock_analysis:
        lines += ["", "---", "## 七、个股深析", ""]
        for s in stock_analysis:
            if "error" in s:
                lines.append(f"### {s['symbol']}  ⚠️ 数据获取失败: {s['error']}")
                continue
            f = s.get("financials", {})
            wb = s.get("writeback_status", "")
            lines += [
                f"### {s['symbol']}  {f.get('name','')}  [{f.get('sector','')}]",
                f"- 营业利润率: **{f.get('operating_margin','N/A')}**  "
                f"净利润率: {f.get('profit_margin_ttm','N/A')}  "
                f"OCF: {f.get('operating_cf','N/A')}",
                f"- Forward PE: {f.get('forward_pe','N/A')}  "
                f"PBR: {f.get('pbr','N/A')}  "
                f"D/E: {f.get('debt_to_equity','N/A')}",
                f"- 营收增长: {f.get('revenue_growth','N/A')}  "
                f"盈利增长: {f.get('earnings_growth','N/A')}  "
                f"股息率: {f.get('dividend_yield','N/A')}",
                f"- 52W: {f.get('52w_low','N/A')} – {f.get('52w_high','N/A')}  "
                f"市值: {f.get('market_cap_jpy','N/A')}",
            ]
            eps = s.get("eps_quarters", [])
            if eps:
                eps_str = "  →  ".join(
                    f"{e['period'][:7]} **{e['diluted_eps']:+.2f}**" for e in eps
                )
                # 判断EPS趋势方向
                vals = [e["diluted_eps"] for e in eps]
                if len(vals) >= 2:
                    trend_note = "📈 改善" if vals[0] > vals[-1] else "📉 恶化" if vals[0] < vals[-1] else "→ 持平"
                    lines.append(f"- 季度EPS趋势({trend_note}): {eps_str}")
                else:
                    lines.append(f"- 季度EPS: {eps_str}")
            if wb:
                lines.append(f"- 模型反馈: {wb}")
            lines.append("")

        # 按日期保存个股深析报告（不覆盖）
        stock_dir = REPORT_DIR / "stock_analysis"
        stock_dir.mkdir(exist_ok=True)
        today_str = report.get("asof", "")
        syms = "_".join(s["symbol"].replace(".","") for s in stock_analysis if "error" not in s)
        if syms:
            stock_path = stock_dir / f"{today_str}_{syms}.md"
            stock_lines = [f"# 个股深析报告  {report.get('generated_at','')}  —  {syms}", ""]
            stock_lines += lines[lines.index("## 七、个股深析"):]
            with open(stock_path, "w", encoding="utf-8") as f:
                f.write("\n".join(stock_lines))
            print(f"[briefing] 个股报告 → {stock_path}")

    lines.append("---")
    lines.append("*本报告由 worker-quant v2 自动生成，数据来源 yfinance（约15分钟延迟）。投资决策请结合实际情况判断。*")

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
    ap.add_argument("--output-version", choices=["v1", "v2"], default="v2",
                    help="报告格式版本: v2（默认，6节结构）| v1（旧版兼容）")
    ap.add_argument("--no-refresh", action="store_true",
                    help="跳过自动数据刷新（新闻/基本面），直接用现有DB数据生成报告")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--strategy_id", default="sprint", help="Strategy ID to report on (e.g. default, sprint)")
    ap.add_argument("--asof", default=None, help="Target date (YYYY-MM-DD). Defaults to today.")
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
    report = build_briefing(args.mode, extra, skip_preflight=args.no_refresh, strategy_id=args.strategy_id, asof=args.asof)
    json_p, md_p = write_report(report)          # 始终写 v1（保持兼容）
    if args.output_version == "v2" or True:      # v2 默认同时输出
        write_report_v2(report)

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
        positions = report.get("live_state", {}).get("positions", [])
        orders = report.get("live_state", {}).get("orders", [])
        if positions:
            print(f"持仓: {len(positions)}只 " + ", ".join(p['symbol'] for p in positions))
        if orders:
            for o in orders:
                print(f"挂单: {o['symbol']} {o['side']} {o['qty']}股 @ {o['limit_price']} [{o['status']}]")
        if not positions and not orders:
            print("当前: 空仓无挂单")


if __name__ == "__main__":
    main()
