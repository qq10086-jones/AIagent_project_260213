from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from trade_schema import connect, ensure_trade_tables

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


SNAPSHOT_NUMERIC_COLUMNS = [
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "book_value_per_share",
    "dividend_per_share",
    "operating_cf",
    "free_cf",
    "total_assets",
    "total_equity",
    "total_debt",
    "shares_outstanding",
    "guidance_revenue",
    "guidance_operating_income",
    "guidance_eps",
]

EVENT_NUMERIC_COLUMNS = [
    "revenue_yoy",
    "operating_income_yoy",
    "eps_yoy",
    "guidance_delta_revenue",
    "guidance_delta_op",
    "guidance_delta_eps",
    "surprise_score",
]


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    try:
        num_f = float(num)
        den_f = float(den)
    except Exception:
        return None
    if abs(den_f) <= 1e-12:
        return None
    return num_f / den_f


def _coerce_float(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _optional_text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _validate_required(frame: pd.DataFrame, required: list[str], fail_closed: bool) -> bool:
    missing = [col for col in required if col not in frame.columns]
    if not missing:
        return True
    message = f"Missing required CSV columns: {', '.join(missing)}"
    if fail_closed:
        raise ValueError(message)
    print(f"[fundamentals] skip import: {message}")
    return False


def _latest_close(conn, symbol: str, asof: str) -> float | None:
    row = conn.execute(
        """
        SELECT close
        FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _build_feature_daily_rows(conn, raw: pd.DataFrame, require_available_ts: bool) -> list[dict]:
    rows: list[dict] = []

    snapshot_mask = raw["fiscal_period_end"].str.strip() != ""
    for rec in raw.loc[snapshot_mask].to_dict(orient="records"):
        available_ts = str(rec.get("available_ts") or "").strip()
        if not available_ts:
            continue
        asof = available_ts[:10]
        symbol = str(rec["symbol"])
        price = _latest_close(conn, symbol, asof)

        # ── Sloan应计比率（反转存储）: (OCF - 净利润) / 总资产 ──────────────
        # 越高越好：OCF >> NI 表示利润由真实现金流支撑（盈利质量高）
        # 越低越差：NI >> OCF 表示利润由会计应计虚增，Sloan(1996)证明预示未来盈利下修
        # 注意：不能用 _safe_div 因为分子需要两项相减
        _ocf = rec.get("operating_cf")
        _ni  = rec.get("net_income")
        _ta  = rec.get("total_assets")
        accruals_inv_val: float | None = None
        if _ocf is not None and _ni is not None and _ta is not None:
            try:
                _ta_f = float(_ta)
                if abs(_ta_f) > 1e-12:
                    accruals_inv_val = (float(_ocf) - float(_ni)) / _ta_f
            except Exception:
                pass

        features = {
            # 账面市值比（价值因子）
            "value_bp":        _safe_div(rec.get("book_value_per_share"), price),

            # 经营性ROA = 营业利润 / 总资产
            # 替代原 quality_roe(净利润/净资产)：
            #   ① 营业利润剔除了并购摊销/重组/非经常损益等一次性扰动
            #   ② 除以总资产而非净资产，消除杠杆倍增效应，行业间可比性更强
            "roa_op":          _safe_div(rec.get("operating_income"), rec.get("total_assets")),

            # 经营现金流资产回报率 = OCF / 总资产
            # 替代原 quality_cfo(OCF/净利润)：
            #   原因子在净利润为负时符号完全反转（OCF正、NI负 → 比率负 → z-score极低），
            #   误将"利润因摊销为负但现金流健康"的公司打为最差；
            #   OCF/总资产始终方向正确，与其他因子尺度一致
            "cfo_assets":      _safe_div(rec.get("operating_cf"), rec.get("total_assets")),

            # Sloan应计比率（反转）= (OCF - 净利润) / 总资产（见上方计算）
            "accruals_inv":    accruals_inv_val,

            # 营业利润率（已有，保持）
            "margin_op":       _safe_div(rec.get("operating_income"), rec.get("revenue")),

            # 财务安全系数 = 净资产 / 有息负债（已有，保持）
            "leverage_safety": _safe_div(rec.get("total_equity"), rec.get("total_debt")),

            # 股息率（已有，保持）
            "dividend_yield":  _safe_div(rec.get("dividend_per_share"), price),
        }
        for feature_name, value in features.items():
            if value is None:
                continue
            rows.append(
                {
                    "asof": asof,
                    "symbol": symbol,
                    "feature_name": feature_name,
                    "value": float(value),
                    "source_fact_ids": f"fundamental_snapshots:{symbol}:{rec.get('fiscal_period_end')}:{rec.get('published_ts')}",
                }
            )

    if "event_type" not in raw.columns:
        return rows

    event_mask = raw["event_type"].fillna("").astype(str).str.strip() != ""
    for rec in raw.loc[event_mask].to_dict(orient="records"):
        available_ts = str(rec.get("available_ts") or "").strip()
        if (not available_ts) and (not require_available_ts):
            available_ts = str(rec.get("published_ts") or "").strip()
        if not available_ts:
            continue
        asof = available_ts[:10]
        symbol = str(rec["symbol"])
        event_features = {
            "growth_rev_yoy": rec.get("revenue_yoy"),
            "growth_op_yoy": rec.get("operating_income_yoy"),
            "guidance_delta": rec.get("guidance_delta_eps"),
        }
        for feature_name, value in event_features.items():
            try:
                value_f = float(value)
            except Exception:
                continue
            rows.append(
                {
                    "asof": asof,
                    "symbol": symbol,
                    "feature_name": feature_name,
                    "value": value_f,
                    "source_fact_ids": f"earnings_events:{symbol}:{rec.get('published_ts')}:{rec.get('event_type')}",
                }
            )
    return rows


def import_csv(db_path: str, csv_path: Path, fail_closed: bool, require_available_ts: bool) -> None:
    if not csv_path.exists():
        message = f"CSV not found: {csv_path}"
        if fail_closed:
            raise FileNotFoundError(message)
        print(f"[fundamentals] {message}; skip import.")
        return

    raw = pd.read_csv(csv_path)
    if raw.empty:
        print("[fundamentals] CSV is empty; nothing to import.")
        return
    if not _validate_required(raw, ["symbol", "published_ts"], fail_closed):
        return

    raw = raw.copy()
    raw["symbol"] = raw["symbol"].astype(str)
    raw["published_ts"] = raw["published_ts"].astype(str)
    raw["available_ts"] = _optional_text(raw, "available_ts")
    raw["fiscal_period_end"] = _optional_text(raw, "fiscal_period_end")
    raw["source"] = _optional_text(raw, "source", default="csv")
    raw["currency"] = _optional_text(raw, "currency", default="JPY")
    raw = _coerce_float(raw, SNAPSHOT_NUMERIC_COLUMNS + EVENT_NUMERIC_COLUMNS)

    if fail_closed and (raw["available_ts"].str.strip() == "").any():
        raise ValueError("available_ts is required in fail-closed mode.")

    snapshot_mask = raw["fiscal_period_end"].str.strip() != ""
    event_mask = raw.get("event_type", pd.Series(index=raw.index, dtype=object)).fillna("").astype(str).str.strip() != ""

    with connect(db_path) as conn:
        ensure_trade_tables(conn)

        if snapshot_mask.any():
            snapshots = raw.loc[snapshot_mask].copy()
            conn.executemany(
                """
                INSERT OR REPLACE INTO fundamental_snapshots (
                  symbol, fiscal_period_end, published_ts, available_ts, source, currency,
                  revenue, operating_income, net_income, eps, book_value_per_share, dividend_per_share,
                  operating_cf, free_cf, total_assets, total_equity, total_debt, shares_outstanding,
                  guidance_revenue, guidance_operating_income, guidance_eps
                ) VALUES (
                  :symbol, :fiscal_period_end, :published_ts, :available_ts, :source, :currency,
                  :revenue, :operating_income, :net_income, :eps, :book_value_per_share, :dividend_per_share,
                  :operating_cf, :free_cf, :total_assets, :total_equity, :total_debt, :shares_outstanding,
                  :guidance_revenue, :guidance_operating_income, :guidance_eps
                )
                """,
                snapshots.to_dict(orient="records"),
            )
            print(f"[fundamentals] imported snapshot rows: {int(snapshot_mask.sum())}")

        if event_mask.any():
            events = raw.loc[event_mask].copy()
            events["event_type"] = events["event_type"].fillna("").astype(str)
            events["headline"] = _optional_text(events, "headline")
            conn.executemany(
                """
                INSERT OR REPLACE INTO earnings_events (
                  symbol, published_ts, event_type, headline,
                  revenue_yoy, operating_income_yoy, eps_yoy,
                  guidance_delta_revenue, guidance_delta_op, guidance_delta_eps,
                  surprise_score, source
                ) VALUES (
                  :symbol, :published_ts, :event_type, :headline,
                  :revenue_yoy, :operating_income_yoy, :eps_yoy,
                  :guidance_delta_revenue, :guidance_delta_op, :guidance_delta_eps,
                  :surprise_score, :source
                )
                """,
                events.to_dict(orient="records"),
            )
            print(f"[fundamentals] imported event rows: {int(event_mask.sum())}")

        feature_rows = _build_feature_daily_rows(conn, raw, require_available_ts=require_available_ts)
        if feature_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO feature_daily (
                  asof, symbol, feature_name, value, source_fact_ids
                ) VALUES (
                  :asof, :symbol, :feature_name, :value, :source_fact_ids
                )
                """,
                feature_rows,
            )
            print(f"[fundamentals] upserted feature_daily rows: {len(feature_rows)}")
        conn.commit()


def _fetch_yfinance_fundamentals(db_path: str) -> pd.DataFrame:
    """通过 yfinance 抓取所有 tickers 的最新季度基本面数据，生成与 import_csv 兼容的 DataFrame。"""
    import sqlite3 as _sq
    import yfinance as yf

    with _sq.connect(db_path) as _conn:
        rows = _conn.execute("SELECT DISTINCT symbol FROM tickers").fetchall()
    symbols = [r[0] for r in rows if r[0]]

    records = []
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            info = tk.info or {}

            # 季度财务报表
            try:
                q_income = tk.quarterly_income_stmt
            except Exception:
                q_income = pd.DataFrame()
            try:
                q_balance = tk.quarterly_balance_sheet
            except Exception:
                q_balance = pd.DataFrame()
            try:
                q_cashflow = tk.quarterly_cashflow
            except Exception:
                q_cashflow = pd.DataFrame()

            # 确定最近季报日期
            if not q_income.empty:
                latest_date = q_income.columns[0]
            elif not q_balance.empty:
                latest_date = q_balance.columns[0]
            else:
                latest_date = pd.Timestamp.now()

            fiscal_period_end = str(latest_date)[:10]
            # yfinance 无真实 PIT 信息，以今日作为 available_ts（保守处理）
            now_ts = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
            available_ts = now_ts
            published_ts = now_ts

            def _get(df: pd.DataFrame, keys: list) -> "float | None":
                """在 DataFrame 中按候选行名依序查找第一列的值。"""
                if df.empty:
                    return None
                for k in keys:
                    if k in df.index:
                        try:
                            return float(df.loc[k].iloc[0])
                        except Exception:
                            pass
                return None

            revenue      = _get(q_income,   ["Total Revenue", "Revenue"])
            op_income    = _get(q_income,   ["Operating Income", "EBIT"])
            net_income   = _get(q_income,   ["Net Income", "Net Income Common Stockholders"])
            total_assets = _get(q_balance,  ["Total Assets"])
            total_equity = _get(q_balance,  ["Stockholders Equity", "Common Stock Equity", "Total Stockholder Equity"])
            total_debt   = _get(q_balance,  ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
            shares_out   = _get(q_balance,  ["Share Issued", "Ordinary Shares Number"])
            op_cf        = _get(q_cashflow, ["Operating Cash Flow", "Cash Flows From Used In Operating Activities Direct Method"])

            # 每股数据从 info 补充
            eps   = info.get("trailingEps")
            bvps  = info.get("bookValue")
            div_r = info.get("dividendRate")
            div_ps = div_r / 4 if div_r else None  # 年化转季度近似
            if shares_out is None:
                shares_out = info.get("sharesOutstanding")
            guidance_eps = info.get("forwardEps")

            records.append({
                "symbol":                    sym,
                "fiscal_period_end":         fiscal_period_end,
                "published_ts":              published_ts,
                "available_ts":              available_ts,
                "source":                    "yfinance",
                "currency":                  "JPY",
                "revenue":                   revenue,
                "operating_income":          op_income,
                "net_income":                net_income,
                "eps":                       eps,
                "book_value_per_share":      bvps,
                "dividend_per_share":        div_ps,
                "operating_cf":              op_cf,
                "free_cf":                   None,
                "total_assets":              total_assets,
                "total_equity":              total_equity,
                "total_debt":                total_debt,
                "shares_outstanding":        shares_out,
                "guidance_revenue":          None,
                "guidance_operating_income": None,
                "guidance_eps":              guidance_eps,
            })
        except Exception as exc:
            print(f"[fundamentals] yfinance error for {sym}: {exc}")

    print(f"[fundamentals] yfinance: 获取 {len(records)}/{len(symbols)} 只标的基本面数据")
    return pd.DataFrame(records) if records else pd.DataFrame()


def import_yfinance(db_path: str, fail_closed: bool, require_available_ts: bool) -> None:
    """通过 yfinance 抓取基本面并写入 DB（零付费依赖）。"""
    try:
        import yfinance  # noqa: F401
    except ImportError as exc:
        msg = f"yfinance 未安装: {exc}"
        if fail_closed:
            raise RuntimeError(msg) from exc
        print(f"[fundamentals] {msg}")
        return

    raw = _fetch_yfinance_fundamentals(db_path)
    if raw.empty:
        print("[fundamentals] yfinance 未返回任何数据。")
        return

    # 通过临时 CSV 复用 import_csv 的全部逻辑（包含 _build_feature_daily_rows）
    temp_csv = Path(db_path).with_name("_tmp_yf_fundamentals.csv")
    raw.to_csv(temp_csv, index=False)
    try:
        import_csv(db_path, temp_csv, fail_closed=fail_closed, require_available_ts=False)
    finally:
        try:
            temp_csv.unlink()
        except Exception:
            pass


def _fetch_jquants_v2_fundamentals(db_path: str) -> pd.DataFrame:
    '''通过 jquantsapi 抓取最新季报基本面数据的强壮版（支持断点续传、429退避）'''
    import sqlite3 as _sq
    import time
    try:
        import jquantsapi
    except ImportError as exc:
        raise RuntimeError(f"jquantsapi 未安装: {exc}")

    api_key = os.getenv("JQUANTS_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 JQUANTS_API_KEY 环境变量。请在配置或环境里加入。")
        
    client = jquantsapi.ClientV2(api_key=api_key)

    with _sq.connect(db_path) as _conn:
        # 获取所有目标股票
        rows = _conn.execute("SELECT DISTINCT symbol FROM tickers").fetchall()
        all_symbols = [r[0] for r in rows if r[0]]
        
        # 断点跳过（Resumable Fetch）机制：保护免费额度
        rs = _conn.execute("""
            SELECT DISTINCT symbol FROM fundamental_snapshots
            WHERE source = 'jquants_v2' 
              AND available_ts > datetime('now', '-14 days')
        """).fetchall()
        fetched_symbols = {r[0] for r in rs if r[0]}

    symbols = sorted([s for s in all_symbols if s not in fetched_symbols])
    print(f"[fundamentals] jquants_v2: 剔除 {len(fetched_symbols)} 只最近已更新的标的，剩余需获取: {len(symbols)}/{len(all_symbols)}")

    records = []
    for i, sym in enumerate(symbols):
        sym_prefix = sym.split(".")[0] if "." in sym else sym
        print(f"[fundamentals] jquants_v2: Fetching {sym} ({i+1}/{len(symbols)})...")
        
        retries = 0
        max_retries = 3
        df = None
        
        while retries <= max_retries:
            try:
                df = client.get_fin_summary(code=sym_prefix)
                break
            except Exception as exc:
                err_msg = str(exc).lower()
                if "429" in err_msg or "rate limit" in err_msg or "too many" in err_msg or "exceeded" in err_msg:
                    sleep_time = 30 * (2 ** retries)
                    print(f"[fundamentals] ⚠️ 触发 429 Rate Limit ({sym}). 休眠 {sleep_time} 秒后重试 ({retries+1}/{max_retries})...")
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    print(f"[fundamentals] jquants_v2 error for {sym}: {exc}")
                    break
        
        if retries > max_retries:
            print(f"[fundamentals] 🚨 J-Quants 遭到持续封禁，自动中止任务。正在将已获取的 {len(records)} 只股票紧急落库...")
            break
            
        if df is not None and not df.empty:
            try:
                df = df.sort_values("DisclosedDate", ascending=True)
                latest = df.iloc[-1]

                published_ts = latest.get("DisclosedDate")
                if not published_ts or type(published_ts) == float:
                    published_ts = __import__("pandas").Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")

                available_ts = published_ts
                fiscal_period_end = latest.get("CurrentPeriodEndDate", latest.get("CurPerEn", ""))

                def _safe_float(val):
                    try: return float(val) if val is not None else None
                    except: return None

                op_income = _safe_float(latest.get("OperatingProfit", latest.get("OP")))
                revenue = _safe_float(latest.get("NetSales", latest.get("Sales")))
                op_cf = _safe_float(latest.get("CashFlowsFromOperatingActivities", latest.get("CFO")))
                net_income = _safe_float(latest.get("ProfitLossAttributableToOwnersOfParent", latest.get("NP")))
                bvps = _safe_float(latest.get("BookValuePerShare", latest.get("BPS")))
                eps = _safe_float(latest.get("EarningsPerShare", latest.get("EPS")))
                total_assets = _safe_float(latest.get("TotalAssets"))
                total_equity = _safe_float(latest.get("Equity", latest.get("TotalEquity")))

                records.append({
                    "symbol":                    sym,
                    "fiscal_period_end":         fiscal_period_end,
                    "published_ts":              published_ts,
                    "available_ts":              available_ts,
                    "source":                    "jquants_v2",
                    "currency":                  "JPY",
                    "revenue":                   revenue,
                    "operating_income":          op_income,
                    "net_income":                net_income,
                    "eps":                       eps,
                    "book_value_per_share":      bvps,
                    "dividend_per_share":        None,
                    "operating_cf":              op_cf,
                    "free_cf":                   None,
                    "total_assets":              total_assets,
                    "total_equity":              total_equity,
                    "total_debt":                None,
                    "shares_outstanding":        None,
                    "guidance_revenue":          None,
                    "guidance_operating_income": None,
                    "guidance_eps":              None,
                })
            except Exception as exc:
                print(f"[fundamentals] jquants_v2 parsing error for {sym}: {exc}")
        
        if i < len(symbols) - 1:
            time.sleep(12.5)

    print(f"[fundamentals] jquants_v2: 实际成功抓取 {len(records)} 只标的")
    import pandas as pd
    return pd.DataFrame(records) if records else pd.DataFrame()
def import_jquants_v2(db_path: str, fail_closed: bool, require_available_ts: bool) -> None:
    """通过 jquantsapi 抓取基本面并写入 DB。"""
    raw = _fetch_jquants_v2_fundamentals(db_path)
    if raw.empty:
        print("[fundamentals] jquants_v2 未返回任何数据。")
        return

    temp_csv = Path(db_path).with_name("_tmp_jq_fundamentals.csv")
    raw.to_csv(temp_csv, index=False)
    try:
        import_csv(db_path, temp_csv, fail_closed=fail_closed, require_available_ts=False)
    finally:
        try:
            temp_csv.unlink()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Initialize or import point-in-time fundamentals")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--source", default="yfinance", choices=["noop", "csv", "yfinance", "jquants_v2"],
                    help="数据源: yfinance（默认，免费）| jquants_v2（高质量季报）| csv（手动导入）| noop（仅建表）")
    ap.add_argument("--csv_path", default=None)
    ap.add_argument("--fail_closed", action="store_true")
    ap.add_argument("--require_available_ts", action="store_true")
    args = ap.parse_args()

    with connect(args.db) as conn:
        ensure_trade_tables(conn)
    print("[fundamentals] PIT tables ensured.")

    if args.source == "csv":
        if not args.csv_path:
            if args.fail_closed:
                raise ValueError("--csv_path is required when --source csv and --fail_closed is set.")
            print("[fundamentals] csv source selected without --csv_path; schema only.")
            return
        import_csv(args.db, Path(args.csv_path), args.fail_closed, args.require_available_ts)
        return

    if args.source == "yfinance":
        import_yfinance(args.db, fail_closed=args.fail_closed, require_available_ts=args.require_available_ts)
        return
        
    if args.source == "jquants_v2":
        import_jquants_v2(args.db, fail_closed=args.fail_closed, require_available_ts=args.require_available_ts)
        return

    print("[fundamentals] noop mode; schema only.")


if __name__ == "__main__":
    main()
