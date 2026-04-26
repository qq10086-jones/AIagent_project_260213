"""Screener: SQLite -> candidate list

Reads OHLCV from SQLite, computes simple quality/liquidity/volatility filters, and outputs
- selected_tickers.json
- (optional) writes signals table back to SQLite

Design goal: fast + explainable + robust.

Usage:
  python screener.py --db japan_market.db --asof 2026-02-01 --out selected_tickers.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from market_db_v2 import MarketDB

_YFINANCE_PER_SYMBOL_TIMEOUT = 8
_YFINANCE_TOTAL_TIMEOUT = 60

@dataclass
class ScreenConfig:
    lookback_days: int = 252
    adv_window: int = 20              # average daily $ volume window
    min_adv: float = 20_000_000       # 20m JPY-ish (proxy if close is JPY) - adjust to your needs
    max_missing: float = 0.02         # allow up to 2% missing in lookback
    vol_window: int = 20
    min_vol: float = 0.005            # 0.5% daily
    max_vol: float = 0.06             # 6% daily
    top_k: int = 50                   # keep best K by score (after hard filters)
    lot_size_default: int = 100       # 日本股票标准手数
    max_cost_per_lot: float = 150_000 # 1手成本上限(JPY)；100万日元需持≥6只才能分散
    version: str = "screener_v2"
    # ── Alpha 评分配置 ──
    use_alpha_scoring: bool = True    # False 时回退旧流动性排序逻辑
    alpha_weights: dict = field(default_factory=lambda: {
        "mom_20": 0.25,
        "mom_60": 0.15,
        "vol_adj_mom20": 0.15,
        "sharpe_20": 0.10,
        "vol_z": 0.15,
        "adv_rank": 0.10,
        "high52w_rank": 0.10,
    })


@dataclass
class FundamentalOverlayConfig:
    """
    基本面叠加层：评分降权而非硬否决。

    设计原则：
    - 核心看营业利润率（operating margin）和经营现金流，而非净利润
    - 净利润受一次性事项（并购摊销、重组费、非经常损益）扭曲，不适合做硬门槛
    - 连续季度 EPS 亏损是信号，但要看是经营性亏损还是账面调整
    - 只有在"营业利润率深度为负 AND 经营现金流为负"时才触发硬否决（真正经营恶化）
    - 其余情况用 fundamental_score(0~1) 乘进技术分，保留排名但降低权重
    """
    enabled: bool = True

    # ── 硬否决条件（同时满足才否决）──────────────────────────────────────
    # 营业利润率低于此值（经营层面持续亏损，不是会计调整）
    hard_veto_operating_margin: float = -0.15    # -15%
    # 且经营现金流为负（现金不撒谎）
    hard_veto_require_negative_ocf: bool = True

    # ── 软降权：营业利润率惩罚 ─────────────────────────────────────────
    # 低于此值开始降权（高于则不惩罚）
    op_margin_warn: float = -0.05               # -5% 以下开始降权
    # 每低 1% 营业利润率扣除的分数
    op_margin_penalty_per_pct: float = 0.03     # 每低1%扣3分

    # ── 软降权：连续季度 EPS 为负 ─────────────────────────────────────
    # 最多检查最近几个季度
    eps_quarters_checked: int = 2
    # 每个亏损季度扣除分数（会计亏损，轻罚）
    eps_loss_penalty_per_quarter: float = 0.08  # 每季扣8%

    # ── 软降权：高杠杆 ────────────────────────────────────────────────
    high_leverage_threshold: float = 150.0      # D/E > 150% 开始降权
    high_leverage_penalty: float = 0.08         # 扣 8%

    # ── 无数据处理 ────────────────────────────────────────────────────
    # 无基本面数据时赋予的默认分数（1.0=不惩罚，<1=保守）
    no_data_score: float = 0.90
    # yfinance 回退
    yfinance_fallback: bool = True


def _asof_str(s: Optional[str]) -> str:
    if s:
        return s
    return date.today().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 基本面否决层
# ---------------------------------------------------------------------------

def _fetch_fundamental_from_db(db_path: str, symbols: List[str], asof: str) -> Dict[str, dict]:
    """从 DB fundamental_snapshots 派生关键指标。"""
    result: Dict[str, dict] = {}
    try:
        conn = sqlite3.connect(db_path)
        ph = ",".join("?" for _ in symbols)
        rows = pd.read_sql_query(
            f"""
            SELECT symbol, fiscal_period_end, net_income, total_equity, total_debt,
                   total_assets, eps, operating_income, revenue, operating_cf
            FROM fundamental_snapshots
            WHERE symbol IN ({ph})
              AND available_ts <= ?
            ORDER BY symbol, fiscal_period_end DESC
            """,
            conn,
            params=list(symbols) + [asof + "T23:59:59"],
        )
        conn.close()
        if rows.empty:
            return result

        for sym, grp in rows.groupby("symbol"):
            latest = grp.iloc[0]
            revenue = float(latest.get("revenue") or 0)
            op_income = float(latest.get("operating_income") or 0)
            ocf = latest.get("operating_cf")
            equity = float(latest.get("total_equity") or 0)
            debt = float(latest.get("total_debt") or 0)

            op_margin = (op_income / revenue) if revenue and revenue != 0 else None
            d2e = (debt / equity) if equity and equity != 0 else None
            ocf_val = float(ocf) if ocf is not None else None
            eps_list = [float(r) for r in grp["eps"].dropna().head(4).tolist()]

            result[str(sym)] = {
                "operating_margin": op_margin,
                "operating_cf": ocf_val,
                "debt_to_equity": d2e,
                "eps_quarters": eps_list,
                "source": "db",
            }
    except Exception as e:
        print(f"[screener][fundamental] DB 读取失败: {e}", file=sys.stderr)
    return result


def _fetch_one_yfinance(sym: str) -> tuple[str, dict | None]:
    try:
        import yfinance as yf
        tk = yf.Ticker(sym)
        info = tk.info or {}
        op_margin = info.get("operatingMargins")
        ocf = info.get("operatingCashflow")
        d2e = info.get("debtToEquity")
        if d2e is not None:
            d2e = d2e / 100.0

        eps_quarters: List[float] = []
        try:
            qi = tk.quarterly_income_stmt
            if not qi.empty and "Diluted EPS" in qi.index:
                eps_quarters = [float(v) for v in qi.loc["Diluted EPS"].dropna().head(4).tolist()]
        except Exception as e:
            print(f"[screener][fundamental] {sym} yfinance EPS fetch failed: {e}", file=sys.stderr)

        return sym, {
            "operating_margin": op_margin,
            "operating_cf": float(ocf) if ocf is not None else None,
            "debt_to_equity": d2e,
            "eps_quarters": eps_quarters,
            "source": "yfinance",
        }
    except Exception as e:
        print(f"[screener][fundamental] {sym} yfinance fetch failed: {e}", file=sys.stderr)
        return sym, None


def _fetch_fundamental_from_yfinance(symbols: List[str]) -> Dict[str, dict]:
    """yfinance fallback with bounded per-symbol and total waits."""
    result: Dict[str, dict] = {}
    if not symbols:
        return result

    deadline = time.monotonic() + _YFINANCE_TOTAL_TIMEOUT
    executor = _cf.ThreadPoolExecutor(max_workers=4)
    fut_to_sym = {executor.submit(_fetch_one_yfinance, sym): sym for sym in symbols}
    try:
        try:
            for fut in _cf.as_completed(fut_to_sym, timeout=_YFINANCE_TOTAL_TIMEOUT):
                sym = fut_to_sym[fut]
                if time.monotonic() > deadline:
                    print("[screener][fundamental] yfinance total timeout reached", file=sys.stderr)
                    break
                try:
                    got_sym, data = fut.result(timeout=_YFINANCE_PER_SYMBOL_TIMEOUT)
                except _cf.TimeoutError:
                    print(f"[screener][fundamental] {sym} yfinance per-symbol timeout", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"[screener][fundamental] {sym} yfinance worker failed: {e}", file=sys.stderr)
                    continue
                if data is not None:
                    result[got_sym] = data
        except _cf.TimeoutError:
            print("[screener][fundamental] yfinance total timeout reached", file=sys.stderr)
    finally:
        for fut, sym in fut_to_sym.items():
            if not fut.done():
                fut.cancel()
                print(f"[screener][fundamental] {sym} yfinance canceled after timeout", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
    return result


def _fetch_fundamental_from_yfinance_legacy(symbols: List[str]) -> Dict[str, dict]:
    """yfinance 回退：拉取营业利润率、经营现金流、D/E、季度EPS。"""
    result: Dict[str, dict] = {}
    try:
        import yfinance as yf
    except ImportError:
        print("[screener][fundamental] yfinance 未安装，跳过回退", file=sys.stderr)
        return result

    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            info = tk.info or {}
            op_margin = info.get("operatingMargins")   # TTM 营业利润率
            ocf = info.get("operatingCashflow")        # TTM 经营现金流
            d2e = info.get("debtToEquity")
            if d2e is not None:
                d2e = d2e / 100.0  # yfinance 返回百分比，统一转为小数比率

            eps_quarters: List[float] = []
            try:
                qi = tk.quarterly_income_stmt
                if not qi.empty and "Diluted EPS" in qi.index:
                    eps_quarters = [float(v) for v in qi.loc["Diluted EPS"].dropna().head(4).tolist()]
            except Exception:
                pass

            result[sym] = {
                "operating_margin": op_margin,
                "operating_cf": float(ocf) if ocf is not None else None,
                "debt_to_equity": d2e,
                "eps_quarters": eps_quarters,
                "source": "yfinance",
            }
        except Exception as e:
            print(f"[screener][fundamental] {sym} yfinance 拉取失败: {e}", file=sys.stderr)
    return result


def compute_fundamental_score(
    fdata: dict,
    ocfg: FundamentalOverlayConfig,
) -> Tuple[float, str]:
    """
    计算单只股票的基本面评分 (0.0–1.0) 及说明。

    评分逻辑：
    - 从 1.0 开始，根据各项风险指标扣分
    - 硬否决（0.0）：营业利润率深度为负 AND 经营现金流为负
    - 其余为软降权，保留排名但减小权重
    返回 (score, reason_str)
    """
    score = 1.0
    reasons: List[str] = []

    op_margin = fdata.get("operating_margin")
    ocf = fdata.get("operating_cf")
    d2e = fdata.get("debt_to_equity")
    eps_q = fdata.get("eps_quarters", [])

    # ── 硬否决：真正经营恶化（营业层面深度亏损 + 现金流告急）──────────
    if (op_margin is not None and np.isfinite(op_margin)
            and op_margin < ocfg.hard_veto_operating_margin):
        if not ocfg.hard_veto_require_negative_ocf or (ocf is not None and ocf < 0):
            return 0.0, f"硬否决:营业利润率{op_margin*100:.1f}%且OCF为负"

    # ── 软降权①：营业利润率低于警戒线 ──────────────────────────────
    if op_margin is not None and np.isfinite(op_margin) and op_margin < ocfg.op_margin_warn:
        shortfall_pct = (ocfg.op_margin_warn - op_margin) * 100  # 超出警戒线的幅度（%）
        penalty = min(shortfall_pct * ocfg.op_margin_penalty_per_pct, 0.40)
        score -= penalty
        reasons.append(f"营业利润率{op_margin*100:.1f}%(降权-{penalty*100:.0f}%)")

    # ── 软降权②：最近季度 EPS 亏损（会计亏损，轻罚）─────────────────
    n_check = min(ocfg.eps_quarters_checked, len(eps_q))
    loss_quarters = sum(1 for e in eps_q[:n_check] if e < 0)
    if loss_quarters > 0:
        penalty = loss_quarters * ocfg.eps_loss_penalty_per_quarter
        score -= penalty
        reasons.append(f"近{n_check}季{loss_quarters}季EPS亏损(降权-{penalty*100:.0f}%)")

    # ── 软降权③：高杠杆 ───────────────────────────────────────────
    if d2e is not None and np.isfinite(d2e) and d2e > ocfg.high_leverage_threshold:
        score -= ocfg.high_leverage_penalty
        reasons.append(f"D/E={d2e:.0f}(降权-{ocfg.high_leverage_penalty*100:.0f}%)")

    score = max(score, 0.05)  # 最低保留 5%，不完全归零（除硬否决外）
    reason_str = "; ".join(reasons) if reasons else "基本面正常"
    return round(score, 4), reason_str


def apply_fundamental_overlay(
    db_path: str,
    candidates: List[str],
    asof: str,
    ocfg: FundamentalOverlayConfig,
) -> Dict[str, Tuple[float, str]]:
    """
    对候选股票列表计算基本面评分。
    返回 {symbol: (fundamental_score, reason)}
    score=1.0 表示基本面健康无惩罚，score=0.0 表示硬否决
    """
    if not ocfg.enabled or not candidates:
        return {s: (1.0, "基本面叠加未启用") for s in candidates}

    db_data = _fetch_fundamental_from_db(db_path, candidates, asof)
    missing_syms = [s for s in candidates if s not in db_data]

    yf_data: Dict[str, dict] = {}
    if missing_syms and ocfg.yfinance_fallback:
        print(f"[screener][fundamental] DB 无数据，yfinance 回退: {missing_syms}")
        yf_data = _fetch_fundamental_from_yfinance(missing_syms)

    results: Dict[str, Tuple[float, str]] = {}
    for sym in candidates:
        fdata = db_data.get(sym) or yf_data.get(sym)
        if fdata is None:
            results[sym] = (ocfg.no_data_score, f"无基本面数据(默认分{ocfg.no_data_score})")
        else:
            results[sym] = compute_fundamental_score(fdata, ocfg)
    return results


# ---------------------------------------------------------------------------
# 主筛选函数
# ---------------------------------------------------------------------------

def _compute_alpha_factors(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """截面 alpha 因子计算（用于 screener 评分排序）。

    返回 DataFrame: index=symbol, columns=[mom_20, mom_60, vol_adj_mom20, sharpe_20, vol_z, high52w_rank]
    所有值为截面 rank percentile (0-1)。

    注意: vol_z 和 high52w_rank 是 sprint_entry_check 的必要条件，
    必须在 screener 层选入，否则 downstream entry gate 会全部 fail。
    """
    ret1 = close.pct_change(fill_method=None)

    # mom_20: 20日收益率
    mom_20_raw = close.iloc[-1] / close.iloc[-21].where(close.iloc[-21] > 0) - 1.0 \
        if len(close) > 21 else pd.Series(0.0, index=close.columns)

    # mom_60: 60日收益率
    mom_60_raw = close.iloc[-1] / close.iloc[-61].where(close.iloc[-61] > 0) - 1.0 \
        if len(close) > 61 else pd.Series(0.0, index=close.columns)

    # vol_adj_mom20: 波动率调整动量
    vol_20 = ret1.tail(20).std()
    vol_adj_mom20_raw = mom_20_raw / (vol_20 + 1e-12)

    # sharpe_20: 20日滚动夏普
    mean_20 = ret1.tail(20).mean()
    sharpe_20_raw = (mean_20 / (vol_20 + 1e-12)) * np.sqrt(252)

    # vol_z: 20日成交量 z-score（sprint_entry_check 要求 > 0.50）
    vol_mean = vol.rolling(60).mean()
    vol_std = vol.rolling(60).std()
    vol_z_raw = ((vol.iloc[-1] - vol_mean.iloc[-1]) / (vol_std.iloc[-1] + 1e-12)) \
        if len(vol) > 60 else pd.Series(0.0, index=close.columns)

    # high52w: 距52周高点距离（sprint_entry_check 要求 > -0.10）
    high_252 = close.tail(252).max() if len(close) >= 252 else close.max()
    high52w_raw = close.iloc[-1] / high_252 - 1.0

    factors = pd.DataFrame({
        "mom_20": mom_20_raw,
        "mom_60": mom_60_raw,
        "vol_adj_mom20": vol_adj_mom20_raw,
        "sharpe_20": sharpe_20_raw,
        "vol_z": vol_z_raw,
        "high52w_rank": high52w_raw,
    })

    # 截面 rank percentile 标准化 (0-1)
    for col in factors.columns:
        factors[col] = factors[col].rank(pct=True, na_option="bottom")

    return factors


def screen(
    db_path: str,
    symbols: Optional[List[str]],
    asof: str,
    out_path: Optional[str],
    cfg: ScreenConfig,
    write_db: bool = True,
    overlay_cfg: Optional[FundamentalOverlayConfig] = None,
) -> Dict:
    db = MarketDB(db_path)

    # universe
    if not symbols:
        symbols = db.list_symbols()

    # Pull close/vol (trim to [asof-lookback, asof])
    end = pd.to_datetime(asof)
    start = end - pd.Timedelta(days=int(cfg.lookback_days * 1.6))  # buffer for non-trading days
    close, vol = db.get_close_vol_multi(symbols, start=start.strftime("%Y-%m-%d"), end=asof)

    if close.empty:
        raise RuntimeError("DB returned empty price data. Run db_update.py first.")

    # keep only last lookback trading rows
    close = close.loc[:end].tail(cfg.lookback_days)
    vol = vol.reindex(close.index)

    # metrics
    ret1 = close.pct_change(fill_method=None)
    vol_d = ret1.rolling(cfg.vol_window).std()

    adv = (close * vol).rolling(cfg.adv_window).mean()   # proxy currency volume
    adv_last = adv.iloc[-1]
    vol_last = vol_d.iloc[-1]
    close_last = close.iloc[-1]
    missing = close.isna().mean()

    # ── Stage 1: Hard Filter (pass/fail, 不参与评分) ──────────────
    rows = []
    for sym in close.columns:
        m_missing = float(missing.get(sym, 1.0))
        m_adv = float(adv_last.get(sym, np.nan))
        m_vol = float(vol_last.get(sym, np.nan))
        m_close = float(close_last.get(sym, np.nan))

        reasons = []
        hard_fail = False

        if not np.isfinite(m_adv) or m_adv < cfg.min_adv:
            hard_fail = True
            reasons.append(f"ADV<{cfg.min_adv:g}")
        if m_missing > cfg.max_missing:
            hard_fail = True
            reasons.append(f"missing>{cfg.max_missing:.2%}")
        if not np.isfinite(m_vol) or (m_vol < cfg.min_vol) or (m_vol > cfg.max_vol):
            hard_fail = True
            reasons.append(f"vol_out_of_range({cfg.min_vol:.3f}-{cfg.max_vol:.3f})")
        if np.isfinite(m_close):
            cost_per_lot = m_close * cfg.lot_size_default
            if cost_per_lot > cfg.max_cost_per_lot:
                hard_fail = True
                reasons.append(f"1手成本过高:{cost_per_lot:,.0f}JPY>{cfg.max_cost_per_lot:,.0f}")

        cost_per_lot = m_close * cfg.lot_size_default if np.isfinite(m_close) else np.nan

        # legacy score（use_alpha_scoring=False 回退用）
        legacy_score = 0.0
        if np.isfinite(m_adv):
            legacy_score += np.log1p(max(m_adv, 0.0))
        if np.isfinite(m_vol):
            legacy_score -= abs(np.log(max(m_vol, 1e-8)) - np.log(0.02))
        legacy_score -= m_missing * 50.0

        rows.append((sym, legacy_score, hard_fail, "; ".join(reasons),
                      m_adv, m_vol, m_missing, m_close, cost_per_lot))

    df = pd.DataFrame(rows, columns=["symbol", "score", "hard_fail", "reason",
                                      "adv", "vol", "missing", "close", "cost_per_lot"])

    # 通过硬过滤的候选（不限 top_k，让 alpha 评分来排序）
    passed = df.loc[~df["hard_fail"]].copy()

    # ── Stage 2: Alpha Score (用于排序) ───────────────────────────
    if cfg.use_alpha_scoring and not passed.empty and len(passed) > 5:
        alpha_factors = _compute_alpha_factors(close, vol)
        passed = passed.merge(
            alpha_factors.reset_index().rename(columns={"index": "symbol"}),
            on="symbol", how="left",
        )
        # ADV rank percentile（流动性加分项）
        passed["adv_rank"] = passed["adv"].rank(pct=True, na_option="bottom")

        # 加权 alpha score
        w = cfg.alpha_weights
        passed["alpha_score"] = 0.0
        for factor_name, weight in w.items():
            if factor_name in passed.columns:
                passed["alpha_score"] += passed[factor_name].fillna(0.5) * weight

        passed["score"] = passed["alpha_score"]
        passed = passed.sort_values("score", ascending=False)
    else:
        # 回���旧逻���
        passed = passed.sort_values("score", ascending=False)

    # 取 top_k
    tech_passed = passed.head(cfg.top_k).copy()

    # ── 基本面叠加层 ─────────────────────────────────────────────
    _ocfg = overlay_cfg if overlay_cfg is not None else FundamentalOverlayConfig()
    if _ocfg.enabled and not tech_passed.empty:
        candidate_syms = tech_passed["symbol"].tolist()
        overlay = apply_fundamental_overlay(db_path, candidate_syms, asof, _ocfg)

        tech_passed = tech_passed.copy()
        tech_passed["fundamental_score"] = tech_passed["symbol"].map(
            lambda s: overlay.get(s, (1.0, ""))[0]
        )
        tech_passed["fundamental_note"] = tech_passed["symbol"].map(
            lambda s: overlay.get(s, (1.0, ""))[1]
        )
        # fundamental_score 参与 alpha scoring 时已作为因子纳入，
        # 此处仅用于硬否决和最终调整
        tech_passed["score_adjusted"] = tech_passed["score"] * tech_passed["fundamental_score"]

        hard_vetoed_mask = tech_passed["fundamental_score"] == 0.0
        if hard_vetoed_mask.any():
            for sym in tech_passed.loc[hard_vetoed_mask, "symbol"]:
                note = overlay[sym][1]
                print(f"[screener][fundamental] 硬否决 {sym}: {note}")

        kept = tech_passed.loc[~hard_vetoed_mask].sort_values("score_adjusted", ascending=False).copy()
        vetoed_df = tech_passed.loc[hard_vetoed_mask].copy()

        downweighted = tech_passed.loc[
            (~hard_vetoed_mask) & (tech_passed["fundamental_score"] < 1.0)
        ]
        for _, row in downweighted.iterrows():
            print(f"[screener][fundamental] 降权 {row['symbol']} "
                  f"(×{row['fundamental_score']:.2f}): {row['fundamental_note']}")
    else:
        kept = tech_passed.copy()
        kept["fundamental_score"] = 1.0
        kept["fundamental_note"] = ""
        kept["score_adjusted"] = kept["score"]
        vetoed_df = pd.DataFrame(columns=kept.columns)

    payload = {
        "asof": asof,
        "version": cfg.version,
        "count": int(len(kept)),
        "symbols": kept["symbol"].tolist(),
        "details": kept.to_dict(orient="records"),
        "hard_vetoed": vetoed_df[["symbol","fundamental_note"]].to_dict(orient="records") if not vetoed_df.empty else [],
        "filters": {
            "lookback_days": cfg.lookback_days,
            "min_adv": cfg.min_adv,
            "max_missing": cfg.max_missing,
            "min_vol": cfg.min_vol,
            "max_vol": cfg.max_vol,
            "top_k": cfg.top_k,
            "lot_size_default": cfg.lot_size_default,
            "max_cost_per_lot": cfg.max_cost_per_lot,
            "use_alpha_scoring": cfg.use_alpha_scoring,
        },
        "screening_stats": {
            "universe_size": len(df),
            "hard_filter_passed": len(passed),
            "top_k_selected": len(tech_passed),
        },
        "fundamental_overlay": {
            "enabled": _ocfg.enabled,
            "hard_vetoed_count": int(len(vetoed_df)),
            "downweighted_count": int((kept["fundamental_score"] < 1.0).sum()) if not kept.empty else 0,
        },
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if write_db:
        try:
            db.save_signals(asof, [(r["symbol"], r["score"], r.get("reason",""), cfg.version) for r in payload["details"]])
        except Exception:
            pass

    db.close()
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--out", default="selected_tickers.json")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--minadv", type=float, default=20_000_000)
    ap.add_argument("--maxcost", type=float, default=150_000,
                    help="Maximum cost per standard lot in JPY")
    ap.add_argument("--symbols", default=None, help="comma-separated symbols (optional)")
    ap.add_argument("--no_db_write", action="store_true")
    # 基本面叠加参数
    ap.add_argument("--no_fundamental_overlay", action="store_true", help="禁用基本面叠加层")
    ap.add_argument("--hard_veto_op_margin", type=float, default=-0.15,
                    help="硬否决营业利润率门槛（默认-0.15即-15%%，需同时OCF为负）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = ScreenConfig(
        top_k=args.topk,
        min_adv=args.minadv,
        max_cost_per_lot=args.maxcost,
    )
    ocfg = FundamentalOverlayConfig(
        enabled=not args.no_fundamental_overlay,
        hard_veto_operating_margin=args.hard_veto_op_margin,
    )
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    asof = _asof_str(args.asof)

    res = screen(args.db, syms, asof, args.out, cfg, write_db=not args.no_db_write, overlay_cfg=ocfg)
    ov = res.get("fundamental_overlay", {})
    print(f"Screener done: {res['count']} tickers -> {args.out}  "
          f"(基本面硬否决:{ov.get('hard_vetoed_count',0)}只  降权:{ov.get('downweighted_count',0)}只)")
