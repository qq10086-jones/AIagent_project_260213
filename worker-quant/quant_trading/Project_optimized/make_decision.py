import argparse
import csv
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd

from compute_ic import (
    TECHNICAL_FACTOR_NAMES,
    RISK_ADJUSTED_FACTOR_NAMES,
    FUNDAMENTAL_FACTOR_NAMES,
    compute_features,
    sector_neutral_zscore,
)
from trade_schema import connect, ensure_trade_tables, get_latest_trading_day
from market_data_utils import refresh_market_data_if_needed, latest_db_date
from kelly_sizer import compute_kelly_params
from sprint_signal import sprint_exit_check_v2


@dataclass
class OrderRow:
    symbol: str
    side: str          # BUY/SELL
    qty: int
    suggested_type: str = "MKT"
    suggested_limit: Optional[float] = None
    est_notional: float = 0.0
    comment: str = "rebalance"


def compute_entry_zone(
    symbol: str,
    signal_side: str,
    last_close: float,
    atr_pct: float | None,
) -> dict:
    """限価区間建議を計算。ATR ベースで target / aggressive / walk_away を算出。"""
    if atr_pct is None or atr_pct <= 0:
        atr_pct = 0.02  # fallback 2%
    atr_pct = min(atr_pct, 0.10)  # cap at 10%

    if signal_side == "BUY":
        target = last_close * (1.0 - atr_pct * 0.3)       # 回調 0.3 ATR 待ち
        aggressive = last_close * (1.0 + atr_pct * 0.2)    # 追い 0.2 ATR まで
        walk_away = last_close * (1.0 + atr_pct * 1.0)     # 1 ATR 超 → 放棄
    else:  # SELL
        target = last_close * (1.0 + atr_pct * 0.3)
        aggressive = last_close * (1.0 - atr_pct * 0.2)
        walk_away = last_close * (1.0 - atr_pct * 1.0)

    return {
        "target_limit": round(target, 1),
        "aggressive_limit": round(aggressive, 1),
        "walk_away_price": round(walk_away, 1),
    }


FACTOR_FAMILY_MAP = {
    "price": list(TECHNICAL_FACTOR_NAMES),
    "risk_adjusted": list(RISK_ADJUSTED_FACTOR_NAMES),
    "fundamental": list(FUNDAMENTAL_FACTOR_NAMES),
}


def _now_iso() -> str:
    # local time is fine; you can pin Asia/Tokyo later if desired
    return datetime.now().isoformat(timespec="seconds")


def _make_run_id(asof: str, config_text: str = "") -> str:
    base = f"{_now_iso()}|{asof}|{config_text}".encode("utf-8")
    h = hashlib.sha1(base).hexdigest()[:10]
    return f"{asof}__{h}"


def _read_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _last_close(conn: sqlite3.Connection, symbol: str, asof: str) -> Tuple[Optional[str], Optional[float]]:
    row = conn.execute(
        """
        SELECT date, close
        FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    if not row:
        return None, None
    return row[0], float(row[1])


def _latest_positions(conn: sqlite3.Connection, asof: str, strategy_id: str = "default") -> Tuple[Optional[str], Dict[str, float]]:
    # pick latest positions date <= asof
    row = conn.execute(
        "SELECT asof FROM positions WHERE asof<=? AND strategy_id=? ORDER BY asof DESC LIMIT 1",
        (asof, strategy_id),
    ).fetchone()
    if not row:
        return None, {}
    pos_date = row[0]
    rows = conn.execute(
        "SELECT symbol, qty FROM positions WHERE asof=? AND strategy_id=?",
        (pos_date, strategy_id),
    ).fetchall()
    return pos_date, {sym: float(q) for sym, q in rows}


def _latest_account_snapshot(conn: sqlite3.Connection, asof: str, strategy_id: str = "default") -> Tuple[Optional[str], Optional[float], Optional[float]]:
    row = conn.execute(
        """
        SELECT asof, cash, nav
        FROM account_snapshots
        WHERE asof<=? AND strategy_id=?
        ORDER BY asof DESC
        LIMIT 1
        """,
        (asof, strategy_id),
    ).fetchone()
    if not row:
        return None, None, None
    snap_asof = str(row[0])

    # ── NAV 鮮度チェック: snapshot 以降に fills があれば自動再構築 ──
    try:
        newer_fills = conn.execute(
            "SELECT COUNT(*) FROM fills WHERE asof > ? AND strategy_id=?",
            (snap_asof, strategy_id),
        ).fetchone()
        if newer_fills and newer_fills[0] > 0:
            print(f"[risk] ⚠️  account_snapshot stale: snapshot asof={snap_asof} but "
                  f"{newer_fills[0]} fills found after that date — auto-rebuilding")
            try:
                from build_account_snapshot import build_account_snapshot as _rebuild_snap
                # 逐日重建从 snap_asof 之后到 asof 的每一天的 snapshot
                fill_dates = conn.execute(
                    "SELECT DISTINCT asof FROM fills WHERE asof > ? AND strategy_id=? AND asof <= ? ORDER BY asof",
                    (snap_asof, strategy_id, asof),
                ).fetchall()
                for (fill_asof,) in fill_dates:
                    # 找该日 fills 的 run_id（取最新的）
                    rid_row = conn.execute(
                        "SELECT run_id FROM fills WHERE asof=? AND strategy_id=? ORDER BY ts DESC LIMIT 1",
                        (fill_asof, strategy_id),
                    ).fetchone()
                    rid = rid_row[0] if rid_row else "manual"
                    _rebuild_snap(conn, rid, fill_asof, strategy_id=strategy_id)
                    print(f"  >> rebuilt snapshot for {fill_asof} (run_id={rid})")
                # 重新查询
                row = conn.execute(
                    "SELECT asof, cash, nav FROM account_snapshots WHERE asof<=? AND strategy_id=? ORDER BY asof DESC LIMIT 1",
                    (asof, strategy_id),
                ).fetchone()
                if row:
                    return str(row[0]), float(row[1]), float(row[2])
            except Exception as e:
                print(f"[risk] ⚠️  auto-rebuild failed ({e}), using stale snapshot — "
                      "run build_account_snapshot.py manually!")
    except sqlite3.OperationalError:
        pass  # fills 表可能还没建

    return snap_asof, float(row[1]), float(row[2])


def _compute_atr_pct(conn: sqlite3.Connection, symbol: str, asof: str, window: int = 20) -> Optional[float]:
    """计算 ATR 占收盘价百分比。数据不足时返回 None。"""
    rows = conn.execute(
        """
        SELECT high, low, close FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT ?
        """,
        (symbol, asof, window + 1),
    ).fetchall()
    if len(rows) < 2:
        return None
    # rows are newest-first; reverse for chronological order
    rows = list(reversed(rows))
    true_ranges = []
    for i in range(1, len(rows)):
        h, l, _ = float(rows[i][0]), float(rows[i][1]), float(rows[i][2])
        prev_c = float(rows[i - 1][2])
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)
    if not true_ranges:
        return None
    atr = sum(true_ranges) / len(true_ranges)
    latest_close = float(rows[-1][2])
    if latest_close <= 0:
        return None
    return atr / latest_close


def _get_positions_with_cost(
    conn: sqlite3.Connection, asof: str, strategy_id: str = "default"
) -> Dict[str, Dict]:
    """读取持仓含 avg_cost 和 high_since_entry。"""
    row = conn.execute(
        "SELECT asof FROM positions WHERE asof<=? AND strategy_id=? ORDER BY asof DESC LIMIT 1",
        (asof, strategy_id),
    ).fetchone()
    if not row:
        return {}
    pos_date = row[0]
    rows = conn.execute(
        """
        SELECT symbol, qty, avg_cost, market_price,
               COALESCE(high_since_entry, market_price) as high_since_entry
        FROM positions WHERE asof=? AND strategy_id=? AND qty > 0
        """,
        (pos_date, strategy_id),
    ).fetchall()
    result = {}
    for sym, qty, avg_cost, mkt_price, high in rows:
        result[str(sym)] = {
            "qty": float(qty),
            "avg_cost": float(avg_cost or 0.0),
            "market_price": float(mkt_price or 0.0),
            "high_since_entry": float(high or mkt_price or 0.0),
        }
    return result


def _check_stop_losses(
    conn: sqlite3.Connection,
    asof: str,
    strategy_id: str,
    stop_loss_config: dict,
) -> Tuple[List[str], List[Dict]]:
    """检查所有持仓是否触发止损/止盈，返回 (forced_exit_symbols, diagnostics)。"""
    positions = _get_positions_with_cost(conn, asof, strategy_id)
    if not positions:
        return [], []

    forced_exits = []
    diagnostics = []
    for sym, pos in positions.items():
        _, current_price = _last_close(conn, sym, asof)
        if current_price is None:
            continue
        atr_pct = _compute_atr_pct(conn, sym, asof, window=int(stop_loss_config.get("atr_window", 20)))

        # 构造最小 row 供 sprint_exit_check_v2 使用
        row = pd.Series({"vol_z": 0.0, "holding_period_target": 999})
        should_exit, reason = sprint_exit_check_v2(
            row,
            holding_days=0,  # 不在这里检查持有天数
            benchmark_state="on",  # 不在这里检查 regime
            avg_cost=pos["avg_cost"],
            current_price=current_price,
            high_since_entry=pos["high_since_entry"],
            atr_pct=atr_pct,
            stop_loss_config=stop_loss_config,
        )

        diag = {
            "symbol": sym,
            "avg_cost": pos["avg_cost"],
            "current_price": current_price,
            "high_since_entry": pos["high_since_entry"],
            "atr_pct": atr_pct,
            "triggered": should_exit,
            "reason": reason,
        }
        diagnostics.append(diag)
        if should_exit:
            forced_exits.append(sym)

    return forced_exits, diagnostics


def _load_target_weights(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected format: symbol,target_weight
    if "symbol" in df.columns and "target_weight" in df.columns:
        out = df[["symbol", "target_weight"]].copy()
        out["target_weight"] = out["target_weight"].astype(float)
        return out
    # fallback: if someone saved a 1-row wide format, try to parse
    if df.shape[0] == 1:
        out = pd.DataFrame({"symbol": df.columns, "target_weight": df.iloc[0].astype(float).values})
        return out
    raise ValueError(f"Unrecognized target weights format: {path}")


def _load_target_meta(reports_dir: Path) -> dict:
    meta_path = reports_dir / "target_weights_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _infer_target_mode(reports_dir: Path, target_meta: dict) -> str:
    mode = str(target_meta.get("primary_mode", "") or "").strip()
    if mode:
        return mode
    promo_path = reports_dir / "promotion_decision.json"
    if promo_path.exists():
        try:
            payload = json.loads(promo_path.read_text(encoding="utf-8"))
            mode = str(payload.get("target_mode", "") or "").strip()
            if mode:
                return mode
        except Exception:
            pass
    return "unknown"


def _load_sector_map(conn: sqlite3.Connection) -> Dict[str, str]:
    try:
        rows = conn.execute("SELECT symbol, sector FROM tickers").fetchall()
    except Exception:
        return {}
    return {str(symbol): str(sector or "Unknown") for symbol, sector in rows}


def _apply_single_name_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    out = weights.astype(float).copy()
    max_weight = float(max_weight)
    if out.empty:
        return out
    if max_weight <= 0.0:
        return out * 0.0
    if max_weight >= 1.0:
        return out
    for _ in range(10):
        over = out > max_weight + 1e-12
        if not bool(over.any()):
            break
        excess = float((out[over] - max_weight).sum())
        out.loc[over] = max_weight
        under = out < max_weight - 1e-12
        capacity = (max_weight - out.loc[under]).clip(lower=0.0)
        capacity_sum = float(capacity.sum())
        alloc = min(excess, capacity_sum)
        if capacity_sum <= 1e-12 or alloc <= 1e-12:
            break
        out.loc[under] += alloc * (capacity / capacity_sum)
        out = out.clip(lower=0.0)
    return out


def _apply_sector_cap(
    weights: pd.Series,
    sector_map: Dict[str, str],
    max_weight: float,
    max_single_weight: float | None = None,
) -> pd.Series:
    out = weights.astype(float).copy()
    max_weight = float(max_weight)
    if out.empty or not sector_map or max_weight <= 0.0 or max_weight >= 1.0:
        return out
    sectors = pd.Series({idx: sector_map.get(idx, "Unknown") for idx in out.index}, dtype=object)
    for _ in range(10):
        changed = False
        for sector_name, idx in sectors.groupby(sectors).groups.items():
            idx = list(idx)
            sector_weight = float(out.loc[idx].sum())
            if sector_weight <= max_weight + 1e-12:
                continue
            excess = sector_weight - max_weight
            out.loc[idx] *= max_weight / max(sector_weight, 1e-12)
            other_idx = [name for name in out.index if name not in idx]
            if other_idx:
                other_sectors = pd.Series({name: sectors.loc[name] for name in other_idx}, dtype=object)
                sector_capacity = {}
                for other_sector, sec_idx in other_sectors.groupby(other_sectors).groups.items():
                    sec_idx = list(sec_idx)
                    current_weight = float(out.loc[sec_idx].sum())
                    sector_capacity[other_sector] = max(max_weight - current_weight, 0.0)
                total_capacity = float(sum(sector_capacity.values()))
                alloc = min(excess, total_capacity)
                if alloc > 1e-12 and total_capacity > 1e-12:
                    for other_sector, sec_idx in other_sectors.groupby(other_sectors).groups.items():
                        sec_idx = list(sec_idx)
                        sector_alloc = alloc * (sector_capacity.get(other_sector, 0.0) / total_capacity)
                        stock_cap_limit = max_weight if max_single_weight is None else min(max_weight, float(max_single_weight))
                        stock_capacity = (stock_cap_limit - out.loc[sec_idx]).clip(lower=0.0)
                        stock_capacity_sum = float(stock_capacity.sum())
                        if stock_capacity_sum > 1e-12 and sector_alloc > 1e-12:
                            out.loc[sec_idx] += sector_alloc * (stock_capacity / stock_capacity_sum)
            changed = True
        out = out.clip(lower=0.0)
        if not changed:
            break
    return out


def enforce_target_weight_limits(
    target_weights: pd.DataFrame,
    sector_map: Dict[str, str],
    max_single_position_pct: float = 0.25,
    max_sector_weight: float = 0.35,
) -> tuple[pd.DataFrame, dict]:
    weights = target_weights.copy()
    if weights.empty:
        return weights, {
            "pre_cap_sum": 0.0,
            "post_cap_sum": 0.0,
            "pre_cap_max_single": 0.0,
            "post_cap_max_single": 0.0,
            "pre_cap_max_sector": 0.0,
            "post_cap_max_sector": 0.0,
            "pre_cap_top_weights": [],
            "post_cap_top_weights": [],
        }

    weights["symbol"] = weights["symbol"].astype(str).str.strip()
    weights["target_weight"] = pd.to_numeric(weights["target_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    grouped = weights.groupby("symbol", as_index=True)["target_weight"].sum().sort_values(ascending=False)
    pre_sum_raw = float(grouped.sum())
    gross = min(pre_sum_raw, 1.0)
    if pre_sum_raw > 1.0 + 1e-12:
        grouped = grouped / pre_sum_raw
        gross = 1.0

    pre_sector = grouped.groupby(grouped.index.map(lambda s: sector_map.get(s, "Unknown"))).sum() if not grouped.empty else pd.Series(dtype=float)
    diagnostics = {
        "pre_cap_sum": float(grouped.sum()),
        "pre_cap_sum_raw": pre_sum_raw,
        "pre_cap_max_single": float(grouped.max()) if not grouped.empty else 0.0,
        "pre_cap_max_sector": float(pre_sector.max()) if not pre_sector.empty else 0.0,
        "pre_cap_top_weights": [
            {"symbol": str(symbol), "target_weight": float(weight)}
            for symbol, weight in grouped.head(5).items()
        ],
    }

    if gross > 1e-12:
        normalized = grouped / gross
        single_cap_norm = min(float(max_single_position_pct) / gross, 1.0)
        sector_cap_norm = min(float(max_sector_weight) / gross, 1.0)
        for _ in range(6):
            prev = normalized.copy()
            normalized = _apply_single_name_cap(normalized, single_cap_norm)
            normalized = _apply_sector_cap(
                normalized,
                sector_map,
                sector_cap_norm,
                max_single_weight=single_cap_norm,
            )
            if normalized.equals(prev):
                break
        grouped = normalized * gross

    post_sector = grouped.groupby(grouped.index.map(lambda s: sector_map.get(s, "Unknown"))).sum() if not grouped.empty else pd.Series(dtype=float)
    diagnostics.update(
        {
            "post_cap_sum": float(grouped.sum()),
            "post_cap_max_single": float(grouped.max()) if not grouped.empty else 0.0,
            "post_cap_max_sector": float(post_sector.max()) if not post_sector.empty else 0.0,
            "post_cap_top_weights": [
                {"symbol": str(symbol), "target_weight": float(weight)}
                for symbol, weight in grouped.sort_values(ascending=False).head(5).items()
            ],
            "max_single_position_pct": float(max_single_position_pct),
            "max_sector_weight": float(max_sector_weight),
        }
    )

    out = grouped.rename("target_weight").reset_index()
    return out, diagnostics


def enforce_lot_feasible_weights(
    target_weights: pd.DataFrame,
    prices: Dict[str, float],
    nav_before: float,
    lot_size: int,
    min_trade_notional: float,
    sector_map: Dict[str, str],
    max_single_position_pct: float = 0.25,
    max_sector_weight: float = 0.35,
) -> tuple[pd.DataFrame, dict]:
    weights = target_weights.copy()
    if weights.empty or nav_before <= 1e-9:
        return weights, {
            "gross_budget": 0.0,
            "selected_count": 0,
            "selected_symbols": [],
            "lot_feasible": False,
            "skipped_expensive_count": 0,
            "budget_left": 0.0,
        }

    weights["symbol"] = weights["symbol"].astype(str).str.strip()
    weights["target_weight"] = pd.to_numeric(weights["target_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weights = weights[weights["symbol"].isin(prices.keys())].copy()
    if weights.empty:
        return weights, {
            "gross_budget": 0.0,
            "selected_count": 0,
            "selected_symbols": [],
            "lot_feasible": False,
            "skipped_expensive_count": 0,
            "budget_left": 0.0,
        }

    gross_budget = min(float(weights["target_weight"].sum()), 1.0)
    min_weight_floor = float(min_trade_notional) / max(float(nav_before), 1e-9)
    rows = []
    for _, row in weights.sort_values(["target_weight", "symbol"], ascending=[False, True]).iterrows():
        symbol = str(row["symbol"])
        price = float(prices.get(symbol, 0.0) or 0.0)
        if price <= 0.0:
            continue
        lot_notional = float(price) * int(lot_size)
        min_weight = max(lot_notional / max(float(nav_before), 1e-9), min_weight_floor)
        rows.append(
            {
                "symbol": symbol,
                "target_weight": float(row["target_weight"]),
                "min_weight": float(min_weight),
                "sector": sector_map.get(symbol, "Unknown"),
            }
        )
    if not rows:
        return weights.iloc[0:0].copy(), {
            "gross_budget": gross_budget,
            "selected_count": 0,
            "selected_symbols": [],
            "lot_feasible": False,
            "skipped_expensive_count": 0,
            "budget_left": gross_budget,
        }

    rows = sorted(rows, key=lambda r: (-r["target_weight"], r["min_weight"], r["symbol"]))
    selected: list[dict] = []
    sector_used: Dict[str, float] = {}
    remaining = gross_budget
    skipped_expensive = 0
    for row in rows:
        if row["min_weight"] > float(max_single_position_pct) + 1e-12:
            skipped_expensive += 1
            continue
        sector_name = str(row["sector"])
        sector_left = float(max_sector_weight) - float(sector_used.get(sector_name, 0.0))
        if row["min_weight"] > sector_left + 1e-12:
            continue
        if row["min_weight"] > remaining + 1e-12:
            continue
        selected.append({**row, "alloc_weight": float(row["min_weight"])})
        sector_used[sector_name] = float(sector_used.get(sector_name, 0.0)) + float(row["min_weight"])
        remaining -= float(row["min_weight"])

    if not selected:
        return weights.iloc[0:0].copy(), {
            "gross_budget": gross_budget,
            "selected_count": 0,
            "selected_symbols": [],
            "lot_feasible": False,
            "skipped_expensive_count": skipped_expensive,
            "budget_left": gross_budget,
        }

    for _ in range(10):
        if remaining <= 1e-9:
            break
        capacity_rows = []
        capacity_sum = 0.0
        for row in selected:
            sector_name = str(row["sector"])
            single_left = float(max_single_position_pct) - float(row["alloc_weight"])
            sector_left = float(max_sector_weight) - float(sector_used.get(sector_name, 0.0))
            cap = max(min(single_left, sector_left), 0.0)
            desired_extra = max(float(row["target_weight"]) - float(row["alloc_weight"]), 0.0)
            effective_cap = min(cap, desired_extra if desired_extra > 1e-12 else cap)
            if effective_cap > 1e-12:
                capacity_rows.append((row, effective_cap, max(float(row["target_weight"]), 1e-12)))
                capacity_sum += max(float(row["target_weight"]), 1e-12)
        if not capacity_rows:
            break
        used_this_round = 0.0
        for row, cap, priority in capacity_rows:
            add = min(remaining * (priority / capacity_sum), cap)
            if add <= 1e-12:
                continue
            row["alloc_weight"] = float(row["alloc_weight"]) + float(add)
            sector_used[row["sector"]] = float(sector_used.get(row["sector"], 0.0)) + float(add)
            used_this_round += float(add)
        if used_this_round <= 1e-12:
            break
        remaining -= used_this_round

    out = pd.DataFrame(
        [{"symbol": row["symbol"], "target_weight": float(row["alloc_weight"])} for row in selected]
    ).sort_values("target_weight", ascending=False)
    diagnostics = {
        "gross_budget": gross_budget,
        "selected_count": int(len(selected)),
        "selected_symbols": [str(row["symbol"]) for row in selected],
        "selected_min_weight_sum": float(sum(float(row["min_weight"]) for row in selected)),
        "budget_left": float(max(remaining, 0.0)),
        "lot_feasible": bool(len(selected) > 0),
        "skipped_expensive_count": int(skipped_expensive),
    }
    return out, diagnostics


def _load_price_windows(
    conn: sqlite3.Connection,
    symbols: List[str],
    asof: str,
    lookback_days: int = 420,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()
    placeholders = ",".join("?" for _ in symbols)
    rows = pd.read_sql_query(
        f"""
        SELECT date, symbol, close, volume
        FROM daily_prices
        WHERE symbol IN ({placeholders})
          AND date <= ?
        ORDER BY date, symbol
        """,
        conn,
        params=list(symbols) + [asof],
    )
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows["date"] = pd.to_datetime(rows["date"])
    close = rows.pivot(index="date", columns="symbol", values="close").sort_index().tail(lookback_days)
    volume = rows.pivot(index="date", columns="symbol", values="volume").sort_index().reindex(close.index)
    return close, volume


def _load_latest_fundamental_values(
    conn: sqlite3.Connection,
    symbols: List[str],
    asof: str,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(index=symbols)
    placeholders_symbol = ",".join("?" for _ in symbols)
    placeholders_feature = ",".join("?" for _ in FUNDAMENTAL_FACTOR_NAMES)
    rows = pd.read_sql_query(
        f"""
        SELECT asof, symbol, feature_name, value
        FROM feature_daily
        WHERE asof <= ?
          AND symbol IN ({placeholders_symbol})
          AND feature_name IN ({placeholders_feature})
        ORDER BY asof DESC
        """,
        conn,
        params=[asof] + list(symbols) + list(FUNDAMENTAL_FACTOR_NAMES),
    )
    if rows.empty:
        return pd.DataFrame(index=symbols)
    rows = rows.drop_duplicates(subset=["symbol", "feature_name"], keep="first")
    wide = rows.pivot(index="symbol", columns="feature_name", values="value")
    return wide.reindex(index=symbols)


def build_factor_family_contributions(
    conn: sqlite3.Connection,
    asof: str,
    target_weights: pd.DataFrame,
    target_mode: str,
) -> tuple[pd.DataFrame, dict]:
    weights = target_weights.copy()
    weights["symbol"] = weights["symbol"].astype(str)
    weights["target_weight"] = pd.to_numeric(weights["target_weight"], errors="coerce").fillna(0.0)
    weights = weights.sort_values("target_weight", ascending=False)
    symbols = weights["symbol"].tolist()
    active_family_map = {
        "ridge": {"price"},
        "shadow_eq": {"price"},
        "shadow_ic": {"price"},
        "shadow_hybrid_ic": {"price", "risk_adjusted", "fundamental"},
    }
    active_families = active_family_map.get(str(target_mode or "").lower(), {"price"})
    empty_summary = {
        "target_mode": target_mode,
        "families": {
            family_name: {
                "configured_factor_count": len(factor_names),
                "active_factor_count": 0,
                "is_active_in_mode": family_name in active_families,
                "portfolio_weighted_score": 0.0,
                "top_symbol": None,
            }
            for family_name, factor_names in FACTOR_FAMILY_MAP.items()
        },
    }
    if not symbols:
        return pd.DataFrame(), empty_summary

    sector_map = _load_sector_map(conn)
    close, volume = _load_price_windows(conn, symbols, asof)
    technical_frames = compute_features(close, volume) if not close.empty else {}
    fundamental_frame = _load_latest_fundamental_values(conn, symbols, asof)

    family_rows: list[dict] = []
    family_summary: dict[str, dict] = {}

    for family_name, factor_names in FACTOR_FAMILY_MAP.items():
        factor_z: dict[str, pd.Series] = {}
        for factor_name in factor_names:
            if family_name == "fundamental":
                if factor_name not in fundamental_frame.columns:
                    continue
                raw = fundamental_frame[factor_name].reindex(symbols)
            else:
                frame = technical_frames.get(factor_name)
                if frame is None or pd.Timestamp(asof) not in frame.index:
                    continue
                raw = frame.loc[pd.Timestamp(asof)].reindex(symbols)
            z = sector_neutral_zscore(
                pd.Series(raw, index=symbols, dtype=float),
                sector_map={sym: sector_map.get(sym, "Unknown") for sym in symbols},
            ).reindex(symbols)
            if z.abs().sum() <= 1e-12 and z.notna().sum() == 0:
                continue
            factor_z[factor_name] = z

        if factor_z:
            family_matrix = pd.DataFrame(factor_z, index=symbols)
            family_score = family_matrix.mean(axis=1, skipna=True).fillna(0.0)
            active_factor_count = int(sum(1 for col in family_matrix.columns if family_matrix[col].notna().any()))
        else:
            family_matrix = pd.DataFrame(index=symbols)
            family_score = pd.Series(0.0, index=symbols, dtype=float)
            active_factor_count = 0

        merged = weights.set_index("symbol").copy()
        merged[f"{family_name}_score"] = family_score.reindex(merged.index).fillna(0.0)
        merged[f"{family_name}_contribution"] = merged["target_weight"] * merged[f"{family_name}_score"]
        for symbol, row in merged.iterrows():
            family_rows.append(
                {
                    "symbol": symbol,
                    "target_weight": float(row["target_weight"]),
                    "family": family_name,
                    "family_score": float(row[f"{family_name}_score"]),
                    "weighted_contribution": float(row[f"{family_name}_contribution"]),
                    "factor_count_configured": len(factor_names),
                    "factor_count_active": active_factor_count,
                    "is_active_in_mode": family_name in active_families,
                }
            )

        family_summary[family_name] = {
            "configured_factor_count": len(factor_names),
            "active_factor_count": active_factor_count,
            "is_active_in_mode": family_name in active_families,
            "portfolio_weighted_score": float(merged[f"{family_name}_contribution"].sum()),
            "top_symbol": None if merged.empty else str(merged[f"{family_name}_contribution"].abs().sort_values(ascending=False).index[0]),
        }

    out_df = pd.DataFrame(family_rows).sort_values(["family", "weighted_contribution"], ascending=[True, False])
    return out_df, {"target_mode": target_mode, "families": family_summary}


def _validate_target_freshness(reports_dir: Path, asof: str) -> dict:
    meta = _load_target_meta(reports_dir)
    if not meta:
        return {}

    exported_asof = meta.get("exported_asof")
    history_last_asof = meta.get("history_last_asof")
    last_row_zero = bool(meta.get("history_last_row_is_zero", False))
    if history_last_asof and str(history_last_asof) != str(asof):
        reason = (
            f"target_weights.csv is stale for decision date {asof}: "
            f"exported_asof={exported_asof}, history_last_asof={history_last_asof}, "
            f"history_last_row_is_zero={last_row_zero}. "
            "The latest model state is not aligned with the requested decision date."
        )
        raise RuntimeError(reason)
    return meta


def build_orders(
    conn: sqlite3.Connection,
    asof: str,
    target_weights: pd.DataFrame,
    cash_jpy: float,
    lot_size: int,
    min_trade_notional: float,
    max_single_position_pct: float = 0.25,
    max_sector_weight: float = 0.35,
    strategy_id: str = "default",
) -> Tuple[List[OrderRow], Dict]:
    # current positions
    pos_date, pos = _latest_positions(conn, asof, strategy_id=strategy_id)
    target_symbols = [str(sym) for sym in target_weights["symbol"]]
    order_universe = list(dict.fromkeys(target_symbols + list(pos.keys())))

    # prices
    px = {}
    px_date = {}
    missing = []
    for sym in order_universe:
        d, p = _last_close(conn, sym, asof)
        if p is None:
            missing.append(sym)
            continue
        px[sym] = p
        px_date[sym] = d

    # NAV estimate
    nav_positions = 0.0
    for sym, q in pos.items():
        d, p = _last_close(conn, sym, asof)
        if p is None:
            continue
        nav_positions += float(q) * float(p)
    nav_before = float(cash_jpy) + float(nav_positions)

    sector_map = _load_sector_map(conn)
    tw = target_weights.copy()
    tw["symbol"] = tw["symbol"].astype(str)
    tw = tw[tw["symbol"].isin(px.keys())].copy()
    tw_capped, cap_diagnostics = enforce_target_weight_limits(
        tw,
        sector_map=sector_map,
        max_single_position_pct=max_single_position_pct,
        max_sector_weight=max_sector_weight,
    )
    tw_tradeable, lot_diagnostics = enforce_lot_feasible_weights(
        tw_capped,
        prices=px,
        nav_before=nav_before,
        lot_size=lot_size,
        min_trade_notional=min_trade_notional,
        sector_map=sector_map,
        max_single_position_pct=max_single_position_pct,
        max_sector_weight=max_sector_weight,
    )
    target_weight_map = {
        str(r["symbol"]): max(float(r["target_weight"]), 0.0)
        for _, r in tw_tradeable.iterrows()
    }
    wsum = float(sum(target_weight_map.values()))

    orders: List[OrderRow] = []
    for sym in order_universe:
        if sym not in px:
            continue
        w = float(target_weight_map.get(sym, 0.0))
        price = float(px[sym])

        cur_qty = float(pos.get(sym, 0.0))
        tgt_value = w * nav_before

        # conservative rounding: floor to lot
        tgt_qty = int((tgt_value // (price * lot_size)) * lot_size)

        diff = tgt_qty - int(cur_qty)

        if diff == 0:
            continue

        side = "BUY" if diff > 0 else "SELL"
        qty = abs(int(diff))

        est_notional = qty * price
        if est_notional < min_trade_notional:
            continue

        orders.append(OrderRow(symbol=sym, side=side, qty=qty, est_notional=est_notional))

    # sort: SELL first then BUY (often safer for cash)
    orders.sort(key=lambda x: (0 if x.side == "SELL" else 1, -x.est_notional))

    info = {
        "asof": asof,
        "positions_asof": pos_date,
        "nav_before": nav_before,
        "cash_input": cash_jpy,
        "missing_symbols": missing,
        "price_dates_sample": {k: px_date[k] for k in list(px_date)[:5]},
        "weights_sum_before_norm": wsum,
        "cap_diagnostics": cap_diagnostics,
        "lot_feasibility": lot_diagnostics,
        "lot_size": lot_size,
        "order_universe_size": len(order_universe),
    }
    return orders, info


def apply_kelly_sizing(
    target_weights: pd.DataFrame,
    suggested_weight: float,
    max_positions: int,
) -> pd.DataFrame:
    if target_weights.empty:
        return target_weights.copy()
    gross_cap = max(float(suggested_weight), 0.0) * max(int(max_positions), 1)
    if gross_cap <= 0.0:
        return target_weights.iloc[0:0].copy()
    tw = target_weights.copy()
    tw["target_weight"] = pd.to_numeric(tw["target_weight"], errors="coerce").fillna(0.0)
    tw = tw[tw["target_weight"] > 0.0].sort_values("target_weight", ascending=False).head(max(int(max_positions), 1)).copy()
    if tw.empty:
        return tw
    total = float(tw["target_weight"].sum())
    if total <= 1e-12:
        return tw.iloc[0:0].copy()
    tw["target_weight"] = tw["target_weight"] / total * gross_cap
    tw["target_weight"] = tw["target_weight"].clip(upper=float(suggested_weight))
    return tw[tw["target_weight"] > 0.0].copy()


def write_db(
    conn: sqlite3.Connection,
    run_id: str,
    asof: str,
    snapshot_path: str,
    orders: List[OrderRow],
    strategy_id: str = "default",
) -> None:
    ts = _now_iso()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO decision_runs(run_id, asof, strategy_id, ts, snapshot_path, status, notes)
            VALUES (?, ?, ?, ?, ?, 'proposed', NULL)
            """,
            (run_id, asof, strategy_id, ts, snapshot_path),
        )

        for i, o in enumerate(orders):
            order_id = f"{run_id}__{i:03d}"
            conn.execute(
                """
                INSERT OR REPLACE INTO orders(
                  order_id, run_id, asof, strategy_id, symbol, side, qty, order_type, limit_price, tif,
                  reason, expected_fee, expected_slippage, expected_value, status, created_ts
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'DAY', ?, NULL, NULL, ?, 'proposed', ?)
                """,
                (
                    order_id, run_id, asof, strategy_id, o.symbol, o.side, o.qty,
                    o.suggested_type, o.suggested_limit, o.comment,
                    float(o.est_notional), ts
                ),
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD, default=latest trading day in DB")
    ap.add_argument("--refresh_data", action="store_true", help="refresh market data before building the decision")
    ap.add_argument("--refresh_lookback", type=int, default=30, help="lookback days used when refresh_data is enabled")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--cash", type=float, default=1000000.0, help="cash used for sizing (manual mode); default=1M JPY")
    ap.add_argument("--lot", type=int, default=100, help="board lot; ETFs often 1, most JP stocks 100")
    ap.add_argument("--peak_nav", type=float, default=None, help="historical peak NAV for drawdown check (JPY); if omitted, no DD check")
    ap.add_argument("--dd_half", type=float, default=0.12, help="drawdown threshold for half-position (default 12%%)")
    ap.add_argument("--dd_full", type=float, default=0.18, help="drawdown threshold for full exit (default 18%%)")
    ap.add_argument("--min_trade", type=float, default=5000.0, help="ignore trades smaller than this notional (JPY)")
    ap.add_argument("--max_single_position_pct", type=float, default=0.25)
    ap.add_argument("--max_sector_weight", type=float, default=0.35)
    ap.add_argument("--strategy_id", default="default")
    ap.add_argument("--position_sizing", default="equal_weight")
    ap.add_argument("--max_positions", type=int, default=12)
    ap.add_argument("--out_dir", default="artifacts/decision")
    ap.add_argument("--stop_loss_vol_mult", type=float, default=3.0, help="ATR multiplier for stop-loss")
    ap.add_argument("--stop_loss_min_pct", type=float, default=0.04, help="minimum stop-loss percentage")
    ap.add_argument("--stop_loss_max_pct", type=float, default=0.12, help="maximum stop-loss percentage")
    ap.add_argument("--trailing_activate_pct", type=float, default=0.03, help="profit threshold to activate trailing stop")
    ap.add_argument("--trailing_stop_pct", type=float, default=0.02, help="trailing stop drawdown from high")
    ap.add_argument("--atr_window", type=int, default=20, help="ATR lookback window")
    args = ap.parse_args()

    requested_asof = args.asof
    if args.refresh_data:
        _before, refreshed_to, _did_refresh = refresh_market_data_if_needed(
            args.db,
            target_date=requested_asof,
            lookback_days=int(args.refresh_lookback),
            force=False,
        )
        if requested_asof and refreshed_to and str(refreshed_to) < str(requested_asof):
            raise RuntimeError(
                f"Market data refresh completed but DB is still behind requested asof={requested_asof}. "
                f"Latest daily_prices date is {refreshed_to}."
            )

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        asof = requested_asof or get_latest_trading_day(conn) or date.today().strftime("%Y-%m-%d")
        db_latest = latest_db_date(args.db)
        if db_latest and str(asof) > str(db_latest):
            raise RuntimeError(
                f"Requested asof={asof} is newer than latest daily_prices date {db_latest}. "
                "Refresh data first or choose an earlier asof."
            )
    finally:
        conn.close()
    reports_dir = Path(args.reports_dir)
    tw_path = reports_dir / "target_weights.csv"
    if not tw_path.exists():
        raise FileNotFoundError(f"target_weights.csv not found at: {tw_path}. Run ss6_sqlite.py first.")
    target_meta = _validate_target_freshness(reports_dir, asof)

    target_weights = _load_target_weights(tw_path)

    # Run-scoped artifact folder (avoid overwrite when multiple runs share the same asof)
    run_id = _make_run_id(asof, config_text=f"cash={args.cash}|lot={args.lot}")
    out_run = Path(args.out_dir) / asof / run_id
    out_run.mkdir(parents=True, exist_ok=True)

    # copy key artifacts for audit
    copied = []
    for fn in [
        "target_weights.csv",
        "weights_history.csv",
        "strategy_report.html",
        "strategy_report_extras.html",
        "weights_heatmap.html",
        "signal_mode_compare.csv",
        "promotion_decision.json",
        "promotion_note.txt",
        "factor_health_report.json",
        "factor_health_report.md",
        "factor_health_families.csv",
        "factor_health_factors.csv",
        "factor_registry_cleanup_candidates.csv",
        "factor_registry_cleanup_report.json",
        "factor_registry_cleanup_report.md",
        "signal_mode_compare_report.json",
        "signal_mode_compare_report.md",
        "zero_exposure_report.json",
        "zero_exposure_report.md",
        "earnings_event_study.json",
        "earnings_event_study.csv",
        "earnings_event_study.md",
        "optimizer_objective_evaluation.json",
        "optimizer_objective_evaluation.md",
    ]:
        p = reports_dir / fn
        if p.exists():
            (out_run / fn).write_bytes(p.read_bytes())
            copied.append(fn)

    conn = connect(args.db)
    ensure_trade_tables(conn)

    try:
        target_mode = _infer_target_mode(reports_dir, target_meta)
        snapshot_asof, snapshot_cash, snapshot_nav = _latest_account_snapshot(conn, asof, strategy_id=args.strategy_id)
        effective_cash = float(snapshot_cash) if snapshot_cash is not None else float(args.cash)
        portfolio_mode = "snapshot" if snapshot_cash is not None else "manual"

        # drawdown check: scale orders if portfolio drawdown exceeds thresholds
        dd_scale = 1.0
        dd_status = "OK"
        
        peak_nav = args.peak_nav
        if peak_nav is None:
            try:
                row = conn.execute(
                    "SELECT MAX(nav) FROM account_snapshots WHERE strategy_id=?",
                    (args.strategy_id,),
                ).fetchone()
                if row and row[0] is not None:
                    peak_nav = float(row[0])
                    print(f"Auto-fetched peak_nav from account_snapshots: {peak_nav:,.0f}")
            except sqlite3.OperationalError:
                pass # table might not exist

        if peak_nav is not None and peak_nav > 0:
            # Estimate current NAV from DB positions + cash
            _, cur_pos = _latest_positions(conn, asof, strategy_id=args.strategy_id)
            cur_nav = float(effective_cash)
            for sym, qty in cur_pos.items():
                _, px = _last_close(conn, sym, asof)
                if px is not None:
                    cur_nav += float(qty) * float(px)
            drawdown = (cur_nav - peak_nav) / peak_nav
            if drawdown < -args.dd_full:
                dd_scale = 0.0
                dd_status = f"FULL_EXIT (DD={drawdown:.1%} < -{args.dd_full:.0%})"
                print(f"[risk] full exit triggered by drawdown: nav={cur_nav:,.0f} peak={peak_nav:,.0f} dd={drawdown:.1%}")
            elif drawdown < -args.dd_half:
                dd_scale = 0.5
                dd_status = f"HALF_POSITION (DD={drawdown:.1%} < -{args.dd_half:.0%})"
                print(f"[risk] half position triggered by drawdown: nav={cur_nav:,.0f} peak={peak_nav:,.0f} dd={drawdown:.1%}")

        # FULL_EXIT: 清空 target_weights，让 build_orders 为所有持仓生成 SELL
        if dd_scale <= 0.0:
            print(f"[risk] FULL_EXIT: clearing all target weights → force SELL all positions")
            target_weights = target_weights.iloc[0:0].copy()

        if args.position_sizing == "half_kelly":
            kelly = compute_kelly_params(conn, strategy_id=args.strategy_id, lookback_days=60, asof=asof)
            target_weights = apply_kelly_sizing(
                target_weights,
                suggested_weight=float(kelly.get("suggested_weight", 0.0) or 0.0),
                max_positions=int(args.max_positions),
            )
        else:
            kelly = {}

        # ── 持仓保全：当 entry_filter 拦截所有新进仓时，不强制清仓现有持仓 ──────
        # 逻辑：
        #   - 目标权重为空 (signal 无法选股)
        #   - 原因是 entry_filter（不是 benchmark=off 或 drawdown 全平）
        #   - 现有持仓未触发止损
        # 此时将现有持仓按当前市值权重注入 target_weights，阻止机械性 rebalance 卖出。
        _benchmark_state = str(target_meta.get("benchmark_state", "on") or "on")
        _tw_is_empty = (
            target_weights.empty
            or float(target_weights.get("target_weight", pd.Series(dtype=float)).fillna(0).sum()) < 1e-9
        )
        # Preserve existing positions when:
        #   - signal returned empty target (entry_filter blocked all candidates)
        #   - benchmark is NOT off (regime allows holding)
        #   - no drawdown full-exit triggered
        # Without this, existing profitable positions get sold mechanically.
        if _tw_is_empty and _benchmark_state != "off" and dd_scale > 0:
            _, _cur_pos = _latest_positions(conn, asof, strategy_id=args.strategy_id)
            if _cur_pos:
                # Compute approximate NAV for weight calculation
                _cur_nav_est = float(effective_cash)
                _held_rows = []
                for _sym, _qty in _cur_pos.items():
                    _, _px = _last_close(conn, _sym, asof)
                    if _px is not None:
                        _val = float(_qty) * float(_px)
                        _cur_nav_est += _val
                        _held_rows.append({"symbol": _sym, "value": _val})
                if _held_rows and _cur_nav_est > 0:
                    _hold_df = pd.DataFrame([
                        {"symbol": r["symbol"], "target_weight": r["value"] / _cur_nav_est}
                        for r in _held_rows
                    ])
                    target_weights = _hold_df
                    _held_syms = [r["symbol"] for r in _held_rows]
                    print(f"[entry_filter_hold] Preserving {len(_held_rows)} existing position(s) "
                          f"instead of rebalancing to zero. Symbols: {_held_syms}")
                    # Cancel any stale SELL orders for preserved positions
                    for _hsym in _held_syms:
                        conn.execute(
                            "UPDATE orders SET status='cancelled' "
                            "WHERE symbol=? AND side='SELL' AND status='proposed' AND strategy_id=?",
                            (_hsym, args.strategy_id),
                        )
                    conn.commit()

        # 止损/止盈检查: 检查现有持仓是否触发退出
        stop_loss_cfg = {
            "stop_loss_vol_mult": float(getattr(args, "stop_loss_vol_mult", 3.0)),
            "stop_loss_min_pct": float(getattr(args, "stop_loss_min_pct", 0.04)),
            "stop_loss_max_pct": float(getattr(args, "stop_loss_max_pct", 0.12)),
            "trailing_activate_pct": float(getattr(args, "trailing_activate_pct", 0.03)),
            "trailing_stop_pct": float(getattr(args, "trailing_stop_pct", 0.02)),
            "atr_window": int(getattr(args, "atr_window", 20)),
        }
        forced_exits, stop_loss_diag = _check_stop_losses(conn, asof, args.strategy_id, stop_loss_cfg)
        if forced_exits:
            print(f"[risk] stop-loss/trailing triggered for: {forced_exits}")
            # 从 target_weights 中移除被止损的股票（强制平仓）
            target_weights = target_weights[~target_weights["symbol"].isin(forced_exits)].copy()

        # build orders
        orders, info = build_orders(
            conn, asof, target_weights, cash_jpy=effective_cash,
            lot_size=args.lot,
            min_trade_notional=args.min_trade,
            max_single_position_pct=args.max_single_position_pct,
            max_sector_weight=args.max_sector_weight,
            strategy_id=args.strategy_id,
        )
        family_contrib_df, family_contrib_summary = build_factor_family_contributions(
            conn=conn,
            asof=asof,
            target_weights=target_weights,
            target_mode=target_mode,
        )

        # Apply drawdown scaling: reduce all BUY quantities proportionally
        if dd_scale < 1.0:
            scaled_orders = []
            for o in orders:
                if o.side == "BUY":
                    new_qty = int(o.qty * dd_scale // args.lot) * args.lot
                    if new_qty <= 0:
                        continue
                    o.qty = new_qty
                    o.est_notional = o.qty * (o.est_notional / max(o.qty / dd_scale if dd_scale > 0 else 1, 1))
                    o.comment = f"rebalance(dd_scale={dd_scale:.0%})"
                scaled_orders.append(o)
            orders = scaled_orders

        orders_csv = out_run / "orders_proposal.csv"

        # compute entry zones for each order
        entry_zones: dict[str, dict] = {}
        for o in orders:
            _, px = _last_close(conn, o.symbol, asof)
            atr = _compute_atr_pct(conn, o.symbol, asof) if px else None
            if px:
                entry_zones[o.symbol] = compute_entry_zone(o.symbol, o.side, px, atr)

        # write orders CSV (with entry zone columns)
        with orders_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "side", "qty", "suggested_type", "suggested_limit",
                         "est_notional", "comment", "target_limit", "aggressive_limit", "walk_away_price"])
            for o in orders:
                ez = entry_zones.get(o.symbol, {})
                w.writerow([o.symbol, o.side, o.qty, o.suggested_type, o.suggested_limit or "",
                            f"{o.est_notional:.2f}", o.comment,
                            ez.get("target_limit", ""), ez.get("aggressive_limit", ""), ez.get("walk_away_price", "")])

        factor_family_csv = out_run / "factor_family_contributions.csv"
        family_contrib_df.to_csv(factor_family_csv, index=False)
        factor_family_json = out_run / "factor_family_summary.json"
        factor_family_json.write_text(
            json.dumps(family_contrib_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # decision snapshot
        snapshot = {
            "run_id": run_id,
            "asof": asof,
            "strategy_id": args.strategy_id,
            "artifact_dir": str(out_run),
            "data": {
                "db_path": args.db,
                "db_last_update": _read_meta(conn, "last_update_run"),
                "price_mode": _read_meta(conn, "price_mode"),
            },
            "model_outputs": {
                "reports_dir": str(reports_dir),
                "exported": copied,
                "target_weights_file": str(out_run / "target_weights.csv"),
                "target_weights_meta": target_meta,
                "factor_family_contributions_file": str(factor_family_csv),
                "factor_family_summary_file": str(factor_family_json),
                "factor_family_summary": family_contrib_summary,
                "zero_exposure_report_file": str(out_run / "zero_exposure_report.json"),
            },
            "portfolio": {
                "mode": portfolio_mode,
                "cash_input": effective_cash,
                "nav_before": info["nav_before"],
                "positions_asof": info["positions_asof"],
                "snapshot_asof": snapshot_asof,
                "snapshot_nav": snapshot_nav,
                "manual_cash_arg": args.cash,
            },
            "orders": {
                "proposal_file": str(orders_csv),
                "count": len(orders),
                "min_trade_notional": args.min_trade,
                "lot_size": args.lot,
                "missing_symbols": info["missing_symbols"],
                "weights_sum_before_norm": info["weights_sum_before_norm"],
                "cap_diagnostics": info["cap_diagnostics"],
                "drawdown_scale": dd_scale,
                "drawdown_status": dd_status,
                "peak_nav_input": peak_nav,
                "position_sizing": args.position_sizing,
                "kelly": kelly,
            },
            "risk_management": {
                "stop_loss_config": stop_loss_cfg,
                "forced_exits": forced_exits,
                "stop_loss_diagnostics": stop_loss_diag,
            },
            "diagnostics": {
                "target_weights_zero_now": bool(float(target_meta.get("export_row_sum", 0.0) or 0.0) <= 1e-12),
                "last_nonzero_asof": target_meta.get("last_nonzero_asof"),
                "last_nonzero_row_sum": target_meta.get("last_nonzero_row_sum"),
                "weights_sum_before_norm": info["weights_sum_before_norm"],
            },
        }

        snapshot_path = out_run / "decision_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        # write DB
        write_db(conn, run_id, asof, str(snapshot_path), orders, strategy_id=args.strategy_id)
        print("=" * 70)
        print("Decision packaged (manual execution mode)")
        print(f"run_id: {run_id}")
        print(f"snapshot: {snapshot_path}")
        print(f"orders:   {orders_csv}  (count={len(orders)})")
        print(f"factors:  {factor_family_csv}")
        if info["missing_symbols"]:
            print(f"missing prices for: {info['missing_symbols']}")
        print("=" * 70)
        return

        print("=" * 70)
        print("[ok] Decision packaged (manual execution mode)")
        print(f"run_id: {run_id}")
        print(f"snapshot: {snapshot_path}")
        print(f"orders:   {orders_csv}  (count={len(orders)})")
        if info["missing_symbols"]:
            print(f"[warn] missing prices for: {info['missing_symbols']}")
        print("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
