from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from make_decision import enforce_lot_feasible_weights, enforce_target_weight_limits
from trade_schema import connect, ensure_trade_tables


def _load_target_weights(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "target_weight"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns or "target_weight" not in df.columns:
        raise ValueError(f"Unrecognized target_weights format: {path}")
    out = df[["symbol", "target_weight"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    return out.groupby("symbol", as_index=False)["target_weight"].sum()


def _load_json(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_asof(conn, table: str, column: str = "asof", strategy_id: str = "default") -> Optional[str]:
    row = conn.execute(
        f"SELECT {column} FROM {table} WHERE strategy_id=? ORDER BY {column} DESC LIMIT 1",
        (strategy_id,),
    ).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _load_positions(conn, asof: str, strategy_id: str = "default") -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl
        FROM positions
        WHERE asof=? AND strategy_id=?
        ORDER BY symbol
        """,
        conn,
        params=(asof, strategy_id),
    )
    if df.empty:
        return pd.DataFrame(columns=["symbol", "qty", "avg_cost", "market_price", "market_value", "unrealized_pnl"])
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    return df


def _load_snapshot(conn, asof: str, strategy_id: str = "default") -> dict:
    row = conn.execute(
        """
        SELECT asof, cash, positions_value, nav, run_id
        FROM account_snapshots
        WHERE asof=? AND strategy_id=?
        LIMIT 1
        """,
        (asof, strategy_id),
    ).fetchone()
    if not row:
        return {"asof": asof, "cash": 0.0, "positions_value": 0.0, "nav": 0.0, "run_id": None}
    return {
        "asof": str(row[0]),
        "cash": float(row[1] or 0.0),
        "positions_value": float(row[2] or 0.0),
        "nav": float(row[3] or 0.0),
        "run_id": row[4],
    }


def _load_latest_prices(conn, symbols: list[str], asof: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "last_date", "last_close"])
    placeholders = ",".join("?" for _ in symbols)
    df = pd.read_sql_query(
        f"""
        SELECT dp.symbol, dp.date AS last_date, dp.close AS last_close
        FROM daily_prices dp
        JOIN (
          SELECT symbol, MAX(date) AS max_date
          FROM daily_prices
          WHERE symbol IN ({placeholders}) AND date<=?
          GROUP BY symbol
        ) mx
          ON dp.symbol = mx.symbol AND dp.date = mx.max_date
        """,
        conn,
        params=symbols + [asof],
    )
    if df.empty:
        return pd.DataFrame(columns=["symbol", "last_date", "last_close"])
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["last_close"] = pd.to_numeric(df["last_close"], errors="coerce")
    return df


def _round_limit_price(px: float) -> float:
    if px >= 1000:
        return float(round(px))
    if px >= 100:
        return float(round(px, 1))
    return float(round(px, 2))


def _limit_price(last_close: float, side: str, buy_bps: float, sell_bps: float) -> float:
    if side == "BUY":
        return _round_limit_price(last_close * (1.0 - buy_bps / 10000.0))
    if side == "SELL":
        return _round_limit_price(last_close * (1.0 + sell_bps / 10000.0))
    return _round_limit_price(last_close)


def build_live_trade_advice(
    conn,
    reports_dir: Path,
    asof: Optional[str] = None,
    strategy_id: str = "default",
    target_weights_path: Optional[Path] = None,
    target_meta_path: Optional[Path] = None,
    promotion_path: Optional[Path] = None,
    lot_size: int = 100,
    min_trade_value: float = 5000.0,
    buy_limit_buffer_bps: float = 20.0,
    sell_limit_buffer_bps: float = 20.0,
    max_single_position_pct: float = 0.25,
    max_sector_weight: float = 0.35,
) -> tuple[dict, pd.DataFrame]:
    ensure_trade_tables(conn)
    asof_used = asof or _latest_asof(conn, "positions", strategy_id=strategy_id) or _latest_asof(conn, "account_snapshots", strategy_id=strategy_id)
    if not asof_used:
        return {
            "status": "warning",
            "headline": "No live portfolio data available.",
            "asof": None,
            "warnings": ["positions/account_snapshots are empty."],
            "actions": ["Import real fills and rebuild positions before requesting advice."],
            "summary": {},
        }, pd.DataFrame()

    target_weights_path = target_weights_path or (reports_dir / "target_weights.csv")
    target_meta_path = target_meta_path or (reports_dir / "target_weights_meta.json")
    promotion_path = promotion_path or (reports_dir / "promotion_decision.json")

    target_weights = _load_target_weights(target_weights_path)
    target_meta = _load_json(target_meta_path)
    promotion = _load_json(promotion_path)
    positions = _load_positions(conn, asof_used, strategy_id=strategy_id)
    snapshot = _load_snapshot(conn, asof_used, strategy_id=strategy_id)
    sector_rows = conn.execute("SELECT symbol, sector FROM tickers").fetchall()
    sector_map = {str(symbol): str(sector or "Unknown") for symbol, sector in sector_rows}
    target_weights, cap_diagnostics = enforce_target_weight_limits(
        target_weights,
        sector_map=sector_map,
        max_single_position_pct=max_single_position_pct,
        max_sector_weight=max_sector_weight,
    )

    union_symbols = sorted(set(target_weights["symbol"].tolist()) | set(positions["symbol"].tolist()))
    prices = _load_latest_prices(conn, union_symbols, asof_used)

    merged = pd.DataFrame({"symbol": union_symbols})
    merged = merged.merge(target_weights, on="symbol", how="left")
    merged = merged.merge(positions[["symbol", "qty", "avg_cost", "market_price", "market_value"]], on="symbol", how="left")
    merged = merged.merge(prices, on="symbol", how="left")
    merged["target_weight"] = pd.to_numeric(merged["target_weight"], errors="coerce").fillna(0.0)
    merged["qty"] = pd.to_numeric(merged["qty"], errors="coerce").fillna(0.0)

    merged["ref_price"] = pd.to_numeric(merged["last_close"], errors="coerce")
    merged.loc[merged["ref_price"].isna(), "ref_price"] = pd.to_numeric(merged["market_price"], errors="coerce")
    merged["current_value"] = merged["qty"] * merged["ref_price"]

    nav = float(snapshot.get("nav") or 0.0)
    if nav <= 0:
        nav = float(merged["current_value"].fillna(0.0).sum()) + float(snapshot.get("cash") or 0.0)

    price_map = {
        str(row.symbol): float(row.last_close)
        for row in prices.itertuples(index=False)
        if pd.notna(row.last_close) and float(row.last_close) > 0.0
    }
    target_weights, lot_diagnostics = enforce_lot_feasible_weights(
        target_weights,
        prices=price_map,
        nav_before=nav,
        lot_size=lot_size,
        min_trade_notional=min_trade_value,
        sector_map=sector_map,
        max_single_position_pct=max_single_position_pct,
        max_sector_weight=max_sector_weight,
    )
    merged = merged.drop(columns=["target_weight"], errors="ignore").merge(target_weights, on="symbol", how="left")
    merged["target_weight"] = pd.to_numeric(merged["target_weight"], errors="coerce").fillna(0.0)

    merged["current_weight"] = merged["current_value"] / nav if nav > 0 else 0.0
    merged["target_value"] = merged["target_weight"] * nav
    merged["delta_value"] = merged["target_value"] - merged["current_value"]

    buy_need = float(merged.loc[merged["delta_value"] > 0, "delta_value"].sum())
    cash_budget = max(float(snapshot.get("cash") or 0.0), 0.0)
    buy_scale = min(1.0, cash_budget / buy_need) if buy_need > 0 else 1.0

    rows: list[dict] = []
    warnings: list[str] = []
    actions: list[str] = []

    for row in merged.itertuples(index=False):
        symbol = str(row.symbol)
        ref_price = float(row.ref_price) if pd.notna(row.ref_price) else 0.0
        current_qty = float(row.qty or 0.0)
        current_weight = float(row.current_weight or 0.0)
        target_weight = float(row.target_weight or 0.0)
        delta_value = float(row.delta_value or 0.0)
        eff_delta_value = delta_value * buy_scale if delta_value > 0 else delta_value

        if ref_price <= 0:
            rows.append(
                {
                    "symbol": symbol,
                    "action": "HOLD",
                    "side": "HOLD",
                    "qty": 0,
                    "suggested_limit_price": None,
                    "current_qty": current_qty,
                    "current_weight_pct": current_weight * 100.0,
                    "target_weight_pct": target_weight * 100.0,
                    "delta_weight_pct": (target_weight - current_weight) * 100.0,
                    "delta_value": eff_delta_value,
                    "reason": "Missing reference price.",
                }
            )
            if current_qty > 0:
                warnings.append(f"{symbol} has a live position but no reference price for advice.")
            continue

        raw_qty = abs(eff_delta_value) / ref_price
        proposed_qty = int(raw_qty // lot_size) * lot_size if lot_size > 0 else int(raw_qty)

        if target_weight <= 1e-9 and current_qty > 0 and current_qty * ref_price >= min_trade_value:
            proposed_qty = int(current_qty)
            eff_delta_value = -current_qty * ref_price

        side = "BUY" if eff_delta_value > 0 else "SELL" if eff_delta_value < 0 else "HOLD"
        if side == "SELL":
            proposed_qty = min(int(current_qty), proposed_qty)

        trade_value = proposed_qty * ref_price
        if proposed_qty <= 0 or trade_value < float(min_trade_value):
            side = "HOLD"
            proposed_qty = 0

        action = "HOLD"
        reason = f"Current {current_weight*100:.2f}% vs target {target_weight*100:.2f}%."
        if side == "BUY":
            action = "ADD"
            reason = (
                f"Underweight by {(target_weight-current_weight)*100:.2f}%. "
                f"Suggested buy sized from cash-aware rebalance."
            )
        elif side == "SELL":
            action = "TRIM" if target_weight > 0 else "EXIT"
            reason = (
                f"Overweight by {(current_weight-target_weight)*100:.2f}%. "
                f"Suggested sell to move toward target/cash state."
            )

        rows.append(
            {
                "symbol": symbol,
                "action": action,
                "side": side,
                "qty": int(proposed_qty),
                "suggested_limit_price": _limit_price(
                    ref_price,
                    side,
                    buy_limit_buffer_bps,
                    sell_limit_buffer_bps,
                ) if side != "HOLD" else None,
                "ref_price": ref_price,
                "current_qty": current_qty,
                "current_weight_pct": current_weight * 100.0,
                "target_weight_pct": target_weight * 100.0,
                "delta_weight_pct": (target_weight - current_weight) * 100.0,
                "delta_value": eff_delta_value,
                "reason": reason,
            }
        )

    advice_df = pd.DataFrame(rows)
    if advice_df.empty:
        advice_df = pd.DataFrame(
            columns=[
                "symbol",
                "action",
                "side",
                "qty",
                "suggested_limit_price",
                "ref_price",
                "current_qty",
                "current_weight_pct",
                "target_weight_pct",
                "delta_weight_pct",
                "delta_value",
                "reason",
            ]
        )

    actionable = advice_df[advice_df["side"].isin(["BUY", "SELL"])].copy()
    buy_count = int((actionable["side"] == "BUY").sum())
    sell_count = int((actionable["side"] == "SELL").sum())
    target_sum = float(target_weights["target_weight"].sum()) if not target_weights.empty else 0.0

    if target_weights.empty:
        warnings.append("No target_weights.csv found. Advice falls back to live holdings only.")
    if target_sum <= 1e-9:
        warnings.append("Latest target weights sum to 0.0. Model is effectively in cash mode.")
    if target_meta.get("history_last_row_is_zero"):
        warnings.append(
            f"Latest target row is zero; last non-zero target date is {target_meta.get('last_nonzero_asof', 'unknown')}."
        )
    if promotion.get("recommendation") == "hold":
        actions.append("Promotion gate is HOLD. Treat advice as execution support, not a model-upgrade signal.")
    if buy_scale < 0.999:
        warnings.append(
            f"Buy recommendations were scaled to {buy_scale*100:.1f}% of desired rebalance due to cash constraint."
        )
    if cap_diagnostics.get("pre_cap_max_single", 0.0) > cap_diagnostics.get("max_single_position_pct", 1.0) + 1e-12:
        warnings.append("Single-name concentration cap adjusted the target weights before advice generation.")
    if cap_diagnostics.get("pre_cap_max_sector", 0.0) > cap_diagnostics.get("max_sector_weight", 1.0) + 1e-12:
        warnings.append("Sector concentration cap adjusted the target weights before advice generation.")
    if lot_diagnostics.get("selected_count", 0) > 0 and target_sum > float(lot_diagnostics.get("gross_budget", 0.0)) - 1e-12:
        warnings.append("Target weights were compressed into a lot-feasible subset for the current account size.")

    if actionable.empty:
        actions.append("Hold current state; no lot-sized rebalance trade clears the threshold.")
        status = "ok"
        headline = f"Live advice for {asof_used}: hold"
    else:
        actions.append(f"{buy_count} buy candidate(s), {sell_count} sell candidate(s) are actionable.")
        status = "warning" if warnings else "ok"
        headline = f"Live advice for {asof_used}: rebalance candidates available"

    summary = {
        "asof": asof_used,
        "nav": nav,
        "cash": float(snapshot.get("cash") or 0.0),
        "positions_value": float(snapshot.get("positions_value") or 0.0),
        "target_weight_sum": target_sum,
        "actionable_count": int(len(actionable)),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "latest_target_mode": target_meta.get("primary_mode") or promotion.get("target_mode"),
        "promotion_recommendation": promotion.get("recommendation"),
        "selected_run_id": snapshot.get("run_id"),
        "cap_diagnostics": cap_diagnostics,
        "lot_feasibility": lot_diagnostics,
    }
    report = {
        "status": status,
        "headline": headline,
        "asof": asof_used,
        "warnings": list(dict.fromkeys(warnings)),
        "actions": actions,
        "summary": summary,
    }
    return report, advice_df.sort_values(["side", "delta_value"], ascending=[True, False]).reset_index(drop=True)


def write_live_trade_advice(report: dict, advice_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "live_trade_advice.json"
    out_csv = out_dir / "live_trade_advice.csv"
    out_md = out_dir / "live_trade_advice.md"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    advice_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# Live Trade Advice",
        "",
        f"- asof: {report.get('asof')}",
        f"- headline: {report.get('headline')}",
        f"- status: {report.get('status')}",
        "",
        "## Summary",
        "",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    if report.get("warnings"):
        lines.extend([f"- {item}" for item in report["warnings"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Recommended Actions", ""])
    for item in report.get("actions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Suggested Orders", ""])
    if advice_df.empty:
        lines.append("_No rows._")
    else:
        try:
            lines.append(advice_df.to_markdown(index=False))
        except ImportError:
            lines.append(advice_df.to_csv(index=False))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_csv, out_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--target_weights", default=None)
    ap.add_argument("--target_meta", default=None)
    ap.add_argument("--promotion", default=None)
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--lot_size", type=int, default=100)
    ap.add_argument("--min_trade_value", type=float, default=5000.0)
    ap.add_argument("--buy_limit_buffer_bps", type=float, default=20.0)
    ap.add_argument("--sell_limit_buffer_bps", type=float, default=20.0)
    ap.add_argument("--max_single_position_pct", type=float, default=0.25)
    ap.add_argument("--max_sector_weight", type=float, default=0.35)
    ap.add_argument("--strategy_id", default="default")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        report, advice_df = build_live_trade_advice(
            conn,
            reports_dir=Path(args.reports_dir),
            asof=args.asof,
            strategy_id=str(args.strategy_id),
            target_weights_path=Path(args.target_weights) if args.target_weights else None,
            target_meta_path=Path(args.target_meta) if args.target_meta else None,
            promotion_path=Path(args.promotion) if args.promotion else None,
            lot_size=int(args.lot_size),
            min_trade_value=float(args.min_trade_value),
            buy_limit_buffer_bps=float(args.buy_limit_buffer_bps),
            sell_limit_buffer_bps=float(args.sell_limit_buffer_bps),
            max_single_position_pct=float(args.max_single_position_pct),
            max_sector_weight=float(args.max_sector_weight),
        )
        out_json, out_csv, out_md = write_live_trade_advice(report, advice_df, Path(args.out_dir))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"csv: {out_csv}")
        print(f"md: {out_md}")
        print(f"json: {out_json}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
