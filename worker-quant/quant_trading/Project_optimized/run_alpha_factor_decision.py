"""run_alpha_factor_decision.py — emit orders from a walk-forward factor.

Given a `strategy_id`, `factors` (single or composite), `top_k`, `min_adv`,
this script:
  1. Loads daily_prices up to `asof` from the DB
  2. Computes the factor score (reuses walk_forward_runner.composite_score
     and alpha_registry_score)
  3. Picks top-K liquid names
  4. Reads the strategy's prior NAV snapshot (or seeds from a sibling)
  5. Sizes orders equal-weight within capital cap
  6. Writes decision_runs + orders rows under the given strategy_id
  7. Optionally invokes paper_execute.py to simulate fills

This is the sibling of make_decision.py but SCOPED to alpha-factor
strategies (high52w, amihud, and all paper lanes). `make_decision.py`
remains the driver for `sprint` (which uses sprint_signal + regime).

Only supports MONTHLY cadence: emits a full rebalance target on the
first trading day of each month, holds between rebalances. Daily runs
on non-rebal days do nothing (skip=True).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import strategy_registry as reg
from walk_forward_runner import (
    alpha_registry_score, composite_score, factor_high52w, factor_mom_consist,
    is_common_stock, liquidity_mask, load_price_panel, pivot, zscore_xs,
)


ROOT = Path(__file__).resolve().parent


def _is_first_trading_day_of_month(conn, asof: str) -> bool:
    ym = asof[:7]
    row = conn.execute(
        "SELECT MIN(date) FROM daily_prices WHERE date LIKE ?",
        (f"{ym}-%",),
    ).fetchone()
    return bool(row and row[0] == asof)


def _latest_nav(conn, strategy_id: str, seed_from: str | None = None) -> float:
    r = conn.execute(
        "SELECT nav FROM account_snapshots WHERE strategy_id=? "
        "ORDER BY asof DESC, ts DESC LIMIT 1",
        (strategy_id,),
    ).fetchone()
    if r and r[0]:
        return float(r[0])
    if seed_from:
        r2 = conn.execute(
            "SELECT nav FROM account_snapshots WHERE strategy_id=? "
            "ORDER BY asof DESC, ts DESC LIMIT 1",
            (seed_from,),
        ).fetchone()
        if r2 and r2[0]:
            return float(r2[0])
    return 0.0


def compute_score(conn, asof: str, factors: list[str], min_adv: float,
                  exclude_etf_reit: bool = True) -> pd.Series:
    """Compute factor score and liquidity mask for a given `asof`."""
    # Load >= 1 year of history ending at asof for rolling windows.
    start_dt = (datetime.fromisoformat(asof) - pd.Timedelta(days=520)).date().isoformat()
    panel = load_price_panel("japan_market.db", start_dt, asof)
    if panel.empty:
        raise RuntimeError(f"empty price panel for asof={asof}")

    close_full = pivot(panel, "close")
    high_full = pivot(panel, "high")
    low_full = pivot(panel, "low")
    vol_full = pivot(panel, "volume")
    if exclude_etf_reit:
        keep = [c for c in close_full.columns if is_common_stock(c)]
        close = close_full[keep]; high = high_full[keep]
        low = low_full[keep]; vol = vol_full[keep]
    else:
        close, high, low, vol = close_full, high_full, low_full, vol_full

    # Composite across requested factors
    parts = []
    for f in factors:
        if f == "high52w":
            parts.append(zscore_xs(factor_high52w(close)).fillna(0))
        elif f == "mom_consist":
            parts.append(zscore_xs(factor_mom_consist(close)).fillna(0))
        else:
            parts.append(alpha_registry_score(close, high, low, vol,
                                              factor_name=f, sign=1.0))
    score = sum(parts) / float(len(parts))

    liq = liquidity_mask(close, vol, min_adv=min_adv)
    masked = score.where(liq)

    # Take the asof date's cross-section.
    if pd.to_datetime(asof) not in masked.index:
        # Fall back to the latest available trading day <= asof.
        avail = masked.index[masked.index <= pd.to_datetime(asof)]
        if len(avail) == 0:
            raise RuntimeError(f"no score rows available on/before {asof}")
        row = masked.loc[avail[-1]].dropna()
    else:
        row = masked.loc[pd.to_datetime(asof)].dropna()
    return row


def _last_close(conn, sym: str, asof: str) -> float | None:
    r = conn.execute(
        "SELECT close FROM daily_prices WHERE symbol=? AND date<=? "
        "ORDER BY date DESC LIMIT 1",
        (sym, asof),
    ).fetchone()
    return float(r[0]) if r and r[0] else None


class DestructiveRerunError(RuntimeError):
    """Raised when a re-run would clobber an already-executed decision."""


def _write_decision(conn, run_id: str, strategy_id: str, asof: str,
                    orders: list[dict]) -> None:
    ts = datetime.now().isoformat()
    # Guard: if run_id already has orders with status != 'proposed'
    # (partial / open / filled / cancelled), a naive REPLACE would
    # destroy the execution history. Refuse.
    existing = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE run_id=? AND status != 'proposed'",
        (run_id,),
    ).fetchone()
    if existing and existing[0] > 0:
        raise DestructiveRerunError(
            f"run_id={run_id!r} already has {existing[0]} non-proposed orders "
            f"(partial/open/filled/cancelled). Re-running would clobber execution "
            f"history. Delete prior run manually or use a distinct run_id."
        )
    # Guard: if any fill row exists for this run_id, also refuse.
    filled = conn.execute(
        "SELECT COUNT(*) FROM fills WHERE run_id=?", (run_id,),
    ).fetchone()
    if filled and filled[0] > 0:
        raise DestructiveRerunError(
            f"run_id={run_id!r} has {filled[0]} fill rows already. "
            f"Aborting to prevent PnL corruption."
        )

    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO decision_runs"
            "(run_id, asof, ts, snapshot_path, status, notes, strategy_id) "
            "VALUES (?, ?, ?, ?, 'proposed', ?, ?)",
            (run_id, asof, ts, "", f"alpha_factor decision ({strategy_id})",
             strategy_id),
        )
        # Clear any prior 'proposed' orders for this run_id — safe because
        # the two guards above ensure no executed orders survive.
        conn.execute("DELETE FROM orders WHERE run_id=? AND status='proposed'",
                     (run_id,))
        for o in orders:
            conn.execute(
                "INSERT INTO orders(order_id, run_id, asof, symbol, side, qty, "
                "order_type, limit_price, tif, reason, expected_fee, "
                "expected_slippage, expected_value, status, created_ts, "
                "strategy_id, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (o["order_id"], run_id, asof, o["symbol"], o["side"],
                 o["qty"], o["order_type"], o["limit_price"], o["tif"],
                 o["reason"], o["expected_fee"], o["expected_slippage"],
                 o["expected_value"], "proposed", ts, strategy_id, "alpha_factor"),
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy_id", required=True)
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD")
    ap.add_argument("--factors", required=True, help="comma-sep factor names")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_adv", type=float, default=50_000_000)
    ap.add_argument("--cash_cap", type=float, default=None,
                    help="per-strategy NAV cap; default=strategy_registry capital_cap_jpy")
    ap.add_argument("--paper", action="store_true",
                    help="simulate fills immediately via paper_execute.py")
    ap.add_argument("--force", action="store_true",
                    help="run even if not first trading day of month")
    ap.add_argument("--lot", type=int, default=100)
    args = ap.parse_args()

    entry = reg.get(args.strategy_id)
    if entry is None:
        print(f"[err] strategy {args.strategy_id} not in registry", file=sys.stderr)
        return 2

    conn = sqlite3.connect("japan_market.db")
    try:
        # ── Capital gate pre-check (2026-04-15, P0 fail-closed) ─────────
        # Real-tier strategies must pass capital_gate before emitting BUYs.
        # Paper lanes skip (exploration is free). On gate error for real
        # tier we fail CLOSED (refuse to emit).
        if entry.tier == "real":
            try:
                import capital_gate
                decision = capital_gate.evaluate(conn, args.strategy_id)
                paused = {"paused", "retired", "paused_sunk_only"}
                if decision.recommended_state in paused:
                    print(f"[capital_gate] BLOCK: {args.strategy_id} "
                          f"state={decision.recommended_state}. No orders emitted.")
                    for r in decision.reasons:
                        print(f"  - {r}")
                    return 0
            except Exception as _ge:
                print(f"[capital_gate] FAIL-CLOSED: evaluation errored "
                      f"({type(_ge).__name__}: {_ge}). Refusing to emit real orders.",
                      file=sys.stderr)
                return 10
        if not args.force and not _is_first_trading_day_of_month(conn, args.asof):
            print(f"[skip] {args.strategy_id}: {args.asof} is not first trading day of month "
                  f"(monthly cadence). Use --force to override.")
            return 0

        factors = [f.strip() for f in args.factors.split(",") if f.strip()]
        score_row = compute_score(conn, args.asof, factors, args.min_adv)
        if score_row.empty:
            print(f"[err] no score rows for {args.asof}", file=sys.stderr)
            return 3

        top = score_row.nlargest(args.top_k).index.tolist()
        top_set = set(top)

        # ── Rotation SELL: names held by this strategy but NOT in new top-K
        # must be liquidated to free capital for new BUYs. This makes the
        # rebalance a true rotation, not an accumulate-only.
        held_rows = conn.execute(
            """
            SELECT symbol, qty FROM positions
            WHERE strategy_id=? AND qty > 0
              AND asof = (SELECT MAX(asof) FROM positions
                          WHERE strategy_id=? AND qty > 0)
            """,
            (args.strategy_id, args.strategy_id),
        ).fetchall()
        rotate_out = [(sym, qty) for (sym, qty) in held_rows if sym not in top_set]

        # Determine capital.
        cash_cap = args.cash_cap
        if cash_cap is None:
            cash_cap = float(entry.capital_cap_jpy) if entry.capital_cap_jpy else 0.0
        if cash_cap <= 0 and entry.tier == "paper":
            # Paper lanes seed from sprint_paper NAV if unset.
            cash_cap = _latest_nav(conn, args.strategy_id, seed_from="sprint_paper")
            if cash_cap <= 0:
                cash_cap = 400_000.0   # canonical R&D unit
        if cash_cap <= 0:
            print(f"[err] no capital cap for {args.strategy_id}", file=sys.stderr)
            return 4

        # Greedy lot-buy: walk top-K in score rank, buy 1 lot of each if
        # it fits remaining budget, else skip to next. This gives a
        # variable-size portfolio (1-K names) that always uses capital
        # discipline — no overshoots, no forced-skip of expensive names
        # that would have fit the total budget.
        per_name_target = cash_cap / max(len(top), 1)
        orders = []
        run_id = f"{args.asof}__{args.strategy_id}__{hashlib.md5(args.asof.encode()).hexdigest()[:8]}"

        # Emit SELLs first so the rotation is ordered correctly.
        for i, (sym, qty) in enumerate(rotate_out):
            px = _last_close(conn, sym, args.asof)
            if px is None or px <= 0:
                continue
            orders.append({
                "order_id": f"{run_id}__sell_{i:03d}",
                "symbol": sym, "side": "SELL",
                "qty": float(qty), "order_type": "MKT",
                "limit_price": None, "tif": "DAY",
                "reason": f"alpha_factor rotation out [{','.join(factors)}]",
                "expected_fee": 0.0, "expected_slippage": 0.0,
                "expected_value": float(qty * px),
            })

        budget_remaining = cash_cap
        # Names already held that are still in new top-K — DON'T re-buy them.
        already_held = {sym for (sym, qty) in held_rows if sym in top_set}
        for i, sym in enumerate(top):
            if sym in already_held:
                continue   # existing position, hold as-is
            px = _last_close(conn, sym, args.asof)
            if px is None or px <= 0:
                continue
            lot_notional = px * args.lot
            if lot_notional > budget_remaining:
                continue   # can't afford 1 lot of this name, try next rank
            # Lots to buy: aim for per_name_target but at least 1.
            n_lots = max(1, int(per_name_target // lot_notional))
            qty = n_lots * args.lot
            notional = qty * px
            if notional > budget_remaining:
                n_lots = int(budget_remaining // lot_notional)
                if n_lots <= 0:
                    continue
                qty = n_lots * args.lot
                notional = qty * px
            budget_remaining -= notional
            orders.append({
                "order_id": f"{run_id}__{i:03d}",
                "symbol": sym, "side": "BUY",
                "qty": float(qty), "order_type": "MKT",
                "limit_price": None, "tif": "DAY",
                "reason": f"alpha_factor [{','.join(factors)}]",
                "expected_fee": 0.0, "expected_slippage": 0.0,
                "expected_value": float(qty * px),
            })

        if not orders:
            print(f"[warn] no executable orders for {args.strategy_id} on {args.asof}")
            return 0

        # Seed an initial account_snapshot if none exists (paper_execute
        # has a bug where --initial_cash is declared but never used).
        existing = conn.execute(
            "SELECT COUNT(*) FROM account_snapshots WHERE strategy_id=?",
            (args.strategy_id,),
        ).fetchone()
        if not existing or existing[0] == 0:
            with conn:
                conn.execute(
                    "INSERT INTO account_snapshots"
                    "(asof, strategy_id, ts, run_id, cash, positions_value, "
                    " nav, net_trade_cashflow, fees, tax, notes) "
                    "VALUES (?, ?, ?, NULL, ?, 0.0, ?, 0.0, 0.0, 0.0, ?)",
                    (args.asof, args.strategy_id,
                     datetime.now().isoformat(),
                     float(cash_cap), float(cash_cap),
                     f"seeded by run_alpha_factor_decision.py cap={cash_cap:.0f}"),
                )
            print(f"[seed] {args.strategy_id}: initial NAV=JPY{cash_cap:,.0f}")

        try:
            _write_decision(conn, run_id, args.strategy_id, args.asof, orders)
        except DestructiveRerunError as _dre:
            print(f"[err] {_dre}", file=sys.stderr)
            return 11
        print(f"[decision] {args.strategy_id} run_id={run_id} orders={len(orders)} "
              f"cap=JPY{cash_cap:,.0f} per_name=JPY{per_name_target:,.0f}")
        for o in orders:
            print(f"   {o['side']} {o['symbol']} x{int(o['qty'])} @ ~JPY{o['expected_value']/o['qty']:.0f}  "
                  f"notional=JPY{o['expected_value']:,.0f}")

        if args.paper:
            # Seed initial_cash from the cap so first paper fill doesn't
            # produce negative NAV.
            seed_cash = cash_cap
            existing = conn.execute(
                "SELECT COUNT(*) FROM account_snapshots WHERE strategy_id=?",
                (args.strategy_id,),
            ).fetchone()
            if existing and existing[0] > 0:
                seed_cash = 0.0   # paper_execute will use the last snapshot
            cmd = [sys.executable, "paper_execute.py",
                   "--run_id", run_id, "--price_mode", "close",
                   "--no_require_approval",
                   "--initial_cash", str(seed_cash)]
            print(f"[paper_execute] {' '.join(cmd)}")
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                print(f"[err] paper_execute rc={rc}", file=sys.stderr)
                return 5
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
