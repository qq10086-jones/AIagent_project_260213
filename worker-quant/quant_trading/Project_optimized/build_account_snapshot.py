import argparse
import os
from datetime import datetime

from trade_schema import connect, ensure_trade_tables
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


_HTR_SSOT_STRATEGIES = frozenset({"etf_buyhold"})


def _refuse_htr_ssot_write(strategy_id: str) -> None:
    """Per ADR-0008 (2026-05-27 HTR cutover): live portfolio state for the
    strategies listed in _HTR_SSOT_STRATEGIES lives in
    ``HotThemeRotator/reports/portfolio/journal/*.jsonl``, not this DB.
    Set ``HTR_CUTOVER_OVERRIDE=1`` only for rare maintenance (e.g. re-running
    the cutover migration itself)."""
    if strategy_id not in _HTR_SSOT_STRATEGIES:
        return
    if os.environ.get("HTR_CUTOVER_OVERRIDE") == "1":
        return
    raise RuntimeError(
        f"Refusing account_snapshots write for strategy_id={strategy_id!r}: "
        f"per ADR-0008 (cutover 2026-05-27), HotThemeRotator journal is the "
        f"single source of truth. Record cash/fills via HTR CLI/API; set "
        f"HTR_CUTOVER_OVERRIDE=1 only for cutover maintenance."
    )


def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def get_prev_snapshot(conn, asof: str, strategy_id: str = "default"):
    row = conn.execute(
        "SELECT asof, cash, nav FROM account_snapshots WHERE asof < ? AND strategy_id=? ORDER BY asof DESC LIMIT 1",
        (asof, strategy_id)
    ).fetchone()
    return row  # (asof, cash, nav) or None

def get_trade_cashflow(conn, run_id: str, asof: str, strategy_id: str = "default"):
    rows = conn.execute(
        "SELECT side, qty, price, COALESCE(fee,0), COALESCE(tax,0) FROM fills WHERE run_id=? AND asof=? AND strategy_id=?",
        (run_id, asof, strategy_id)
    ).fetchall()

    buy_notional = sell_notional = 0.0
    fees = tax = 0.0
    for side, qty, price, fee, tx in rows:
        side = str(side).upper()
        qty = float(qty); price = float(price)
        fee = float(fee); tx = float(tx)
        fees += fee; tax += tx
        notional = qty * price
        if side == "BUY":
            buy_notional += notional
        elif side == "SELL":
            sell_notional += notional
        else:
            raise ValueError(f"Unknown side: {side}")

    # net cashflow from trades:
    # SELL increases cash, BUY decreases cash, fees/tax decrease cash
    net_trade_cashflow = sell_notional - buy_notional - fees - tax
    return net_trade_cashflow, fees, tax, buy_notional, sell_notional, len(rows)


def get_cash_ledger_delta(conn, asof: str) -> float:
    """Sum cash ledger for a given asof (deposits/dividends positive, withdrawals negative)."""
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount),0) FROM cash_ledger WHERE asof=?",
            (asof,),
        ).fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0

def get_positions_value(conn, asof: str, strategy_id: str = "default"):
    # prefer market_value if present; if null, treat as 0 (dashboard will warn separately)
    row = conn.execute(
        "SELECT COALESCE(SUM(COALESCE(market_value, 0)), 0) FROM positions WHERE asof=? AND strategy_id=?",
        (asof, strategy_id)
    ).fetchone()
    return float(row[0]) if row else 0.0


def build_account_snapshot(conn, run_id: str, asof: str, initial_cash: float = 0.0, strategy_id: str = "default") -> dict:
    """Write account_snapshots for asof, using prior snapshot cash as starting point.

    Returns a dict ...
    """
    _refuse_htr_ssot_write(strategy_id)
    ensure_trade_tables(conn)
    prev = get_prev_snapshot(conn, asof, strategy_id=strategy_id)
    
    # Determine cash_start: prefer previous snapshot, then initial_cash fallback
    if prev is not None and initial_cash <= 0:
        # Normal path: derive from yesterday's actual cash
        prev_asof, cash_start, _prev_nav = prev
        cash_start = float(cash_start)
    elif prev is not None and initial_cash > 0:
        # Re-run for today: caller already knows today's starting cash
        cash_start = float(initial_cash)
        prev_asof = prev[0]
    else:
        # No prior snapshot at all (first ever run)
        cash_start = float(initial_cash)
        prev_asof = None

    net_cf, fees, tax, buy_notional, sell_notional, nfills = get_trade_cashflow(conn, run_id, asof, strategy_id=strategy_id)
    cash_ledger_delta = get_cash_ledger_delta(conn, asof)
    cash_end = cash_start + net_cf + cash_ledger_delta
    pos_val = get_positions_value(conn, asof, strategy_id=strategy_id)
    nav = cash_end + pos_val

    with conn:
        conn.execute("DELETE FROM account_snapshots WHERE asof=? AND strategy_id=?", (asof, strategy_id))
        conn.execute(
            """
            INSERT INTO account_snapshots(
              asof, strategy_id, ts, run_id, cash, positions_value, nav, net_trade_cashflow, fees, tax, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asof, strategy_id, now_iso(), run_id,
                cash_end, pos_val, nav, net_cf, fees, tax,
                f"cash_start={cash_start}; prev_asof={prev_asof}; buy={buy_notional}; sell={sell_notional}; fills={nfills}; cash_ledger_delta={cash_ledger_delta}"
            )
        )

    return {
        "asof": asof,
        "run_id": run_id,
        "strategy_id": strategy_id,
        "prev_asof": prev_asof,
        "cash_start": cash_start,
        "net_trade_cashflow": net_cf,
        "cash_ledger_delta": cash_ledger_delta,
        "cash_end": cash_end,
        "positions_value": pos_val,
        "nav": nav,
        "fees": fees,
        "tax": tax,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "n_fills": nfills,
    }

def _collect_candidate_asofs(conn, *, after_asof: str | None, through_asof: str, strategy_id: str) -> list[str]:
    """Return the sorted set of dates strictly > after_asof and <= through_asof that
    warrant a snapshot entry: any date with fills, positions, or cash_ledger activity
    for the given strategy. Hold-only days appear via positions; deposit-only days via
    cash_ledger. ``after_asof=None`` means "from the beginning"."""
    lower = after_asof or "0000-00-00"
    dates: set[str] = set()

    rows = conn.execute(
        "SELECT DISTINCT asof FROM fills WHERE asof > ? AND asof <= ? AND strategy_id=?",
        (lower, through_asof, strategy_id),
    ).fetchall()
    dates.update(r[0] for r in rows)

    rows = conn.execute(
        "SELECT DISTINCT asof FROM positions WHERE asof > ? AND asof <= ? AND strategy_id=?",
        (lower, through_asof, strategy_id),
    ).fetchall()
    dates.update(r[0] for r in rows)

    try:
        rows = conn.execute(
            "SELECT DISTINCT asof FROM cash_ledger WHERE asof > ? AND asof <= ?",
            (lower, through_asof),
        ).fetchall()
        dates.update(r[0] for r in rows)
    except Exception:
        pass

    return sorted(dates)


def rebuild_snapshot_chain(conn, *, asof: str, strategy_id: str = "default",
                           initial_cash: float = 0.0) -> list[dict]:
    """Advance the account_snapshots chain for ``strategy_id`` through ``asof``.

    Strategy:
      * Find the latest existing snapshot for the strategy at or before ``asof``.
      * Collect every date strictly after that snapshot (through ``asof``) that has
        fills, positions, or cash_ledger activity — including **hold-only days**
        where positions were merely marked-to-market and no fills occurred.
      * Rebuild a snapshot for each such day in chronological order so cash and
        NAV flow correctly day-by-day. On a hold-only day the existing
        ``build_account_snapshot`` yields ``net_trade_cashflow=0``, carrying the
        prior cash forward and refreshing ``positions_value`` / ``nav`` from the
        day's ``positions`` rows.

    Returns the list of per-day snapshot result dicts (empty if nothing to do).
    """
    ensure_trade_tables(conn)

    prev = get_prev_snapshot(conn, asof, strategy_id=strategy_id)
    latest_snap_asof = prev[0] if prev else None

    # Include `asof` itself only if there's no snapshot for it yet, to avoid
    # re-writing the same day unnecessarily when caller is idempotent.
    candidates = _collect_candidate_asofs(
        conn,
        after_asof=latest_snap_asof,
        through_asof=asof,
        strategy_id=strategy_id,
    )

    # If no snapshot existed and the target asof has no activity either, still
    # anchor a snapshot at ``asof`` using initial_cash so the chain has a root.
    if not prev and not candidates:
        candidates = [asof]

    results: list[dict] = []
    for i, d in enumerate(candidates):
        rid_row = conn.execute(
            "SELECT run_id FROM fills WHERE asof=? AND strategy_id=? ORDER BY ts DESC LIMIT 1",
            (d, strategy_id),
        ).fetchone()
        rid = rid_row[0] if rid_row else "manual-rebuild"
        # initial_cash only applies when anchoring the very first snapshot in the
        # chain (prev is None and we're on the first rebuilt date).
        first_anchor = (prev is None and i == 0)
        results.append(
            build_account_snapshot(
                conn,
                rid,
                d,
                initial_cash=float(initial_cash) if first_anchor else 0.0,
                strategy_id=strategy_id,
            )
        )

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", required=True)  # YYYY-MM-DD
    ap.add_argument("--initial_cash", type=float, default=0.0, help="Used only if no previous snapshot exists")
    ap.add_argument("--strategy_id", default="default")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        res = build_account_snapshot(
            conn,
            args.run_id,
            args.asof,
            initial_cash=float(args.initial_cash),
            strategy_id=args.strategy_id,
        )
        print("✅ account_snapshot saved")
        print(f"asof={res['asof']} run_id={res['run_id']}")
        print(f"cash_start={res['cash_start']:,.0f} net_trade_cf={res['net_trade_cashflow']:,.0f} cash_ledger={res['cash_ledger_delta']:,.0f} cash_end={res['cash_end']:,.0f}")
        print(f"positions_value={res['positions_value']:,.0f} nav={res['nav']:,.0f}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
