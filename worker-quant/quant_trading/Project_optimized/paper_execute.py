"""Paper-trading execution bridge.

Turns proposed orders into simulated fills using daily_prices, then refreshes
positions / NAV / execution report so the paper account can move end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from execution_model import sbi_fee

from trade_schema import connect, ensure_trade_tables, get_run_meta, resolve_run_artifact_dir
from import_fills import import_fills_df
from build_positions import build_positions
from build_account_snapshot import build_account_snapshot
from execution_report import generate_execution_report
from market_data_utils import (
    refresh_market_data_if_needed,
    latest_db_date,
    refresh_intraday_if_needed,
)


def post_trade_analytics(conn, run_id: str, strategy_id: str = "default") -> dict:
    rows = conn.execute(
        """
        SELECT side, qty, price, COALESCE(fee,0), COALESCE(price_validated,0), COALESCE(quote_close, price)
        FROM fills
        WHERE run_id=? AND strategy_id=?
        """,
        (run_id, strategy_id),
    ).fetchall()
    fill_count = len(rows)
    total_commission = 0.0
    validated = 0
    slippages = []
    for side, qty, price, fee, price_validated, quote_close in rows:
        total_commission += float(fee or 0.0)
        validated += int(price_validated or 0)
        price = float(price or 0.0)
        ref = float(quote_close or price or 0.0)
        if price > 0.0 and ref > 0.0:
            direction = 1.0 if str(side).upper() == "BUY" else -1.0
            slippages.append(direction * (price / ref - 1.0) * 10000.0)
    avg_slippage_bps = float(sum(slippages) / len(slippages)) if slippages else 0.0
    fill_validation_rate = float(validated / fill_count) if fill_count else 1.0
    return {
        "run_id": run_id,
        "strategy_id": strategy_id,
        "fill_count": fill_count,
        "avg_slippage_bps": avg_slippage_bps,
        "total_commission": total_commission,
        "fill_validation_rate": fill_validation_rate,
    }


def _resolve_run_id(conn, run_id: str | None) -> str:
    if run_id:
        return run_id
    row = conn.execute(
        """
        SELECT run_id
        FROM decision_runs
        WHERE status IN ('proposed', 'partial')
        ORDER BY ts DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise RuntimeError("No proposed/partial decision_run found.")
    return str(row[0])


def _existing_cash_snapshot(conn, asof: str, strategy_id: str = "default") -> float | None:
    row = conn.execute(
        """
        SELECT cash
        FROM account_snapshots
        WHERE asof=? AND strategy_id=?
        LIMIT 1
        """,
        (asof, strategy_id),
    ).fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])


def _load_orders(conn, run_id: str, strategy_id: str = "default") -> list[tuple]:
    return conn.execute(
        """
        SELECT order_id, symbol, side, qty, order_type, limit_price
        FROM orders
        WHERE run_id=? AND strategy_id=? AND status IN ('proposed', 'partial')
        ORDER BY created_ts, order_id
        """,
        (run_id, strategy_id),
    ).fetchall()


def _fingerprint_orders(orders: list[tuple] | list[dict]) -> str:
    """Canonical hash over (order_id, symbol, side, qty, order_type, limit_price).

    Used to detect order-set drift between pending_approval artifact and DB
    state at approve time.
    """
    canonical: list[tuple] = []
    for o in orders:
        if isinstance(o, dict):
            canonical.append((
                str(o.get("order_id") or ""),
                str(o.get("symbol") or ""),
                str(o.get("side") or ""),
                float(o.get("qty") or 0),
                str(o.get("order_type") or ""),
                float(o["limit_price"]) if o.get("limit_price") is not None else None,
            ))
        else:
            canonical.append((
                str(o[0] or ""),
                str(o[1] or ""),
                str(o[2] or ""),
                float(o[3] or 0),
                str(o[4] or ""),
                float(o[5]) if o[5] is not None else None,
            ))
    canonical.sort(key=lambda x: x[0])
    blob = json.dumps(canonical, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _pending_artifact_paths(reports_dir: Path | str, run_id: str) -> tuple[Path, Path, Path, Path]:
    """Return (per_run_json, per_run_md, latest_json, latest_md).

    Per-run-id files are the source of truth for approve validation (never
    overwritten by other concurrent runs). latest_* are human-facing pointers
    refreshed on every new pending artifact for convenience.
    """
    rd = Path(reports_dir)
    safe = run_id.replace("/", "_").replace("\\", "_")
    return (
        rd / f"pending_approval_{safe}.json",
        rd / f"PENDING_APPROVAL_{safe}.md",
        rd / "pending_approval.json",
        rd / "PENDING_APPROVAL.md",
    )


def write_pending_approval(
    reports_dir: Path | str,
    run_id: str,
    asof: str,
    orders: list[tuple],
    strategy_id: str,
) -> tuple[Path, Path]:
    """Emit pending approval artifacts instead of auto-filling.

    Returns (per_run_json_path, per_run_md_path). Also refreshes
    pending_approval.json / PENDING_APPROVAL.md as latest-pointer convenience
    files. The per-run-id artifact is the source of truth consulted by the
    approve path.
    """
    rd = Path(reports_dir)
    rd.mkdir(parents=True, exist_ok=True)
    order_rows = [
        {
            "order_id": o[0],
            "symbol": o[1],
            "side": o[2],
            "qty": float(o[3] or 0),
            "order_type": o[4],
            "limit_price": float(o[5]) if o[5] is not None else None,
        }
        for o in orders
    ]
    fingerprint = _fingerprint_orders(orders)
    payload = {
        "status": "awaiting_approval",
        "run_id": run_id,
        "asof": asof,
        "strategy_id": strategy_id,
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "orders": order_rows,
        "order_fingerprint": fingerprint,
        "approve_command": (
            f"python paper_execute.py --run_id {run_id} --asof {asof} --approve"
        ),
    }
    per_run_json, per_run_md, latest_json, latest_md = _pending_artifact_paths(rd, run_id)
    blob = json.dumps(payload, ensure_ascii=False, indent=2)
    per_run_json.write_text(blob, encoding="utf-8")
    latest_json.write_text(blob, encoding="utf-8")  # convenience pointer

    lines = [
        f"# Pending paper approval — {asof}",
        "",
        f"- run_id: `{run_id}`",
        f"- strategy_id: `{strategy_id}`",
        f"- generated: {payload['generated_at_utc']}",
        f"- order_fingerprint: `{fingerprint}`",
        "",
        "## Proposed orders",
        "",
        "| symbol | side | qty | type | limit |",
        "|---|---|---|---|---|",
    ]
    for o in order_rows:
        lp = "" if o["limit_price"] is None else f"{o['limit_price']:.2f}"
        lines.append(f"| {o['symbol']} | {o['side']} | {o['qty']:.0f} | {o['order_type']} | {lp} |")
    lines += [
        "",
        "## Approve",
        "",
        "```",
        payload["approve_command"],
        "```",
        "",
        "Until approved, no simulated fills are generated and the run stays in",
        "`awaiting_approval` state. Approve will abort if DB orders have drifted",
        "from the fingerprint above.",
        "",
    ]
    md_blob = "\n".join(lines)
    per_run_md.write_text(md_blob, encoding="utf-8")
    latest_md.write_text(md_blob, encoding="utf-8")
    return per_run_json, per_run_md


def _update_run_status_after_paper(conn, run_id: str, order_count: int, fills_inserted: int) -> str:
    if fills_inserted > 0:
        status = "paper_filled"
    elif order_count <= 0:
        status = "paper_no_orders"
    else:
        status = "paper_checked_no_fill"
    conn.execute("UPDATE decision_runs SET status=? WHERE run_id=?", (status, run_id))
    conn.commit()
    return status


def _market_quote(conn, symbol: str, asof: str, price_mode: str) -> dict | None:
    if price_mode == "latest":
        row = conn.execute(
            """
            SELECT ts, price, open, high, low, close, source
            FROM intraday_quotes
            WHERE symbol=? AND asof=?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (symbol, asof),
        ).fetchone()
        if row and row[1] is not None:
            return {
                "price": float(row[1]),
                "price_ts": str(row[0]),
                "price_source": str(row[6] or "intraday_quotes"),
                "price_mode": price_mode,
                "quote_open": float(row[2]) if row[2] is not None else None,
                "quote_high": float(row[3]) if row[3] is not None else None,
                "quote_low": float(row[4]) if row[4] is not None else None,
                "quote_close": float(row[5]) if row[5] is not None else None,
            }

    col = "open" if price_mode == "open" else "close"
    row = conn.execute(
        f"""
        SELECT date, open, high, low, close
        FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    if not row:
        return None
    idx = {"open": 1, "close": 4}[col]
    if row[idx] is None:
        return None
    return {
        "price": float(row[idx]),
        "price_ts": f"{row[0]} 09:00:00" if price_mode == "open" else f"{row[0]} 15:00:00",
        "price_source": "daily_prices",
        "price_mode": price_mode,
        "quote_open": float(row[1]) if row[1] is not None else None,
        "quote_high": float(row[2]) if row[2] is not None else None,
        "quote_low": float(row[3]) if row[3] is not None else None,
        "quote_close": float(row[4]) if row[4] is not None else None,
    }


def _fill_price(base_price: float, side: str, slippage_bps: float) -> float:
    direction = 1.0 if side.upper() == "BUY" else -1.0
    return base_price * (1.0 + direction * slippage_bps / 10000.0)


def _validate_fill(fill_price: float, quote: dict) -> tuple[int, str]:
    q_low = quote.get("quote_low")
    q_high = quote.get("quote_high")
    if q_low is None or q_high is None:
        return 0, "missing quote range"
    if q_low - 1e-9 <= fill_price <= q_high + 1e-9:
        return 1, "validated against quote range"
    return 0, f"fill price {fill_price:.6f} outside quote range [{q_low:.6f}, {q_high:.6f}]"


def simulate_fills(
    conn,
    run_id: str,
    asof: str,
    price_mode: str,
    slippage_bps: float,
    fee_bps: float,
    fill_ratio: float,
    strategy_id: str = "default",
    fee_mode: str = "sbi_zero",
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    ts = f"{asof} 09:00:00" if price_mode == "open" else f"{asof} 15:00:00"

    for order_id, symbol, side, qty, _order_type, _limit_price in _load_orders(conn, run_id, strategy_id=strategy_id):
        quote = _market_quote(conn, str(symbol), asof, price_mode)
        if quote is None:
            missing.append(str(symbol))
            continue

        order_qty = float(qty or 0.0)
        fill_qty = int(order_qty * fill_ratio)
        if fill_qty <= 0:
            continue

        price = _fill_price(float(quote["price"]), str(side), slippage_bps)
        notional = fill_qty * price
        if fee_mode == "flat_bps":
            fee = notional * fee_bps / 10000.0
        else:
            fee = sbi_fee(notional, fee_mode)
        price_validated, validation_note = _validate_fill(price, quote)
        if not price_validated:
            # 盘中数据只有单一价位（high=low）时滑点会越界；clamp 到行情区间而非中断
            q_low = quote.get("quote_low")
            q_high = quote.get("quote_high")
            if q_low is not None and q_high is not None:
                price = max(q_low, min(q_high, price))
                print(f"[paper_execute] ⚠️  {symbol} {side}: {validation_note} → clamped to {price:.4f}")
            else:
                raise RuntimeError(
                    f"Paper fill validation failed for {symbol} {side}: {validation_note} "
                    f"(source={quote['price_source']} ts={quote['price_ts']})"
                )

        rows.append(
            {
                "ts": quote["price_ts"] if price_mode == "latest" else ts,
                "symbol": str(symbol),
                "side": str(side).upper(),
                "qty": float(fill_qty),
                "price": float(round(price, 6)),
                "fee": float(round(fee, 6)),
                "tax": 0.0,
                "external_ref": f"paper::{run_id}::{order_id}",
                "order_id": str(order_id),
                "price_source": str(quote["price_source"]),
                "price_ts": str(quote["price_ts"]),
                "price_mode": str(quote["price_mode"]),
                "quote_open": quote.get("quote_open"),
                "quote_high": quote.get("quote_high"),
                "quote_low": quote.get("quote_low"),
                "quote_close": quote.get("quote_close"),
                "price_validated": int(price_validated),
                "validation_note": validation_note,
            }
        )

    return pd.DataFrame(rows), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", default=None, help="default: latest proposed/partial run")
    ap.add_argument("--asof", default=None, help="default: run asof")
    ap.add_argument("--price_mode", choices=["latest", "open", "close"], default="latest")
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--fee_mode", default="sbi_zero", choices=["sbi_zero", "sbi_standard", "flat_bps"])
    ap.add_argument("--fill_ratio", type=float, default=1.0, help="0-1 simulated participation ratio")
    ap.add_argument("--initial_cash", type=float, default=0.0, help="used only for first account snapshot")
    ap.add_argument("--refresh_data", action="store_true", help="refresh market data before paper execution")
    ap.add_argument("--refresh_lookback", type=int, default=30, help="lookback days used when refresh_data is enabled")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument(
        "--require_approval",
        dest="require_approval",
        action="store_true",
        default=True,
        help="(default) skip fills, emit pending_approval artifacts, exit without executing",
    )
    ap.add_argument(
        "--no_require_approval",
        dest="require_approval",
        action="store_false",
        help="legacy auto-fill behaviour (not recommended — pollutes paper/live reconciliation)",
    )
    ap.add_argument(
        "--approve",
        action="store_true",
        default=False,
        help="user has reviewed the pending orders and explicitly approves paper fills for this run",
    )
    args = ap.parse_args()

    if not (0.0 <= args.fill_ratio <= 1.0):
        raise ValueError("--fill_ratio must be between 0 and 1")

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        run_id = _resolve_run_id(conn, args.run_id)
        meta = get_run_meta(conn, run_id)
        asof = args.asof or (meta.get("asof") if meta else None)
        strategy_id = str((meta or {}).get("strategy_id", "default") or "default")
        if not asof:
            raise ValueError("asof is required (pass --asof or ensure decision_runs has asof for this run_id)")
    finally:
        conn.close()

    # Approval gate runs BEFORE market-data refresh: the whole point is we do
    # not touch paper state (or depend on live data) until the human has seen
    # which orders would fill.
    if args.require_approval and not args.approve:
        conn = connect(args.db)
        ensure_trade_tables(conn)
        try:
            orders = _load_orders(conn, run_id, strategy_id=strategy_id)
            if not orders:
                print(f"No proposed/partial orders for run_id={run_id}. Nothing to approve.")
                conn.execute(
                    "UPDATE decision_runs SET status=? WHERE run_id=?",
                    ("paper_no_orders", run_id),
                )
                conn.commit()
                return
            json_path, md_path = write_pending_approval(
                args.reports_dir, run_id=run_id, asof=asof, orders=orders, strategy_id=strategy_id,
            )
            conn.execute(
                "UPDATE decision_runs SET status=? WHERE run_id=?",
                ("awaiting_approval", run_id),
            )
            conn.commit()
            print("=" * 70)
            print("Paper execution PAUSED — approval required")
            print(f"run_id: {run_id}  asof: {asof}  strategy: {strategy_id}")
            print(f"orders awaiting approval: {len(orders)}")
            print(f"review: {md_path}")
            print(f"approve: python paper_execute.py --run_id {run_id} --asof {asof} --approve")
            print("=" * 70)
        finally:
            conn.close()
        return

    # Approve path: before any fill work, (1) verify the DB orders still match
    # the pending_approval artifact the user reviewed, and (2) atomically
    # transition decision_runs.status from awaiting_approval -> approved_in_progress
    # so a concurrent approve cannot double-fill.
    if args.approve:
        per_run_json, _per_run_md, _latest_json, _latest_md = _pending_artifact_paths(
            args.reports_dir, run_id
        )
        if not per_run_json.exists():
            raise RuntimeError(
                f"approve aborted: no pending_approval artifact at {per_run_json}. "
                "Run paper_execute.py without --approve first to generate the review artifact."
            )
        pending = json.loads(per_run_json.read_text(encoding="utf-8"))
        expected_fp = pending.get("order_fingerprint")
        if not expected_fp:
            raise RuntimeError(
                f"approve aborted: pending artifact {per_run_json} missing order_fingerprint "
                "(regenerate by rerunning without --approve)."
            )

        conn = connect(args.db)
        ensure_trade_tables(conn)
        try:
            db_orders = _load_orders(conn, run_id, strategy_id=strategy_id)
            actual_fp = _fingerprint_orders(db_orders)
            if actual_fp != expected_fp:
                raise RuntimeError(
                    f"approve aborted: order-set drift for run_id={run_id}. "
                    f"pending fingerprint={expected_fp} but DB now={actual_fp}. "
                    f"Re-review by rerunning paper_execute without --approve."
                )
            # CAS: only one approve may advance an awaiting_approval run.
            cur = conn.execute(
                "UPDATE decision_runs SET status=? "
                "WHERE run_id=? AND status=?",
                ("approved_in_progress", run_id, "awaiting_approval"),
            )
            if cur.rowcount != 1:
                (current_status,) = conn.execute(
                    "SELECT status FROM decision_runs WHERE run_id=?", (run_id,)
                ).fetchone() or (None,)
                raise RuntimeError(
                    f"approve aborted: run_id={run_id} is not in awaiting_approval "
                    f"(current status={current_status!r}). Another approve may already be "
                    "running, or the run has been expired."
                )
            conn.commit()
        finally:
            conn.close()

    if args.refresh_data:
        _before, refreshed_to, _did_refresh = refresh_market_data_if_needed(
            args.db,
            target_date=asof,
            lookback_days=int(args.refresh_lookback),
            force=False,
        )
        if refreshed_to and str(refreshed_to) < str(asof):
            raise RuntimeError(
                f"Market data refresh completed but DB is still behind requested asof={asof}. "
                f"Latest daily_prices date is {refreshed_to}."
            )
        if args.price_mode == "latest":
            refresh_intraday_if_needed(
                args.db,
                target_date=asof,
                symbols_arg=None,
                force=False,
            )

    db_latest = latest_db_date(args.db)
    if db_latest and str(asof) > str(db_latest):
        raise RuntimeError(
            f"Requested asof={asof} is newer than latest daily_prices date {db_latest}. "
            "Refresh data first or choose an earlier asof."
        )

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        meta = get_run_meta(conn, run_id)
        source_strategy = str((meta or {}).get("strategy_id", "default") or "default")
        # v3 C-1: paper track is physically separated from live to stop
        # paper simulator from polluting the real SBI strategy state.
        # Orders are still READ from source_strategy (that is where the
        # signal chain wrote them); fills / positions / account snapshots
        # are WRITTEN under <source>_paper unless already suffixed.
        if source_strategy.endswith("_paper"):
            paper_strategy = source_strategy
        else:
            paper_strategy = f"{source_strategy}_paper"
        order_count = len(_load_orders(conn, run_id, strategy_id=source_strategy))
        fills_df, missing = simulate_fills(
            conn,
            run_id=run_id,
            asof=asof,
            price_mode=args.price_mode,
            slippage_bps=float(args.slippage_bps),
            fee_bps=float(args.fee_bps),
            fill_ratio=float(args.fill_ratio),
            strategy_id=source_strategy,
            fee_mode=str(args.fee_mode),
        )

        inserted = 0
        if not fills_df.empty:
            inserted = import_fills_df(
                conn,
                run_id=run_id,
                asof=asof,
                df=fills_df,
                venue="PAPER",
                force=False,
                source="paper_simulator",
                strategy_id=paper_strategy,
            )

        prev_asof, rows_out, missing_px = build_positions(conn, run_id, asof, strategy_id=paper_strategy)
        starting_cash = _existing_cash_snapshot(conn, asof, strategy_id=paper_strategy)
        if starting_cash is not None:
            # Today's snapshot already exists (re-run) — pass it so it is preserved
            snap = build_account_snapshot(conn, run_id, asof, initial_cash=starting_cash, strategy_id=paper_strategy)
        else:
            # No snapshot for today yet — let build_account_snapshot derive
            # cash_start from the previous day's snapshot.  Only fall back to
            # args.initial_cash when there is no prior snapshot at all.
            snap = build_account_snapshot(conn, run_id, asof, initial_cash=0.0, strategy_id=paper_strategy)

        artifact_dir = resolve_run_artifact_dir(meta.get("snapshot_path") if meta else None)
        if artifact_dir is None:
            artifact_dir = Path("artifacts/decision") / asof / run_id
        md_path, csv_path = generate_execution_report(conn, run_id, asof, artifact_dir,
                                                       strategy_id=paper_strategy)
        analytics = post_trade_analytics(conn, run_id, strategy_id=paper_strategy)
        reports_dir = Path(args.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "execution_quality.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_status = _update_run_status_after_paper(conn, run_id, order_count=order_count, fills_inserted=inserted)

        print("=" * 70)
        print("Paper execution complete")
        print(f"run_id: {run_id}")
        print(f"asof: {asof}")
        print(f"run_status: {run_status}")
        print(f"fills_inserted: {inserted}")
        print(f"avg_slippage_bps: {analytics['avg_slippage_bps']:.2f}")
        print(f"fill_validation_rate: {analytics['fill_validation_rate']:.2%}")
        print(f"positions: {len(rows_out)} (prev_positions_asof={prev_asof})")
        print(f"nav: {snap['nav']:,.2f} cash={snap['cash_end']:,.2f} positions_value={snap['positions_value']:,.2f}")
        if missing:
            print(f"missing market price for orders: {missing}")
        if missing_px:
            print(f"missing valuation price for positions: {missing_px}")
        print(f"report_md: {md_path}")
        print(f"report_csv: {csv_path}")
        print("=" * 70)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
