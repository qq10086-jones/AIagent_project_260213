"""Intraday decision issuer — signals for user to mirror-trade into SBI.

Runs ~14:45 JST (30-ish minutes before the 15:00 Tokyo close). Refreshes
intraday quotes, reads the most recent model target_weights, compares to
REAL SBI positions (source='sbi_actual' only — never paper_simulator), and
emits an actionable BUY/SELL sheet plus a Discord notification.

Core rule: this tool does NOT write fills, does NOT mutate positions. It
produces a human-readable signal for the user to execute at SBI.

Usage
-----
    # full run with Discord alert
    python intraday_decision.py

    # dry preview (no webhook, still writes file)
    python intraday_decision.py --no_webhook

    # skip intraday quote refresh (fast, uses last cached)
    python intraday_decision.py --no_refresh
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

JST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------- DB helpers


def _sbi_positions(conn: sqlite3.Connection) -> dict[str, dict]:
    """Compute real SBI holdings from sbi_actual fills only.

    Returns: {symbol: {"qty": float, "avg_cost": float}}
    """
    rows = conn.execute(
        """
        SELECT symbol,
               SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END) AS net_qty,
               SUM(CASE WHEN side='BUY' THEN qty*price ELSE 0 END) AS buy_notional,
               SUM(CASE WHEN side='BUY' THEN qty ELSE 0 END) AS total_bought
        FROM fills
        WHERE source='sbi_actual'
        GROUP BY symbol
        HAVING net_qty > 0
        """
    ).fetchall()
    out = {}
    for sym, qty, buy_notional, total_bought in rows:
        avg_cost = (buy_notional / total_bought) if total_bought else 0.0
        out[sym] = {"qty": float(qty), "avg_cost": float(avg_cost)}
    return out


def _sbi_cash(conn: sqlite3.Connection) -> float:
    """Latest cash from account_snapshots labelled as SBI actual."""
    row = conn.execute(
        """
        SELECT cash FROM account_snapshots
        WHERE strategy_id='sprint' AND (notes LIKE '%SBI actual%' OR run_id LIKE 'sbi-%')
        ORDER BY asof DESC LIMIT 1
        """
    ).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return 0.0


def _intraday_price(conn: sqlite3.Connection, symbol: str) -> tuple[float | None, str | None]:
    row = conn.execute(
        "SELECT price, ts FROM intraday_quotes WHERE symbol=? ORDER BY ts DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT close, date FROM daily_prices WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    if row and row[0] is not None:
        return float(row[0]), str(row[1])
    return None, None


# ---------------------------------------------------------------- trade logic


def _compute_trades(
    conn: sqlite3.Connection,
    target_weights: dict[str, float],
    lot_size: int = 100,
    min_notional: float = 5000.0,
) -> dict:
    positions = _sbi_positions(conn)
    cash = _sbi_cash(conn)

    # Mark-to-market current holdings with intraday prices
    held_mv = 0.0
    holdings_detail: list[dict] = []
    for sym, p in positions.items():
        px, ts = _intraday_price(conn, sym)
        mv = (px or 0.0) * p["qty"]
        held_mv += mv
        holdings_detail.append({
            "symbol": sym, "qty": p["qty"], "avg_cost": p["avg_cost"],
            "price": px, "price_ts": ts, "market_value": mv,
            "unrealized_pnl": (px - p["avg_cost"]) * p["qty"] if px else 0.0,
        })
    nav = cash + held_mv

    # Target $ per symbol
    targets_jpy: dict[str, float] = {sym: w * nav for sym, w in target_weights.items()}
    current_qty = {sym: positions.get(sym, {}).get("qty", 0.0) for sym in set(target_weights) | set(positions)}

    trades: list[dict] = []
    for sym in sorted(set(target_weights) | set(positions)):
        px, ts = _intraday_price(conn, sym)
        if px is None or px <= 0:
            continue
        target_qty_raw = targets_jpy.get(sym, 0.0) / px
        # Round to lots
        target_qty = round(target_qty_raw / lot_size) * lot_size
        cur_qty = current_qty.get(sym, 0.0)
        delta = target_qty - cur_qty
        if delta == 0:
            continue
        notional = abs(delta) * px
        if notional < min_notional:
            continue
        side = "BUY" if delta > 0 else "SELL"
        trades.append({
            "symbol": sym,
            "side": side,
            "qty": int(abs(delta)),
            "limit_price_suggest": round(px, 1),
            "intraday_ref_price": px,
            "intraday_ref_ts": ts,
            "target_weight": target_weights.get(sym, 0.0),
            "current_qty": cur_qty,
            "target_qty": target_qty,
            "notional": notional,
        })
    return {
        "nav": nav,
        "cash": cash,
        "held_market_value": held_mv,
        "holdings": holdings_detail,
        "trades": trades,
    }


def _load_target_weights(reports_dir: Path) -> dict[str, float]:
    path = reports_dir / "target_weights.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        sym = str(r.get("symbol", "")).strip()
        w = float(r.get("target_weight", 0.0) or 0.0)
        if sym and w > 0:
            out[sym] = w
    return out


# ---------------------------------------------------------------- outputs


def _write_signal_file(reports_dir: Path, payload: dict, asof: str) -> Path:
    out = reports_dir / f"TRADE_SIGNAL_{asof}.md"
    now_jst = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    expire = (datetime.now(JST) + timedelta(minutes=15)).strftime("%H:%M JST")

    lines = [
        f"# Trade Signal — {asof}",
        "",
        f"_generated {now_jst}_  |  **valid until ~{expire}**  |  source: SBI-actual only",
        "",
        f"## Account",
        f"- NAV (real): ¥{payload['nav']:,.0f}",
        f"- Cash: ¥{payload['cash']:,.0f}",
        f"- Held MV: ¥{payload['held_market_value']:,.0f}",
        "",
    ]
    if payload["holdings"]:
        lines += ["## Current SBI positions", "",
                  "| symbol | qty | avg_cost | price | MV | uPnL |",
                  "|---|---|---|---|---|---|"]
        for h in payload["holdings"]:
            lines.append(f"| {h['symbol']} | {h['qty']:.0f} | ¥{h['avg_cost']:,.2f} | "
                         f"¥{(h['price'] or 0):,.2f} | ¥{h['market_value']:,.0f} | ¥{h['unrealized_pnl']:+,.0f} |")
        lines.append("")
    else:
        lines += ["## Current SBI positions", "", "_flat — all cash_", ""]

    lines += ["## Signals", ""]
    if not payload["trades"]:
        lines += ["_no trades — already at target_", ""]
    else:
        lines += ["| # | side | symbol | qty | limit (¥) | notional | weight |",
                  "|---|---|---|---|---|---|---|"]
        for i, t in enumerate(payload["trades"], 1):
            lines.append(
                f"| {i} | **{t['side']}** | {t['symbol']} | {t['qty']} | "
                f"{t['limit_price_suggest']:.1f} | ¥{t['notional']:,.0f} | {t['target_weight']:.2f} |"
            )
        lines.append("")

    lines += [
        "## How to act",
        "",
        "1. Place these as **limit orders** at SBI with the suggested price (或 ±1%)",
        "2. Must complete **before 15:00 JST close**",
        "3. After SBI fills, import back via `import_fills.py --source sbi_actual`",
        "",
        "_This tool never mutates your account; it only suggests. Paper track runs separately post-close._",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _post_discord(webhook_url: str, payload: dict, asof: str, signal_path: Path) -> tuple[bool, str]:
    """Concise Discord embed — title + top-3 trades + link hint."""
    try:
        import urllib.request
    except Exception as e:  # pragma: no cover
        return False, str(e)

    trades = payload["trades"]
    desc_lines = []
    if not trades:
        desc_lines.append("**NO TRADES** — already at target, or signal quiet.")
    else:
        for t in trades[:6]:
            desc_lines.append(
                f"**{t['side']}** `{t['symbol']}` ×{t['qty']} @¥{t['limit_price_suggest']:.1f}  "
                f"(¥{t['notional']:,.0f})"
            )
        if len(trades) > 6:
            desc_lines.append(f"... and {len(trades)-6} more (see file)")

    now_jst = datetime.now(JST).strftime("%H:%M JST")
    embed = {
        "title": f"🔔 Trade Signal — {asof} ({now_jst})",
        "description": "\n".join(desc_lines),
        "color": 0x3498DB if trades else 0x95A5A6,
        "fields": [
            {"name": "NAV", "value": f"¥{payload['nav']:,.0f}", "inline": True},
            {"name": "Cash", "value": f"¥{payload['cash']:,.0f}", "inline": True},
            {"name": "Execute before", "value": "15:00 JST close", "inline": True},
        ],
        "footer": {"text": f"file: {signal_path.name}"},
    }
    body = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300, f"status {resp.status}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD; default: today JST")
    ap.add_argument("--lot", type=int, default=100)
    ap.add_argument("--min_notional", type=float, default=5000.0)
    ap.add_argument("--no_refresh", action="store_true",
                    help="skip intraday quote refresh (faster, uses cached)")
    ap.add_argument("--no_webhook", action="store_true",
                    help="do not post to Discord")
    ap.add_argument("--symbols", default=None,
                    help="override symbols to refresh (comma-sep); default: targets + holdings")
    ap.add_argument("--watchlist_only", action="store_true",
                    default=True,
                    help="(default 2026-04-15) emit watchlist/risk-alert, NOT buy orders. "
                         "The walk-forward evidence supports MONTHLY rebalance only; "
                         "daily BUY emissions are unsupported by 23-month OOS data. "
                         "Use --emit_orders to override.")
    ap.add_argument("--emit_orders", dest="watchlist_only", action="store_false",
                    help="emit BUY/SELL signals for daily execution (legacy). "
                         "Requires explicit opt-in; default is watchlist-only.")
    args = ap.parse_args()

    asof = args.asof or datetime.now(JST).strftime("%Y-%m-%d")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_weights = _load_target_weights(reports_dir)
    if not target_weights:
        print(f"[intraday] WARNING: no target_weights.csv or all zero at {reports_dir}")

    # Refresh intraday for target symbols + current holdings
    if not args.no_refresh:
        conn0 = sqlite3.connect(args.db)
        try:
            held_syms = set(_sbi_positions(conn0).keys())
        finally:
            conn0.close()
        syms = args.symbols or ",".join(sorted(set(target_weights) | held_syms))
        if syms:
            print(f"[intraday] refreshing intraday quotes: {syms}")
            cmd = [
                sys.executable, "intraday_update.py",
                "--db", args.db, "--symbols", syms,
                "--period", "1d", "--interval", "1m",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=Path(__file__).parent,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(f"[intraday] WARN: refresh failed rc={r.returncode}: {r.stderr[-300:]}")
            else:
                print("[intraday] refresh OK")

    conn = sqlite3.connect(args.db)
    try:
        payload = _compute_trades(conn, target_weights, lot_size=args.lot,
                                   min_notional=args.min_notional)
    finally:
        conn.close()

    # 2026-04-15 downgrade: in watchlist_only mode, the signal message
    # is informational — we replace `trades` with watchlist alerts and
    # mark the payload so Discord / markdown render as "observation only".
    if args.watchlist_only:
        raw_trades = payload.get("trades", [])
        watchlist = []
        for t in raw_trades:
            watchlist.append({
                "side": t.get("side", "?"),
                "symbol": t.get("symbol"),
                "qty_at_target": t.get("qty"),
                "last_price": t.get("limit_price_suggest"),
                "notional": t.get("notional"),
                "target_weight": t.get("target_weight"),
                "note": "WATCHLIST ONLY — walk-forward evidence does not support daily BUY. "
                        "Monthly rebalance only.",
            })
        payload["trades"] = []
        payload["watchlist"] = watchlist
        payload["mode"] = "watchlist_only"
        print(f"[intraday] watchlist_only mode: {len(watchlist)} names observed, 0 orders emitted.")

    signal_path = _write_signal_file(reports_dir, payload, asof)
    print(f"[intraday] signal written → {signal_path}")
    print(f"  nav=JPY {payload['nav']:,.0f}  cash=JPY {payload['cash']:,.0f}  "
          f"trades={len(payload['trades'])}")
    for t in payload["trades"]:
        print(f"  {t['side']} {t['symbol']} x{t['qty']} @JPY {t['limit_price_suggest']:.1f}")

    if not args.no_webhook:
        url = os.environ.get("WORKER_QUANT_ALERT_WEBHOOK_URL", "")
        if not url:
            # fall back to .env file
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("WORKER_QUANT_ALERT_WEBHOOK_URL="):
                        url = line.split("=", 1)[1].strip()
                        break
        if url:
            ok, msg = _post_discord(url, payload, asof, signal_path)
            print(f"[intraday] discord: {'OK' if ok else 'FAIL'} ({msg})")
        else:
            print("[intraday] WARN: no webhook URL set; skipping discord")

    return 0


if __name__ == "__main__":
    sys.exit(main())
