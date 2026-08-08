"""P34-08 — six-arm trend-overlay SHADOW report for Sleeve A.

    python tools/tsmom_shadow_report.py --asof 2026-08-08

Runs six arms on a real JP equity index series and reports them side by side.

**This changes nothing.** Sleeve A's exposure band is owner-declared (Rule 4) and
this report proposes no allocation, no switch, and no mandate amendment. It exists
so that a future discussion about timing Sleeve A starts from measured numbers
rather than from the intuition that trend-following "should" help.

One thing it must not be used for: the retrospective of 2026-08-04 found 17/17
observed sessions BELOW the authorized band, and classified that as an execution
failure rather than a design choice. A timing rule discovered afterwards must not
become a story that makes the under-deployment look intentional. The report says
so in its own output.

Instrument note: 1568.T (the held 2x ETF) has too little history for a trend
study, so the comparison runs on the unlevered index and SIMULATES leverage.
Simulated leverage understates real cost — the fund's tracking error is not
modelled.

Rule 3 / Rule 4: shadow only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.trend_overlay import (  # noqa: E402
    TrendOverlayError,
    compare_arms,
    detect_price_jumps,
    longest_clean_segment,
)

DB_REL = "data/raw/htr_market.db"


def _series(conn, symbol: str) -> list[tuple[str, float]]:
    return [(d, float(c)) for d, c in conn.execute(
        "select date, close from daily_prices where symbol=? and close>0 order by date",
        (symbol,))]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    ap.add_argument("--symbol", default="1306.T", help="unlevered index proxy")
    ap.add_argument("--held-symbol", default="1568.T", help="the actual Sleeve A instrument")
    ap.add_argument("--trend-lookback", type=int, default=245)
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    conn = sqlite3.connect(str(base / DB_REL))
    idx = _series(conn, args.symbol)
    held = _series(conn, args.held_symbol)
    conn.close()

    if not idx:
        print(f"no price series for {args.symbol}", file=sys.stderr)
        return 1

    all_prices = [c for _, c in idx]
    # The price store keeps RAW closes, so splits appear as one-day collapses
    # (1306.T falls 90.1% on 2026-03-30 on a 10:1 split). Restrict to the longest
    # contiguous split-free window rather than compounding through the artifact.
    jumps = detect_price_jumps(all_prices)
    a, b = longest_clean_segment(all_prices)
    prices = all_prices[a:b]
    clean_span = [idx[a][0], idx[b - 1][0]]
    try:
        out = compare_arms(prices, trend_lookback=args.trend_lookback,
                           sma_window=int(245 * 10 / 12), vol_window=60,
                           switch_cost_bp=20.0)
    except TrendOverlayError as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return 1
    out["price_integrity"] = {
        "raw_span": [idx[0][0], idx[-1][0]],
        "raw_n": len(all_prices),
        "unadjusted_jumps_detected": [
            {"date": idx[i][0], "move": r} for i, r in jumps],
        "clean_segment_used": clean_span,
        "clean_n": len(prices),
        "note": (
            "daily_prices stores RAW closes (auto_adjust=False), so corporate "
            "actions appear as returns. The comparison runs ONLY on the longest "
            "split-free segment; the discarded history is not a modelling choice "
            "but a data defect."
        ),
    }

    out.update({
        "asof": args.asof,
        "generated_by": "tools/tsmom_shadow_report.py",
        "index_symbol": args.symbol,
        "index_span": [idx[0][0], idx[-1][0]],
        "held_symbol": args.held_symbol,
        "held_span": [held[0][0], held[-1][0]] if held else None,
        "held_n_bars": len(held),
        "why_simulated": (
            f"{args.held_symbol} has {len(held)} bars"
            + (f" ({held[0][0]}..{held[-1][0]})" if held else "")
            + f", far too few for a {args.trend_lookback}-period trend study, so "
              f"the comparison runs on {args.symbol} with SIMULATED leverage"),
        "governance": {
            "task": "P34-08",
            "rules": ["Rule 3 advice-only", "Rule 4 owner-declared mandate"],
            "mandate_change": "none",
            "under_deployment_note": (
                "The 2026-08-04 retrospective recorded 17/17 observed sessions "
                "below the authorized band and classified this as an EXECUTION "
                "FAILURE, not a design. No arm in this report may be cited as a "
                "retrospective justification for that under-deployment."
            ),
        },
    })

    order = sorted(out["arms"].values(), key=lambda a: a["name"])
    print(f"index      : {args.symbol}  raw {out['index_span'][0]} .. {out['index_span'][1]}")
    print(f"integrity  : {len(jumps)} unadjusted jump(s); using clean segment "
          f"{clean_span[0]} .. {clean_span[1]} ({out['n_periods']} periods)")
    print(f"held       : {out['why_simulated']}")
    print(f"sample     : {out['independent_lookback_windows']:.1f} independent "
          f"{args.trend_lookback}-period windows -> {out['sample_adequacy']}")
    print()
    print(f"{'arm':<28}{'total':>9}{'ann':>9}{'vol':>8}{'sharpe~':>9}"
          f"{'maxDD':>9}{'inMkt':>7}{'sw':>5}")
    for a in order:
        print(f"{a['name']:<28}{a['total_return']:>8.1%}{a['annualized_return']:>9.1%}"
              f"{a['annualized_vol']:>8.1%}{a['sharpe_like']:>9.2f}"
              f"{a['max_drawdown']:>9.1%}{a['time_in_market']:>7.0%}"
              f"{a['n_switches']:>5}")
    print()
    for c in out["caveats"]:
        print(f"  - {c}")

    outp = base / "reports" / "research" / f"tsmom_shadow_{args.asof}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
