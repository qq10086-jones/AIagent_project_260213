"""Daily risk-mandate snapshot + trace (P25-05, Section 17 / ADR-0012).

Assembles the sleeve panel (Section 17) from the journal-derived portfolio
state and the owner-declared mandate config, then writes:

- reports/observability/risk_mandate/{asof}.json   full panel snapshot
- reports/observability/risk_mandate_trace.jsonl   one summary row per run
  (nav, exposure ratio, band status, kill-switch buffer, flag counts)

Read-only / advice-only (Rule 3): it computes and records, never acts. Wired
into daily_routine afterclose as a NON-FATAL diagnostic (a diagnostic must
never block collection). Fail-open: without the mandate config or portfolio
state it reports "unavailable" and exits 0 — never fabricated state
(Rule 11.9.4).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    DEFAULT_STRATEGY_ID,
    PositionAdapterError,
    default_db_path,
    load_portfolio_state,
)
from hot_theme_rotator.risk.sleeve_engine import build_risk_mandate_panel  # noqa: E402


def _flag_ages(trace_path: Path, current_flags: dict) -> dict:
    """Rule 17.4.7 — how many CONSECUTIVE prior sessions each open (sleeve, flag)
    has already been present, read from the append-only trace. Returns a map
    {(sleeve, flag): prior_sessions}. Fail-open to {} on unreadable history."""
    if not current_flags:
        return {}
    try:
        rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        return {}
    ages: dict = {}
    for sid, fl in current_flags.items():
        for f in fl:
            n = 0
            for row in reversed(rows):
                if f in ((row.get("flags") or {}).get(sid) or []):
                    n += 1
                else:
                    break
            ages[(sid, f)] = n
    return ages


def _positions_dict() -> dict:
    """Serialize portfolio state in the same shape the dashboard uses."""
    try:
        state = load_portfolio_state(default_db_path(), strategy_id=DEFAULT_STRATEGY_ID)
    except PositionAdapterError as exc:
        return {"available": False, "error": str(exc), "holdings": []}
    return {
        "available": True,
        "nav": state.nav,
        "cash": state.cash,
        "holdings": [
            {
                "symbol": h.symbol,
                "qty": h.qty,
                "avg_cost": h.avg_cost,
                "market_price": h.market_price,
                "market_value": h.market_value,
                "unrealized_pnl": h.unrealized_pnl,
                "unrealized_return_pct": h.unrealized_return_pct,
            }
            for h in state.holdings
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", default=None, help="ISO date stamp for the artifact (default: today).")
    ap.add_argument("--base-dir", default=str(ROOT))
    ap.add_argument("--no-write", action="store_true", help="Print only; do not write artifacts.")
    args = ap.parse_args(argv)
    asof = args.asof or _dt.date.today().isoformat()

    panel = build_risk_mandate_panel(_positions_dict(), base_dir=args.base_dir)
    if panel is None:
        # Fail-open: absent mandate/portfolio is a reportable state, not an error.
        print(f"risk mandate panel unavailable (no mandate config or portfolio) asof={asof}")
        return 0

    exp = panel["exposure"]
    ks = panel["killSwitch"]
    flags = {s["id"]: s.get("flags", []) for s in panel["sleeves"]}
    n_flags = sum(len(v) for v in flags.values())
    print(f"=== RISK MANDATE SNAPSHOT asof={asof} (Section 17; read-only; advice-only) ===")
    print(f"  NAV ¥{panel['navJpy']:,.0f}  cash ¥{panel['cashJpy']:,.0f}")
    print(f"  exposure {exp['betaAdjustedJpy']:,.0f} = {exp['ratio']}x  [{exp['bandStatus']}]")
    print(f"  kill-switch buffer ¥{ks['bufferJpy']:,.0f} ({ks['bufferPct']}%)  breached={ks['breached']}")
    for sid, fl in flags.items():
        if fl:
            print(f"  [{sid}] flags: {', '.join(fl)}")

    # Sleeve C exit brackets (Rule 17.4.6) — surface armed/triggered state.
    for s in panel["sleeves"]:
        for h in s.get("holdings", []):
            br = h.get("exitBracket")
            if br:
                lo = f"{br['lowerJpy']:,.0f}" if br.get("lowerJpy") is not None else "—"
                up = f"{br['upperJpy']:,.0f}" if br.get("upperJpy") is not None else "—"
                print(f"  [{s['id']}] {h['symbol']} exit-bracket [{lo} / {up}] "
                      f"@close ¥{br['priceEvaluatedJpy']:,.0f} → {br['status']}: {br['note']}")

    # Sector look-through (Rule 17.7) — penetrated concentration, top themes.
    lt = panel.get("sectorLookThrough") or []
    if lt:
        top = ", ".join(f"{r['theme']} ¥{r['totalJpy']:,.0f} ({r['fracNav']*100:.1f}% NAV)" for r in lt[:4])
        print(f"  sector look-through: {top}")

    # Flag-sunset escalation (Rule 17.4.7) — read history BEFORE writing today's row.
    obs = Path(args.base_dir) / "reports" / "observability"
    sunset_n = int(panel.get("mandate", {}).get("flagSunsetSessions") or 7)
    ages = _flag_ages(obs / "risk_mandate_trace.jsonl", {k: v for k, v in flags.items() if v})
    escalations = []
    for (sid, f), prior in ages.items():
        open_sessions = prior + 1  # incl. today's still-open flag
        if open_sessions >= sunset_n:
            escalations.append({"sleeve": sid, "flag": f, "openSessions": open_sessions})
            print(f"  ⚠ SUNSET [{sid}] '{f}' open {open_sessions} sessions ≥ {sunset_n} "
                  f"— Rule 17.4.7 demands resolve (write thesis / re-underwrite / exit)")

    if not args.no_write:
        (obs / "risk_mandate").mkdir(parents=True, exist_ok=True)
        (obs / "risk_mandate" / f"{asof}.json").write_text(
            json.dumps({"asof": asof, **panel}, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {
            "asof": asof,
            "nav_jpy": panel["navJpy"],
            "exposure_ratio": exp["ratio"],
            "band_status": exp["bandStatus"],
            "ks_buffer_pct": ks["bufferPct"],
            "ks_breached": ks["breached"],
            "flags": {k: v for k, v in flags.items() if v},
            "n_flags": n_flags,
            "sunset": escalations,
        }
        with (obs / "risk_mandate_trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {obs / 'risk_mandate' / (asof + '.json')} + trace row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
