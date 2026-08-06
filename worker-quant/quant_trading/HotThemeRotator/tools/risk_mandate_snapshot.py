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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.data.jpx_calendar import (  # noqa: E402
    calendar_covers,
    is_trading_day,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    DEFAULT_STRATEGY_ID,
    PositionAdapterError,
    default_db_path,
    load_portfolio_state,
)
from hot_theme_rotator.risk.sleeve_engine import build_risk_mandate_panel  # noqa: E402


@dataclass(frozen=True)
class FlagAge:
    """Rule 17.4.7 age for one open (sleeve, flag).

    ``prior_observed_sessions`` counts only sessions where the trace ACTUALLY
    recorded the flag open. ``observation_gap_sessions`` counts eligible JPX
    sessions with no row at all — unobserved, not closed, so they neither
    increment nor reset the age (Rule 11.9.4: absence of data is reported, never
    imputed in either direction).
    """

    prior_observed_sessions: int
    observation_gap_sessions: int


def _effective_trace_rows(trace_path: Path) -> tuple[dict[_dt.date, dict], tuple[str, ...]]:
    """Return the last parseable row per ``asof`` plus non-fatal diagnostics.

    Collapsing by date is what makes the age a SESSION count rather than a row
    count: same-date reruns and duplicate historical lines contribute once.
    Bad lines are skipped WITH a warning — a single malformed row must never
    silently disable the whole escalation.
    """
    if not trace_path.exists():
        return {}, ()
    try:
        raw_lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, (f"flag_age_trace_unreadable:{type(exc).__name__}",)

    effective: dict[_dt.date, dict] = {}
    warnings: list[str] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            warnings.append(f"malformed_trace_line:{line_number}")
            continue
        try:
            asof = _dt.date.fromisoformat(str(row["asof"]))
        except (KeyError, TypeError, ValueError):
            warnings.append(f"invalid_trace_asof:{line_number}")
            continue
        effective[asof] = row
    return effective, tuple(warnings)


def _previous_trading_day(d: _dt.date) -> _dt.date:
    """Previous JPX session before ``d`` (bounded backstep, mirrors the calendar)."""
    prior = d - _dt.timedelta(days=1)
    for _ in range(10):
        if is_trading_day(prior):
            return prior
        prior -= _dt.timedelta(days=1)
    return prior


def _flag_ages(
    trace_path: Path,
    current_flags: dict,
    *,
    current_asof: str,
) -> tuple[dict[tuple[str, str], FlagAge], tuple[str, ...]]:
    """Rule 17.4.7 — prior CONSECUTIVE covered JPX sessions each open flag was
    observed open, walking the calendar backwards from ``current_asof``.

    Continuity ends only at an explicit observation of the flag ABSENT. Missing
    rows degrade confidence (reported via warnings + ``observation_gap_sessions``)
    but never reset the age: a skipped afterclose run is not evidence that the
    owner resolved anything. Fail-open to ``{}`` with a warning when the current
    date is outside the verified holiday table, so an uncovered operating year
    reduces the escalation to silence LOUDLY rather than silently.
    """
    if not current_flags:
        return {}, ()
    try:
        current_date = _dt.date.fromisoformat(current_asof)
    except ValueError:
        return {}, (f"flag_age_invalid_current_asof:{current_asof}",)
    if not calendar_covers(current_date) or not is_trading_day(current_date):
        return {}, (f"flag_age_calendar_uncovered:{current_asof}",)

    effective, load_warnings = _effective_trace_rows(trace_path)
    # Bound the walk by the oldest ELIGIBLE past row: rows outside the covered
    # calendar or in the future are irrelevant to this flag's history and must
    # not poison it.
    past_dates = [
        d for d in effective
        if d < current_date and calendar_covers(d) and is_trading_day(d)
    ]
    earliest = min(past_dates) if past_dates else current_date

    ages: dict[tuple[str, str], FlagAge] = {}
    warnings = list(load_warnings)
    for sid, flags in current_flags.items():
        for flag in flags:
            count = 0
            gaps = 0
            cursor = _previous_trading_day(current_date)
            while cursor >= earliest:
                if not calendar_covers(cursor):
                    warnings.append(f"flag_age_calendar_uncovered:{cursor.isoformat()}")
                    break
                row = effective.get(cursor)
                if row is None:
                    gaps += 1
                    cursor = _previous_trading_day(cursor)
                    continue
                row_flags = ((row.get("flags") or {}).get(sid) or [])
                if flag not in row_flags:
                    break
                count += 1
                cursor = _previous_trading_day(cursor)
            ages[(sid, flag)] = FlagAge(
                prior_observed_sessions=count,
                observation_gap_sessions=gaps,
            )
            if gaps:
                warnings.append(
                    f"flag_age_degraded:{sid}:{flag}:missing_sessions={gaps}"
                )
    return ages, tuple(dict.fromkeys(warnings))


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
    ages, age_warnings = _flag_ages(
        obs / "risk_mandate_trace.jsonl",
        {k: v for k, v in flags.items() if v},
        current_asof=asof,
    )
    for warning in age_warnings:
        print(f"  WARNING {warning}")
    escalations = []
    for (sid, f), age in ages.items():
        # Inclusive Rule 17.4.7 count: prior OBSERVED-open sessions + today.
        # Unobserved sessions are reported alongside, never folded in.
        open_sessions = age.prior_observed_sessions + 1
        if open_sessions >= sunset_n:
            escalations.append({
                "sleeve": sid,
                "flag": f,
                "openSessions": open_sessions,
                "observationGapSessions": age.observation_gap_sessions,
                "ageQuality": "degraded" if age.observation_gap_sessions else "complete",
            })
            print(f"  ⚠ SUNSET [{sid}] '{f}' open {open_sessions} observed sessions ≥ {sunset_n} "
                  f"(gaps={age.observation_gap_sessions}) "
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
        if age_warnings:
            row["age_warnings"] = list(age_warnings)
        with (obs / "risk_mandate_trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {obs / 'risk_mandate' / (asof + '.json')} + trace row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
