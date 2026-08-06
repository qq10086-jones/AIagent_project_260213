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
    previous_trading_day as _previous_trading_day,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    DEFAULT_STRATEGY_ID,
    PositionAdapterError,
    default_db_path,
    load_portfolio_state,
)
from hot_theme_rotator.decision_queue import open_item, queue_report  # noqa: E402
from hot_theme_rotator.risk.sleeve_engine import build_risk_mandate_panel  # noqa: E402


# Rule 17.4 flags that constitute BINDING advice — a declared condition the
# owner said they would act on. Advisory flags (thesis_missing, cap_breached)
# stay off the queue on purpose: queueing every state turns the queue into the
# dashboard nobody opens, which is the failure P29 exists to fix.
_BINDING_FLAG_RULES = {
    "exit_triggered": ("17.4.6", "declared exit bracket breached on close"),
    "review_required": ("17.4.4", "review drawdown reached; re-underwrite required"),
}
_HEALTHY_BAND = "within_band"


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
    # Earliest session in the current unbroken run where the flag was actually
    # observed open. Stable while the condition persists, so the P29 decision
    # queue can key one advice item to one unresolved condition instead of
    # minting a fresh item (and a fresh age) every afterclose. ``None`` when
    # today is the first sighting.
    first_observed_asof: str | None = None


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
            count, gaps, first_observed, walk_warnings = _backward_run(
                effective,
                current_date,
                earliest,
                lambda row, _sid=sid, _flag=flag: _flag in
                ((row.get("flags") or {}).get(_sid) or []),
            )
            warnings.extend(walk_warnings)
            ages[(sid, flag)] = FlagAge(
                prior_observed_sessions=count,
                observation_gap_sessions=gaps,
                first_observed_asof=first_observed.isoformat() if first_observed else None,
            )
            if gaps:
                warnings.append(
                    f"flag_age_degraded:{sid}:{flag}:missing_sessions={gaps}"
                )
    return ages, tuple(dict.fromkeys(warnings))


def _backward_run(
    effective: dict[_dt.date, dict],
    current_date: _dt.date,
    earliest: _dt.date,
    predicate,
) -> tuple[int, int, _dt.date | None, list[str]]:
    """Walk sessions backwards while ``predicate(row)`` holds.

    Returns (observed sessions, unobserved gaps, earliest observed date,
    warnings). Shared by flag ages and band-status ages so both use one
    definition of "consecutive" — a second walk with its own gap semantics is
    how the two would silently diverge.
    """
    count = 0
    gaps = 0
    first_observed: _dt.date | None = None
    warnings: list[str] = []
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
        if not predicate(row):
            break
        count += 1
        first_observed = cursor
        cursor = _previous_trading_day(cursor)
    return count, gaps, first_observed, warnings


def _band_first_observed(trace_path: Path, current_asof: str, band_status: str) -> str:
    """Earliest session of the current unbroken out-of-band run.

    Keys the band-breach advice item so a breach standing since 2026-07-13 ages
    from 07-13, rather than resetting to zero every afterclose.
    """
    try:
        current_date = _dt.date.fromisoformat(current_asof)
    except ValueError:
        return current_asof
    effective, _ = _effective_trace_rows(trace_path)
    past = [d for d in effective
            if d < current_date and calendar_covers(d) and is_trading_day(d)]
    if not past:
        return current_asof
    _, _, first_observed, _ = _backward_run(
        effective, current_date, min(past),
        lambda row, _s=band_status: row.get("band_status") == _s,
    )
    return first_observed.isoformat() if first_observed else current_asof


def _queue_sync(
    queue_path: Path,
    trace_path: Path,
    *,
    asof: str,
    flags: dict,
    ages: dict,
    band_status: str | None,
) -> list[str]:
    """Open one P29 advice item per binding, unresolved mandate condition.

    Each item is keyed to the session the condition FIRST appeared, so a
    persistent breach is one item that ages rather than a new item every day.

    NON-FATAL by contract (Rule 15.5): the snapshot is a diagnostic wired into
    afterclose, and a diagnostic must never block data collection. Any failure
    here degrades to a printed warning and an empty result.
    """
    opened: list[str] = []
    try:
        for sleeve_id, sleeve_flags in (flags or {}).items():
            for flag in sleeve_flags:
                mapping = _BINDING_FLAG_RULES.get(flag)
                if mapping is None:
                    continue
                rule, summary = mapping
                age = ages.get((sleeve_id, flag))
                created = (age.first_observed_asof if age else None) or asof
                opened.append(open_item(
                    queue_path,
                    source_rule=rule,
                    subject=f"sleeve_{sleeve_id}",
                    summary=summary,
                    created_asof=created,
                    severity="binding",
                    evidence_ref=f"reports/observability/risk_mandate/{asof}.json",
                ))
        if band_status and band_status != _HEALTHY_BAND:
            opened.append(open_item(
                queue_path,
                source_rule="17.2",
                subject="portfolio",
                summary=(
                    f"β-adjusted exposure {band_status}; declared band requires "
                    "deploy, a Rule 4 band amendment, or a dated time-bounded "
                    "exception — silence is not a fourth option"
                ),
                created_asof=_band_first_observed(trace_path, asof, band_status),
                severity="binding",
                evidence_ref=f"reports/observability/risk_mandate/{asof}.json",
            ))
    except Exception as exc:  # noqa: BLE001 - diagnostic must never block
        print(f"  WARNING queue_sync_failed: {type(exc).__name__}: {exc}")
        return []
    return opened


def _semantic_trace_row(row: dict) -> dict:
    """The row's state, stripped of the bookkeeping fields revisions add."""
    return {
        key: value
        for key, value in row.items()
        if key not in {"asof_revision", "supersedes_revision"}
    }


def _append_trace_row(trace_path: Path, row: dict) -> str:
    """Append once per semantic state; preserve changed same-date revisions.

    A rerun with identical state writes nothing — that duplicate row is what
    inflated the Rule 17.4.7 age. A rerun with CHANGED state is real new
    information and is appended as an explicit revision, never an overwrite:
    §14 append-only history is preserved and ``_effective_trace_rows`` treats
    the last row for a date as authoritative. Historical duplicates are left
    in place, auditable, and no longer counted.
    """
    existing: list[dict] = []
    if trace_path.exists():
        try:
            existing = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError):
            existing = []

    same_date = [item for item in existing if item.get("asof") == row.get("asof")]
    prior_revision = max(
        (int(item.get("asof_revision", 1)) for item in same_date),
        default=0,
    )
    if same_date and _semantic_trace_row(same_date[-1]) == _semantic_trace_row(row):
        return "unchanged"

    output = {**row, "asof_revision": prior_revision + 1}
    if prior_revision:
        output["supersedes_revision"] = prior_revision
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(output, ensure_ascii=False) + "\n")
    return "revised" if prior_revision else "appended"


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
        # P29: binding mandate conditions become first-class, ageable advice.
        queue_path = obs / "decision_queue.jsonl"
        opened = _queue_sync(
            queue_path, obs / "risk_mandate_trace.jsonl",
            asof=asof,
            flags={k: v for k, v in flags.items() if v},
            ages=ages,
            band_status=exp.get("bandStatus"),
        )
        if opened:
            try:
                report = queue_report(queue_path, asof=_dt.date.fromisoformat(asof))
                oldest = report["oldest_open_sessions"]
                print(f"  decision queue: {report['open_count']} open, oldest "
                      f"{oldest if oldest is not None else 'n/a'} sessions "
                      f"(tools/decision_queue_cli.py list)")
            except Exception as exc:  # noqa: BLE001 - diagnostic must never block
                print(f"  WARNING queue_report_failed: {type(exc).__name__}: {exc}")

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
        trace_result = _append_trace_row(obs / "risk_mandate_trace.jsonl", row)
        print(f"wrote {obs / 'risk_mandate' / (asof + '.json')} + trace {trace_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
