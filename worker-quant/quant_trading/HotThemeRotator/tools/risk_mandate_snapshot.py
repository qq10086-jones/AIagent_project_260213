"""Daily risk-mandate snapshot + trace (P25-05, Section 17 / ADR-0012).

Assembles the sleeve panel (Section 17) from the journal-derived portfolio
state and the owner-declared mandate config, then writes:

- reports/observability/risk_mandate/{asof}.json   full panel snapshot
- reports/observability/risk_mandate_trace.jsonl   one summary row per run
  (nav, exposure ratio, band status, kill-switch buffer, flag counts)

Read-only / advice-only (Rule 3): it computes and records, never acts. Wired
into daily_routine afterclose as a NON-FATAL diagnostic (a diagnostic must
never block collection). Fail-open: without the mandate config or portfolio
state it reports "unavailable" and exits 0 -- never fabricated state
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

from hot_theme_rotator.common.console import enable_console_fallback  # noqa: E402
from hot_theme_rotator.portfolio.nav_publication import annotate_nav_record  # noqa: E402
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
from hot_theme_rotator.decision_queue import (  # noqa: E402
    DecisionQueueError,
    TERMINAL_STATES,
    is_position_bound,
    load_queue,
    open_item,
    queue_report,
    transition,
)
from hot_theme_rotator.portfolio.reconciliation import reconcile_positions  # noqa: E402
from hot_theme_rotator.risk.sleeve_engine import (  # noqa: E402
    build_risk_mandate_panel,
    load_mandate,
)


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


def _supersede_orphaned_advice(
    queue_path: Path,
    queue_items: dict,
    subjects: list[str],
    *,
    asof: str,
    closed: list[dict],
) -> list[str]:
    """Terminalize open position-bound advice whose position is gone (P37-00).

    Append-only supersession, never deletion: the erroneous items stay in the
    queue as the audit evidence that the system generated binding advice about
    a reported-closed position. Only Rule 17.4.x items are eligible — a Rule
    17.2 band breach survives a disposition, and in fact usually worsens.

    NON-FATAL by contract, like everything else this diagnostic does.
    """
    if not subjects:
        return []
    symbols = ", ".join(sorted({str(r.get("symbol")) for r in closed})) or "n/a"
    superseded: list[str] = []
    for aid, item in sorted(queue_items.items()):
        if item.state in TERMINAL_STATES or item.subject not in subjects:
            continue
        if not is_position_bound(item.source_rule):
            continue
        # The terminal date follows this producer's convention — the session
        # being reported on — but may never precede the item's own creation:
        # regenerating an OLD snapshot would otherwise stamp a supersession
        # before the item existed and hand the age counter a negative span.
        terminal_asof = max(asof, item.created_asof or asof)
        try:
            transition(
                queue_path, aid, "superseded", asof=terminal_asof,
                note=(
                    f"P37-00: position {symbols} is CLOSED_PENDING_PRICE (owner-reported "
                    "execution recorded in the queue; §14 journal fill still outstanding). "
                    "Position-bound advice cannot stand without the position. Superseded "
                    "append-only — the original rows are retained as audit evidence that "
                    "this item was generated against a phantom holding."
                ),
            )
        except DecisionQueueError as exc:
            print(f"  WARNING supersede_failed:{aid}: {exc}")
            continue
        superseded.append(aid)
    return superseded


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



def _reconcile_for_snapshot(root, asof, panel):
    """Reconcile the panel against a broker snapshot file, if one exists.

    The snapshot lives at reports/observability/broker_snapshot/{asof}.json and
    is written by hand from the broker's own pages. There is no automated feed,
    which is precisely why the default must be `missing_broker_snapshot` rather
    than silence.
    """
    from hot_theme_rotator.portfolio.broker_reconciliation import (
        BrokerPosition, BrokerSnapshot, JournalView, reconcile_against_broker)
    from datetime import date as _date

    path = root / "reports" / "observability" / "broker_snapshot" / f"{asof}.json"
    if not path.is_file():
        return reconcile_against_broker(
            JournalView(asof=_date.fromisoformat(asof), cash=panel["cashJpy"],
                        quantities={}, mark_time=None), None)
    raw = json.loads(path.read_text(encoding="utf-8"))
    snap = BrokerSnapshot(
        asof=_date.fromisoformat(raw["asof"]), cash=float(raw["cash"]),
        positions={s: BrokerPosition(s, float(p["qty"]), float(p["mark"]), float(p["value"]))
                   for s, p in raw["positions"].items()},
        total_assets=float(raw["total_assets"]), source=raw["source"],
        mark_time=raw.get("mark_time"))
    holdings = {h["symbol"]: float(h.get("qty") or 0) for h in (panel.get("holdings") or [])}
    return reconcile_against_broker(
        JournalView(asof=_date.fromisoformat(asof), cash=panel["cashJpy"],
                    quantities=holdings or snap.quantities,
                    mark_time=raw.get("mark_time")), snap)


def main(argv=None) -> int:
    # Sleeve labels, theses and band notes are legitimately Japanese; a cp932
    # console cannot encode them and would kill the run MID-PRINT, leaving a
    # truncated report that looks finished. Losing a glyph beats losing the
    # report. (The tool's own text stays ASCII — that is the other half of the
    # rule, enforced by tests/unit/test_cli_console_encoding.py.)
    enable_console_fallback()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", default=None, help="ISO date stamp for the artifact (default: today).")
    ap.add_argument("--base-dir", default=str(ROOT))
    ap.add_argument("--no-write", action="store_true", help="Print only; do not write artifacts.")
    args = ap.parse_args(argv)
    asof = args.asof or _dt.date.today().isoformat()

    obs = Path(args.base_dir) / "reports" / "observability"
    queue_path = obs / "decision_queue.jsonl"

    # P37-00: reconcile EXECUTION truth (decision queue) against ACCOUNTING truth
    # (Section 14 journal) BEFORE the panel is assembled. A disposition the owner
    # reported but the journal has not priced must not reach the sleeve engine,
    # because everything downstream of it — exposure, discipline flags, and the
    # advice those flags open — would then be about a position nobody holds.
    mandate = load_mandate(args.base_dir)
    queue_items = load_queue(queue_path)
    recon = reconcile_positions(
        _positions_dict(),
        queue_items,
        sleeve_map=(mandate or {}).get("sleeve_map"),
    )
    for warning in recon.warnings:
        print(f"  WARNING {warning}")

    panel = build_risk_mandate_panel(recon.positions, base_dir=args.base_dir, mandate=mandate)
    if panel is None:
        # Fail-open: absent mandate/portfolio is a reportable state, not an error.
        print(f"risk mandate panel unavailable (no mandate config or portfolio) asof={asof}")
        return 0
    panel["reconciliation"] = recon.as_dict()

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

    # P37-00: name the provisional part of NAV out loud. An exposure ratio that
    # silently rests on unpriced proceeds is exactly the kind of number that
    # gets quoted later as if it were settled.
    for row in recon.closed_pending_price:
        if row.get("proceedsAtLastMarkJpy") is None:
            # Quantity left regardless; only the proceeds estimate is missing.
            value = "proceeds UNKNOWN (no usable mark — not estimated, not zeroed)"
        else:
            value = (f"¥{row['proceedsAtLastMarkJpy']:,.0f} of NAV is PROVISIONAL proceeds "
                     f"at the ¥{row['lastMarkJpy']:,.0f} mark")
        print(f"  [reconciled] {row['symbol']} {row['state']} qty {row['qtyClosed']:g} "
              f"reported {row['executionReportedAt']} (advice {row['adviceId']}) — "
              f"{value}; blocked on {', '.join(row['blockedOn'])}")
    for row in recon.unreconciled_executions:
        print(f"  ⚠ UNRECONCILED EXECUTION [{row['advice_id']}] Rule {row['source_rule']} "
              f"{row['subject']}: {row['reason']} — nothing was closed (fail-closed); "
              "add linkage via tools/decision_queue_cli.py annotate")

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

    # P29: binding mandate conditions become first-class, ageable advice.
    # Opening items WRITES, so it honours --no-write; reporting is read-only
    # and always runs, because an item aging quietly is exactly what the daily
    # surface must show.
    opened: list[str] = []
    superseded: list[str] = []
    if not args.no_write:
        # P37-00: retire orphaned position-bound advice BEFORE opening new items,
        # so a single run cannot report an item as both live and orphaned.
        superseded = _supersede_orphaned_advice(
            queue_path, queue_items, recon.supersede_subjects,
            asof=asof, closed=recon.closed_pending_price,
        )
        for aid in superseded:
            print(f"  [reconciled] advice {aid} superseded — its position is "
                  "CLOSED_PENDING_PRICE (original rows retained as audit evidence)")
        # Event, not state: an idempotent rerun retires nothing and correctly
        # reports []. Which advice STANDS orphaned is `supersedeSubjects`, and
        # the queue itself remains the SSoT for who was superseded and when.
        panel["reconciliation"]["supersededThisRun"] = superseded
        opened = _queue_sync(
            queue_path, obs / "risk_mandate_trace.jsonl",
            asof=asof,
            flags={k: v for k, v in flags.items() if v},
            ages=ages,
            band_status=exp.get("bandStatus"),
        )
    if opened or queue_path.exists():
        try:
            report = queue_report(queue_path, asof=_dt.date.fromisoformat(asof))
            oldest = report["oldest_open_sessions"]
            terminal = ", ".join(
                f"{k}={v}" for k, v in sorted(report["terminal_counts"].items())
            ) or "none"
            seen = report["median_trigger_to_seen_sessions"]
            print(f"  decision queue: {report['open_count']} open, oldest "
                  f"{oldest if oldest is not None else 'n/a'} sessions; "
                  f"terminal [{terminal}]; median trigger->seen "
                  f"{seen if seen is not None else 'unobserved'} "
                  f"(tools/decision_queue_cli.py list)")
            for row in report["open_items"][:3]:
                print(f"    [{row['advice_id']}] Rule {row['source_rule']} "
                      f"{row['subject']} age {row['age_sessions']} sessions")
        except Exception as exc:  # noqa: BLE001 - diagnostic must never block
            print(f"  WARNING queue_report_failed: {type(exc).__name__}: {exc}")

    if not args.no_write:
        (obs / "risk_mandate").mkdir(parents=True, exist_ok=True)
        # P37-05: every published NAV carries the reconciliation state it was
        # born in. Without a broker snapshot on disk that state is
        # `missing_broker_snapshot` and `official` is False - the honest
        # default, because a NAV nobody checked against the account is a
        # computation, not a measurement. An unlabelled NAV in this trace is
        # indistinguishable from a settled one, which is how JPY 394,724 once
        # sat here next to a JPY 393,998 account.
        record = annotate_nav_record({"asof": asof, **panel},
                                     _reconcile_for_snapshot(ROOT, asof, panel))
        (obs / "risk_mandate" / f"{asof}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
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
        if recon.applied or recon.unreconciled_executions:
            # A ratio computed on provisional NAV must carry that fact into the
            # trace too — the trace is what later analyses read, and a bare
            # 0.42x there would be indistinguishable from a settled one.
            row["reconciliation"] = {
                "closed_pending_price": [r["symbol"] for r in recon.closed_pending_price],
                "provisional_cash_jpy": recon.provisional_cash_jpy,
                "nav_is_provisional": recon.applied,
                "unreconciled_executions": [
                    r["advice_id"] for r in recon.unreconciled_executions
                ],
            }
        if age_warnings:
            row["age_warnings"] = list(age_warnings)
        trace_result = _append_trace_row(obs / "risk_mandate_trace.jsonl", row)
        print(f"wrote {obs / 'risk_mandate' / (asof + '.json')} + trace {trace_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
