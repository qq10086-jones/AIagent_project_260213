"""Three-ledger scorecard - account / research / execution, never blended (P33).

Design Section 4.3 requires three SEPARATE cards. Blending them is outcome bias with a
number attached: a good month launders a broken process, a bad month buries a
correct one. So this tool publishes three cards side by side and computes no
overall figure of merit at all.

    Account outcome       NAV return, benchmark return, active return, drawdown
                          -> ``unavailable`` while the ledger is unreconciled
    Research validity     live date clusters, trial count, DSR/PBO, cost
                          hurdle, promotion verdict
                          -> ``insufficient``, never zero-as-a-verdict
    Execution reliability open-item age, trigger-to-seen, trigger-to-terminal,
                          band compliance, ledger lag
                          -> ``not_applicable`` when the denominator is zero

Every metric renders its numerator, denominator, unit, source and as-of date
even when it has no value, so an absent number is legible instead of missing.
A metric whose denominator is undefined renders the definition, not a number
(Rule 11.9: honest absence beats a plausible-looking value).

Two states are easy to confuse and are kept apart deliberately:

``insufficient``
    The measurement is defined and the collection is running; there is not yet
    enough of it. Zero matured 63D date clusters means "not measured", not
    "the signal did not survive".
``not_applicable``
    The denominator is zero. No open advice items is not 0% compliance.

Ledger lag and band compliance are measured ONLY on days with a valid reading
and a reconciled position. As of 2026-08-06 the account ledger is NOT
reconciled - a Rule 17.4.6 exit advice has been open since 2026-07-24 while the
Section 14 journal stops at 2026-07-14 - so the account card renders
``unavailable`` and the band-compliance denominator stops at the last day the
position was verifiable.

Some execution inputs come from the P29 decision queue, which may not exist
yet. Those metrics render their definition plus ``unavailable:
input_not_present`` and start reporting on their own once the file appears.
This tool never creates it.

Read-only / advice-only (Rule 3). No probability, win-rate or forward-return
language (Rule 8.3). Fail-open: missing inputs give an honest report, exit 0.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.common.console import (  # noqa: E402
    enable_console_fallback,
)

from hot_theme_rotator.research.cost_model import (  # noqa: E402
    COST_MODEL_REL,
    resolve_cost_model,
)
from hot_theme_rotator.data.jpx_calendar import (  # noqa: E402
    calendar_covers,
    is_trading_day,
    previous_trading_day,
    sessions_between,
)

METRIC_STATES = ("ok", "unavailable", "insufficient", "not_applicable")

# Rule 8.3 vocabulary guard. The scorecard reports realised, observed
# quantities; it must never speak in forward-looking odds.
FORBIDDEN_TERMS = (
    "win rate", "win-rate", "winrate", "probability", "probabilities",
    "expected return", "expected value", "odds", "胜率", "概率", "预期收益", "期望收益",
)

REL_JOURNAL = "reports/portfolio/journal"
REL_RISK_TRACE = "reports/observability/risk_mandate_trace.jsonl"
REL_QUEUE = "reports/observability/decision_queue.jsonl"
REL_VALUE_LIVELOG = "reports/observability/value_livelog"
REL_BENCHMARK = "reports/observability/benchmark_trace.jsonl"
REL_TRIAL_FAMILY = "reports/research/trial_family.json"
REL_COST_MODEL = COST_MODEL_REL   # shared contract, not a local path
REL_EVIDENCE_REVIEW_DIR = "reports/observability/evidence_review_63d"
REL_OUT_DIR = "reports/observability/three_ledger"
REL_OUT_TRACE = "reports/observability/three_ledger_trace.jsonl"

# A Rule 17.4.6 exit advice is the only trace flag that implies an owner ACTION
# was due; the others (thesis_missing, review_required, cap_breached) demand
# writing or deciding, not a fill, so they cannot desynchronise the ledger.
RECONCILIATION_FLAGS = ("exit_triggered",)
TERMINAL_STATES = ("executed", "declined", "expired", "superseded")

SEPARATION_NOTE = (
    "Three scorecards are published side by side and never combined: no blended "
    "grade is computed. Process quality is judged on what was knowable at "
    "decision time; the account result stays visible and is not excused by it."
)


# --- small helpers --------------------------------------------------------

def _trial_family_from_evidence_review(
    base: Path, asof: str, warnings: list,
) -> tuple[dict | None, str]:
    """Fall back to P31's frozen trial family (`tools/evidence_review_63d.py`).

    P31 counts the family before computing any statistic, which is the property
    that makes the count usable here. Reporting ``input_not_present`` for a
    number a sibling tool already publishes would understate the scorecard.

    Uses the newest artifact at or before ``asof`` — never a later one, which
    would leak a count computed after the reporting date. Takes the INCLUSIVE
    count because over-counting search breadth is the fail-closed direction.
    """
    directory = base / REL_EVIDENCE_REVIEW_DIR
    if not directory.is_dir():
        return None, REL_TRIAL_FAMILY
    candidates = sorted(p for p in directory.glob("*.json") if p.stem <= asof)
    if not candidates:
        return None, REL_TRIAL_FAMILY
    latest = candidates[-1]
    payload = _read_json(latest, warnings, "evidence_review_63d")
    family = (payload or {}).get("trial_family")
    if not isinstance(family, dict):
        return None, REL_TRIAL_FAMILY
    count = family.get("n_trials_inclusive")
    if not isinstance(count, int):
        return None, REL_TRIAL_FAMILY
    return (
        {"count": count, "frozen_asof": payload.get("asof")},
        f"{REL_EVIDENCE_REVIEW_DIR}/{latest.name}",
    )


def _metric(
    name: str,
    *,
    numerator: str,
    denominator: str,
    unit: str,
    source: str,
    asof: str,
    value=None,
    state: str = "unavailable",
    reason: str | None = None,
    **extra,
) -> dict:
    """One metric with its definition attached, value or not."""
    if state not in METRIC_STATES:
        raise ValueError(f"unknown metric state: {state}")
    if state != "ok":
        value = None
        reason = reason or state
    else:
        reason = None
    return {
        "metric": name,
        "definition": {"numerator": numerator, "denominator": denominator, "unit": unit},
        "value": value,
        "state": state,
        "reason": reason,
        "asof": asof,
        "source": source,
        **extra,
    }


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _read_jsonl(path: Path, warnings: list[str], label: str) -> list[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"{label}_unreadable:{type(exc).__name__}")
        return []
    rows: list[dict] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            warnings.append(f"malformed_{label}_line:{number}")
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            warnings.append(f"malformed_{label}_line:{number}")
    return rows


def _read_json(path: Path, warnings: list[str], label: str) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"{label}_unreadable:{type(exc).__name__}")
        return None
    return data if isinstance(data, dict) else None


def _iso(value) -> str | None:
    try:
        return _dt.date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError):
        return None


def _effective_rows(rows: list[dict]) -> list[tuple[str, dict]]:
    """Last row per ``asof``, in date order - same-date reruns count once."""
    latest: dict[str, dict] = {}
    for row in rows:
        iso = _iso(row.get("asof"))
        if iso:
            latest[iso] = row
    return sorted(latest.items())


def _sessions(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        return sessions_between(_dt.date.fromisoformat(start), _dt.date.fromisoformat(end))
    except (TypeError, ValueError):
        return None


# --- ledger reconciliation ------------------------------------------------

def _journal_dates(base: Path) -> list[str]:
    root = base / REL_JOURNAL
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("*.jsonl")):
        iso = _iso(p.stem)
        if iso:
            out.append(iso)
    return out


def _first_open_flag_date(trace: list[tuple[str, dict]]) -> str | None:
    """Earliest CONSECUTIVE observation of a still-open exit advice.

    Walks back from the latest row while the flag stays observed. A gap ends
    nothing; only the latest contiguous run is claimed, because that is all the
    trace actually witnessed.
    """
    if not trace:
        return None
    last_flags = trace[-1][1].get("flags") or {}
    open_flags = {f for flags in last_flags.values() if isinstance(flags, list) for f in flags}
    if not any(f in open_flags for f in RECONCILIATION_FLAGS):
        return None
    first = trace[-1][0]
    for iso, row in reversed(trace[:-1]):
        flags = row.get("flags") or {}
        present = {f for fl in flags.values() if isinstance(fl, list) for f in fl}
        if not any(f in present for f in RECONCILIATION_FLAGS):
            break
        first = iso
    return first


def _reconciliation(base: Path, asof: str, trace: list[tuple[str, dict]], queue: dict | None) -> dict:
    """Is the position ledger trustworthy enough to price the account?

    Positive proof of reconciliation does not exist in this repository — there
    is no broker feed to diff against. What CAN be established is the absence
    of a contradiction, and one specific contradiction is detectable: an exit
    advice that has been due for sessions with nothing recorded after it. In
    that state "already sold, not written down" and "not sold yet" are
    indistinguishable, and a NAV computed on either reading would be asserted,
    not measured.
    """
    definition = (
        "reconciled_no_contradicting_evidence = a journal exists and no executed "
        "or overdue exit advice lacks a matching journal entry. undetermined = an "
        "exit advice has been open since a date the journal does not reach, so an "
        "executed-but-unrecorded fill cannot be excluded. Absence of contradiction "
        "is not positive confirmation."
    )
    journal_dates = _journal_dates(base)
    if not journal_dates:
        return {
            "state": "unavailable",
            "reason": f"input_not_present:{REL_JOURNAL}",
            "definition": definition,
            "journal_last_event_date": None,
            "open_exit_advice_since": None,
            "reconciled_through": None,
            "unrecorded_executed_advice": [],
        }
    last_event = journal_dates[-1]

    if queue and queue.get("available"):
        unrecorded = []
        for item in queue["items"].values():
            if item.get("state") != "executed":
                continue
            executed_on = item.get("terminal_asof")
            if not executed_on:
                continue
            if not any(d >= executed_on for d in journal_dates):
                unrecorded.append({"advice_id": item.get("advice_id"), "executed_asof": executed_on})
        if unrecorded:
            # Days BEFORE the first unrecorded execution are still verifiable;
            # only from that session on is the position in question. Leaving the
            # boundary open would let unreconciled days into every rate below.
            earliest = min(u["executed_asof"] for u in unrecorded)
            try:
                through = previous_trading_day(_dt.date.fromisoformat(earliest)).isoformat()
            except (TypeError, ValueError):
                through = None
            return {
                "state": "unreconciled",
                "reason": "executed_advice_without_journal_entry",
                "definition": definition,
                "journal_last_event_date": last_event,
                "open_exit_advice_since": None,
                "unrecorded_executed_since": earliest,
                "reconciled_through": through,
                "unrecorded_executed_advice": unrecorded,
            }
        return {
            "state": "reconciled_no_contradicting_evidence",
            "reason": None,
            "definition": definition,
            "journal_last_event_date": last_event,
            "open_exit_advice_since": None,
            "reconciled_through": asof,
            "unrecorded_executed_advice": [],
        }

    open_since = _first_open_flag_date(trace)
    if open_since and open_since > last_event:
        through = None
        try:
            through = previous_trading_day(_dt.date.fromisoformat(open_since)).isoformat()
        except (TypeError, ValueError):
            through = None
        return {
            "state": "undetermined",
            "reason": "open_exit_advice_without_subsequent_journal_entry",
            "definition": definition,
            "journal_last_event_date": last_event,
            "open_exit_advice_since": open_since,
            "reconciled_through": through,
            "unrecorded_executed_advice": [],
        }
    return {
        "state": "reconciled_no_contradicting_evidence",
        "reason": None,
        "definition": definition,
        "journal_last_event_date": last_event,
        "open_exit_advice_since": open_since,
        "reconciled_through": asof,
        "unrecorded_executed_advice": [],
    }


# --- decision queue (P29 substrate; may not exist yet) --------------------

def _load_queue(base: Path, warnings: list[str]) -> dict:
    """Fold the append-only queue into per-advice current state.

    Parsed defensively rather than through the P29 module: this tool must keep
    working both before that module exists and if its API moves.
    """
    path = base / REL_QUEUE
    if not path.exists():
        return {"available": False, "reason": f"input_not_present:{REL_QUEUE}", "items": {}}
    rows = _read_jsonl(path, warnings, "queue")
    items: dict[str, dict] = {}
    for row in rows:
        advice_id = row.get("advice_id")
        if not advice_id:
            continue
        created = _iso(row.get("created_asof")) or _iso(row.get("asof"))
        item = items.setdefault(str(advice_id), {
            "advice_id": str(advice_id),
            "created_asof": created,
            "state": "open",
            "source_rule": row.get("source_rule"),
            "subject": row.get("subject"),
            "seen_asof": None,
            "terminal_asof": None,
        })
        if item["created_asof"] is None:
            item["created_asof"] = created
        state = row.get("state")
        stamp = _iso(row.get("asof")) or created
        if state == "acknowledged" and item["seen_asof"] is None:
            item["seen_asof"] = stamp
        if state in TERMINAL_STATES:
            item["terminal_asof"] = stamp
        if isinstance(state, str):
            item["state"] = state
    return {"available": True, "reason": None, "items": items}


# --- account outcome ------------------------------------------------------

def _return_pct(series: list[tuple[str, float]]) -> float | None:
    if len(series) < 2 or series[0][1] == 0:
        return None
    return round((series[-1][1] - series[0][1]) / series[0][1] * 100.0, 6)


def _max_drawdown_pct(series: list[tuple[str, float]]) -> float | None:
    if len(series) < 2:
        return None
    peak = series[0][1]
    worst = 0.0
    for _, v in series:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak * 100.0)
    return round(worst, 6)


def _account_card(base: Path, asof: str, trace: list[tuple[str, dict]], recon: dict,
                  benchmark_path: Path, window_start: str | None, warnings: list[str]) -> dict:
    nav_series = [(iso, float(row["nav_jpy"])) for iso, row in trace
                  if _is_number(row.get("nav_jpy"))]
    if window_start:
        nav_series = [p for p in nav_series if p[0] >= window_start]
    window = {"start": nav_series[0][0] if nav_series else None,
              "end": nav_series[-1][0] if nav_series else None}

    gated = recon["state"] not in ("reconciled_no_contradicting_evidence",)
    if recon["state"] == "unavailable":
        gate_reason = f"ledger_reconciliation_unavailable:{recon['reason']}"
    else:
        gate_reason = f"ledger_unreconciled:{recon['state']}"

    def gate(state: str, value=None, reason: str | None = None):
        if gated:
            return "unavailable", None, gate_reason
        return state, value, reason

    nav_return = _return_pct(nav_series)
    state, value, reason = gate(
        "ok" if nav_return is not None else "insufficient",
        nav_return,
        None if nav_return is not None else "fewer_than_two_valid_nav_observations",
    )
    metrics = {"nav_return_pct": _metric(
        "nav_return_pct",
        numerator="NAV(window end) - NAV(window start), JPY, journal-derived marks",
        denominator="NAV(window start), JPY",
        unit="percent",
        source=REL_RISK_TRACE, asof=asof, value=value, state=state, reason=reason,
        window=window, observations=len(nav_series))}

    bench_rows = _read_jsonl(benchmark_path, warnings, "benchmark")
    bench_series = [(iso, float(row["close"])) for iso, row in _effective_rows(bench_rows)
                    if _is_number(row.get("close"))]
    if window["start"]:
        bench_series = [p for p in bench_series if p[0] >= window["start"]]
    try:
        benchmark_rel = benchmark_path.relative_to(base).as_posix()
    except ValueError:
        benchmark_rel = benchmark_path.as_posix()
    if not benchmark_path.exists():
        b_state, b_value, b_reason = "unavailable", None, f"input_not_present:{benchmark_rel}"
    else:
        bench_return = _return_pct(bench_series)
        b_state, b_value, b_reason = (
            ("ok", bench_return, None) if bench_return is not None
            else ("insufficient", None, "fewer_than_two_valid_benchmark_observations"))
    if gated:
        b_state, b_value, b_reason = "unavailable", None, gate_reason
    metrics["benchmark_return_pct"] = _metric(
        "benchmark_return_pct",
        numerator="benchmark close(window end) - close(window start), JPY",
        denominator="benchmark close(window start), JPY",
        unit="percent",
        source=benchmark_rel, asof=asof, value=b_value, state=b_state, reason=b_reason,
        window=window, observations=len(bench_series))

    if gated:
        a_state, a_value, a_reason = "unavailable", None, gate_reason
    elif metrics["nav_return_pct"]["state"] == "ok" and b_state == "ok":
        a_state, a_value, a_reason = "ok", round(
            metrics["nav_return_pct"]["value"] - b_value, 6), None
    else:
        a_state, a_value, a_reason = "unavailable", None, (
            b_reason or metrics["nav_return_pct"]["reason"] or "component_metric_unavailable")
    metrics["active_return_pp"] = _metric(
        "active_return_pp",
        numerator="nav_return_pct - benchmark_return_pct over the same window",
        denominator="1 (difference of two percentages; the unit is percentage points)",
        unit="percentage_points",
        source=f"{REL_RISK_TRACE} + {REL_BENCHMARK}", asof=asof,
        value=a_value, state=a_state, reason=a_reason, window=window)

    dd = _max_drawdown_pct(nav_series)
    state, value, reason = gate(
        "ok" if dd is not None else "insufficient", dd,
        None if dd is not None else "fewer_than_two_valid_nav_observations")
    metrics["max_drawdown_pct"] = _metric(
        "max_drawdown_pct",
        numerator="max over the window of (running peak NAV - NAV), JPY",
        denominator="the running peak NAV at that point, JPY",
        unit="percent",
        source=REL_RISK_TRACE, asof=asof, value=value, state=state, reason=reason,
        window=window, observations=len(nav_series))

    card_state = "unavailable" if gated else (
        "ok" if any(m["state"] == "ok" for m in metrics.values()) else "insufficient")
    return {
        "scorecard": "account_outcome",
        "state": card_state,
        "reason": gate_reason if gated else None,
        "empty_state_rule": "unavailable while the ledger is unreconciled",
        "reconciliation": recon,
        "metrics": metrics,
    }


# --- research validity ----------------------------------------------------

def _latest_livelog(base: Path, asof: str) -> tuple[Path | None, dict | None, list[str]]:
    warnings: list[str] = []
    root = base / REL_VALUE_LIVELOG
    if not root.is_dir():
        return None, None, warnings
    candidates = sorted(p for p in root.glob("*.json") if _iso(p.stem) and p.stem <= asof)
    if not candidates:
        return None, None, warnings
    path = candidates[-1]
    return path, _read_json(path, warnings, "value_livelog"), warnings


def _research_card(base: Path, asof: str, warnings: list[str]) -> dict:
    path, livelog, sub_warnings = _latest_livelog(base, asof)
    warnings.extend(sub_warnings)
    source = path.relative_to(base).as_posix() if path else REL_VALUE_LIVELOG
    result = (livelog or {}).get("result") or {}
    metrics: dict[str, dict] = {}
    unmet: list[str] = []

    for signal in ("earnings_yield", "value_bp"):
        for horizon in ("21", "63"):
            name = f"live_date_clusters_{signal}_{horizon}d"
            block = ((result.get(signal) or {}).get(horizon) or {}) if result else {}
            n_dates = block.get("n_dates")
            observed = {
                "n_dates": n_dates,
                "matured": block.get("matured"),
                "unmatured": block.get("unmatured"),
            }
            if not livelog:
                state, value, reason = "unavailable", None, f"input_not_present:{REL_VALUE_LIVELOG}"
                unmet.append(f"{name}:input_not_present")
            elif isinstance(n_dates, int) and n_dates > 0:
                state, value, reason = "ok", n_dates, None
            else:
                state, value, reason = "insufficient", None, "zero_matured_date_clusters"
                unmet.append(f"{name}:zero_matured_date_clusters")
            metrics[name] = _metric(
                name,
                numerator=(f"count of distinct live cross-sectional dates whose {horizon}D "
                           f"forward window has matured for {signal}"),
                denominator=("1 (a count, not a ratio; the independent-cluster count IS "
                             "the quantity)"),
                unit="date_clusters",
                source=source, asof=asof, value=value, state=state, reason=reason,
                observed=observed)

    trials = _read_json(base / REL_TRIAL_FAMILY, warnings, "trial_family")
    trial_source = REL_TRIAL_FAMILY
    if trials is None:
        # P31 already freezes and counts the trial family. Prefer its artifact
        # over reporting input_not_present for a number the repo does compute —
        # a scorecard that ignores an available input under-reports itself.
        trials, trial_source = _trial_family_from_evidence_review(base, asof, warnings)
    if trials is None:
        t_state, t_value, t_reason = "unavailable", None, f"input_not_present:{REL_TRIAL_FAMILY}"
        unmet.append(f"trial_family_count:input_not_present:{REL_TRIAL_FAMILY}")
    else:
        listed = trials.get("trials")
        count = trials.get("count")
        if isinstance(listed, list):
            t_state, t_value, t_reason = "ok", len(listed), None
        elif isinstance(count, int):
            t_state, t_value, t_reason = "ok", count, None
        else:
            t_state, t_value, t_reason = "insufficient", None, "trial_family_not_enumerated"
            unmet.append("trial_family_count:not_enumerated")
    metrics["trial_family_count"] = _metric(
        "trial_family_count",
        numerator="count of variants in the FROZEN trial family (all attempted, not just kept)",
        denominator="1 (a count; freezing it before computing is what makes DSR interpretable)",
        unit="trials",
        source=trial_source, asof=asof, value=t_value, state=t_state, reason=t_reason,
        frozen_asof=(trials or {}).get("frozen_asof"))

    # Same contract the P31 evidence review consumes, so the two reports can
    # never disagree about whether the Rule 16.0 hurdle is computable. Nothing
    # is defaulted: an assumed cost that clears the hurdle is the failure mode
    # Rule 16.0 exists to prevent.
    cost = resolve_cost_model(base, horizon=63)
    warnings.extend(cost.warnings)
    if cost.round_trip_cost is None:
        c_state, c_value, c_reason = (
            "unavailable", None, f"input_not_present:{REL_COST_MODEL}")
        unmet.append(f"cost_hurdle_bp:input_not_present:{REL_COST_MODEL}")
    else:
        c_state, c_value, c_reason = "ok", round(cost.round_trip_cost * 10_000, 4), None
    metrics["cost_hurdle_bp"] = _metric(
        "cost_hurdle_bp",
        numerator="declared round-trip execution cost of the traded basket, basis points",
        denominator="1 (a hurdle level the realised gross spread must clear, not a ratio)",
        unit="basis_points",
        source=REL_COST_MODEL, asof=asof, value=c_value, state=c_state, reason=c_reason,
        cost_model=cost.as_dict())

    hurdle = cost.hurdle()
    metrics["cost_hurdle_ic"] = _metric(
        "cost_hurdle_ic",
        numerator="tau * c_rt (per-rebalance turnover x round-trip cost)",
        denominator="sigma_r at the 63D label horizon (cross-sectional dispersion)",
        unit="information_coefficient",
        source=REL_COST_MODEL, asof=asof,
        value=round(hurdle, 6) if hurdle is not None else None,
        state="ok" if hurdle is not None else "unavailable",
        reason=None if hurdle is not None else f"missing:{','.join(cost.missing)}",
        cost_model=cost.as_dict())
    if hurdle is None:
        unmet.append(f"cost_hurdle_ic:missing:{','.join(cost.missing)}")

    for name, label, numerator in (
        ("dsr", "Deflated Sharpe Ratio",
         "realised Sharpe of the live series, deflated by the frozen trial count and "
         "the observed higher moments"),
        ("pbo", "Bailey-Lopez de Prado backtest-overfitting metric",
         "share of CPCV splits whose in-sample best variant ranks below median "
         "out-of-sample"),
    ):
        if trials is None:
            state, reason = "unavailable", f"input_not_present:{REL_TRIAL_FAMILY}"
        else:
            state, reason = "insufficient", "no_matured_63d_date_clusters"
        unmet.append(f"{name}:{reason}")
        metrics[name] = _metric(
            name, numerator=numerator,
            denominator=("the frozen trial family and the matured live sample; both are "
                         "required before this quantity is defined"),
            unit="ratio", source=f"{REL_TRIAL_FAMILY} + {source}", asof=asof,
            state=state, reason=reason, label=label)

    if unmet:
        v_state, v_value, v_reason = "insufficient", None, "rule_16_6_gate_inputs_incomplete"
    else:
        v_state, v_value, v_reason = "ok", "confirm", None
    metrics["promotion_verdict"] = _metric(
        "promotion_verdict",
        numerator="Rule 16.6 promotion requirements met",
        denominator="Rule 16.6 promotion requirements declared",
        unit="verdict",
        source=source, asof=asof, value=v_value, state=v_state, reason=v_reason,
        allowed_verdicts=["confirm", "fail", "insufficient"],
        unmet_requirements=unmet,
        note=("2026-08-26 is the EARLIEST review date, not a guaranteed verdict date; "
              "an unmet gate is not a negative verdict."))

    card_state = "ok" if any(m["state"] == "ok" for m in metrics.values()) else "insufficient"
    if livelog is None and trials is None and cost is None:
        card_state = "insufficient"
    return {
        "scorecard": "research_validity",
        "state": card_state,
        "reason": None if card_state == "ok" else "collection_in_progress",
        "empty_state_rule": "insufficient, never zero-as-a-verdict",
        "metrics": metrics,
    }


# --- execution reliability ------------------------------------------------

def _flag_run_sessions(trace: list[tuple[str, dict]]) -> tuple[int | None, str | None, str | None]:
    """Elapsed sessions of the still-open exit advice, as OBSERVED.

    Measured to the last session the trace actually witnessed, not to today: a
    routine that has not run since is a gap in observation, and stretching the
    age across it would assert something never seen.
    """
    first = _first_open_flag_date(trace)
    if not first or not trace:
        return None, None, None
    last_seen = trace[-1][0]
    return _sessions(first, last_seen), first, last_seen


def _execution_card(base: Path, asof: str, trace: list[tuple[str, dict]], recon: dict,
                    queue: dict, warnings: list[str]) -> dict:
    metrics: dict[str, dict] = {}
    queue_missing_reason = queue.get("reason") or f"input_not_present:{REL_QUEUE}"
    items = list(queue.get("items", {}).values())
    open_items = [i for i in items if i.get("state") not in TERMINAL_STATES]

    if not queue.get("available"):
        state, value, reason = "unavailable", None, queue_missing_reason
    else:
        state, value, reason = "ok", len(open_items), None
    metrics["open_item_count"] = _metric(
        "open_item_count",
        numerator="advice items whose latest recorded state is not terminal",
        denominator="1 (a count of live items, not a ratio)",
        unit="items", source=REL_QUEUE, asof=asof, value=value, state=state, reason=reason)

    ages = [a for a in (_sessions(i.get("created_asof"), asof) for i in open_items) if a is not None]
    if queue.get("available"):
        age_source = REL_QUEUE
        if ages:
            state, value, reason = "ok", max(ages), None
        else:
            state, value, reason = "not_applicable", None, "no_open_items"
        first_open = last_seen = None
    else:
        run, first_open, last_seen = _flag_run_sessions(trace)
        age_source = REL_RISK_TRACE
        if run is None:
            state, value, reason = "not_applicable", None, "no_open_advice_observed"
        else:
            state, value, reason = "ok", run, None
    metrics["open_item_age_sessions_max"] = _metric(
        "open_item_age_sessions_max",
        numerator=("JPX sessions elapsed after the item was raised, up to asof (queue) or "
                   "up to the last session the trace observed it open (trace fallback)"),
        denominator="1 (a session count; the creation session itself is age 0)",
        unit="jpx_sessions", source=age_source, asof=asof, value=value, state=state,
        reason=reason, observed_open_since=first_open, observed_through=last_seen)

    for name, key, label in (
        ("trigger_to_seen_sessions_median", "seen",
         "sessions from the advice being raised to the owner acknowledging it"),
        ("trigger_to_terminal_sessions_median", "terminal",
         "sessions from the advice being raised to a terminal state"),
    ):
        if not queue.get("available"):
            state, value, reason = "unavailable", None, queue_missing_reason
            n = 0
        else:
            stamps = [_sessions(i.get("created_asof"), i.get(f"{key}_asof"))
                      for i in items if i.get(f"{key}_asof")]
            observed = [s for s in stamps if s is not None]
            n = len(observed)
            if observed:
                state, value, reason = "ok", round(statistics.median(observed), 2), None
            else:
                state, value, reason = "not_applicable", None, f"no_items_reached_{key}"
        metrics[name] = _metric(
            name, numerator=f"median of: {label}",
            denominator=(f"advice items that actually reached {key}; an item never "
                         f"{key} contributes no value and is not counted as zero"),
            unit="jpx_sessions", source=REL_QUEUE, asof=asof, value=value, state=state,
            reason=reason, denominator_value=n)

    journal_dates = _journal_dates(base)
    if not queue.get("available"):
        state, value, reason, n = "unavailable", None, queue_missing_reason, 0
    else:
        lags = []
        for item in items:
            executed_on = item.get("terminal_asof")
            if item.get("state") != "executed" or not executed_on:
                continue
            later = [d for d in journal_dates if d >= executed_on]
            if not later:
                continue
            lag = _sessions(executed_on, min(later))
            if lag is not None:
                lags.append(lag)
        n = len(lags)
        if lags:
            state, value, reason = "ok", max(lags), None
        else:
            state, value, reason = "not_applicable", None, "no_executed_advice_with_a_journal_entry"
    metrics["ledger_lag_sessions_max"] = _metric(
        "ledger_lag_sessions_max",
        numerator=("JPX sessions from an advice reaching executed to the first journal "
                   "entry on or after that session"),
        denominator=("executed advice items that have a matching journal entry; an "
                     "executed item with NO entry is a reconciliation contradiction, "
                     "reported on the account card rather than counted as lag 0"),
        unit="jpx_sessions", source=f"{REL_QUEUE} + {REL_JOURNAL}", asof=asof,
        value=value, state=state, reason=reason, denominator_value=n)

    through = recon.get("reconciled_through")
    eligible = []
    # No known-reconciled boundary means no day qualifies. Treating an unknown
    # boundary as "everything counts" is exactly the measurement that would
    # score days whose positions cannot be trusted.
    for iso, row in (trace if through else []):
        if iso > through:
            continue
        try:
            day = _dt.date.fromisoformat(iso)
        except ValueError:
            continue
        if not calendar_covers(day) or not is_trading_day(day):
            continue
        if not _is_number(row.get("nav_jpy")) or not _is_number(row.get("exposure_ratio")):
            continue
        if not isinstance(row.get("band_status"), str) or not row["band_status"]:
            continue
        eligible.append(row)
    in_band = sum(1 for row in eligible if row["band_status"] == "in_band")
    if not eligible:
        state, value, reason = "not_applicable", None, (
            "position_reconciliation_boundary_unknown" if not through
            else "no_days_with_a_valid_reading_and_a_reconciled_position")
    else:
        state, value, reason = "ok", round(in_band / len(eligible) * 100.0, 4), None
    metrics["band_compliance_rate_pct"] = _metric(
        "band_compliance_rate_pct",
        numerator="eligible sessions whose recorded band status is in_band",
        denominator=("covered JPX sessions with a finite NAV, a finite exposure ratio, a "
                     "recorded band status, and a position reconciled through that day"),
        unit="percent", source=REL_RISK_TRACE, asof=asof, value=value, state=state,
        reason=reason, numerator_value=in_band, denominator_value=len(eligible),
        eligible_through=through)

    staleness = _sessions(recon.get("journal_last_event_date"), asof)
    if staleness is None:
        state, value, reason = "unavailable", None, (
            recon.get("reason") or f"input_not_present:{REL_JOURNAL}")
    else:
        state, value, reason = "ok", staleness, None
    metrics["journal_staleness_sessions"] = _metric(
        "journal_staleness_sessions",
        numerator="JPX sessions from the newest journal entry to asof",
        denominator="1 (an observation of ledger freshness, not a lag measure)",
        unit="jpx_sessions", source=REL_JOURNAL, asof=asof, value=value, state=state,
        reason=reason,
        note=("A quiet ledger is normal when nothing was due; this becomes evidence only "
              "alongside an advice that was due."))

    card_state = "ok" if any(m["state"] == "ok" for m in metrics.values()) else "unavailable"
    return {
        "scorecard": "execution_reliability",
        "state": card_state,
        "reason": None if card_state == "ok" else queue_missing_reason,
        "empty_state_rule": "not_applicable when the denominator is zero",
        "metrics": metrics,
    }


# --- assembly -------------------------------------------------------------

def build_scorecards(base_dir: str | Path, *, asof: str,
                     benchmark_series: str | Path | None = None,
                     window_start: str | None = None) -> dict:
    base = Path(base_dir)
    warnings: list[str] = []
    trace = _effective_rows(_read_jsonl(base / REL_RISK_TRACE, warnings, "risk_trace"))
    queue = _load_queue(base, warnings)
    recon = _reconciliation(base, asof, trace, queue)
    benchmark_path = Path(benchmark_series) if benchmark_series else base / REL_BENCHMARK

    report = {
        "asof": asof,
        "tool": "three_ledger_scorecard",
        "base_dir": str(base),
        "separation_note": SEPARATION_NOTE,
        "metric_states": list(METRIC_STATES),
        "inputs": {
            "journal": REL_JOURNAL,
            "risk_trace": REL_RISK_TRACE,
            "decision_queue": REL_QUEUE,
            "value_livelog": REL_VALUE_LIVELOG,
            "benchmark_series": benchmark_path.as_posix(),
            "trial_family": REL_TRIAL_FAMILY,
            "cost_model": REL_COST_MODEL,
        },
        "account_outcome": _account_card(base, asof, trace, recon, benchmark_path,
                                         window_start, warnings),
        "research_validity": _research_card(base, asof, warnings),
        "execution_reliability": _execution_card(base, asof, trace, recon, queue, warnings),
        "warnings": warnings,
    }
    return report


def _render_metric(m: dict) -> list[str]:
    head = f"    {m['metric']}: "
    if m["state"] == "ok":
        head += f"{m['value']} {m['definition']['unit']}"
    else:
        head += f"[{m['state']}] {m['reason']}"
    lines = [head]
    if "numerator_value" in m and "denominator_value" in m:
        lines.append(f"        n/d = {m['numerator_value']}/{m['denominator_value']}")
    lines.append(f"        num: {m['definition']['numerator']}")
    lines.append(f"        den: {m['definition']['denominator']}")
    lines.append(f"        src: {m['source']}  asof={m['asof']}")
    return lines


def render_text(report: dict) -> str:
    out = [f"=== THREE-LEDGER SCORECARD asof={report['asof']} "
           f"(read-only; advice-only; Rule 3) ===",
           f"  {report['separation_note']}"]
    recon = report["account_outcome"]["reconciliation"]
    out.append(f"  ledger reconciliation: {recon['state']} "
               f"(journal last event {recon['journal_last_event_date']}, "
               f"reconciled through {recon['reconciled_through']})")
    if recon.get("unrecorded_executed_advice"):
        for item in recon["unrecorded_executed_advice"]:
            out.append(f"      executed advice {item['advice_id']} on "
                       f"{item['executed_asof']} has no journal entry on or after it")
    elif recon.get("open_exit_advice_since"):
        out.append(f"      exit advice open since {recon['open_exit_advice_since']}, "
                   f"journal does not reach it")
    for card_key in ("account_outcome", "research_validity", "execution_reliability"):
        card = report[card_key]
        out.append(f"  --- {card_key} [{card['state']}] "
                   f"empty-state rule: {card['empty_state_rule']} ---")
        for m in card["metrics"].values():
            out.extend(_render_metric(m))
    for w in report["warnings"]:
        out.append(f"  WARNING {w}")
    return "\n".join(out)


def _trace_row(report: dict) -> dict:
    def card(key):
        return {
            "state": report[key]["state"],
            "metrics": {name: {"value": m["value"], "state": m["state"]}
                        for name, m in report[key]["metrics"].items()},
        }
    return {
        "asof": report["asof"],
        "reconciliation": report["account_outcome"]["reconciliation"]["state"],
        "account_outcome": card("account_outcome"),
        "research_validity": card("research_validity"),
        "execution_reliability": card("execution_reliability"),
    }


def _append_trace(trace_path: Path, row: dict) -> str:
    existing: list[dict] = []
    if trace_path.exists():
        try:
            existing = [json.loads(line) for line in
                        trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError):
            existing = []
    same_date = [r for r in existing if r.get("asof") == row.get("asof")]
    prior = max((int(r.get("asof_revision", 1)) for r in same_date), default=0)
    if same_date and {k: v for k, v in same_date[-1].items()
                      if k not in {"asof_revision", "supersedes_revision"}} == row:
        return "unchanged"
    payload = {**row, "asof_revision": prior + 1}
    if prior:
        payload["supersedes_revision"] = prior
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return "revised" if prior else "appended"


def main(argv=None) -> int:
    
    # Data-sourced text (rule titles, theses) may be Japanese; degrade rather
    # than die mid-print on a cp932 console.
    enable_console_fallback()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", default=None, help="ISO date stamp (default: today).")
    ap.add_argument("--base-dir", default=str(ROOT))
    ap.add_argument("--no-write", action="store_true", help="Print only; write nothing.")
    ap.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    ap.add_argument("--benchmark-series", default=None,
                    help=f"JSONL of {{asof, close}} rows (default: {REL_BENCHMARK}).")
    ap.add_argument("--window-start", default=None,
                    help="ISO date; restrict the account window (default: first NAV row).")
    args = ap.parse_args(argv)
    asof = args.asof or _dt.date.today().isoformat()
    base = Path(args.base_dir)

    try:
        report = build_scorecards(base, asof=asof, benchmark_series=args.benchmark_series,
                                  window_start=args.window_start)
    except Exception as exc:  # fail-open: a diagnostic must never block the day
        print(f"three-ledger scorecard unavailable asof={asof}: {type(exc).__name__}: {exc}")
        return 0

    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report))

    if not args.no_write:
        out_dir = base / REL_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{asof}.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        result = _append_trace(base / REL_OUT_TRACE, _trace_row(report))
        print(f"wrote {out_path} + trace {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
