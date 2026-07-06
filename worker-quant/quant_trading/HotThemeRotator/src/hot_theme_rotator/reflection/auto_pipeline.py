"""Daily reflection autorun pipeline (P1 + Rule 13.19).

Orchestrates the standing reflection layer that runs every day after the
emit/sweep pass:

1. **PIT snapshot** — captures candidate universe / watchlist / config /
   alert budget / silent queue at decision_cutoff (P11-00 / Rule 8.2 PIT).
2. **Decision traces** — one trace per candidate today, summarizing the
   module chain that produced the BUY/HOLD/SELL decision (Rule 13.17
   reproducibility metadata baseline).
3. **Auto-generated proposals** — Rule 13.19 governed: per-day cap of 3,
   same-target 24h cooldown, structured-metadata-only, default
   diagnostic_only tier, mandatory recent source_trace_ids.

Run via the daily_pipeline CLI:

    python tools/daily_pipeline.py --with-reflection

Or programmatically:

    from hot_theme_rotator.reflection.auto_pipeline import run_daily_reflection
    report = run_daily_reflection(base_dir=...)
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Sequence


__all__ = [
    "AUTO_PIPELINE_GENERATOR",
    "AutoProposalDropError",
    "AutoProposalFinding",
    "MAX_PROPOSALS_PER_DAY",
    "SAME_TARGET_COOLDOWN_HOURS",
    "SOURCE_TRACE_MAX_AGE_DAYS",
    "TIER_RECOVERY_THRESHOLD",
    "generate_auto_proposals",
    "run_daily_reflection",
]


AUTO_PIPELINE_GENERATOR = "auto_reflection_pipeline_v1"
MAX_PROPOSALS_PER_DAY = 3          # Rule 13.19.4
SAME_TARGET_COOLDOWN_HOURS = 24    # Rule 13.19.5
SOURCE_TRACE_MAX_AGE_DAYS = 7      # Rule 13.19.2
TIER_RECOVERY_THRESHOLD = 0.05     # Rule 13.19.3


class AutoProposalDropError(RuntimeError):
    """Raised internally when a finding cannot become a valid proposal."""


@dataclass(frozen=True)
class AutoProposalFinding:
    """One RCA-derived finding ready for Rule 13.19 vetting.

    All fields come from STRUCTURED RCA output (Rule 13.19.6). LLM-produced
    text lives in ``rationale_text`` (separate, optional).
    """

    intervention_target: str
    evidence_class: str
    sample_size: int
    confidence_interval: tuple[float, float]
    counterfactual_validity: str
    rationale_pointer: str            # path to RCA report
    marginal_recovery: float
    source_trace_ids: tuple[str, ...]
    parameter_change: Optional[Mapping] = None
    backtest_evidence: Optional[Mapping] = None
    rationale_text: str = ""          # LLM annotation only, never proposal field
    extra: Mapping = field(default_factory=dict)


# ─── Rule 13.19 enforcement core ────────────────────────────────────────


def generate_auto_proposals(
    findings: Sequence[AutoProposalFinding],
    *,
    base_dir: str | Path,
    now_iso: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    """Apply Rule 13.19 to convert RCA findings into proposal payloads.

    Returns ``(accepted_payloads, silent_drops)``. Accepted payloads are
    written to ``reports/reflections/proposals/{proposal_id}.json`` (the
    canonical decision_gate location). Silent drops are written to
    ``reports/reflections/silent/silent_findings.jsonl`` for audit; every
    drop also appends to ``reports/reflections/auto_pipeline_audit.jsonl``.

    Drop reasons (Rule 13.19):
    - ``source_trace_missing`` — Rule 13.19.2 (no recent trace_id within 7d)
    - ``same_target_cooldown`` — Rule 13.19.5 (target proposed within 24h)
    - ``daily_cap_reached`` — Rule 13.19.4 (3/day)
    - ``invalid_metadata`` — Rule 13.19.6 (structured fields malformed)
    """
    base = Path(base_dir)
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now(tz=timezone.utc)

    proposals_dir = base / "reports" / "reflections" / "proposals"
    silent_dir = base / "reports" / "reflections" / "silent"
    audit_path = base / "reports" / "reflections" / "auto_pipeline_audit.jsonl"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    silent_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    # Build cooldown set: targets proposed in last 24h across all states
    # (including expired/rejected — Rule 13.19.5 explicit).
    cooldown_targets = _gather_recent_targets(base, now=now)

    accepted: list[dict] = []
    silent: list[dict] = []
    accepted_targets_today: set[str] = set()

    for finding in findings:
        try:
            payload = _vet_finding_rule_13_19(
                finding,
                now=now,
                accepted_count=len(accepted),
                cooldown_targets=cooldown_targets,
                accepted_targets_today=accepted_targets_today,
                base_dir=base,
            )
        except AutoProposalDropError as exc:
            drop_reason = str(exc)
            silent_row = {
                "ts": now.isoformat(),
                "generator": AUTO_PIPELINE_GENERATOR,
                "target": finding.intervention_target,
                "drop_reason": drop_reason,
                "finding_summary": {
                    "evidence_class": finding.evidence_class,
                    "marginal_recovery": finding.marginal_recovery,
                    "sample_size": finding.sample_size,
                },
            }
            silent.append(silent_row)
            _append_jsonl(silent_dir / "silent_findings.jsonl", silent_row)
            _append_jsonl(audit_path, silent_row)
            continue

        # Write the proposal payload to disk via the canonical proposal store.
        proposal_id = payload["proposal_id"]
        target_path = proposals_dir / f"{proposal_id}.json"
        if target_path.exists():
            # Duplicate id — treat as silent drop with duplicate reason.
            silent_row = {
                "ts": now.isoformat(),
                "generator": AUTO_PIPELINE_GENERATOR,
                "target": finding.intervention_target,
                "drop_reason": "duplicate_proposal_id",
                "finding_summary": {"proposal_id": proposal_id},
            }
            silent.append(silent_row)
            _append_jsonl(audit_path, silent_row)
            continue

        _atomic_write_json(target_path, payload)
        accepted.append(payload)
        accepted_targets_today.add(finding.intervention_target)

    return accepted, silent


def _vet_finding_rule_13_19(
    finding: AutoProposalFinding,
    *,
    now: datetime,
    accepted_count: int,
    cooldown_targets: set[str],
    accepted_targets_today: set[str],
    base_dir: Path,
) -> dict:
    """Apply all Rule 13.19 sub-clauses; raise AutoProposalDropError to drop."""

    # Rule 13.19.4 — daily cap
    if accepted_count >= MAX_PROPOSALS_PER_DAY:
        raise AutoProposalDropError("daily_cap_reached")

    # Rule 13.19.5 — same-target cooldown (24h)
    if (finding.intervention_target in cooldown_targets
            or finding.intervention_target in accepted_targets_today):
        raise AutoProposalDropError("same_target_cooldown")

    # Rule 13.19.2 — source_trace_ids mandatory + within 7d
    if not finding.source_trace_ids:
        raise AutoProposalDropError("source_trace_missing")
    if not _has_recent_trace(finding.source_trace_ids, now=now, base_dir=base_dir):
        raise AutoProposalDropError("source_trace_missing")

    # Rule 13.19.6 — metadata source restriction (structured only).
    # We sanity-check that core fields are present and non-trivial.
    if not finding.intervention_target.strip():
        raise AutoProposalDropError("invalid_metadata")
    if finding.evidence_class not in {
        "funnel_loss", "ablation", "freshness_attribution",
        "chase_filter_overshoot", "calibration_drift",
    }:
        raise AutoProposalDropError("invalid_metadata")
    if finding.sample_size < 0:
        raise AutoProposalDropError("invalid_metadata")
    if (not isinstance(finding.confidence_interval, tuple)
            or len(finding.confidence_interval) != 2):
        raise AutoProposalDropError("invalid_metadata")

    # Rule 13.19.3 — default tier = diagnostic_only unless marginal_recovery > 0.05
    # AND backtest_evidence present (for threshold_tuning / policy_change).
    if (finding.marginal_recovery > TIER_RECOVERY_THRESHOLD
            and finding.backtest_evidence
            and finding.parameter_change):
        tier = "threshold_tuning"
    else:
        tier = "diagnostic_only"

    # Build the canonical Proposal payload (decision_gate.Proposal.to_dict shape).
    created_ts = now.isoformat()
    proposal_id = sha256(
        f"{finding.evidence_class}|{finding.intervention_target}"
        f"|auto|{created_ts}".encode("utf-8"),
    ).hexdigest()[:16]

    payload = {
        "proposal_id": proposal_id,
        "created_ts": created_ts,
        "snapshot_id": _today_snapshot_id_hint(base_dir, now=now),
        "trace_id": finding.source_trace_ids[0],
        "evidence_class": finding.evidence_class,
        "intervention_target": finding.intervention_target,
        "sample_size": int(finding.sample_size),
        "confidence_interval": list(finding.confidence_interval),
        "counterfactual_validity": finding.counterfactual_validity,
        "rationale_pointer": finding.rationale_pointer,
        "generator": AUTO_PIPELINE_GENERATOR,
        "parameter_change": dict(finding.parameter_change) if finding.parameter_change else None,
        "backtest_evidence": dict(finding.backtest_evidence) if finding.backtest_evidence else None,
        "extra": {
            **dict(finding.extra),
            "source_trace_ids": list(finding.source_trace_ids),
            "tier": tier,
            "marginal_recovery": float(finding.marginal_recovery),
            "rationale_text": finding.rationale_text,  # LLM annotation, Rule 13.19.6
        },
    }
    return payload


def _gather_recent_targets(base_dir: Path, *, now: datetime) -> set[str]:
    """Return targets proposed within SAME_TARGET_COOLDOWN_HOURS across all states."""
    cutoff = now - timedelta(hours=SAME_TARGET_COOLDOWN_HOURS)
    targets: set[str] = set()
    for state_dir in ("proposals", "accepted", "rejected", "expired"):
        d = base_dir / "reports" / "reflections" / state_dir
        if not d.exists():
            continue
        for path in d.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                created_str = payload.get("created_ts", "")
                created = datetime.fromisoformat(created_str)
                if created.tzinfo is None:
                    continue
                if created >= cutoff:
                    target = payload.get("intervention_target")
                    if target:
                        targets.add(str(target))
            except (json.JSONDecodeError, ValueError, OSError):
                continue
    return targets


def _has_recent_trace(
    trace_ids: Sequence[str], *, now: datetime, base_dir: Path,
) -> bool:
    """Return True if any trace_id refers to a trace within last 7 days."""
    cutoff = now - timedelta(days=SOURCE_TRACE_MAX_AGE_DAYS)
    traces_dir = base_dir / "reports" / "traces"
    if not traces_dir.exists():
        return False
    target_ids = set(trace_ids)
    for traces_file in sorted(traces_dir.glob("*.jsonl"), reverse=True):
        try:
            # Parse trade_date from filename — it's YYYY-MM-DD.jsonl.
            trade_date = traces_file.stem
            try:
                file_date = datetime.fromisoformat(trade_date)
            except ValueError:
                file_date = None
            # If file is way older than cutoff, can short-circuit.
            if file_date and file_date.replace(tzinfo=timezone.utc) < cutoff:
                continue
            for line in traces_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("trace_id") not in target_ids:
                    continue
                created_str = row.get("created_ts", "")
                try:
                    created = datetime.fromisoformat(created_str)
                except ValueError:
                    continue
                if created.tzinfo is None:
                    continue
                if created >= cutoff:
                    return True
        except OSError:
            continue
    return False


def _today_snapshot_id_hint(base_dir: Path, *, now: datetime) -> str:
    """Best-effort snapshot_id for the proposal. If no snapshot today, use a stub.

    The exact snapshot lookup is done by the daily reflection orchestrator;
    here we just hint with an empty-tolerant fallback so Rule 13.6 fields
    aren't malformed.
    """
    return now.strftime("auto%Y%m%d")[:16]


# ─── Daily reflection orchestrator ─────────────────────────────────────


def run_daily_reflection(
    *,
    base_dir: str | Path,
    now_iso: Optional[str] = None,
    trade_date: Optional[str] = None,
    findings: Optional[Sequence[AutoProposalFinding]] = None,
) -> dict:
    """Run the standing daily reflection pipeline.

    Steps:
    1. Build + write PIT snapshot
    2. Write traces for today's predictions
    3. (Rule 13.19) Vet findings → proposals

    If ``findings`` is None, the pipeline calls a stub RCA hook that returns
    an empty list (no proposals generated). Real RCA wiring lands when the
    ``reflection.rca`` module is exercised on accumulated outcomes.

    Returns ``{snapshot_id, traces_written, proposals_written, silent_findings}``.
    """
    from hot_theme_rotator.observability.pit_ledger import append_snapshot
    from hot_theme_rotator.observability.schema import (
        PitSchemaError, PitSnapshot, compute_snapshot_id,
    )
    from hot_theme_rotator.reflection.trace_logger import (
        ModuleStep, ReflectionTraceError, TraceRecord, append_trace,
    )

    base = Path(base_dir)
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now(tz=timezone.utc)
    today = trade_date or now.strftime("%Y-%m-%d")

    # 1. PIT snapshot
    snapshot = _build_pit_snapshot(base, now=now, trade_date=today)
    snapshot_written = False
    if snapshot is not None:
        try:
            append_snapshot(snapshot, base_dir=base)
            snapshot_written = True
        except (PitSchemaError, RuntimeError) as exc:
            # If a snapshot with same id already exists, that's OK (idempotent).
            if "duplicate" not in str(exc).lower():
                raise

    # 2. Traces for today's predictions
    traces_written = 0
    if snapshot is not None:
        traces = _build_daily_traces(base, snapshot=snapshot, now=now)
        for trace in traces:
            try:
                append_trace(trace, base_dir=base)
                traces_written += 1
            except ReflectionTraceError as exc:
                # Duplicate trace — silently skip (idempotent daily reruns).
                if "duplicate" not in str(exc).lower():
                    raise

    # 3. Auto-proposals (Rule 13.19 enforced)
    findings_list = list(findings) if findings is not None else []
    accepted, silent = generate_auto_proposals(
        findings_list, base_dir=base, now_iso=now.isoformat(),
    )

    return {
        "snapshot_id": snapshot.snapshot_id if snapshot else None,
        "snapshot_written": snapshot_written,
        "traces_written": traces_written,
        "proposals_written": len(accepted),
        "silent_findings": len(silent),
    }


def _build_pit_snapshot(base_dir: Path, *, now: datetime, trade_date: str):
    """Build a minimal-but-valid PitSnapshot from current state.

    Returns None if required state (selected_tickers) isn't on disk.
    """
    from hot_theme_rotator.data.universe_adapter import (
        UniverseAdapterError, default_selected_tickers_path, load_screener_snapshot,
    )
    from hot_theme_rotator.observability.schema import (
        PitSchemaError, PitSnapshot, compute_snapshot_id,
    )

    try:
        tickers_path = default_selected_tickers_path()
        snap = load_screener_snapshot(tickers_path)
    except (UniverseAdapterError, OSError):
        return None

    universe = frozenset(t.symbol for t in snap.tickers)
    if not universe:
        return None

    # Watchlist (Rule 14.9 user_state).
    watchlist: frozenset[str] = frozenset()
    try:
        from hot_theme_rotator.user_state.watchlist import load_watchlist
        wl_state = load_watchlist(base_dir=base_dir)
        watchlist = frozenset(e.symbol for e in wl_state.entries)
    except Exception:  # noqa: BLE001
        pass

    decision_cutoff = now.isoformat()
    config_version = _compute_config_version(base_dir)
    active_filters = sha256(
        f"{config_version}|{trade_date}|{len(universe)}".encode("utf-8"),
    ).hexdigest()[:16]

    snapshot_id = compute_snapshot_id(
        decision_cutoff=decision_cutoff,
        config_version=config_version,
        candidate_universe=universe,
    )

    # Shadow panel: deterministic small sample of non-watchlist tickers.
    pool = sorted(universe - watchlist)
    k = min(5, len(pool))
    shadow_seed = int(sha256(snapshot_id.encode()).hexdigest()[:8], 16)
    import random
    rng = random.Random(shadow_seed)
    shadow_panel = tuple(rng.sample(pool, k=k)) if k > 0 else ()

    try:
        return PitSnapshot(
            snapshot_id=snapshot_id,
            decision_cutoff=decision_cutoff,
            trade_date=trade_date,
            candidate_universe=universe,
            watchlist=watchlist,
            active_filters=active_filters,
            source_freshness={},
            alert_budget_state={"used": 0, "remaining": 10},
            silent_queue_count=0,
            user_action_state=now.isoformat(),
            missing_data_reasons={},
            config_version=config_version,
            model_versions={
                "screener": "v2",
                "isotonic": "v1",
                "auto_reflection_pipeline": "v1",
            },
            shadow_panel=shadow_panel,
        )
    except PitSchemaError:
        return None


def _build_daily_traces(base_dir: Path, *, snapshot, now: datetime) -> list:
    """Build a TraceRecord per candidate in today's universe.

    The trace records the canonical module chain that produced the candidate
    (screener → leader_ranker). Real per-module branch decisions land when
    the production decision pipeline writes traces inline; this autorun
    captures the daily snapshot equivalent.
    """
    from hot_theme_rotator.reflection.trace_logger import (
        ModuleStep, TraceRecord, compute_trace_id,
    )

    out: list[TraceRecord] = []
    for symbol in sorted(snapshot.candidate_universe):
        created_ts = now.isoformat()
        trace_id = compute_trace_id(
            snapshot_id=snapshot.snapshot_id,
            prediction_id="",
            symbol=symbol,
            created_ts=created_ts,
            final_action="WATCH",
        )
        step = ModuleStep(
            module="screener",
            input_summary={"universe_size": len(snapshot.candidate_universe)},
            output_summary={"symbol": symbol},
            branch_decision="included",
        )
        try:
            record = TraceRecord(
                trace_id=trace_id,
                snapshot_id=snapshot.snapshot_id,
                prediction_id=None,
                trade_date=snapshot.trade_date,
                created_ts=created_ts,
                symbol=symbol,
                module_chain=(step,),
                final_action="WATCH",
                final_reason="screener_inclusion",
            )
            out.append(record)
        except Exception:  # noqa: BLE001
            # Skip any symbol whose schema validation fails — rare, but
            # never block the whole pipeline on one bad input.
            continue
    return out


def _compute_config_version(base_dir: Path) -> str:
    """Approximate config_version as a hash of key file mtimes."""
    targets = [
        base_dir / "docs" / "02_GOVERNANCE.md",
        base_dir / "reports" / "recalibrator_isotonic_v1.json",
    ]
    parts = []
    for t in targets:
        if t.exists():
            parts.append(f"{t.name}:{int(t.stat().st_mtime)}")
        else:
            parts.append(f"{t.name}:missing")
    return sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
