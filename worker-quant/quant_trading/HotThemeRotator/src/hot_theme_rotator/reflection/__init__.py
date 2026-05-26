"""Reflection system (P11, ADR-0007).

7-layer architecture per ADR-0007:

- L0 PIT Observability Ledger → ``observability/`` (P11-00 done)
- L1 Decision Trace Logger → this module's ``trace_logger`` (P11-01)
- L2 Event Detector + CUSUM + ARL bootstrap (P11-02, pending)
- L3 Policy Replay Engine (P11-03, pending)
- L4 Root Cause Analysis (P11-04, pending)
- L5 LLM Reflection Report (P11-05, pending)
- L6 Human Decision Gate (P11-06, pending)
- L7 Meta-Reflection (P11-07, pending)

This module currently exposes only the L1 surface.
"""
from __future__ import annotations

from hot_theme_rotator.reflection.bootstrap_arl import (
    ArlCalibration,
    block_bootstrap_indices,
    default_h_grid,
    derive_threshold_for_target_arl,
    estimate_arl_on_sequence,
)
from hot_theme_rotator.reflection.cusum import (
    CusumState,
    cusum_breached,
    reset_cusum,
    run_cusum,
    step_cusum,
)
from hot_theme_rotator.reflection.event_detector import (
    ALLOWED_KPI_KINDS,
    CusumThreshold,
    FamilyDetection,
    KpiSeries,
    derive_kpi_threshold,
    detect_family_events,
    holm_correction,
    robust_returns_stats,
)
from hot_theme_rotator.reflection.ablation import (
    ALLOWED_INTERVENTIONS,
    AblationContribution,
    AblationError,
    AblationResult,
    compute_ablation,
    rank_contributions,
)
from hot_theme_rotator.reflection.funnel import (
    ALLOWED_STAGE_NAMES,
    FunnelError,
    FunnelReport,
    FunnelStage,
    build_funnel_report,
    stage_loss,
    total_loss_ratio,
)
from hot_theme_rotator.reflection.policy_replay import (
    PolicyConfig,
    PolicyReplayError,
    PolicyReplayResult,
    RecordedScannerOutput,
    ReplayCellResult,
    compute_pareto_frontier,
    data_freshness_gate,
    replay_under_policy_grid,
)
from hot_theme_rotator.reflection.rca import (
    RcaError,
    RcaReport,
    build_rca_report,
)
from hot_theme_rotator.reflection.validity_class import (
    VALIDITY_CLASSES,
    conditional_language_prefix,
    is_publishable,
    is_stronger_than,
)
from hot_theme_rotator.reflection.trace_logger import (
    ModuleStep,
    ReflectionTraceError,
    TraceRecord,
    append_trace,
    compute_trace_id,
    read_traces,
    traces_path,
)

__all__ = [
    "ALLOWED_INTERVENTIONS",
    "ALLOWED_KPI_KINDS",
    "ALLOWED_STAGE_NAMES",
    "AblationContribution",
    "AblationError",
    "AblationResult",
    "ArlCalibration",
    "CusumState",
    "CusumThreshold",
    "FamilyDetection",
    "FunnelError",
    "FunnelReport",
    "FunnelStage",
    "KpiSeries",
    "ModuleStep",
    "PolicyConfig",
    "PolicyReplayError",
    "PolicyReplayResult",
    "RcaError",
    "RcaReport",
    "RecordedScannerOutput",
    "ReflectionTraceError",
    "ReplayCellResult",
    "TraceRecord",
    "VALIDITY_CLASSES",
    "append_trace",
    "block_bootstrap_indices",
    "build_funnel_report",
    "build_rca_report",
    "compute_ablation",
    "compute_pareto_frontier",
    "compute_trace_id",
    "conditional_language_prefix",
    "cusum_breached",
    "data_freshness_gate",
    "default_h_grid",
    "derive_kpi_threshold",
    "derive_threshold_for_target_arl",
    "detect_family_events",
    "estimate_arl_on_sequence",
    "holm_correction",
    "is_publishable",
    "is_stronger_than",
    "rank_contributions",
    "read_traces",
    "replay_under_policy_grid",
    "reset_cusum",
    "robust_returns_stats",
    "run_cusum",
    "stage_loss",
    "step_cusum",
    "total_loss_ratio",
    "traces_path",
]
