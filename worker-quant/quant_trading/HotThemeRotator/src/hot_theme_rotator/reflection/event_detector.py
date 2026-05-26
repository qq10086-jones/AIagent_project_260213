"""Multi-KPI event detector — CUSUM + bootstrap-derived h + Holm correction.

Codex review §3 hard requirements:

- Per-KPI ``h`` derived from block bootstrap on historical baseline (NOT
  generic σ multiple) — see ``bootstrap_arl``.
- Target family-level ARL_0 of 1-3 months across ~10 KPIs. Per-KPI ARL_0 of
  100 days would yield family false-alarm ~once per 10 trading days — too
  noisy. We expose per-KPI ``target_arl`` and let callers inflate it
  (Bonferroni-style) when a Holm step-down isn't being applied.
- Bernoulli outcomes (hit/miss): use proportion-aware ``k`` (NOT Gaussian σ).
- Returns: use robust stats (median, MAD) instead of mean and std.
- Holm step-down correction across the family (preferred over Bonferroni).

Output language: any reflection consumer of these detections MUST phrase
conclusions conditional on validity class (per ADR-0007 §5). This module
emits raw breach data; the language-shaping is the LLM Reflection Report
(P11-05) layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hot_theme_rotator.reflection.bootstrap_arl import (
    ArlCalibration,
    derive_threshold_for_target_arl,
)
from hot_theme_rotator.reflection.cusum import CusumState, run_cusum


__all__ = [
    "ALLOWED_KPI_KINDS",
    "CusumThreshold",
    "FamilyDetection",
    "KpiSeries",
    "derive_kpi_threshold",
    "detect_family_events",
    "holm_correction",
    "robust_returns_stats",
]


ALLOWED_KPI_KINDS = frozenset({"bernoulli", "returns"})


@dataclass(frozen=True)
class KpiSeries:
    """One KPI's baseline + ongoing observations.

    - ``kind='bernoulli'``: values are 0/1 hits. ``k`` is derived from the
      base rate via ``sqrt(p(1-p))``.
    - ``kind='returns'``: values are real-valued (returns, PnL, ratios).
      Robust stats (median + MAD) feed the CUSUM target and ``k``.
    """

    name: str
    kind: str
    baseline: tuple[float, ...]
    observed: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.kind not in ALLOWED_KPI_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(ALLOWED_KPI_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.baseline, tuple) or len(self.baseline) < 2:
            raise ValueError("baseline must be a tuple of length >= 2")
        if not isinstance(self.observed, tuple):
            raise ValueError("observed must be a tuple")
        if self.kind == "bernoulli":
            for v in self.baseline + self.observed:
                if v not in (0, 1, 0.0, 1.0):
                    raise ValueError(
                        f"bernoulli values must be 0/1, got {v!r} in {self.name!r}"
                    )


@dataclass(frozen=True)
class CusumThreshold:
    """Calibrated CUSUM parameters for one KPI."""

    kpi_name: str
    kind: str
    target: float
    k: float
    h: float
    target_arl: float
    calibration: ArlCalibration


@dataclass(frozen=True)
class FamilyDetection:
    """Per-KPI breach result before and after family multiplicity correction."""

    kpi_name: str
    breached: bool
    breach_index: int | None
    final_state: CusumState
    survival_score: float        # smaller = more anomalous (1.0 = never breached)
    significant_after_holm: bool


# ─── robust stats ──────────────────────────────────────────────────────────


def robust_returns_stats(values: Sequence[float]) -> tuple[float, float]:
    """Return (median, MAD*1.4826) for a returns-like KPI baseline.

    MAD scaled by 1.4826 is a consistent estimator of σ under normality and
    much more outlier-resistant than the sample std. We use this as the σ
    proxy when computing the CUSUM reference value ``k``.
    """
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    sorted_v = sorted(values)
    median = _median_sorted(sorted_v)
    abs_dev = sorted(abs(v - median) for v in values)
    mad = _median_sorted(abs_dev)
    return median, mad * 1.4826


def _median_sorted(sorted_v: Sequence[float]) -> float:
    n = len(sorted_v)
    if n % 2 == 1:
        return float(sorted_v[n // 2])
    return 0.5 * (float(sorted_v[n // 2 - 1]) + float(sorted_v[n // 2]))


# ─── per-KPI threshold derivation ───────────────────────────────────────────


def derive_kpi_threshold(
    kpi: KpiSeries,
    *,
    target_arl: float,
    n_bootstrap: int,
    block_size: int,
    seed: int,
    h_grid: Sequence[float] | None = None,
) -> CusumThreshold:
    """Compute target + k + h for ``kpi`` using its baseline."""
    if kpi.kind == "bernoulli":
        p = sum(kpi.baseline) / len(kpi.baseline)
        target = p
        sigma_proxy = (p * (1.0 - p)) ** 0.5
        # Edge case: p=0 or p=1 → sigma_proxy=0 → CUSUM trivially detects any
        # change. Use a tiny floor so block_bootstrap can still derive a
        # meaningful h on noisy real data.
        sigma_proxy = max(sigma_proxy, 1e-3)
        k = 0.5 * sigma_proxy
    else:
        median, sigma_proxy = robust_returns_stats(kpi.baseline)
        target = median
        sigma_proxy = max(sigma_proxy, 1e-9)
        k = 0.5 * sigma_proxy

    calibration = derive_threshold_for_target_arl(
        historical_values=kpi.baseline,
        target_arl=target_arl,
        target=target,
        k=k,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        seed=seed,
        h_grid=h_grid,
    )

    return CusumThreshold(
        kpi_name=kpi.name,
        kind=kpi.kind,
        target=target,
        k=k,
        h=calibration.selected_h,
        target_arl=target_arl,
        calibration=calibration,
    )


# ─── Holm multiplicity correction ───────────────────────────────────────────


def holm_correction(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[bool, ...]:
    """Holm-Bonferroni step-down. Returns flags aligned with input order.

    Procedure: sort p-values ascending; for rank r (1-based), declare
    significant iff ``p_(r) ≤ alpha / (m - r + 1)`` AND every smaller rank
    also passed. First non-passing rank stops the cascade — all subsequent
    ranks are non-significant. Provides strong family-wise error rate
    control at ``alpha`` and is uniformly more powerful than Bonferroni.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    m = len(p_values)
    if m == 0:
        return ()
    for p in p_values:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p}")

    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    significant = [False] * m
    for r, (orig_idx, p) in enumerate(ranked, start=1):
        threshold = alpha / (m - r + 1)
        if float(p) <= threshold:
            significant[orig_idx] = True
        else:
            # Step-down: every higher-rank p stays non-significant.
            break
    return tuple(significant)


# ─── orchestration ─────────────────────────────────────────────────────────


def detect_family_events(
    kpis: Sequence[KpiSeries],
    thresholds: Sequence[CusumThreshold],
    *,
    alpha: float = 0.05,
) -> tuple[FamilyDetection, ...]:
    """Run CUSUM per KPI; apply Holm across the family."""
    if len(kpis) != len(thresholds):
        raise ValueError(
            f"kpis and thresholds must align: {len(kpis)} vs {len(thresholds)}"
        )

    raw_results: list[tuple[KpiSeries, int | None, CusumState, float]] = []
    survival_scores: list[float] = []
    for kpi, thr in zip(kpis, thresholds):
        if kpi.name != thr.kpi_name:
            raise ValueError(
                f"kpi/threshold name mismatch: {kpi.name!r} vs {thr.kpi_name!r}"
            )
        breach_idx, state, _ = run_cusum(
            kpi.observed, target=thr.target, k=thr.k, h=thr.h,
        )
        # Survival score: ratio of run length to target ARL. Smaller = more
        # anomalous. Cap at 1.0 (never breached or breached beyond ARL).
        if breach_idx is None:
            score = 1.0
        else:
            score = min(1.0, (breach_idx + 1) / float(thr.target_arl))
        raw_results.append((kpi, breach_idx, state, score))
        survival_scores.append(score)

    # Treat survival_score as a p-value-like quantity for Holm. Under H0 the
    # observed run length is geometrically distributed with mean ≈ target_arl,
    # so P(run_length ≤ k) ≈ 1 − exp(−k / target_arl). For our use case the
    # ratio score itself orders the KPIs correctly by anomaly strength and
    # supplies a conservative (upper-bound) p-value.
    sig_flags = holm_correction(survival_scores, alpha=alpha)

    return tuple(
        FamilyDetection(
            kpi_name=kpi.name,
            breached=(breach_idx is not None),
            breach_index=breach_idx,
            final_state=state,
            survival_score=score,
            significant_after_holm=sig,
        )
        for (kpi, breach_idx, state, score), sig in zip(raw_results, sig_flags)
    )
