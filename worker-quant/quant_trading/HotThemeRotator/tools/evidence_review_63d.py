"""Locked 63D evidence review protocol (P31, design 2026-08-06 Section 4.4 / Section 4.5).

Purpose: stop a CALENDAR from moving a research gate. 2026-08-26 is the
EARLIEST date at which the 63D value/E-P live reading may be reviewed; it is
never a guaranteed verdict date and it can never upgrade an immature gate.

What this emits, in this order (the order matters):

1. The FROZEN trial family - every attempted variant counted BEFORE a single
   statistic is computed. Search breadth may not be hidden (Rule 8.2.2), and it
   is the denominator the Deflated Sharpe deflates over.
2. The data inventory - raw rows, independent date clusters, maturity coverage,
   and missingness, per signal and per horizon.
3. The check battery - PIT, survivorship, cost hurdle (Rule 16.0), purge,
   embargo, DSR, PBO/CPCV, and the Harvey-style t-stat bar (Rule 16.6 /
   ADR-0010). A check that cannot be computed from what is on disk reports
   ``insufficient`` WITH the reason. It never reports a pass it did not earn
   and never reports a failure it did not observe (Rule 11.9).
4. A three-valued verdict: ``confirm`` / ``fail`` / ``insufficient``.
   ``insufficient`` is the default and is NOT a failure and NOT a zero - it
   means the minimum evidence needed to decide is absent and collection
   continues without any capital change.
5. ``signal_verdict`` and ``deployment_verdict`` kept SEPARATE (design Section 4.5).
   Sleeve B holds no capital, so the deployment question is ``not_started``
   until a real Sleeve B fill exists in the journal, and the declared
   ``unwind_to_A`` response is non-operative on an empty sleeve.

E/P (``earnings_yield``) and B/P (``value_bp``) are reported INDEPENDENTLY. No
composite is emitted: a composite could average away the B/P sign reversal that
the live log has already recorded at 21D, and hiding a contrary observation is
the specific failure this protocol exists to prevent.

Read-only / advice-only (Rule 3): it computes and records, never acts. No
capital and no config change follows from this report. Fail-open: absent inputs
produce an honest "unavailable" report and exit 0, never a traceback
(Rule 11.9.4). Output carries no probability, win-rate, or expected-return
claim (Rule 8.3) - it reports IC, t-stat, DSR, and counts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from math import sqrt
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.common.console import (  # noqa: E402
    enable_console_fallback,
)

from hot_theme_rotator.calibration.overfit_gate import promote_gate  # noqa: E402
from hot_theme_rotator.research.cost_model import (  # noqa: E402
    COST_MODEL_REL,
    read_declared_cost_model,
    resolve_from_declared,
)
from hot_theme_rotator.risk.sleeve_engine import load_mandate  # noqa: E402

# --------------------------------------------------------------------------
# locked constants (changing any of these is a Rule 4 change, not a code tweak)
# --------------------------------------------------------------------------
EARLIEST_REVIEW_DATE = "2026-08-26"
REVIEW_HORIZON = 63                 # the locked label horizon of the experiment
CONTEXT_HORIZON = 21                # advance reading only; never the verdict
SIGNALS = ("earnings_yield", "value_bp")
VERDICTS = ("confirm", "fail", "insufficient")
DEPLOYMENT_NOT_STARTED = "not_started"

MIN_DATE_CLUSTERS = 20              # purged_walk_forward.min_effective_clusters
MIN_EFFECTIVE_OBS = 60              # overfit_gate.promote_gate.min_obs
HARVEY_T = 3.0                      # ADR-0010 Harvey-style t-stat bar
DSR_THRESHOLD = 0.95                # overfit_gate.promote_gate.dsr_threshold
TURNOVER = 0.7                      # signal_library.evaluate_signal default tau
DEFAULT_ROUND_TRIP_COST = 0.0005    # signal_library.S_KABU_COST (Rule 5.2)

REQUIRED_CHECKS = (
    "maturity",
    "date_clusters",
    "pit",
    "survivorship",
    "cost_hurdle",
    "purge",
    "embargo",
    "dsr",
    "pbo_cpcv",
    "harvey_t",
    "expected_sign",
    "ic_decay",
)

# --------------------------------------------------------------------------
# the FROZEN trial family
# --------------------------------------------------------------------------
# Counted before any statistic is computed. Each entry names the file whose
# declared grid produced it, so the count is auditable rather than asserted.
# ``in_hypothesis_lineage`` marks the searches from which the surviving E/P
# candidate was actually selected; the other families are still counted in the
# inclusive total because the inclusive total is the CONSERVATIVE deflation
# denominator, and under-counting search breadth is the failure mode Rule 8.2.2
# forbids. Grids overlap between studies (the live table re-runs configurations
# the historical sweeps also searched); the overlap is retained rather than
# netted out, again because over-counting deflates harder and is therefore the
# safe direction.
FROZEN_TRIAL_FAMILY = (
    {
        "study": "factor_zoo_v1_historical",
        "source": "tools/backtest_factor_zoo_history.py",
        "venue": "historical",
        "grid": "12 factors x 3 horizons (5/20/60D)",
        "n_trials": 36,
        "in_hypothesis_lineage": True,
        "note": "value_bp and the quality factors were first searched here (2026-07-02).",
    },
    {
        "study": "value_quality_v2_historical_P23B",
        "source": "tools/backtest_value_quality_history.py",
        "venue": "historical",
        "grid": "7 factors x 2 horizons (21/63D)",
        "n_trials": 14,
        "in_hypothesis_lineage": True,
        "note": "the sweep that selected earnings_yield@63D as the candidate (2026-07-03).",
    },
    {
        "study": "value_livelog_forward_P23F",
        "source": "tools/backtest_value_on_livelog.py",
        "venue": "live",
        "grid": "2 signals x 2 horizons (21/63D)",
        "n_trials": 4,
        "in_hypothesis_lineage": True,
        "note": "the live forward track under review here.",
    },
    {
        "study": "fundamental_yield_live_table",
        "source": "tools/forward_signal_report.py",
        "venue": "live",
        "grid": "2 signals x 3 horizons (1/3/5D)",
        "n_trials": 6,
        "in_hypothesis_lineage": True,
        "note": "same two signals at the short horizons in the daily forward table.",
    },
    {
        "study": "price_reversal_historical",
        "source": "tools/backtest_price_reversal_history.py",
        "venue": "historical",
        "grid": "5 lookbacks x 3 horizons",
        "n_trials": 15,
        "in_hypothesis_lineage": False,
        "note": "separate family; counted in the inclusive denominator only.",
    },
    {
        "study": "price_reversal_live_grid",
        "source": "tools/forward_signal_report.py",
        "venue": "live",
        "grid": "5 lookbacks x 3 horizons (recorded n_trials=15 in the live gate)",
        "n_trials": 15,
        "in_hypothesis_lineage": False,
        "note": "separate family; counted in the inclusive denominator only.",
    },
    {
        "study": "disclosure_drift_historical",
        "source": "tools/backtest_disclosure_drift_history.py",
        "venue": "historical",
        "grid": "4 horizons (1/3/5/10D)",
        "n_trials": 4,
        "in_hypothesis_lineage": False,
        "note": "separate family; counted in the inclusive denominator only.",
    },
    {
        "study": "screener_score_family_live",
        "source": "tools/forward_signal_report.py",
        "venue": "live",
        "grid": "2 signals (screener_buy, reversal_of_score) x 3 horizons",
        "n_trials": 6,
        "in_hypothesis_lineage": False,
        "note": "separate family; counted in the inclusive denominator only.",
    },
)

TRIAL_COUNT_INCLUSIVE = sum(s["n_trials"] for s in FROZEN_TRIAL_FAMILY)
TRIAL_COUNT_LINEAGE = sum(
    s["n_trials"] for s in FROZEN_TRIAL_FAMILY if s["in_hypothesis_lineage"]
)


def trial_family() -> dict:
    """The frozen family, counted before any statistic is computed."""
    return {
        "frozen": True,
        "counted_before_statistics": True,
        "n_trials_inclusive": TRIAL_COUNT_INCLUSIVE,
        "n_trials_lineage": TRIAL_COUNT_LINEAGE,
        "deflation_denominator": "n_trials_inclusive",
        "studies": [dict(s) for s in FROZEN_TRIAL_FAMILY],
        "note": (
            "Grids overlap between studies; the overlap is retained because "
            "over-counting search breadth deflates harder and is the safe "
            "direction. Both counts are reported so the denominator cannot be "
            "quietly reduced later."
        ),
    }


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _num(value):
    """Return ``value`` as a float when it is a real number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _horizon_row(result: dict | None, horizon: int) -> dict:
    """Fetch one (signal, horizon) row tolerating int/str JSON keys."""
    if not isinstance(result, dict):
        return {}
    row = result.get(str(horizon))
    if row is None:
        row = result.get(horizon)
    return row if isinstance(row, dict) else {}


def _check(name, status, detail, *, limiter=None, kill_criterion=False, **extra) -> dict:
    """One check row. ``limiter`` says WHY a check is not a pass:

    - ``sample``   the protocol exists, the sample is not yet large enough;
    - ``protocol`` the evidence needed is not produced by any current tool, so
      no amount of waiting will turn it into a pass.
    """
    row = {
        "check": name,
        "status": status,
        "detail": detail,
        "limiter": limiter,
        "kill_criterion": kill_criterion,
    }
    row.update(extra)
    return row


def aggregate_verdict(checks, *, confirm_adequate: bool, fail_adequate: bool):
    """Three-valued aggregation. Order matters and is deliberate.

    - ``fail`` only when a PREDECLARED kill criterion is observed to fail AND
      there is enough evidence to say so. A kill needs less evidence than a
      confirmation (a decisive contrary observation is cheaper than a proof of
      skill) — but it still needs evidence.
    - ``confirm`` only when every check passes AND the full locked sample bar
      is met. One uncomputable check is enough to withhold confirmation.
    - ``insufficient`` otherwise. It is the DEFAULT, not a failure, not a zero.
    """
    reasons: list[str] = []
    killed = [c for c in checks if c.get("kill_criterion") and c.get("status") == "fail"]
    if killed and fail_adequate:
        reasons = [f"kill criterion failed: {c['check']} - {c.get('detail', '')}" for c in killed]
        return "fail", reasons
    if killed and not fail_adequate:
        reasons.append(
            "a kill criterion is failing but the sample is below the minimum "
            "needed to declare it; recorded, not acted on"
        )
    not_pass = [c for c in checks if c.get("status") != "pass"]
    if not not_pass and confirm_adequate:
        return "confirm", ["every locked check passed with an adequate sample"]
    if not confirm_adequate:
        reasons.append("sample below the locked confirmation bar")
    for c in not_pass:
        reasons.append(f"{c['check']}={c.get('status')}: {c.get('detail', '')}")
    return "insufficient", reasons


# --------------------------------------------------------------------------
# observed cross-trial Sharpe dispersion (the DSR needs a real sr_std)
# --------------------------------------------------------------------------
def _observed_trial_sharpes(livelog: dict | None, forward_eval: dict | None):
    """Per-observation Sharpes for every trial whose statistics are on disk.

    Convention matches the existing gates (``tools/backtest_factor_zoo_history``
    and the live ``price_reversal_gate``): ``sr = t_stat / sqrt(n_days)``.
    """
    out: list[float] = []
    result = (livelog or {}).get("result") or {}
    for signal, per_h in result.items():
        if not isinstance(per_h, dict):
            continue
        for _h, row in per_h.items():
            t = _num((row or {}).get("t_stat"))
            n = _num((row or {}).get("n_dates"))
            if t is not None and n and n > 0:
                out.append(t / sqrt(n))
    table = (forward_eval or {}).get("table") or {}
    for signal, block in table.items():
        for _h, row in ((block or {}).get("horizons") or {}).items():
            t = _num((row or {}).get("t_stat"))
            n = _num((row or {}).get("n_days"))
            if t is not None and n and n > 0:
                out.append(t / sqrt(n))
    return out


# --------------------------------------------------------------------------
# per-signal assessment
# --------------------------------------------------------------------------
def _inventory_row(row: dict, raw_rows, horizon: int) -> dict:
    matured = _num(row.get("matured")) or 0.0
    unmatured = _num(row.get("unmatured")) or 0.0
    n_dates = int(_num(row.get("n_dates")) or 0)
    scored = matured + unmatured
    coverage = (matured / scored) if scored > 0 else None
    rows_unscored = (raw_rows - scored) if isinstance(raw_rows, (int, float)) else None
    unscored_fraction = (
        (rows_unscored / raw_rows) if (rows_unscored is not None and raw_rows) else None
    )
    return {
        "independent_date_clusters": n_dates,
        "matured": int(matured),
        "unmatured": int(unmatured),
        "rows_scored": int(scored),
        "maturity_coverage": coverage,
        "rows_unscored": int(rows_unscored) if rows_unscored is not None else None,
        "unscored_fraction": unscored_fraction,
        "mean_ic": _num(row.get("mean_ic")),
        "t_stat": _num(row.get("t_stat")),
        # Overlapping windows: only every ``horizon``-th cluster is independent.
        # See ``effective_sample_protocol`` for the estimator and its rivals —
        # this integer is a floor, not the only defensible number.
        "n_obs_effective": n_dates // horizon if horizon else None,
        "n_obs_effective_continuous": (n_dates / horizon) if horizon else None,
        "n_obs_effective_method": "disjoint_blocks",
    }


JPX_SESSIONS_PER_YEAR = 245        # ~245 trading sessions/yr, used only to
                                   # express a cluster count as calendar time.


def effective_sample_protocol(n_dates: int, horizon: int, min_obs: int) -> dict:
    """Candidate effective-sample estimators for overlapping labels, with their
    assumptions attached. PROPOSED, not locked.

    ``n_obs_effective`` was originally a bare ``n_dates // horizon`` with no
    recorded justification. A first attempt to justify it OVERREACHED and is
    retracted here (2026-08-06), because getting this wrong in the confident
    direction is worse than leaving it unexplained:

    - it computed ``1 + 2*sum_{k<h}(1 - k/h) = h`` and labelled that
      "Newey-West/Hansen-Hodrick". That factor is the TRUE variance inflation
      of a mean of overlapping h-sums under a triangular ACF — it is NOT the
      Newey-West estimator, which applies BARTLETT WEIGHTS and yields
      ``1 + 2*sum_{k<h}(1 - k/h)^2 = (2h^2+1)/(3h) = 42.0`` at h=63
      (Newey & West 1987, DOI 10.2307/1913610);
    - it therefore claimed the estimators "agree" and the bar "cannot be
      relieved by switching estimators". Both claims are withdrawn: the NW
      estimator implies ~2,520 clusters where disjoint blocks implies ~3,780;
    - ``rho_k = 1 - k/h`` itself holds for overlapping sums of iid equal-weight
      increments. A 63D cross-sectional Rank-IC series is not that, so the
      triangular ACF is an ASSUMPTION about this data, never a derivation from
      it. Hansen & Hodrick (1980, DOI 10.1086/260910) is a different overlapping
      -forecast setting again.

    What survives: ``floor(n/h)`` is a defensible NON-OVERLAPPING threshold, and
    the cluster counts below are conditional on whichever estimator is adopted.
    Which one governs the gate is a Rule 4 owner decision. Engineering may
    propose it; engineering may not mark it locked, which is what the previous
    version did.
    """
    horizon = max(int(horizon), 1)
    n_dates = max(int(n_dates), 0)
    true_vif = float(horizon)
    nw_vif = 1 + 2 * sum((1 - k / horizon) ** 2 for k in range(1, horizon))

    def required(vif: float) -> dict:
        clusters = int(round(min_obs * vif))
        return {
            "date_clusters_required": clusters,
            "years_of_daily_cross_sections_required": round(
                clusters / JPX_SESSIONS_PER_YEAR, 1),
        }

    return {
        "status": "proposed",
        "requires_owner_approval": (
            "Rule 4 — engineering may propose an effective-sample estimator "
            "but may not adopt one on its own authority"
        ),
        "gate_estimator_in_use": "disjoint_blocks",
        "gate_estimator_basis": (
            "the pre-existing implementation, retained pending an owner "
            "decision; it is a choice, not a result"
        ),
        "estimators": {
            "disjoint_blocks": {
                "n_obs_effective": n_dates // horizon,
                "variance_inflation_factor": true_vif,
                "assumption": (
                    "only every h-th cluster starts a non-overlapping window; "
                    "partially overlapping windows contribute nothing"
                ),
                **required(true_vif),
            },
            "unweighted_triangular_acf": {
                "n_obs_effective": n_dates / true_vif,
                "variance_inflation_factor": true_vif,
                "assumption": (
                    "overlapping sums of iid equal-weight increments give "
                    "rho_k = 1 - k/h, whose unweighted long-run variance "
                    "inflation is exactly h. NOT established for a "
                    "cross-sectional Rank-IC series"
                ),
                **required(true_vif),
            },
            "newey_west_bartlett": {
                "n_obs_effective": n_dates / nw_vif,
                "variance_inflation_factor": nw_vif,
                "assumption": (
                    "Newey-West HAC with Bartlett weights at bandwidth h-1 "
                    "applied to the same triangular ACF; downweights high lags "
                    "by construction to guarantee a PSD estimate "
                    "(Newey & West 1987, DOI 10.2307/1913610)"
                ),
                **required(nw_vif),
            },
            "naive_ignores_overlap": {
                "n_obs_effective": n_dates,
                "variance_inflation_factor": 1.0,
                "assumption": (
                    "overlapping windows treated as independent; reported only "
                    "to size the error it would introduce, never used"
                ),
                **required(1.0),
            },
        },
        "min_effective_obs": min_obs,
        "retracted_claims": [
            "that the unweighted factor h is the Newey-West estimator",
            "that the candidate estimators agree analytically",
            "that the bar cannot be relieved by switching estimators",
            "that ~15.4 years is an unconditional statistical necessity",
        ],
        "note": (
            "Every cluster requirement below is CONDITIONAL on the estimator "
            "adopted. Under disjoint blocks the min_obs bar needs "
            f"{required(true_vif)['date_clusters_required']} clusters; under "
            f"Newey-West-Bartlett it needs "
            f"{required(nw_vif)['date_clusters_required']}. Neither is a fact "
            "about the data until an estimator is chosen under Rule 4."
        ),
    }


def _leakage_checks(leakage_audit, leakage_audit_asof, window_end):
    """PIT + survivorship, driven by the Rule 9.4.2 leakage-audit artifact.

    An out-of-scope audit is never mapped onto this experiment in EITHER
    direction: it cannot manufacture a pass, and it cannot manufacture a
    failure the experiment was never audited for.
    """
    if not isinstance(leakage_audit, dict):
        detail = (
            "no Rule 9.4.2 leakage-audit artifact at or before this asof under "
            "reports/calibration/leakage_audit_*.json; the PIT contract of the "
            "live-log read is a code property, not recorded evidence"
        )
        return (
            _check("pit", "insufficient", detail, limiter="protocol"),
            _check(
                "survivorship", "insufficient",
                "no leakage-audit artifact records a survivorship verdict for "
                "the live-log universe; the append-only forward log is not a "
                "reconstructed universe, but that too is a code property rather "
                "than recorded evidence",
                limiter="protocol",
            ),
        )

    verdict = str(leakage_audit.get("verdict", "unknown"))
    vectors = {
        str(v.get("vector", "")): v
        for v in (leakage_audit.get("vectors") or [])
        if isinstance(v, dict)
    }
    covers = bool(leakage_audit_asof and window_end and leakage_audit_asof >= window_end)

    def _one(name, vector_prefix, subject):
        vec = next(
            (v for k, v in vectors.items() if k.startswith(vector_prefix)), None
        )
        vec_status = str((vec or {}).get("status", "absent"))
        if verdict == "clean" and covers and vec_status == "pass":
            return _check(
                name, "pass",
                f"leakage audit {leakage_audit_asof} verdict={verdict}, "
                f"{vector_prefix} status=pass, and the audit covers the "
                f"evaluation window end {window_end}",
                limiter=None, audit_asof=leakage_audit_asof, audit_verdict=verdict,
                vector_status=vec_status,
            )
        gaps = []
        if verdict != "clean":
            gaps.append(f"audit verdict={verdict}")
        if not covers:
            gaps.append(
                f"audit asof {leakage_audit_asof} does not cover the evaluation "
                f"window end {window_end}"
            )
        if vec_status != "pass":
            gaps.append(f"{vector_prefix} status={vec_status}")
        return _check(
            name, "insufficient",
            f"{subject} not verified for this experiment: " + "; ".join(gaps)
            + ". The audit on disk is scoped to the backdated calibration "
              "sample, not to the value live-log read, so neither a pass nor a "
              "failure may be transferred to it",
            limiter="protocol", audit_asof=leakage_audit_asof,
            audit_verdict=verdict, vector_status=vec_status,
        )

    return (_one("pit", "V3", "point-in-time contract"),
            _one("survivorship", "V2", "survivorship"))


def _assess_signal(
    signal: str,
    *,
    livelog: dict | None,
    forward_eval: dict | None,
    inventory: dict,
    leakage_pair,
    sr_std,
    n_sr_observed,
    declared_cost_model=None,
) -> dict:
    locked = inventory.get(str(REVIEW_HORIZON), {})
    matured = locked.get("matured", 0)
    n_dates = locked.get("independent_date_clusters", 0)
    n_obs_eff = locked.get("n_obs_effective", 0)
    ic = locked.get("mean_ic")
    t = locked.get("t_stat")

    confirm_adequate = (
        matured > 0 and n_dates >= MIN_DATE_CLUSTERS and n_obs_eff >= MIN_EFFECTIVE_OBS
    )
    fail_adequate = matured > 0 and n_dates >= MIN_DATE_CLUSTERS

    checks: list[dict] = []

    checks.append(_check(
        "maturity",
        "pass" if matured > 0 else "insufficient",
        f"{matured} matured rows at {REVIEW_HORIZON}D "
        f"({locked.get('unmatured', 0)} still open); zero matured is an absent "
        f"reading, not a zero reading",
        limiter="sample", matured=matured, unmatured=locked.get("unmatured", 0),
    ))

    checks.append(_check(
        "date_clusters",
        "pass" if n_dates >= MIN_DATE_CLUSTERS else "insufficient",
        f"{n_dates} independent date clusters vs {MIN_DATE_CLUSTERS} minimum "
        f"(same-day names share a market factor and are one cluster, not N rows)",
        limiter="sample", n_dates=n_dates, minimum=MIN_DATE_CLUSTERS,
    ))

    pit_check, surv_check = leakage_pair
    checks.append(dict(pit_check))
    checks.append(dict(surv_check))

    # Rule 16.0 cost hurdle at the LOCKED horizon only.
    fe_row = _horizon_row(
        ((forward_eval or {}).get("table") or {}).get(signal, {}).get("horizons"),
        REVIEW_HORIZON,
    )
    # Rule 16.0 inputs come from the SHARED contract so this report and the P33
    # scorecard can never disagree about whether the hurdle is computable.
    # Note the removed behaviour: the round-trip cost used to fall back to a
    # module default, which meant the hurdle could "pass" on an assumed cost.
    cost = resolve_from_declared(
        declared_cost_model, horizon=REVIEW_HORIZON, observed=fe_row)
    hurdle = cost.hurdle()
    if hurdle is not None and ic is not None:
        net = ic * cost.sigma_r - cost.turnover * cost.round_trip_cost
        # A hurdle computed from observed per-run values is not the same
        # evidence as one computed from a declared model, and cannot clear a
        # governance gate on its own.
        verdict = ("pass" if (ic > 0 and net > 0) else "fail") \
            if cost.fully_declared else "insufficient"
        detail = (
            f"Rule 16.0 hurdle {hurdle:.5f} vs mean IC {ic:+.5f} "
            f"(sigma_r={cost.sigma_r:.5f}, tau={cost.turnover}, "
            f"c_rt={cost.round_trip_cost})"
        )
        if not cost.fully_declared:
            detail += (
                f"; computed from OBSERVED values, not a declared cost model "
                f"({COST_MODEL_REL} absent or partial), so it is reported and "
                f"not scored as a pass"
            )
        checks.append(_check(
            "cost_hurdle", verdict, detail,
            limiter="sample" if cost.fully_declared else "protocol",
            hurdle=hurdle, mean_ic=ic, net_ic_after_cost=net,
            cost_model=cost.as_dict(),
        ))
    else:
        available = sorted(
            str(k) for k in (
                ((forward_eval or {}).get("table") or {})
                .get(signal, {}).get("horizons") or {}
            )
        )
        checks.append(_check(
            "cost_hurdle", "insufficient",
            f"Rule 16.0 hurdle inputs absent: {', '.join(cost.missing)}. "
            f"The shared contract {COST_MODEL_REL} is the canonical source; "
            f"forward-eval horizons on disk: "
            f"{', '.join(available) if available else 'none'}. Nothing is "
            f"assumed - a defaulted cost that clears the hurdle is the failure "
            f"mode Rule 16.0 exists to prevent",
            limiter="protocol", sigma_r_available_horizons=available,
            cost_model=cost.as_dict(),
        ))

    checks.append(_check(
        "purge", "insufficient",
        "the live-log read performs no train/test split, so no purge protocol "
        "is recorded in the artifact; absence of a fitted parameter is a code "
        "property of tools/backtest_value_on_livelog.py, not recorded evidence, "
        "and is not scored as a pass",
        limiter="protocol",
    ))

    protocol = effective_sample_protocol(n_dates, REVIEW_HORIZON, MIN_EFFECTIVE_OBS)
    checks.append(_check(
        "embargo",
        "pass" if n_obs_eff >= MIN_EFFECTIVE_OBS else "insufficient",
        f"overlapping {REVIEW_HORIZON}D windows: {n_dates} date clusters give "
        f"{n_obs_eff} effective observations under the "
        f"{protocol['gate_estimator_in_use']} estimator vs {MIN_EFFECTIVE_OBS} "
        f"required. That estimator is PROPOSED, not locked (Rule 4). The cluster "
        f"requirement is conditional on it: "
        f"{protocol['estimators']['disjoint_blocks']['date_clusters_required']} "
        f"under disjoint blocks vs "
        f"{protocol['estimators']['newey_west_bartlett']['date_clusters_required']} "
        f"under Newey-West-Bartlett. An earlier claim that these agree, and that "
        f"the bar could not be relieved by switching estimators, is RETRACTED",
        limiter="sample", horizon_days=REVIEW_HORIZON, n_obs_effective=n_obs_eff,
        min_effective_obs=MIN_EFFECTIVE_OBS,
        date_clusters_required_for_min_obs=(
            protocol["estimators"]["disjoint_blocks"]["date_clusters_required"]),
        date_clusters_observed=n_dates,
        effective_sample_protocol=protocol,
    ))

    # Deflated Sharpe over the FROZEN family (counted first, above).
    if t is None or not n_dates:
        checks.append(_check(
            "dsr", "insufficient",
            f"no {REVIEW_HORIZON}D t-stat on disk; nothing to deflate",
            limiter="sample", n_trials=TRIAL_COUNT_INCLUSIVE,
            n_trials_lineage=TRIAL_COUNT_LINEAGE, threshold=DSR_THRESHOLD,
        ))
    elif not sr_std or sr_std <= 0:
        checks.append(_check(
            "dsr", "insufficient",
            f"cross-trial Sharpe dispersion could not be estimated "
            f"({n_sr_observed} trials with recorded statistics); the gate "
            f"fails closed rather than deflating against an assumed dispersion",
            limiter="sample", n_trials=TRIAL_COUNT_INCLUSIVE,
            n_trials_lineage=TRIAL_COUNT_LINEAGE, threshold=DSR_THRESHOLD,
            n_trials_with_observed_sr=n_sr_observed,
        ))
    else:
        sr = t / sqrt(n_dates)
        gate_inc = promote_gate(
            sr, n_trials=TRIAL_COUNT_INCLUSIVE, n_obs=max(n_obs_eff, 2),
            sr_std=sr_std, dsr_threshold=DSR_THRESHOLD, min_obs=MIN_EFFECTIVE_OBS,
        )
        gate_lin = promote_gate(
            sr, n_trials=TRIAL_COUNT_LINEAGE, n_obs=max(n_obs_eff, 2),
            sr_std=sr_std, dsr_threshold=DSR_THRESHOLD, min_obs=MIN_EFFECTIVE_OBS,
        )
        checks.append(_check(
            "dsr",
            "pass" if gate_inc["pass"] else "fail",
            f"DSR {gate_inc['dsr']:.4f} vs {DSR_THRESHOLD} threshold over "
            f"{TRIAL_COUNT_INCLUSIVE} frozen trials "
            f"(sr_per_obs {sr:+.4f}, sr_std {sr_std:.4f}, "
            f"noise-max {gate_inc['expectedMaxSharpe']:.4f}, "
            f"n_obs_eff {gate_inc['nObs']})",
            limiter="sample", n_trials=TRIAL_COUNT_INCLUSIVE,
            n_trials_lineage=TRIAL_COUNT_LINEAGE, threshold=DSR_THRESHOLD,
            dsr_inclusive=gate_inc["dsr"], dsr_lineage=gate_lin["dsr"],
            sr_per_obs=sr, sr_std_observed=sr_std,
            n_trials_with_observed_sr=n_sr_observed,
            expected_max_sharpe_noise_bound=gate_inc["expectedMaxSharpe"],
            gate_reasons=list(gate_inc["reasons"]),
        ))

    checks.append(_check(
        "pbo_cpcv", "insufficient",
        "no combinatorially-purged cross-validation path matrix exists in this "
        "repo, so the PBO (backtest-overfitting) statistic is not "
        "computed by any current tool (see the overfit_gate module docstring: "
        "PBO/CPCV is wired once real trial paths exist). Until it exists, the "
        "ADR-0010 gate cannot be fully cleared and no confirmation is available",
        limiter="protocol",
    ))

    if t is None:
        harvey_status = "insufficient"
        harvey_detail = f"no {REVIEW_HORIZON}D t-stat on disk"
    elif t >= HARVEY_T:
        harvey_status = "pass"
        harvey_detail = f"t={t:+.3f} clears the Harvey bar |t|>={HARVEY_T} in the declared direction"
    elif t <= -HARVEY_T:
        harvey_status = "fail"
        harvey_detail = (
            f"t={t:+.3f} clears |t|>={HARVEY_T} in the OPPOSITE direction to the "
            f"declared hypothesis"
        )
    else:
        harvey_status = "fail" if confirm_adequate else "insufficient"
        harvey_detail = f"t={t:+.3f} below the Harvey bar |t|>={HARVEY_T}"
    checks.append(_check(
        "harvey_t", harvey_status, harvey_detail, limiter="sample",
        t_stat=t, bar=HARVEY_T,
    ))

    # The predeclared kill criterion: the declared direction is POSITIVE for
    # both a yield (E/P) and a book-to-price (B/P). A reversal established at
    # the Harvey bar is the one observation that can end the experiment.
    if ic is None:
        sign_status, sign_detail = "insufficient", f"no {REVIEW_HORIZON}D IC on disk"
    elif ic > 0:
        sign_status, sign_detail = "pass", f"mean IC {ic:+.5f} has the declared positive sign"
    elif t is not None and abs(t) >= HARVEY_T:
        sign_status = "fail"
        sign_detail = (
            f"SIGN REVERSAL: mean IC {ic:+.5f} with t={t:+.3f} establishes the "
            f"opposite of the declared direction at the Harvey bar"
        )
    else:
        sign_status = "insufficient"
        sign_detail = (
            f"mean IC {ic:+.5f} is not positive but is not established at the "
            f"Harvey bar either (t={t if t is None else round(t, 3)})"
        )
    checks.append(_check(
        "expected_sign", sign_status, sign_detail, limiter="sample",
        kill_criterion=True, mean_ic=ic, declared_direction="positive",
    ))

    decay = {h: row.get("mean_ic") for h, row in inventory.items()}
    n_with_ic = sum(1 for v in decay.values() if v is not None)
    checks.append(_check(
        "ic_decay",
        "pass" if n_with_ic >= 2 else "insufficient",
        f"IC across horizons {sorted(decay)}: "
        + ", ".join(
            f"{h}D=" + ("n/a" if v is None else f"{v:+.4f}") for h, v in sorted(decay.items())
        )
        + f" ({n_with_ic} horizons with a reading; Rule 16.6 requires decay to be reported)",
        limiter="sample", horizons_with_ic=n_with_ic, decay=decay,
    ))

    verdict, reasons = aggregate_verdict(
        checks, confirm_adequate=confirm_adequate, fail_adequate=fail_adequate
    )
    return {
        "signal": signal,
        "locked_horizon": REVIEW_HORIZON,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "evidence_adequacy": {
            "confirm_adequate": confirm_adequate,
            "fail_adequate": fail_adequate,
            "confirm_rule": (
                f"matured>0 AND date_clusters>={MIN_DATE_CLUSTERS} AND "
                f"n_obs_effective>={MIN_EFFECTIVE_OBS}"
            ),
            "fail_rule": f"matured>0 AND date_clusters>={MIN_DATE_CLUSTERS}",
        },
        "locked_horizon_reading": locked,
        "context_horizons": {
            h: row for h, row in inventory.items() if h != str(REVIEW_HORIZON)
        },
        "checks": checks,
    }


# --------------------------------------------------------------------------
# deployment (design §4.5) — a separate question from the signal
# --------------------------------------------------------------------------
def _effective_fills(entries):
    """Journal fills after Rule 14.4 corrections, mirroring portfolio.derive.

    A ``source='correction'`` entry is bookkeeping and is skipped; the entry it
    corrects never happened and is skipped too.
    """
    skip: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        if str(e.get("source")) == "correction":
            eid = e.get("entry_id")
            if eid:
                skip.add(str(eid))
            corrected = e.get("corrects")
            if corrected:
                skip.add(str(corrected))
    return [
        e for e in entries
        if isinstance(e, dict)
        and e.get("_type") == "fill"
        and str(e.get("entry_id")) not in skip
    ]


def _deployment_block(mandate, journal_entries):
    if not isinstance(mandate, dict) or not isinstance(mandate.get("sleeve_map"), dict):
        return {
            "available": False,
            "reason": (
                "configs/risk_mandate.json is absent or has no sleeve_map; "
                "Sleeve B membership cannot be resolved, so the deployment "
                "question is unanswerable rather than answered 'not started'"
            ),
            "sleeve_b_symbols": None,
            "sleeve_b_fill_count": None,
            "sleeve_b_net_qty": {},
            "sleeve_b_symbols_held": [],
            "verdict": "insufficient",
        }
    sleeve_map = mandate["sleeve_map"]
    b_symbols = sorted(s for s, sleeve in sleeve_map.items() if str(sleeve).upper() == "B")
    fills = [f for f in _effective_fills(journal_entries) if f.get("symbol") in set(b_symbols)]
    net: dict[str, int] = {}
    for f in fills:
        qty = _num(f.get("qty")) or 0.0
        side = str(f.get("side", "")).upper()
        delta = int(qty) if side == "BUY" else -int(qty)
        net[str(f.get("symbol"))] = net.get(str(f.get("symbol")), 0) + delta
    held = sorted(s for s, q in net.items() if q > 0)
    return {
        "available": True,
        "reason": None,
        "sleeve_b_symbols": b_symbols,
        "sleeve_b_fill_count": len(fills),
        "sleeve_b_net_qty": {s: q for s, q in sorted(net.items()) if q != 0},
        "sleeve_b_symbols_held": held,
        "verdict": DEPLOYMENT_NOT_STARTED if not fills else "insufficient",
        "note": (
            "deployment evidence means execution, slippage, adherence, and "
            "holding-period records from a REAL Sleeve B portfolio; with no "
            "fill there is nothing to measure and the question is not started, "
            "which is distinct from a failed experiment"
            if not fills else
            "Sleeve B fills exist; execution/slippage/adherence evaluation is "
            "not implemented by this report and is therefore not claimed"
        ),
    }


def _precommitment_block(mandate, deployment):
    sleeve_b = ((mandate or {}).get("sleeves") or {}).get("B") or {}
    pre = sleeve_b.get("precommitment") or {}
    if not pre:
        return {
            "available": False,
            "reason": "no Sleeve B pre-commitment declared in the mandate config",
        }
    held = deployment.get("sleeve_b_symbols_held") or []
    operative = bool(held)
    declared = pre.get("verdict_date")
    return {
        "available": True,
        "declared_verdict_date": declared,
        "declared_verdict_date_is_earliest_review": declared == EARLIEST_REVIEW_DATE,
        "declared_verdict_date_reading": (
            "the mandate's 'verdict_date' is read as the EARLIEST review date; "
            "it does not schedule a verdict and cannot upgrade an immature gate"
        ),
        "cap_jpy": sleeve_b.get("cap_jpy"),
        "on_confirm_cap_jpy": pre.get("on_confirm_cap_jpy"),
        "on_fail": pre.get("on_fail"),
        "on_fail_operative": operative,
        "on_fail_note": (
            "non-operative: Sleeve B is empty, so unwinding it is a no-op and "
            "must never be recorded as a completed pre-commitment response"
            if not operative else
            "operative: Sleeve B holds a position, so the declared response "
            "would move real capital and requires a Rule 4 owner action"
        ),
    }


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
def build_review(
    *,
    asof: str,
    livelog: dict | None = None,
    livelog_asof: str | None = None,
    livelog_path: str | None = None,
    trace_rows=(),
    forward_eval: dict | None = None,
    forward_eval_asof: str | None = None,
    journal_entries=(),
    mandate: dict | None = None,
    leakage_audit: dict | None = None,
    leakage_audit_asof: str | None = None,
    declared_cost_model: dict | None = None,
    warnings=(),
) -> dict:
    """Pure assembly of the locked review artifact from already-loaded inputs."""
    # 1. Trial family FIRST — counted before any statistic is computed.
    family = trial_family()

    reached = bool(asof >= EARLIEST_REVIEW_DATE)
    review_window = {
        "asof": asof,
        "earliest_review_date": EARLIEST_REVIEW_DATE,
        "earliest_review_date_reached": reached,
        "is_guaranteed_verdict_date": False,
        "language": (
            "2026-08-26 is the EARLIEST date at which the 63D reading may be "
            "reviewed. It is not a guaranteed verdict date. A date can never "
            "override a failed or immature gate."
        ),
        "locked_horizon_days": REVIEW_HORIZON,
    }

    # 2. Inventory.
    raw_rows = _num((livelog or {}).get("n_rows"))
    trade_days = _num((livelog or {}).get("trade_days"))
    if not isinstance(livelog, dict):
        inventory = {
            "available": False,
            "reason": (
                "no value/E-P live-log artifact at or before this asof under "
                "reports/observability/value_livelog/; the reading is "
                "unavailable, which is not the same as a reading of zero"
            ),
            "source_artifact": None,
            "source_artifact_asof": None,
            "raw_rows_in_log": None,
            "trade_days": None,
            "per_signal": {},
        }
        per_signal_inv = {s: {} for s in SIGNALS}
    else:
        per_signal_inv = {}
        result = livelog.get("result") or {}
        for signal in SIGNALS:
            rows = {}
            for h in (CONTEXT_HORIZON, REVIEW_HORIZON):
                rows[str(h)] = _inventory_row(
                    _horizon_row(result.get(signal), h),
                    int(raw_rows) if raw_rows is not None else None,
                    h,
                )
            per_signal_inv[signal] = rows
        inventory = {
            "available": True,
            "reason": None,
            "source_artifact": livelog_path,
            "source_artifact_asof": livelog_asof,
            "artifact_lag_days": _date_gap(livelog_asof, asof),
            "raw_rows_in_log": int(raw_rows) if raw_rows is not None else None,
            "trade_days": int(trade_days) if trade_days is not None else None,
            "per_signal": per_signal_inv,
            "missingness_note": (
                "rows_unscored = rows in the live log minus rows the signal "
                "could score; the upstream artifact records no per-row drop "
                "reason, so the count is reported without an attributed cause"
            ),
        }

    # forward-eval short horizons are context for the Rule 16.6 decay report.
    fe_table = (forward_eval or {}).get("table") or {}
    for signal in SIGNALS:
        horizons = (fe_table.get(signal) or {}).get("horizons") or {}
        for h, row in horizons.items():
            if str(h) == str(REVIEW_HORIZON):
                continue
            per_signal_inv.setdefault(signal, {})[str(h)] = {
                "independent_date_clusters": int(_num(row.get("n_days")) or 0),
                "matured": None,
                "unmatured": None,
                "rows_scored": None,
                "maturity_coverage": None,
                "rows_unscored": None,
                "unscored_fraction": None,
                "mean_ic": _num(row.get("mean_ic")),
                "t_stat": _num(row.get("t_stat")),
                "n_obs_effective": None,
                "source": "forward_signal_eval",
            }

    # 3. Checks.
    window_end = livelog_asof
    leakage_pair = _leakage_checks(leakage_audit, leakage_audit_asof, window_end)
    sharpes = _observed_trial_sharpes(livelog, forward_eval)
    sr_std = pstdev(sharpes) if len(sharpes) >= 2 else 0.0

    signals = {
        signal: _assess_signal(
            signal,
            livelog=livelog,
            forward_eval=forward_eval,
            inventory=per_signal_inv.get(signal, {}),
            leakage_pair=leakage_pair,
            sr_std=sr_std,
            n_sr_observed=len(sharpes),
            declared_cost_model=declared_cost_model,
        )
        for signal in SIGNALS
    }

    # 4. Sign-reversal watch — the B/P reversal must never be averageable away.
    observations = []
    for signal in SIGNALS:
        for h, row in sorted((per_signal_inv.get(signal) or {}).items(), key=lambda kv: int(kv[0])):
            ic = row.get("mean_ic")
            t = row.get("t_stat")
            established = ic is not None and ic < 0 and t is not None and abs(t) >= HARVEY_T
            observations.append({
                "signal": signal,
                "horizon": h,
                "mean_ic": ic,
                "t_stat": t,
                "sign": "unavailable" if ic is None else ("negative" if ic < 0 else "positive"),
                "declared_direction": "positive",
                "established_at_harvey_bar": bool(established),
            })
    watch = {
        "observations": observations,
        "any_reversal_established": any(o["established_at_harvey_bar"] for o in observations),
        "note": (
            "B/P (value_bp) is tracked here on its own axis. A contrary sign at "
            "any horizon is recorded and never netted against E/P"
        ),
    }

    # 5. Confirmability — which blockers no amount of waiting can clear.
    blocking = sorted({
        c["check"]
        for s in signals.values()
        for c in s["checks"]
        if c.get("limiter") == "protocol" and c.get("status") != "pass"
    })
    confirmability = {
        "confirm_reachable": not blocking,
        "blocking_checks": blocking,
        "note": (
            "these checks are blocked by a missing PROTOCOL, not by a small "
            "sample: waiting for more data cannot turn them into a pass. Until "
            "they exist, 'confirm' is not an available verdict and the honest "
            "ceiling of this review is 'insufficient'"
        ),
    }

    deployment = _deployment_block(mandate, list(journal_entries))
    precommitment = _precommitment_block(mandate, deployment)

    signal_verdict = signals["earnings_yield"]["verdict"]
    report = {
        "_kind": "evidence_review_63d",
        "asof": asof,
        "generated_by": "tools/evidence_review_63d.py",
        "governance": {
            "rules": ["3", "8.2", "8.3", "11.9", "16.0", "16.3", "16.6", "17"],
            "design": "docs/superpowers/specs/2026-08-06-evidence-based-retrospective-remediation-design.md",
            "task": "P31",
        },
        "review_window": review_window,
        "trial_family": family,
        "data_inventory": inventory,
        "trace_continuity": _trace_summary(trace_rows),
        "signals": signals,
        "sign_reversal_watch": watch,
        "composite": {
            "emitted": False,
            "reason": (
                "no composite of E/P and B/P is emitted: averaging them could "
                "hide the recorded B/P sign reversal, and a metric that can "
                "hide a contrary observation is not evidence"
            ),
        },
        "confirmability": confirmability,
        "deployment": deployment,
        "precommitment": precommitment,
        "verdicts": {
            "signal_verdict": signal_verdict,
            "signal_verdict_basis": (
                f"earnings_yield @ {REVIEW_HORIZON}D - the Section 17 / ADR-0012 "
                f"Sleeve B pre-commitment hook"
            ),
            "per_signal": {s: signals[s]["verdict"] for s in SIGNALS},
            "deployment_verdict": deployment["verdict"],
            "deployment_basis": (
                "presence of a real Sleeve B fill in reports/portfolio/journal/"
            ),
            "vocabulary": list(VERDICTS),
            "insufficient_means": (
                "the minimum evidence needed to decide is absent; collection "
                "continues and no capital moves. It is not a failure and not a zero"
            ),
        },
        "actions": {
            "advice_only": True,
            "capital_change": "none",
            "config_change": "none",
            "note": (
                "this report is read-only. Any response to it is an owner "
                "action under Rule 4, never an automatic consequence"
            ),
        },
        "warnings": list(warnings),
    }
    return report


def _date_gap(earlier: str | None, later: str | None):
    try:
        return (_dt.date.fromisoformat(later) - _dt.date.fromisoformat(earlier)).days
    except (TypeError, ValueError):
        return None


def _trace_summary(trace_rows) -> dict:
    rows = [r for r in trace_rows if isinstance(r, dict)]
    if not rows:
        return {"available": False, "reason": "no value_livelog trace rows on disk"}
    dates = sorted({str(r.get("asof")) for r in rows if r.get("asof")})
    matured_63 = [
        str(r.get("asof")) for r in rows
        if isinstance(r.get("ey_63d_ndates"), (int, float)) and r["ey_63d_ndates"] > 0
    ]
    return {
        "available": True,
        "rows": len(rows),
        "distinct_asof": len(dates),
        "first_asof": dates[0] if dates else None,
        "last_asof": dates[-1] if dates else None,
        "first_asof_with_63d_clusters": matured_63[0] if matured_63 else None,
        "note": (
            "the trace is the append-only record of daily readings; a date with "
            "no row is unobserved, not zero"
        ),
    }


# --------------------------------------------------------------------------
# IO (fail-open; never raises)
# --------------------------------------------------------------------------
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _latest_at_or_before(directory: Path, asof: str):
    """Newest ``{date}.json`` in ``directory`` with date <= asof (PIT)."""
    if not directory.is_dir():
        return None, None
    best = None
    try:
        entries = list(directory.iterdir())
    except OSError:
        return None, None
    for p in entries:
        if p.suffix != ".json":
            continue
        stem = p.stem
        try:
            _dt.date.fromisoformat(stem)
        except ValueError:
            continue
        if stem <= asof and (best is None or stem > best.stem):
            best = p
    if best is None:
        return None, None
    return _read_json(best), best


def _read_jsonl(path: Path, warnings: list[str], label: str):
    rows = []
    if not path.is_file():
        return rows
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"{label}_unreadable:{type(exc).__name__}")
        return rows
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            warnings.append(f"malformed_{label}_line:{path.name}:{i}")
    return rows


def load_inputs(base_dir, asof: str) -> dict:
    """Read every input this review needs. Missing input -> ``None`` + warning."""
    base = Path(base_dir)
    warnings: list[str] = []
    obs = base / "reports" / "observability"

    livelog, livelog_path = _latest_at_or_before(obs / "value_livelog", asof)
    if livelog is None:
        warnings.append("value_livelog_artifact_unavailable")
    forward_eval, fe_path = _latest_at_or_before(obs / "forward_signal_eval", asof)
    if forward_eval is None:
        warnings.append("forward_signal_eval_artifact_unavailable")

    trace_rows = _read_jsonl(obs / "value_livelog_trace.jsonl", warnings, "trace")

    # Rule 16.0 inputs come from the shared contract, loaded once here so
    # build_review stays a pure assembly of already-read inputs.
    declared_cost_model, cost_warnings = read_declared_cost_model(base)
    warnings.extend(cost_warnings)
    if declared_cost_model is None:
        warnings.append(f"declared_cost_model_absent:{COST_MODEL_REL}")

    journal_entries: list[dict] = []
    jdir = base / "reports" / "portfolio" / "journal"
    if jdir.is_dir():
        try:
            files = sorted(p for p in jdir.iterdir() if p.suffix == ".jsonl")
        except OSError:
            files = []
        for p in files:
            journal_entries.extend(_read_jsonl(p, warnings, "journal"))
    else:
        warnings.append("portfolio_journal_unavailable")

    mandate = load_mandate(base)
    if mandate is None:
        warnings.append("risk_mandate_config_unavailable")

    audit, audit_path = None, None
    cdir = base / "reports" / "calibration"
    if cdir.is_dir():
        try:
            cands = sorted(
                p for p in cdir.iterdir()
                if p.name.startswith("leakage_audit_") and p.suffix == ".json"
                and p.stem[len("leakage_audit_"):] <= asof
            )
        except OSError:
            cands = []
        if cands:
            audit_path = cands[-1]
            audit = _read_json(audit_path)
    if audit is None:
        warnings.append("leakage_audit_artifact_unavailable")

    return {
        "livelog": livelog,
        "livelog_asof": livelog_path.stem if livelog_path else None,
        "livelog_path": str(livelog_path) if livelog_path else None,
        "forward_eval": forward_eval,
        "forward_eval_asof": fe_path.stem if fe_path else None,
        "trace_rows": trace_rows,
        "journal_entries": journal_entries,
        "mandate": mandate,
        "leakage_audit": audit,
        "leakage_audit_asof": (
            audit_path.stem[len("leakage_audit_"):] if audit_path else None
        ),
        "declared_cost_model": declared_cost_model,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# rendering (ASCII only, so a cp932 console cannot turn a report into a crash)
# --------------------------------------------------------------------------
def _fmt(value, spec="{:+.4f}"):
    return spec.format(value) if isinstance(value, (int, float)) else "n/a"


def render_text(report: dict) -> str:
    rw = report["review_window"]
    tf = report["trial_family"]
    inv = report["data_inventory"]
    lines = [
        f"=== LOCKED 63D EVIDENCE REVIEW asof={report['asof']} "
        f"(P31; read-only; advice-only) ===",
        f"  review window : earliest={rw['earliest_review_date']} "
        f"reached={rw['earliest_review_date_reached']} "
        f"guaranteed_verdict_date={rw['is_guaranteed_verdict_date']}",
        f"  trial family  : {tf['n_trials_inclusive']} frozen trials inclusive "
        f"({tf['n_trials_lineage']} in the E/P hypothesis lineage), counted before any statistic",
    ]
    if not inv.get("available"):
        lines.append(f"  evidence      : UNAVAILABLE - {inv.get('reason')}")
    else:
        lines.append(
            f"  evidence      : artifact {inv.get('source_artifact_asof')} "
            f"({inv.get('raw_rows_in_log')} rows over {inv.get('trade_days')} trade days)"
        )

    for signal, block in report["signals"].items():
        locked = block.get("locked_horizon_reading") or {}
        lines.append(
            f"  [{signal}] {REVIEW_HORIZON}D verdict={block['verdict'].upper()}  "
            f"IC={_fmt(locked.get('mean_ic'))} t={_fmt(locked.get('t_stat'), '{:+.2f}')} "
            f"clusters={locked.get('independent_date_clusters')} "
            f"matured={locked.get('matured')} unmatured={locked.get('unmatured')}"
        )
        for c in block["checks"]:
            if c["status"] != "pass":
                lines.append(f"      - {c['check']}: {c['status'].upper()} - {c['detail']}")

    watch = report["sign_reversal_watch"]
    for o in watch["observations"]:
        if o["established_at_harvey_bar"]:
            lines.append(
                f"  ! SIGN REVERSAL {o['signal']} @ {o['horizon']}D "
                f"IC={_fmt(o['mean_ic'])} t={_fmt(o['t_stat'], '{:+.2f}')} "
                f"(declared direction: {o['declared_direction']})"
            )
    lines.append(
        f"  composite     : not emitted - {report['composite']['reason']}"
    )

    conf = report["confirmability"]
    lines.append(
        f"  confirmable   : {conf['confirm_reachable']}"
        + (f"  blocked by: {', '.join(conf['blocking_checks'])}" if conf["blocking_checks"] else "")
    )

    dep = report["deployment"]
    v = report["verdicts"]
    lines.append(
        f"  deployment    : {v['deployment_verdict'].upper()}  "
        f"sleeve_B fills={dep.get('sleeve_b_fill_count')} "
        f"held={dep.get('sleeve_b_symbols_held')}"
    )
    pre = report["precommitment"]
    if pre.get("available"):
        lines.append(
            f"  precommitment : on_fail={pre.get('on_fail')} "
            f"operative={pre.get('on_fail_operative')} - {pre.get('on_fail_note')}"
        )
    lines.append(
        f"  VERDICTS      : signal={v['signal_verdict'].upper()} "
        f"deployment={v['deployment_verdict'].upper()}   "
        f"(insufficient = evidence absent, not a failure and not a zero)"
    )
    lines.append("  no capital and no config change follows from this report (Rule 3).")
    for w in report.get("warnings", []):
        lines.append(f"  WARNING {w}")
    return "\n".join(lines)


def main(argv=None) -> int:

    # Data-sourced text (rule titles, theses) may be Japanese; degrade rather
    # than die mid-print on a cp932 console.
    enable_console_fallback()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--asof", default=None, help="ISO date stamp for the artifact (default: today).")
    ap.add_argument("--base-dir", default=str(ROOT))
    ap.add_argument("--no-write", action="store_true", help="Print only; do not write artifacts.")
    args = ap.parse_args(argv)
    asof = args.asof or _dt.date.today().isoformat()

    try:
        loaded = load_inputs(args.base_dir, asof)
    except Exception as exc:  # fail-open: a broken input is a reportable state
        print(f"evidence review unavailable (input load failed: {type(exc).__name__}) asof={asof}")
        return 0

    try:
        report = build_review(
            asof=asof,
            livelog=loaded["livelog"],
            livelog_asof=loaded["livelog_asof"],
            livelog_path=loaded["livelog_path"],
            trace_rows=loaded["trace_rows"],
            forward_eval=loaded["forward_eval"],
            forward_eval_asof=loaded["forward_eval_asof"],
            journal_entries=loaded["journal_entries"],
            mandate=loaded["mandate"],
            leakage_audit=loaded["leakage_audit"],
            leakage_audit_asof=loaded["leakage_audit_asof"],
            declared_cost_model=loaded["declared_cost_model"],
            warnings=loaded["warnings"],
        )
    except Exception as exc:  # fail-open
        print(f"evidence review unavailable (assembly failed: {type(exc).__name__}) asof={asof}")
        return 0

    print(render_text(report))

    if not args.no_write:
        out_dir = Path(args.base_dir) / "reports" / "observability" / "evidence_review_63d"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{asof}.json"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {path}")
        except OSError as exc:
            print(f"artifact not written ({type(exc).__name__}); the reading above still stands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
