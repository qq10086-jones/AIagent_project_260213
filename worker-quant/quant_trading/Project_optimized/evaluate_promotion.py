from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


PATCH_NOTE = (
    "Sharpe is treated as a governance metric and promotion gate in this patch. "
    "It is not used here as a standalone regression target."
)

MODE_FACTOR_FAMILIES = {
    "ridge": {
        "technical": ["mom_consist", "rsi14", "vol_adj_mom20", "ret20", "slope60"],
        "risk_adjusted": [],
        "fundamental": [],
    },
    "shadow_eq": {
        "technical": ["mom_consist", "rsi14", "vol_adj_mom20", "ret20", "slope60"],
        "risk_adjusted": [],
        "fundamental": [],
    },
    "shadow_ic": {
        "technical": ["mom_consist", "rsi14", "vol_adj_mom20", "ret20", "slope60"],
        "risk_adjusted": [],
        "fundamental": [],
    },
    "shadow_hybrid_ic": {
        "technical": ["mom_consist", "rsi14", "vol_adj_mom20", "ret20", "slope60"],
        "risk_adjusted": ["sharpe_20", "sharpe_60", "sortino_60", "vol_stability"],
        "fundamental": [
            "value_bp",
            "quality_roe",
            "quality_cfo",
            "margin_op",
            "growth_rev_yoy",
            "growth_op_yoy",
            "guidance_delta",
            "leverage_safety",
            "dividend_yield",
        ],
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def _load_mode_from_snapshot(snapshot_path: str | None) -> str | None:
    if not snapshot_path:
        return None
    p = Path(snapshot_path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_outputs = payload.get("model_outputs", {})
    target_meta = model_outputs.get("target_weights_meta", {})
    mode = target_meta.get("primary_mode")
    return str(mode) if mode else None


def _parse_learning_notes(notes: str | None) -> dict[str, Any]:
    text = str(notes or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    out: dict[str, Any] = {}
    match_t = re.search(r"t=([0-9.+-]+)", text)
    match_n = re.search(r"\bn=([0-9]+)", text)
    match_xs = re.search(r"xs_n=([0-9.+-]+)", text)
    match_guard = re.search(r"guard=(PASS|FAIL)", text)
    if match_t:
        out["t_stat"] = _safe_float(match_t.group(1))
    if match_n:
        out["n_periods"] = int(match_n.group(1))
    if match_xs:
        out["avg_cross_section_n"] = _safe_float(match_xs.group(1))
    if match_guard:
        out["guard"] = match_guard.group(1)
    return out


def _paper_mode_stats(db_path: str, target_mode: str) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        run_rows = conn.execute(
            """
            SELECT run_id, asof, status, snapshot_path
            FROM decision_runs
            ORDER BY asof, ts
            """
        ).fetchall()
        fill_rows = conn.execute(
            """
            SELECT DISTINCT run_id, asof
            FROM fills
            WHERE source='paper_simulator'
            """
        ).fetchall()
        snapshot_rows = conn.execute(
            """
            SELECT DISTINCT run_id, asof
            FROM account_snapshots
            WHERE run_id IS NOT NULL
            """
        ).fetchall()

    filled_by_run = {str(run_id): str(asof) for run_id, asof in fill_rows}
    snap_by_run = {str(run_id): str(asof) for run_id, asof in snapshot_rows}
    matched_runs = []
    paper_days = set()
    for run_id, asof, status, snapshot_path in run_rows:
        mode = _load_mode_from_snapshot(snapshot_path)
        if mode != target_mode:
            continue
        run_id = str(run_id)
        has_fill = run_id in filled_by_run
        has_snapshot = run_id in snap_by_run
        is_papered = has_fill or has_snapshot
        if is_papered:
            paper_days.add(str(filled_by_run.get(run_id) or snap_by_run.get(run_id) or asof))
        matched_runs.append(
            {
                "run_id": run_id,
                "asof": str(asof),
                "status": str(status) if status is not None else None,
                "paper_executed": is_papered,
                "has_fills": has_fill,
                "has_snapshot": has_snapshot,
            }
        )
    return {
        "mode": target_mode,
        "decision_run_count": len(matched_runs),
        "paper_days": len(paper_days),
        "paper_asofs": sorted(paper_days),
        "recent_runs": matched_runs[-10:],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _learning_gate_stats(db_path: str, target_mode: str) -> dict[str, Any]:
    families = MODE_FACTOR_FAMILIES.get(target_mode, MODE_FACTOR_FAMILIES["ridge"])
    selected_factors = []
    for family_name in ["technical", "risk_adjusted", "fundamental"]:
        selected_factors.extend(families.get(family_name, []))
    if not selected_factors:
        return {
            "mode": target_mode,
            "factor_count": 0,
            "family_metrics": {},
            "aggregate": {
                "mean_ic": 0.0,
                "mean_t_stat": 0.0,
                "positive_ic_ratio": 0.0,
                "min_family_t_stat": 0.0,
            },
        }

    placeholders = ",".join("?" for _ in selected_factors)
    with sqlite3.connect(db_path) as conn:
        registry_rows = conn.execute(
            f"""
            SELECT factor_name, ic_mean, ic_std, icir, weight, n_observations
            FROM factor_registry
            WHERE factor_name IN ({placeholders})
            """,
            selected_factors,
        ).fetchall()
        audit_rows = conn.execute(
            f"""
            SELECT factor_name, notes, created_at
            FROM learning_audit
            WHERE factor_name IN ({placeholders})
            ORDER BY created_at DESC, id DESC
            """,
            selected_factors,
        ).fetchall()

    registry_by_factor = {
        str(factor_name): {
            "ic_mean": _safe_float(ic_mean),
            "ic_std": _safe_float(ic_std, 1.0),
            "icir": _safe_float(icir),
            "weight": _safe_float(weight),
            "n_observations": int(n_observations or 0),
        }
        for factor_name, ic_mean, ic_std, icir, weight, n_observations in registry_rows
    }
    latest_audit_by_factor: dict[str, dict[str, Any]] = {}
    for factor_name, notes, created_at in audit_rows:
        key = str(factor_name)
        if key in latest_audit_by_factor:
            continue
        parsed = _parse_learning_notes(notes)
        parsed["created_at"] = str(created_at) if created_at is not None else None
        latest_audit_by_factor[key] = parsed

    family_metrics: dict[str, Any] = {}
    aggregate_ic = []
    aggregate_t = []
    positive_ic_flags = []
    for family_name, family_factors in families.items():
        rows = []
        for factor in family_factors:
            reg = registry_by_factor.get(factor, {})
            audit = latest_audit_by_factor.get(factor, {})
            ic_mean = _safe_float(reg.get("ic_mean"))
            t_stat = _safe_float(audit.get("t_stat"))
            rows.append(
                {
                    "factor_name": factor,
                    "ic_mean": ic_mean,
                    "t_stat": t_stat,
                    "guard": audit.get("guard"),
                    "n_observations": int(reg.get("n_observations", 0) or 0),
                }
            )
            if reg:
                aggregate_ic.append(ic_mean)
                positive_ic_flags.append(1.0 if ic_mean > 0.0 else 0.0)
            if audit:
                aggregate_t.append(t_stat)
        if not rows:
            continue
        ic_vals = [row["ic_mean"] for row in rows if row["n_observations"] > 0]
        t_vals = [row["t_stat"] for row in rows if row["t_stat"] > 0.0]
        family_metrics[family_name] = {
            "factor_count": len(rows),
            "observed_factor_count": int(sum(1 for row in rows if row["n_observations"] > 0)),
            "mean_ic": float(sum(ic_vals) / len(ic_vals)) if ic_vals else 0.0,
            "mean_t_stat": float(sum(t_vals) / len(t_vals)) if t_vals else 0.0,
            "positive_ic_ratio": float(sum(1.0 for row in rows if row["ic_mean"] > 0.0) / len(rows)),
            "factors": rows,
        }

    positive_ic_ratio = float(sum(positive_ic_flags) / len(positive_ic_flags)) if positive_ic_flags else 0.0
    min_family_t_stat = min(
        (
            metrics["mean_t_stat"]
            for metrics in family_metrics.values()
            if metrics.get("observed_factor_count", 0) > 0
        ),
        default=0.0,
    )
    return {
        "mode": target_mode,
        "factor_count": len(selected_factors),
        "family_metrics": family_metrics,
        "aggregate": {
            "mean_ic": float(sum(aggregate_ic) / len(aggregate_ic)) if aggregate_ic else 0.0,
            "mean_t_stat": float(sum(aggregate_t) / len(aggregate_t)) if aggregate_t else 0.0,
            "positive_ic_ratio": positive_ic_ratio,
            "min_family_t_stat": float(min_family_t_stat),
        },
    }


def evaluate(compare_df: pd.DataFrame, target_mode: str, baseline_mode: str, db_path: str, reports_dir: Path, thresholds: dict[str, Any]) -> dict[str, Any]:
    if "mode" not in compare_df.columns:
        raise ValueError("signal_mode_compare.csv must contain a mode column")
    if target_mode not in set(compare_df["mode"].astype(str)):
        raise ValueError(f"Target mode {target_mode} not found in signal_mode_compare.csv")

    target_row = compare_df.loc[compare_df["mode"].astype(str) == target_mode].iloc[0]
    baseline_row = compare_df.loc[compare_df["mode"].astype(str) == baseline_mode].iloc[0] if baseline_mode in set(compare_df["mode"].astype(str)) else None
    paper_stats = _paper_mode_stats(db_path, target_mode)
    learning_stats = _learning_gate_stats(db_path, target_mode)
    compare_report = _load_json(reports_dir / "signal_mode_compare_report.json")
    zero_report = _load_json(reports_dir / "zero_exposure_report.json")

    target_sharpe = _safe_float(target_row.get("sharpe"))
    target_sortino = _safe_float(target_row.get("sortino"))
    target_dd = _safe_float(target_row.get("max_drawdown_pct"))
    target_turnover_cv = _safe_float(target_row.get("turnover_cv"))
    target_avg_turnover = _safe_float(target_row.get("avg_turnover_notional"))
    baseline_sharpe = _safe_float(baseline_row.get("sharpe")) if baseline_row is not None else 0.0

    min_backtest_sharpe = _safe_float(thresholds.get("min_backtest_sharpe"), 0.0)
    max_drawdown_pct = _safe_float(thresholds.get("max_drawdown_pct"), 999.0)
    paper_days_required = int(thresholds.get("paper_days_required", 0) or 0)
    min_sharpe_improvement = _safe_float(thresholds.get("min_sharpe_improvement"), 0.0)
    min_production_ic = _safe_float(thresholds.get("min_production_ic"), 0.0)
    min_t_stat = _safe_float(thresholds.get("min_t_stat"), 0.0)
    min_eligible_factors = int(thresholds.get("min_eligible_factors", 0) or 0)
    max_zero_exposure_days = int(thresholds.get("max_zero_exposure_days", 9999) or 9999)
    require_actionable_mode = bool(int(thresholds.get("require_actionable_mode", 1) or 0))
    max_turnover_cv = thresholds.get("max_turnover_cv")
    max_turnover_cv = None if max_turnover_cv is None else _safe_float(max_turnover_cv, 0.0)
    eligible_factor_count = 0
    for metrics in learning_stats.get("family_metrics", {}).values():
        for row in metrics.get("factors", []):
            if str(row.get("guard")) == "PASS" and int(row.get("n_observations", 0) or 0) >= 80:
                eligible_factor_count += 1
    actionable_mode_count = int(compare_report.get("summary", {}).get("actionable_mode_count", 0) or 0)
    latest_weights_zero = bool(zero_report.get("latest_weights_zero", False))
    zero_days = int(zero_report.get("days_since_last_nonzero", 0) or 0) if latest_weights_zero else 0

    gates = {
        "production_ic": {
            "passed": _safe_float(learning_stats["aggregate"].get("mean_ic")) >= min_production_ic,
            "actual": _safe_float(learning_stats["aggregate"].get("mean_ic")),
            "threshold": min_production_ic,
        },
        "t_stat": {
            "passed": _safe_float(learning_stats["aggregate"].get("min_family_t_stat")) >= min_t_stat,
            "actual": _safe_float(learning_stats["aggregate"].get("min_family_t_stat")),
            "threshold": min_t_stat,
        },
        "backtest_sharpe": {
            "passed": target_sharpe >= min_backtest_sharpe,
            "actual": target_sharpe,
            "threshold": min_backtest_sharpe,
        },
        "max_drawdown": {
            "passed": target_dd <= max_drawdown_pct,
            "actual": target_dd,
            "threshold": max_drawdown_pct,
        },
        "sharpe_vs_baseline": {
            "passed": (target_sharpe - baseline_sharpe) >= min_sharpe_improvement,
            "actual": target_sharpe - baseline_sharpe,
            "threshold": min_sharpe_improvement,
            "baseline_mode": baseline_mode,
            "baseline_sharpe": baseline_sharpe,
        },
        "paper_days": {
            "passed": int(paper_stats["paper_days"]) >= paper_days_required,
            "actual": int(paper_stats["paper_days"]),
            "threshold": paper_days_required,
        },
        "eligible_factors": {
            "passed": eligible_factor_count >= min_eligible_factors,
            "actual": eligible_factor_count,
            "threshold": min_eligible_factors,
        },
        "zero_exposure_window": {
            "passed": (not latest_weights_zero) or zero_days < max_zero_exposure_days,
            "actual": zero_days,
            "threshold": max_zero_exposure_days,
            "latest_weights_zero": latest_weights_zero,
        },
        "actionable_mode_available": {
            "passed": (actionable_mode_count > 0) if require_actionable_mode else True,
            "actual": actionable_mode_count,
            "threshold": 1 if require_actionable_mode else 0,
        },
    }
    if max_turnover_cv is None:
        gates["turnover_stability"] = {
            "passed": True,
            "actual": target_turnover_cv,
            "threshold": None,
            "note": "No max_turnover_cv configured; reported only.",
        }
    else:
        gates["turnover_stability"] = {
            "passed": target_avg_turnover > 0.0 and target_turnover_cv <= max_turnover_cv,
            "actual": target_turnover_cv,
            "threshold": max_turnover_cv,
            "avg_turnover_notional": target_avg_turnover,
        }

    eligible = all(bool(v["passed"]) for v in gates.values())
    return {
        "target_mode": target_mode,
        "baseline_mode": baseline_mode,
        "recommendation": "eligible_for_promotion" if eligible else "hold",
        "patch_note": PATCH_NOTE,
        "target_metrics": {
            "sharpe": target_sharpe,
            "sortino": target_sortino,
            "max_drawdown_pct": target_dd,
            "avg_turnover_notional": target_avg_turnover,
            "turnover_cv": target_turnover_cv,
            "total_return_pct": _safe_float(target_row.get("total_return_pct")),
            "annual_vol_pct": _safe_float(target_row.get("annual_vol_pct")),
        },
        "paper_stats": paper_stats,
        "learning_stats": learning_stats,
        "qa_stats": {
            "eligible_factor_count": eligible_factor_count,
            "actionable_mode_count": actionable_mode_count,
            "latest_zero_exposure_days": zero_days,
            "latest_weights_zero": latest_weights_zero,
        },
        "gates": gates,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--target_mode", required=True)
    ap.add_argument("--baseline_mode", default="ridge")
    ap.add_argument("--min_backtest_sharpe", type=float, default=1.0)
    ap.add_argument("--max_drawdown_pct", type=float, default=20.0)
    ap.add_argument("--paper_days_required", type=int, default=20)
    ap.add_argument("--min_sharpe_improvement", type=float, default=0.0)
    ap.add_argument("--min_production_ic", type=float, default=0.0)
    ap.add_argument("--min_t_stat", type=float, default=1.5)
    ap.add_argument("--max_turnover_cv", type=float, default=None)
    ap.add_argument("--min_eligible_factors", type=int, default=3)
    ap.add_argument("--max_zero_exposure_days", type=int, default=3)
    ap.add_argument("--require_actionable_mode", type=int, default=1)
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    compare_path = reports_dir / "signal_mode_compare.csv"
    if not compare_path.exists():
        raise FileNotFoundError(f"signal_mode_compare.csv not found at {compare_path}")
    compare_df = pd.read_csv(compare_path)

    result = evaluate(
        compare_df=compare_df,
        target_mode=args.target_mode,
        baseline_mode=args.baseline_mode,
        db_path=args.db,
        reports_dir=reports_dir,
        thresholds={
            "min_backtest_sharpe": args.min_backtest_sharpe,
            "max_drawdown_pct": args.max_drawdown_pct,
            "paper_days_required": args.paper_days_required,
            "min_sharpe_improvement": args.min_sharpe_improvement,
            "min_production_ic": args.min_production_ic,
            "min_t_stat": args.min_t_stat,
            "max_turnover_cv": args.max_turnover_cv,
            "min_eligible_factors": args.min_eligible_factors,
            "max_zero_exposure_days": args.max_zero_exposure_days,
            "require_actionable_mode": args.require_actionable_mode,
        },
    )

    out_json = reports_dir / "promotion_decision.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "promotion_note.txt").write_text(PATCH_NOTE + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
