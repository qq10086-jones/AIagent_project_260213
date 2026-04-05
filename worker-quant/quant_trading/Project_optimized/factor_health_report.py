from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compute_ic import load_factor_tiers, load_promotion_rules
from evaluate_promotion import MODE_FACTOR_FAMILIES, _learning_gate_stats, _safe_float


MIN_PRODUCTION_OBS = 80


def build_report(db_path: str, reports_dir: Path, target_mode: str) -> dict:
    learning_stats = _learning_gate_stats(db_path, target_mode)

    compare_path = reports_dir / "signal_mode_compare.csv"
    compare_df = pd.read_csv(compare_path) if compare_path.exists() else pd.DataFrame()
    compare_row = (
        compare_df.loc[compare_df["mode"].astype(str) == str(target_mode)].iloc[0].to_dict()
        if (not compare_df.empty and target_mode in set(compare_df["mode"].astype(str)))
        else {}
    )

    promotion_path = reports_dir / "promotion_decision.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8")) if promotion_path.exists() else {}
    target_metrics = promotion.get("target_metrics", {})

    family_rows = []
    for family_name, metrics in learning_stats.get("family_metrics", {}).items():
        factor_count = int(metrics.get("factor_count", 0))
        observed_count = int(metrics.get("observed_factor_count", 0))
        family_rows.append(
            {
                "family": family_name,
                "factor_count": factor_count,
                "observed_factor_count": observed_count,
                "mean_ic": _safe_float(metrics.get("mean_ic")),
                "mean_t_stat": _safe_float(metrics.get("mean_t_stat")),
                "positive_ic_ratio": _safe_float(metrics.get("positive_ic_ratio")),
                "coverage_ratio": 0.0 if factor_count <= 0 else observed_count / factor_count,
            }
        )

    family_df = pd.DataFrame(family_rows).sort_values("family") if family_rows else pd.DataFrame(
        columns=["family", "factor_count", "observed_factor_count", "mean_ic", "mean_t_stat", "positive_ic_ratio", "coverage_ratio"]
    )

    tier_map = load_factor_tiers()
    promo_rules = load_promotion_rules()
    min_promo_obs = int(promo_rules.get("min_observations", 100))
    min_promo_t = float(promo_rules.get("min_t_stat", 1.5))
    min_promo_ic = float(promo_rules.get("min_ic_mean", 0.01))
    demote_t = float(promo_rules.get("demotion_t_stat", 0.5))

    factor_rows = []
    for family_name in ["technical", "risk_adjusted", "fundamental"]:
        metrics = learning_stats.get("family_metrics", {}).get(family_name, {})
        for row in metrics.get("factors", []):
            n_obs = int(row.get("n_observations", 0) or 0)
            guard = row.get("guard")
            factor_name = row.get("factor_name", "")
            tier = tier_map.get(factor_name, "candidate")
            ic_mean = _safe_float(row.get("ic_mean"))
            t_stat_val = _safe_float(row.get("t_stat"))

            # promotion_eligible: candidate/fundamental_pending that meets all promotion criteria
            promotion_eligible = (
                tier in ("candidate", "fundamental_pending")
                and n_obs >= min_promo_obs
                and t_stat_val >= min_promo_t
                and abs(ic_mean) >= min_promo_ic
            )
            # demotion_risk: core factor with t-stat below demotion threshold
            demotion_risk = (
                tier == "core"
                and n_obs >= min_promo_obs
                and t_stat_val < demote_t
            )

            factor_rows.append(
                {
                    "family": family_name,
                    "factor_name": factor_name,
                    "tier": tier,
                    "ic_mean": ic_mean,
                    "t_stat": t_stat_val,
                    "guard": guard,
                    "n_observations": n_obs,
                    "production_eligible": bool(str(guard) == "PASS" and n_obs >= MIN_PRODUCTION_OBS),
                    "promotion_eligible": promotion_eligible,
                    "demotion_risk": demotion_risk,
                }
            )
    factor_df = pd.DataFrame(factor_rows).sort_values(["family", "factor_name"]) if factor_rows else pd.DataFrame(
        columns=["family", "factor_name", "tier", "ic_mean", "t_stat", "guard", "n_observations",
                 "production_eligible", "promotion_eligible", "demotion_risk"]
    )

    eligible_factors = (
        factor_df.loc[factor_df["production_eligible"].astype(bool), "factor_name"].astype(str).tolist()
        if not factor_df.empty
        else []
    )
    cleanup_df = (
        factor_df.assign(
            recommended_action=lambda df_: df_["production_eligible"].map(
                lambda ok: "keep" if bool(ok) else "exclude_from_production"
            ),
            exclusion_reason=lambda df_: df_.apply(
                lambda row: (
                    "insufficient_observations_and_guard_fail"
                    if int(row["n_observations"]) < MIN_PRODUCTION_OBS and str(row["guard"]) != "PASS"
                    else "insufficient_observations"
                    if int(row["n_observations"]) < MIN_PRODUCTION_OBS
                    else "guard_fail"
                )
                if not bool(row["production_eligible"])
                else "eligible"
                ,
                axis=1,
            ),
        )
        if not factor_df.empty
        else pd.DataFrame(
            columns=[
                "family",
                "factor_name",
                "guard",
                "n_observations",
                "production_eligible",
                "recommended_action",
                "exclusion_reason",
            ]
        )
    )

    report = {
        "target_mode": target_mode,
        "families_expected": MODE_FACTOR_FAMILIES.get(target_mode, MODE_FACTOR_FAMILIES.get("ridge", {})),
        "aggregate_learning": learning_stats.get("aggregate", {}),
        "production_eligibility_rule": {
            "guard_must_pass": True,
            "min_observations": MIN_PRODUCTION_OBS,
        },
        "production_eligible_factors": eligible_factors,
        "cleanup_summary": {
            "eligible_count": int(len(eligible_factors)),
            "exclude_count": int((~cleanup_df["production_eligible"].astype(bool)).sum()) if not cleanup_df.empty else 0,
            "exclude_candidates": (
                cleanup_df.loc[~cleanup_df["production_eligible"].astype(bool), "factor_name"].astype(str).tolist()
                if not cleanup_df.empty
                else []
            ),
        },
        "backtest_metrics": {
            "total_return_pct": _safe_float(compare_row.get("total_return_pct")),
            "annual_vol_pct": _safe_float(compare_row.get("annual_vol_pct")),
            "sharpe": _safe_float(compare_row.get("sharpe", target_metrics.get("sharpe"))),
            "sortino": _safe_float(compare_row.get("sortino", target_metrics.get("sortino"))),
            "max_drawdown_pct": _safe_float(compare_row.get("max_drawdown_pct", target_metrics.get("max_drawdown_pct"))),
            "avg_turnover_notional": _safe_float(compare_row.get("avg_turnover_notional")),
            "avg_turnover_pct": _safe_float(compare_row.get("avg_turnover_pct")),
            "turnover_cv": _safe_float(compare_row.get("turnover_cv")),
            "avg_cost_paid": _safe_float(compare_row.get("avg_cost_paid")),
        },
        "promotion_recommendation": promotion.get("recommendation"),
        "qa_stats": promotion.get("qa_stats", {}),
    }

    out_json = reports_dir / "factor_health_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    family_df.to_csv(reports_dir / "factor_health_families.csv", index=False)
    factor_df.to_csv(reports_dir / "factor_health_factors.csv", index=False)
    cleanup_df.to_csv(reports_dir / "factor_registry_cleanup_candidates.csv", index=False)
    (reports_dir / "factor_registry_cleanup_report.json").write_text(
        json.dumps(
            {
                "target_mode": target_mode,
                "eligibility_rule": report["production_eligibility_rule"],
                "summary": report["cleanup_summary"],
                "rows": cleanup_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Factor Health Report: {target_mode}",
        "",
        f"- Promotion recommendation: {promotion.get('recommendation', 'unknown')}",
        f"- Mean IC: {_safe_float(learning_stats.get('aggregate', {}).get('mean_ic')):.4f}",
        f"- Mean t-stat: {_safe_float(learning_stats.get('aggregate', {}).get('mean_t_stat')):.4f}",
        f"- Sharpe: {_safe_float(compare_row.get('sharpe', target_metrics.get('sharpe'))):.4f}",
        f"- Max drawdown %: {_safe_float(compare_row.get('max_drawdown_pct', target_metrics.get('max_drawdown_pct'))):.2f}",
        f"- Avg turnover %: {_safe_float(compare_row.get('avg_turnover_pct')):.2f}",
        f"- Production-eligible factors ({MIN_PRODUCTION_OBS}+ obs + PASS): {', '.join(eligible_factors) if eligible_factors else 'none'}",
        f"- QA eligible factor count: {int(report.get('qa_stats', {}).get('eligible_factor_count', 0) or 0)}",
        f"- QA actionable mode count: {int(report.get('qa_stats', {}).get('actionable_mode_count', 0) or 0)}",
        f"- QA latest zero exposure days: {int(report.get('qa_stats', {}).get('latest_zero_exposure_days', 0) or 0)}",
        "",
        "## Family Summary",
        "",
    ]
    if family_df.empty:
        lines.append("No family metrics available.")
    else:
        for _, row in family_df.iterrows():
            lines.append(
                f"- {row['family']}: IC={row['mean_ic']:.4f}, t={row['mean_t_stat']:.4f}, "
                f"coverage={row['observed_factor_count']}/{row['factor_count']}, positive_ic_ratio={row['positive_ic_ratio']:.2f}"
            )
    lines.extend(["", "## Factor Eligibility", ""])
    if factor_df.empty:
        lines.append("No factor-level metrics available.")
    else:
        for _, row in factor_df.iterrows():
            flags = []
            if row.get("promotion_eligible"):
                flags.append("PROMO_READY")
            if row.get("demotion_risk"):
                flags.append("DEMOTE_RISK")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"- {row['factor_name']} | tier={row.get('tier', '?')} | family={row['family']} | guard={row['guard']} | "
                f"obs={int(row['n_observations'])} | eligible={bool(row['production_eligible'])}{flag_str}"
            )
    (reports_dir / "factor_health_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cleanup_lines = [
        f"# Factor Registry Cleanup Candidates: {target_mode}",
        "",
        f"- Eligible count: {report['cleanup_summary']['eligible_count']}",
        f"- Exclude count: {report['cleanup_summary']['exclude_count']}",
        "",
    ]
    if cleanup_df.empty:
        cleanup_lines.append("No factor-level metrics available.")
    else:
        for _, row in cleanup_df.iterrows():
            cleanup_lines.append(
                f"- {row['factor_name']} | action={row['recommended_action']} | reason={row['exclusion_reason']} | "
                f"guard={row['guard']} | obs={int(row['n_observations'])}"
            )
    (reports_dir / "factor_registry_cleanup_report.md").write_text("\n".join(cleanup_lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--target_mode", required=True)
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.db, reports_dir, args.target_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
