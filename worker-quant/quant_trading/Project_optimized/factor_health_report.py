from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_promotion import MODE_FACTOR_FAMILIES, _learning_gate_stats, _safe_float


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

    factor_rows = []
    for family_name in ["technical", "risk_adjusted", "fundamental"]:
        metrics = learning_stats.get("family_metrics", {}).get(family_name, {})
        for row in metrics.get("factors", []):
            factor_rows.append(
                {
                    "family": family_name,
                    "factor_name": row.get("factor_name"),
                    "ic_mean": _safe_float(row.get("ic_mean")),
                    "t_stat": _safe_float(row.get("t_stat")),
                    "guard": row.get("guard"),
                    "n_observations": int(row.get("n_observations", 0) or 0),
                }
            )
    factor_df = pd.DataFrame(factor_rows).sort_values(["family", "factor_name"]) if factor_rows else pd.DataFrame(
        columns=["family", "factor_name", "ic_mean", "t_stat", "guard", "n_observations"]
    )

    report = {
        "target_mode": target_mode,
        "families_expected": MODE_FACTOR_FAMILIES.get(target_mode, MODE_FACTOR_FAMILIES.get("ridge", {})),
        "aggregate_learning": learning_stats.get("aggregate", {}),
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
    }

    out_json = reports_dir / "factor_health_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    family_df.to_csv(reports_dir / "factor_health_families.csv", index=False)
    factor_df.to_csv(reports_dir / "factor_health_factors.csv", index=False)

    lines = [
        f"# Factor Health Report: {target_mode}",
        "",
        f"- Promotion recommendation: {promotion.get('recommendation', 'unknown')}",
        f"- Mean IC: {_safe_float(learning_stats.get('aggregate', {}).get('mean_ic')):.4f}",
        f"- Mean t-stat: {_safe_float(learning_stats.get('aggregate', {}).get('mean_t_stat')):.4f}",
        f"- Sharpe: {_safe_float(compare_row.get('sharpe', target_metrics.get('sharpe'))):.4f}",
        f"- Max drawdown %: {_safe_float(compare_row.get('max_drawdown_pct', target_metrics.get('max_drawdown_pct'))):.2f}",
        f"- Avg turnover %: {_safe_float(compare_row.get('avg_turnover_pct')):.2f}",
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
    (reports_dir / "factor_health_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
