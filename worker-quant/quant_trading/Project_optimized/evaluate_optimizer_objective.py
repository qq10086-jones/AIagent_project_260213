from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(reports_dir: Path, target_mode: str) -> dict:
    promotion = {}
    promotion_path = reports_dir / "promotion_decision.json"
    if promotion_path.exists():
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    factor_health = {}
    factor_health_path = reports_dir / "factor_health_report.json"
    if factor_health_path.exists():
        factor_health = json.loads(factor_health_path.read_text(encoding="utf-8"))

    sharpe = float(promotion.get("target_metrics", {}).get("sharpe", 0.0) or 0.0)
    turnover_cv = float(promotion.get("target_metrics", {}).get("turnover_cv", 0.0) or 0.0)
    mean_ic = float(factor_health.get("aggregate_learning", {}).get("mean_ic", 0.0) or 0.0)
    mean_t_stat = float(factor_health.get("aggregate_learning", {}).get("mean_t_stat", 0.0) or 0.0)

    ready = sharpe >= 1.0 and mean_ic > 0.02 and mean_t_stat >= 1.5 and turnover_cv <= 1.5
    report = {
        "target_mode": target_mode,
        "recommendation": "defer_optimizer_sharpe_objective" if not ready else "candidate_for_optimizer_sharpe_objective_trial",
        "rationale": {
            "sharpe": sharpe,
            "mean_ic": mean_ic,
            "mean_t_stat": mean_t_stat,
            "turnover_cv": turnover_cv,
        },
        "decision_rule": {
            "required_sharpe": 1.0,
            "required_mean_ic": 0.02,
            "required_mean_t_stat": 1.5,
            "max_turnover_cv": 1.5,
        },
    }
    (reports_dir / "optimizer_objective_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "optimizer_objective_evaluation.md").write_text(
        "\n".join(
            [
                "# Optimizer Objective Evaluation",
                "",
                f"- Target mode: {target_mode}",
                f"- Recommendation: {report['recommendation']}",
                f"- Sharpe: {sharpe:.4f}",
                f"- Mean IC: {mean_ic:.4f}",
                f"- Mean t-stat: {mean_t_stat:.4f}",
                f"- Turnover CV: {turnover_cv:.4f}",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--target_mode", required=True)
    args = ap.parse_args()
    report = evaluate(Path(args.reports_dir), args.target_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
