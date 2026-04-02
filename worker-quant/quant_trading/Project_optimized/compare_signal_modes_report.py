from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_mode_meta(reports_dir: Path, mode: str) -> dict:
    path = reports_dir / f"target_weights_{mode}_meta.json"
    if not path.exists() and mode == "ridge":
        path = reports_dir / "target_weights_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_zero_exposure_report(reports_dir: Path) -> dict:
    path = reports_dir / "zero_exposure_report.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_report(reports_dir: Path) -> dict:
    compare_path = reports_dir / "signal_mode_compare.csv"
    if not compare_path.exists():
        raise FileNotFoundError(f"Missing {compare_path}")
    df = pd.read_csv(compare_path)
    if df.empty:
        raise RuntimeError("signal_mode_compare.csv is empty")

    if "mode" not in df.columns:
        raise RuntimeError("signal_mode_compare.csv must contain mode")
    for col in ["total_return_pct", "max_drawdown_pct", "sharpe", "sortino", "avg_turnover_pct", "turnover_cv"]:
        if col not in df.columns:
            df[col] = 0.0

    df["score"] = (
        df["sharpe"].fillna(0.0) * 3.0
        + df["sortino"].fillna(0.0) * 1.0
        + df["total_return_pct"].fillna(0.0) * 0.05
        - df["max_drawdown_pct"].fillna(0.0) * 0.08
        - df["avg_turnover_pct"].fillna(0.0) * 0.03
        - df["turnover_cv"].fillna(0.0) * 0.5
    )

    meta_rows = []
    for mode in df["mode"].astype(str):
        meta = _load_mode_meta(reports_dir, mode)
        meta_rows.append(
            {
                "mode": mode,
                "current_weights_zero": bool(meta.get("history_last_row_is_zero", False)),
                "last_nonzero_asof": meta.get("last_nonzero_asof"),
                "last_nonzero_row_sum": float(meta.get("last_nonzero_row_sum", 0.0) or 0.0),
                "export_row_sum": float(meta.get("export_row_sum", 0.0) or 0.0),
                "actionable_now": not bool(meta.get("history_last_row_is_zero", False)),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    if not meta_df.empty:
        df = df.merge(meta_df, on="mode", how="left")
    else:
        df["current_weights_zero"] = False
        df["last_nonzero_asof"] = None
        df["last_nonzero_row_sum"] = 0.0
        df["export_row_sum"] = 0.0
        df["actionable_now"] = True

    ranked = df.sort_values(["score", "sharpe", "total_return_pct"], ascending=[False, False, False]).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()

    actionable_count = int(df["actionable_now"].fillna(False).astype(bool).sum())
    all_zero_now = bool((~df["actionable_now"].fillna(False).astype(bool)).all())
    zero_exposure = _load_zero_exposure_report(reports_dir)
    zero_cause = str(zero_exposure.get("primary_cause", "") or "")
    report = {
        "recommended_mode": str(best["mode"]),
        "recommendation_note": (
            (
                f"All compared modes currently export zero target weights. Shared latest cause: {zero_cause}."
                if zero_cause
                else "All compared modes currently export zero target weights."
            )
            if all_zero_now
            else "Recommendation is based on current comparison metrics."
        ),
        "ranking": ranked.to_dict(orient="records"),
        "summary": {
            "mode_count": int(len(ranked)),
            "actionable_mode_count": actionable_count,
            "all_modes_zero_now": all_zero_now,
            "best_sharpe_mode": str(df.sort_values("sharpe", ascending=False).iloc[0]["mode"]),
            "best_return_mode": str(df.sort_values("total_return_pct", ascending=False).iloc[0]["mode"]),
            "lowest_drawdown_mode": str(df.sort_values("max_drawdown_pct", ascending=True).iloc[0]["mode"]),
        },
    }
    (reports_dir / "signal_mode_compare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Signal Mode Comparison Report",
        "",
        f"- Recommended mode: {report['recommended_mode']}",
        f"- Note: {report['recommendation_note']}",
        f"- Actionable mode count: {report['summary']['actionable_mode_count']}",
        f"- All modes zero now: {report['summary']['all_modes_zero_now']}",
        f"- Best Sharpe mode: {report['summary']['best_sharpe_mode']}",
        f"- Best return mode: {report['summary']['best_return_mode']}",
        f"- Lowest drawdown mode: {report['summary']['lowest_drawdown_mode']}",
        "",
        "## Ranking",
        "",
    ]
    for idx, row in ranked.iterrows():
        lines.append(
            f"{idx + 1}. {row['mode']} | score={row['score']:.4f} | sharpe={row['sharpe']:.4f} | "
            f"return={row['total_return_pct']:.2f}% | maxDD={row['max_drawdown_pct']:.2f}% | "
            f"turnover={row['avg_turnover_pct']:.2f}% | actionable_now={bool(row.get('actionable_now', False))} | "
            f"last_nonzero_asof={row.get('last_nonzero_asof')}"
        )
    (reports_dir / "signal_mode_compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()
    report = build_report(Path(args.reports_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
