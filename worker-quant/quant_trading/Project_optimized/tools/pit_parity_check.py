"""D-8 Step 3 — empirical PIT parity check for price features.

Runs compute_price_features under three paths for the same historical asof
and diffs the outputs:

  A. NEW_PIT:    load_prices_from_db(conn, asof=T)                → SQL-bounded
  B. LEGACY:     load_prices_from_db(conn) then slice <= T        → pre-fix shape
  C. DB_STORED:  existing feature_daily rows for asof=T            → what routing has been using

If A ≡ B the SQL fix is a no-op for correctness (only performance).
If A ≡ C the stored feature_daily is already PIT-clean → no rebuild needed.
Any divergence is the contamination magnitude.

Writes: reports/pit_parity_<asof>.md
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compute_price_features import compute_features_for_symbol, load_prices_from_db  # noqa: E402


def _load_stored(conn: sqlite3.Connection, asof: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT symbol, feature_name, value FROM feature_daily "
        "WHERE asof=? AND source_fact_ids='compute_price_features'",
        conn,
        params=(asof,),
    )
    if df.empty:
        return df
    return df.pivot(index="symbol", columns="feature_name", values="value")


def _compute_features_for_symbol_at_asof(close_all: pd.DataFrame, vol_all: pd.DataFrame, asof: str) -> pd.DataFrame:
    asof_dt = pd.Timestamp(asof)
    c = close_all[close_all.index <= asof_dt]
    v = vol_all[vol_all.index <= asof_dt]
    out = {}
    for sym in c.columns:
        close_ser = c[sym].dropna()
        if close_ser.empty:
            continue
        vol_ser = v[sym].dropna() if sym in v.columns else None
        feats = compute_features_for_symbol(close_ser, vol_ser)
        if feats.empty:
            continue
        out[sym] = feats.iloc[-1]
    return pd.DataFrame(out).T


def _diff_stats(a: pd.DataFrame, b: pd.DataFrame, label: str) -> dict:
    common_idx = a.index.intersection(b.index)
    common_cols = a.columns.intersection(b.columns)
    a2 = a.loc[common_idx, common_cols]
    b2 = b.loc[common_idx, common_cols]
    diff = (a2 - b2).abs()
    total = diff.size
    n_nan = int(diff.isna().sum().sum())
    n_zero = int((diff == 0).sum().sum())
    finite = diff.replace([np.inf, -np.inf], np.nan).dropna()
    max_diff = float(finite.values.max()) if finite.size else 0.0
    mean_diff = float(finite.values.mean()) if finite.size else 0.0
    return {
        "label": label,
        "rows_compared": len(common_idx),
        "cols_compared": len(common_cols),
        "cells_total": total,
        "cells_nan_diff": n_nan,
        "cells_zero_diff": n_zero,
        "cells_nonzero": total - n_nan - n_zero,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def run(db: str, asof: str, reports_dir: Path) -> dict:
    conn = sqlite3.connect(db)
    try:
        # Path A: new PIT-bounded SQL
        close_a, vol_a = load_prices_from_db(conn, asof=asof)
        feats_a = _compute_features_for_symbol_at_asof(close_a, vol_a, asof)

        # Path B: legacy — load all then slice (simulates pre-fix caller)
        close_b, vol_b = load_prices_from_db(conn)  # no asof
        feats_b = _compute_features_for_symbol_at_asof(close_b, vol_b, asof)

        # Path C: stored in DB
        feats_c = _load_stored(conn, asof)
    finally:
        conn.close()

    stats_ab = _diff_stats(feats_a, feats_b, "A vs B (new-PIT vs legacy-slice)")
    stats_ac = _diff_stats(feats_a, feats_c, "A vs C (new-PIT vs DB-stored)")

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_md = reports_dir / f"pit_parity_{asof}.md"
    lines = [
        f"# PIT Parity Check — asof={asof}",
        "",
        f"Database: `{db}`  |  n_symbols in A: {len(feats_a)}  |  stored in DB: {len(feats_c)}",
        "",
        "## A vs B — does SQL asof bound change what the current code computes?",
        "",
        f"- rows compared: {stats_ab['rows_compared']}  cols: {stats_ab['cols_compared']}  cells: {stats_ab['cells_total']}",
        f"- cells with zero diff: {stats_ab['cells_zero_diff']}",
        f"- cells with NaN diff: {stats_ab['cells_nan_diff']}",
        f"- cells non-zero diff: {stats_ab['cells_nonzero']}",
        f"- max abs diff: {stats_ab['max_abs_diff']:.6g}",
        f"- mean abs diff: {stats_ab['mean_abs_diff']:.6g}",
        "",
        "**Interpretation**: if non-zero cells == 0 and max_abs_diff == 0, the SQL",
        "fix is a performance-only change; no contamination existed via this path",
        "(caller already did `close[close.index <= asof_dt]` slicing).",
        "",
        "## A vs C — does existing feature_daily match the PIT-correct recompute?",
        "",
        f"- rows compared: {stats_ac['rows_compared']}  cols: {stats_ac['cols_compared']}  cells: {stats_ac['cells_total']}",
        f"- cells with zero diff: {stats_ac['cells_zero_diff']}",
        f"- cells with NaN diff: {stats_ac['cells_nan_diff']}",
        f"- cells non-zero diff: {stats_ac['cells_nonzero']}",
        f"- max abs diff: {stats_ac['max_abs_diff']:.6g}",
        f"- mean abs diff: {stats_ac['mean_abs_diff']:.6g}",
        "",
        "**Interpretation**: if the stored feature_daily matches a fresh PIT-",
        "correct recompute, no historical rebuild is required.",
        "",
        "## Conclusion",
        "",
    ]
    if stats_ab["cells_nonzero"] == 0 and stats_ab["max_abs_diff"] == 0:
        lines.append("- **A ≡ B**: SQL asof bound is performance-only. Existing compute path was already PIT-correct at the caller level.")
    else:
        lines.append(f"- **A ≠ B**: {stats_ab['cells_nonzero']} cells differ. The legacy call path was returning PIT-inconsistent features.")
    if stats_ac["cells_nonzero"] == 0 and stats_ac["max_abs_diff"] == 0:
        lines.append("- **A ≡ C**: feature_daily stored values match PIT recompute. No rebuild needed.")
    else:
        lines.append(f"- **A ≠ C**: {stats_ac['cells_nonzero']} stored cells differ (max |Δ|={stats_ac['max_abs_diff']:.4g}). Historical rebuild of feature_daily is warranted before trusting factor IC.")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return {"asof": asof, "ab": stats_ab, "ac": stats_ac, "report": str(out_md)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD to verify")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()
    out = run(args.db, args.asof, Path(args.reports_dir))
    print(f"A vs B: {out['ab']}")
    print(f"A vs C: {out['ac']}")
    print(f"report: {out['report']}")


if __name__ == "__main__":
    main()
