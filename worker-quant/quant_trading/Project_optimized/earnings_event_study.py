from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


WINDOWS = [1, 3, 5]


def build_event_study(db_path: str, out_dir: Path) -> dict:
    with sqlite3.connect(db_path) as conn:
        events = pd.read_sql_query(
            """
            SELECT symbol, published_ts, event_type, headline,
                   revenue_yoy, operating_income_yoy, eps_yoy,
                   guidance_delta_revenue, guidance_delta_op, guidance_delta_eps,
                   surprise_score
            FROM earnings_events
            ORDER BY published_ts
            """,
            conn,
        )
        prices = pd.read_sql_query(
            """
            SELECT date, symbol, close
            FROM daily_prices
            ORDER BY date, symbol
            """,
            conn,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if events.empty or prices.empty:
        report = {"event_count": 0, "summary": {}}
        (out_dir / "earnings_event_study.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(out_dir / "earnings_event_study.csv", index=False)
        return report

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"])
    px = prices.pivot(index="date", columns="symbol", values="close").sort_index()

    rows = []
    events["published_ts"] = pd.to_datetime(events["published_ts"], errors="coerce")
    for rec in events.to_dict(orient="records"):
        symbol = str(rec["symbol"])
        if symbol not in px.columns or pd.isna(rec["published_ts"]):
            continue
        event_date = pd.Timestamp(rec["published_ts"]).normalize()
        idx = px.index.searchsorted(event_date)
        if idx >= len(px.index):
            continue
        trade_dt = px.index[idx]
        base_px = px.at[trade_dt, symbol]
        if pd.isna(base_px) or float(base_px) <= 0:
            continue

        row = {
            "symbol": symbol,
            "event_date": event_date.strftime("%Y-%m-%d"),
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "event_type": rec.get("event_type"),
            "headline": rec.get("headline"),
            "surprise_score": rec.get("surprise_score"),
            "guidance_delta_eps": rec.get("guidance_delta_eps"),
            "eps_yoy": rec.get("eps_yoy"),
        }
        for window in WINDOWS:
            target_idx = idx + window
            if target_idx >= len(px.index):
                row[f"ret_{window}d_pct"] = None
                continue
            end_px = px.iat[target_idx, px.columns.get_loc(symbol)]
            row[f"ret_{window}d_pct"] = None if pd.isna(end_px) else (float(end_px) / float(base_px) - 1.0) * 100.0
        rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "earnings_event_study.csv", index=False)

    summary = {"event_count": int(len(detail))}
    if not detail.empty:
        summary["all_events"] = {
            f"avg_ret_{window}d_pct": float(detail[f"ret_{window}d_pct"].dropna().mean()) if detail[f"ret_{window}d_pct"].notna().any() else 0.0
            for window in WINDOWS
        }
        positive_guidance = detail[pd.to_numeric(detail["guidance_delta_eps"], errors="coerce") > 0]
        negative_guidance = detail[pd.to_numeric(detail["guidance_delta_eps"], errors="coerce") < 0]
        summary["positive_guidance"] = {
            "count": int(len(positive_guidance)),
            **{
                f"avg_ret_{window}d_pct": float(positive_guidance[f"ret_{window}d_pct"].dropna().mean()) if not positive_guidance.empty else 0.0
                for window in WINDOWS
            },
        }
        summary["negative_guidance"] = {
            "count": int(len(negative_guidance)),
            **{
                f"avg_ret_{window}d_pct": float(negative_guidance[f"ret_{window}d_pct"].dropna().mean()) if not negative_guidance.empty else 0.0
                for window in WINDOWS
            },
        }

    report = {"event_count": int(len(detail)), "summary": summary}
    (out_dir / "earnings_event_study.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Earnings Event Study",
        "",
        f"- Event count: {report['event_count']}",
    ]
    if summary:
        for key, payload in summary.items():
            if key == "event_count":
                continue
            lines.append(f"- {key}: {json.dumps(payload, ensure_ascii=False)}")
    (out_dir / "earnings_event_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()
    report = build_event_study(args.db, Path(args.out_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
