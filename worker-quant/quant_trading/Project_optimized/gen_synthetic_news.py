"""Generate synthetic news CSV for testing ss7 news overlay.

Produces a CSV with columns: date, symbol, sent, weight, conf
Sentiment is derived from price momentum with added noise — realistic enough
to test the overlay machinery without real news data.

Usage:
  python gen_synthetic_news.py --db japan_market.db --out data/synthetic_news.csv
  python gen_synthetic_news.py --db japan_market.db --out data/synthetic_news.csv --coverage 0.2
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--out", default="data/synthetic_news.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coverage", type=float, default=0.25,
                    help="Fraction of symbol-days that get at least one news item")
    ap.add_argument("--items_per_day", type=int, default=2,
                    help="Average news items per covered symbol-day")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    conn = sqlite3.connect(args.db)
    prices_df = pd.read_sql_query(
        "SELECT date, symbol, close FROM daily_prices ORDER BY symbol, date",
        conn,
    )
    conn.close()

    if prices_df.empty:
        print("No data in daily_prices. Run db_update.py first.")
        return

    prices_df["date"] = pd.to_datetime(prices_df["date"])
    prices_df = prices_df.sort_values(["symbol", "date"])

    # Base sentiment signal: 5-day return, compressed via tanh
    # tanh(ret5 / 0.05): a 5% move maps to ~0.76 sentiment
    prices_df["ret5"] = prices_df.groupby("symbol")["close"].pct_change(5)
    prices_df = prices_df.dropna(subset=["ret5"])
    prices_df["base_sent"] = np.tanh(prices_df["ret5"] / 0.05)

    rows = []
    for (symbol,), grp in prices_df.groupby(["symbol"]):
        for _, row in grp.iterrows():
            if rng.random() > args.coverage:
                continue

            n_items = max(1, int(rng.poisson(args.items_per_day)))
            for _ in range(n_items):
                noise = rng.normal(0, 0.35)
                sent = float(np.clip(float(row["base_sent"]) + noise, -1.0, 1.0))
                # Weight: log-normal (most items are small, occasional big ones)
                weight = float(np.clip(rng.lognormal(0, 0.8), 0.1, 10.0))
                # Confidence: beta-distributed, skewed toward high confidence
                conf = float(np.clip(rng.beta(3, 2), 0.05, 1.0))
                rows.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "ticker": symbol,   # ss7 load_news_items expects 'ticker' column
                        "sent": round(sent, 4),
                        "weight": round(weight, 4),
                        "conf": round(conf, 4),
                    }
                )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_symbols = prices_df["symbol"].nunique()
    date_range = f"{out_df['date'].min()} to {out_df['date'].max()}"
    print(f"Generated {len(out_df):,} synthetic news items")
    print(f"Tickers: {n_symbols}  |  Date range: {date_range}")
    print(f"Saved to: {out_path}")
    print()
    print("To enable news overlay in the backtest, set in config.yaml:")
    print("  model:")
    print(f"    news_csv: {out_path}")
    print()
    print("Or pass SS6_NEWS_CSV env var when calling ss7_sqlite_news_overlay.py directly.")


if __name__ == "__main__":
    main()
