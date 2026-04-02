import sqlite3
import pandas as pd

conn = sqlite3.connect('japan_market.db')
day = '2026-03-26'
print(f"Checking specific target symbols from {day} to today:")
for sym in ['4005.T', '9432.T']:
    px_old = conn.execute("SELECT close FROM daily_prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1", (sym, day)).fetchone()
    px_new = conn.execute("SELECT close FROM daily_prices WHERE symbol=? ORDER BY date DESC LIMIT 1", (sym,)).fetchone()
    if px_old and px_new:
        po = px_old[0]
        pn = px_new[0]
        pct = (pn - po) / po * 100
        print(f"  {sym}: price {po:.1f} -> {pn:.1f} ({pct:+.2f}%)")
