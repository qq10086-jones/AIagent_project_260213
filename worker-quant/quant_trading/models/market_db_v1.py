import sqlite3
from datetime import datetime, date
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd


class MarketDB:
    """
    A tiny SQLite wrapper for Japan market data.

    Tables:
      - tickers(symbol PK, name, sector, memo, last_updated)
      - daily_prices(symbol, date, open, high, low, close, volume, PK(symbol,date))
      - meta(key PK, value)   [optional, safe to ignore]
    """

    def __init__(self, db_path: str = "japan_market.db"):
        # check_same_thread=False helps if you later add threads; harmless for single-thread usage.
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        self._apply_pragmas()
        self.create_tables()

    def _apply_pragmas(self) -> None:
        # WAL is a big win for read/write mixes; NORMAL is a good safety/perf tradeoff for local DBs.
        with self.conn:
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")
            self.conn.execute("PRAGMA temp_store = MEMORY;")
            self.conn.execute("PRAGMA cache_size = -20000;")  # ~20MB page cache (negative means KB)

    def create_tables(self) -> None:
        """Create tables & indexes (idempotent)."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tickers (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    memo TEXT,
                    last_updated TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_prices (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,         -- YYYY-MM-DD
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol ON daily_prices(symbol)")

            # Optional metadata table (safe for future expansions: price mode, last run, etc.)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

    # ---------- Tickers ----------

    def save_tickers(self, ticker_list: Sequence[Tuple[str, str, str, str, datetime]]) -> None:
        """Upsert tickers."""
        if not ticker_list:
            return
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO tickers (symbol, name, sector, memo, last_updated)
                VALUES (?, ?, ?, ?, ?)
                """,
                ticker_list,
            )
        print(f"✅ 已更新 {len(ticker_list)} 只股票的基础信息")

    # ---------- Prices ----------

    def get_latest_date(self, symbol: str) -> Optional[date]:
        """Return the latest stored trading date for a symbol, or None if not found."""
        cur = self.conn.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = ?", (symbol,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        # Stored format: YYYY-MM-DD
        return datetime.strptime(row[0], "%Y-%m-%d").date()

    def save_prices(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Save OHLCV prices for one symbol.

        Expected df:
          - index is DatetimeIndex (or has a Date column)
          - columns include Open/High/Low/Close/Volume (case-insensitive)

        Returns: number of rows inserted/replaced.
        """
        if df is None or df.empty:
            return 0

        data = df.copy()

        # Normalize: ensure we have a DatetimeIndex.
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try common index/column patterns
            data = data.reset_index()
            cols_lower = {c.lower(): c for c in data.columns}
            date_col = cols_lower.get("date") or cols_lower.get("index")
            if date_col is None:
                raise ValueError("Input dataframe has no DatetimeIndex and no 'Date' column.")
            data.rename(columns={date_col: "date"}, inplace=True)
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data.set_index("date", inplace=True)

        # Drop rows with invalid dates
        data = data[~data.index.isna()]
        if data.empty:
            return 0

        # Normalize column names
        data.columns = [str(c).lower().strip() for c in data.columns]

        # yfinance sometimes uses 'adj close' if auto_adjust=False; we keep 'close' as provided.
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for {symbol}: {missing}. Got: {list(data.columns)}")

        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": data.index.strftime("%Y-%m-%d"),
                "open": data["open"].astype("float64"),
                "high": data["high"].astype("float64"),
                "low": data["low"].astype("float64"),
                "close": data["close"].astype("float64"),
                "volume": pd.to_numeric(data["volume"], errors="coerce"),
            }
        )

        # Keep rows where at least close exists
        out = out.dropna(subset=["close"])
        if out.empty:
            return 0

        records = list(out.itertuples(index=False, name=None))

        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO daily_prices (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

        print(f"📈 {symbol}: 已存储 {len(records)} 条K线数据")
        return len(records)

    def get_price_df(self, symbol: str) -> pd.DataFrame:
        """Read OHLCV as a DataFrame indexed by date."""
        query = """
            SELECT date, open, high, low, close, volume
            FROM daily_prices
            WHERE symbol = ?
            ORDER BY date
        """
        return pd.read_sql(query, self.conn, params=(symbol,), index_col="date", parse_dates=["date"])

    # ---------- Meta (optional) ----------

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_meta(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    db = MarketDB()
    print("数据库已初始化完成：japan_market.db")
    db.close()
