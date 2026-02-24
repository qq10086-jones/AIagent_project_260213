-- Decision runs: one batch = one day pipeline output (snapshot + orders proposal etc.)
CREATE TABLE IF NOT EXISTS decision_runs (
  run_id TEXT PRIMARY KEY,
  asof TEXT NOT NULL,
  ts TEXT NOT NULL,
  snapshot_path TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'created',   -- created/proposed/executed/filled/closed
  notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_decision_runs_asof ON decision_runs(asof);

-- Orders: system-generated proposal for manual execution
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  asof TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,             -- BUY/SELL
  qty REAL NOT NULL,              -- shares/units
  order_type TEXT NOT NULL,       -- MKT/LMT (suggested)
  limit_price REAL,
  tif TEXT DEFAULT 'DAY',
  reason TEXT,                    -- rebalance / risk_off / adjust
  expected_fee REAL,
  expected_slippage REAL,
  expected_value REAL,
  status TEXT NOT NULL DEFAULT 'proposed', -- proposed/executed/cancelled
  created_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id);
CREATE INDEX IF NOT EXISTS idx_orders_asof ON orders(asof);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);

-- Fills: actual executions (imported manually from broker statements)
CREATE TABLE IF NOT EXISTS fills (
  fill_id TEXT PRIMARY KEY,
  order_id TEXT,                  -- nullable if you don't map 1:1
  run_id TEXT NOT NULL,
  asof TEXT NOT NULL,
  ts TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,             -- BUY/SELL
  qty REAL NOT NULL,
  price REAL NOT NULL,
  fee REAL DEFAULT 0,
  tax REAL DEFAULT 0,
  venue TEXT DEFAULT 'SBI',
  external_ref TEXT               -- e.g. 約定番号
);
CREATE INDEX IF NOT EXISTS idx_fills_run ON fills(run_id);
CREATE INDEX IF NOT EXISTS idx_fills_asof ON fills(asof);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);

-- Positions: end-of-day snapshot, derived from fills + prior positions
CREATE TABLE IF NOT EXISTS positions (
  asof TEXT NOT NULL,
  symbol TEXT NOT NULL,
  qty REAL NOT NULL,
  avg_cost REAL,
  market_price REAL,
  market_value REAL,
  unrealized_pnl REAL,
  PRIMARY KEY (asof, symbol)
);
CREATE INDEX IF NOT EXISTS idx_positions_asof ON positions(asof);

-- Optional: cash ledger (recommended)
CREATE TABLE IF NOT EXISTS cash_ledger (
  asof TEXT NOT NULL,
  ts TEXT NOT NULL,
  amount REAL NOT NULL,           -- +in / -out
  currency TEXT NOT NULL DEFAULT 'JPY',
  reason TEXT,
  run_id TEXT,
  PRIMARY KEY (asof, ts, amount, reason)
);
CREATE INDEX IF NOT EXISTS idx_cash_ledger_asof ON cash_ledger(asof);
