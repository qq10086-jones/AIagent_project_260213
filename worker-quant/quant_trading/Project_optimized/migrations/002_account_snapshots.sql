CREATE TABLE IF NOT EXISTS account_snapshots (
  asof TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  run_id TEXT,
  cash REAL NOT NULL,
  positions_value REAL NOT NULL,
  nav REAL NOT NULL,
  net_trade_cashflow REAL NOT NULL,  -- cash change caused by trades (SELL - BUY - fee - tax)
  fees REAL NOT NULL,
  tax REAL NOT NULL,
  notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_account_snapshots_asof ON account_snapshots(asof);
