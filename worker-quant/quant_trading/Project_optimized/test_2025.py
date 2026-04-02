import sys
sys.path.append('.')
import ss7_sqlite_news_overlay as ss7
import pandas as pd
import numpy as np

# Tickers used currently
tickers = ["7267.T","9432.T","8604.T","9434.T","7261.T","7211.T","9501.T","4755.T","6471.T","5020.T","7201.T","4005.T","5401.T"]

config = ss7.BTConfig(tickers=tickers)
exec_cfg = ss7.ExecConfig(initial_capital=1000000, fee_bps=10.0, slippage_bps=5.0)
db_path = "japan_market.db"

print("Loading data...")
high, low, close, vol = ss7.load_ohlcv_from_sqlite(db_path, tickers + ["1321.T"], "2020-01-01", "2025-10-01")
print("Building features...")
ret_fwd, X_all = ss7.make_features(high, low, close, vol, H=config.H, train_window=config.train_window, benchmark_ticker="1321.T")
X_all = ss7.merge_feature_row(X_all, db_path, tickers, use_fundamental=True)

modes = ["ridge", "shadow_ic"]

for mode in modes:
    print(f"\nRunning walk-forward backtest [{mode}]...")
    # Get equity curve from backtest
    pf_val, _, _, _, _ = ss7.backtest(
        mode,
        ret_fwd, X_all, close, vol,
        train_window=config.train_window,
        rebalance_every=config.rebalance_every,
        H=config.H,
        benchmark_ticker="1321.T",
        exec_cfg=exec_cfg,
        db_path=db_path,
        news_cfg=ss7.NewsConfig(enabled=True)
    )
    
    # We want profitability between 2025-04-01 and 2025-10-01
    s_eq = pd.Series(pf_val).dropna()
    if s_eq.empty:
        print("Empty equity curve.")
        continue

    # Filter dates
    mask = (s_eq.index >= pd.Timestamp('2025-04-01').tz_localize(s_eq.index.tz)) & (s_eq.index <= pd.Timestamp('2025-10-01').tz_localize(s_eq.index.tz))
    period_eq = s_eq.loc[mask]

    if len(period_eq) < 2:
        print("Not enough data in the requested period.")
        continue

    start_eq = period_eq.iloc[0]
    end_eq = period_eq.iloc[-1]
    ret = (end_eq - start_eq) / start_eq * 100
    
    print(f"[{mode}] Account Equity on 2025-04-01: {start_eq:,.0f} JPY")
    print(f"[{mode}] Account Equity on 2025-10-01: {end_eq:,.0f} JPY")
    print(f"[{mode}] Profitability (2025-04-01 -> 2025-10-01): {ret:+.2f}%")
