# Execution Report (2026-02-13)

- run_id: `2026-02-13__4832668ef4`
- orders count: 1 / fills count: 1 / positions count: 2
- expected notional (orders sum): 56,230
- filled notional (fills sum): 56,258
- fee: 56 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | BUY    |   100 |        100 |          0 | 562.581 |         56258.1 | 56.2581 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 161770c1833b5a29 | 4005.T   | BUY    |   100 | 562.581 | 56.2581 |     0 | 2026-02-13 15:00:00 | PAPER   | paper::2026-02-13__4832668ef4::2026-02-13__4832668ef4__000 |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 4005.T   |   100 |    562.581 |          562.3 |          56230 |          -28.115 |
| 7201.T   |   100 |    416.8   |          447   |          44700 |         3020     |