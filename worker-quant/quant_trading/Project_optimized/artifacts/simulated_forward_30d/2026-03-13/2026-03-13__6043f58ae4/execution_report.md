# Execution Report (2026-03-13)

- run_id: `2026-03-13__6043f58ae4`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 109,210
- filled notional (fills sum): 109,155
- fee: 109 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 498.651 |         49865.1 | 49.8651 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 592.903 |         59290.3 | 59.2903 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 0bc3cd85747888e6 | 5401.T   | SELL   |   100 | 592.903 | 59.2903 |     0 | 2026-03-13 15:00:00 | PAPER   | paper::2026-03-13__6043f58ae4::2026-03-13__6043f58ae4__000 |
| f4464dabc13fdcdd | 4005.T   | SELL   |   100 | 498.651 | 49.8651 |     0 | 2026-03-13 15:00:00 | PAPER   | paper::2026-03-13__6043f58ae4::2026-03-13__6043f58ae4__001 |

## End-of-day Positions

_No positions found._