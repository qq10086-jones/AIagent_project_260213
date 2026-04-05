# Execution Report (2026-03-03)

- run_id: `2026-03-03__71174f8dce`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 113,140
- filled notional (fills sum): 113,083
- fee: 113 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 510.844 |         51084.4 | 51.0844 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 619.99  |         61999   | 61.999  |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| ccf8118b71ef1a3e | 5401.T   | SELL   |   100 | 619.99  | 61.999  |     0 | 2026-03-03 15:00:00 | PAPER   | paper::2026-03-03__71174f8dce::2026-03-03__71174f8dce__000 |
| 4964bd3ed25a50ff | 4005.T   | SELL   |   100 | 510.844 | 51.0844 |     0 | 2026-03-03 15:00:00 | PAPER   | paper::2026-03-03__71174f8dce::2026-03-03__71174f8dce__001 |

## End-of-day Positions

_No positions found._