# Execution Report (2026-03-06)

- run_id: `2026-03-06__d84683b9e9`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 111,130
- filled notional (fills sum): 111,074
- fee: 111 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 507.246 |         50724.6 | 50.7246 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 603.498 |         60349.8 | 60.3498 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 64342227ef16a11c | 5401.T   | SELL   |   100 | 603.498 | 60.3498 |     0 | 2026-03-06 15:00:00 | PAPER   | paper::2026-03-06__d84683b9e9::2026-03-06__d84683b9e9__000 |
| 82c8e8ee8c5c2509 | 4005.T   | SELL   |   100 | 507.246 | 50.7246 |     0 | 2026-03-06 15:00:00 | PAPER   | paper::2026-03-06__d84683b9e9::2026-03-06__d84683b9e9__001 |

## End-of-day Positions

_No positions found._