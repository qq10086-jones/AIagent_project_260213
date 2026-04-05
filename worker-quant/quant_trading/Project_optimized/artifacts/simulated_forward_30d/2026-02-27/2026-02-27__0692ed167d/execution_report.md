# Execution Report (2026-02-27)

- run_id: `2026-02-27__0692ed167d`
- orders count: 2 / fills count: 2 / positions count: 1
- expected notional (orders sum): 120,570
- filled notional (fills sum): 120,573
- fee: 121 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 569.415 |         56941.5 | 56.9415 |     0 |         1 | MKT          |               |
| 5401.T   | BUY    |   100 |        100 |          0 | 636.318 |         63631.8 | 63.6318 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| b39cd07c58669f9e | 4005.T   | SELL   |   100 | 569.415 | 56.9415 |     0 | 2026-02-27 15:00:00 | PAPER   | paper::2026-02-27__0692ed167d::2026-02-27__0692ed167d__000 |
| b90c0dd87c210845 | 5401.T   | BUY    |   100 | 636.318 | 63.6318 |     0 | 2026-02-27 15:00:00 | PAPER   | paper::2026-02-27__0692ed167d::2026-02-27__0692ed167d__001 |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 5401.T   |   100 |    636.318 |            636 |          63600 |            -31.8 |