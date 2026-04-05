# Execution Report (2026-03-11)

- run_id: `2026-03-11__86d97043b2`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 110,540
- filled notional (fills sum): 110,485
- fee: 110 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 493.853 |         49385.3 | 49.3853 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 610.994 |         61099.4 | 61.0994 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 630a469d56364659 | 5401.T   | SELL   |   100 | 610.994 | 61.0994 |     0 | 2026-03-11 15:00:00 | PAPER   | paper::2026-03-11__86d97043b2::2026-03-11__86d97043b2__000 |
| 61d523b6dc1b9176 | 4005.T   | SELL   |   100 | 493.853 | 49.3853 |     0 | 2026-03-11 15:00:00 | PAPER   | paper::2026-03-11__86d97043b2::2026-03-11__86d97043b2__001 |

## End-of-day Positions

_No positions found._