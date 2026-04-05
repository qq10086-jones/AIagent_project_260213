# Execution Report (2026-03-10)

- run_id: `2026-03-10__4e3df80ffe`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 109,050
- filled notional (fills sum): 108,995
- fee: 109 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 490.255 |         49025.5 | 49.0255 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 599.7   |         59970   | 59.97   |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 38e50e131ead79a8 | 5401.T   | SELL   |   100 | 599.7   | 59.97   |     0 | 2026-03-10 15:00:00 | PAPER   | paper::2026-03-10__4e3df80ffe::2026-03-10__4e3df80ffe__000 |
| 4996f101ef036d72 | 4005.T   | SELL   |   100 | 490.255 | 49.0255 |     0 | 2026-03-10 15:00:00 | PAPER   | paper::2026-03-10__4e3df80ffe::2026-03-10__4e3df80ffe__001 |

## End-of-day Positions

_No positions found._