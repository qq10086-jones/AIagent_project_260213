# Execution Report (2026-03-09)

- run_id: `2026-03-09__29f37b1dbb`
- orders count: 2 / fills count: 2 / positions count: 1
- expected notional (orders sum): 100,010
- filled notional (fills sum): 99,960
- fee: 100 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4755.T   | SELL   |   100 |        100 |          0 | 785.907 |         78590.7 | 78.5907 |     0 |         1 | MKT          |               |
| 9434.T   | SELL   |   100 |        100 |          0 | 213.693 |         21369.3 | 21.3693 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 3c9ddb7e29b3c01c | 4755.T   | SELL   |   100 | 785.907 | 78.5907 |     0 | 2026-03-09 15:00:00 | PAPER   | paper::2026-03-09__29f37b1dbb::2026-03-09__29f37b1dbb__000 |
| 5fd904b61171dd33 | 9434.T   | SELL   |   100 | 213.693 | 21.3693 |     0 | 2026-03-09 15:00:00 | PAPER   | paper::2026-03-09__29f37b1dbb::2026-03-09__29f37b1dbb__001 |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 5401.T   |   100 |    626.013 |          590.6 |          59060 |         -3541.29 |