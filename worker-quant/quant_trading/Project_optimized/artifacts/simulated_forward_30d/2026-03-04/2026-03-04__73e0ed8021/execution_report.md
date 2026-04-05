# Execution Report (2026-03-04)

- run_id: `2026-03-04__73e0ed8021`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 108,050
- filled notional (fills sum): 107,996
- fee: 108 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 475.062 |         47506.2 | 47.5062 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 604.897 |         60489.7 | 60.4897 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 555abf50951fa002 | 5401.T   | SELL   |   100 | 604.897 | 60.4897 |     0 | 2026-03-04 15:00:00 | PAPER   | paper::2026-03-04__73e0ed8021::2026-03-04__73e0ed8021__000 |
| 80e32966f75714d6 | 4005.T   | SELL   |   100 | 475.062 | 47.5062 |     0 | 2026-03-04 15:00:00 | PAPER   | paper::2026-03-04__73e0ed8021::2026-03-04__73e0ed8021__001 |

## End-of-day Positions

_No positions found._