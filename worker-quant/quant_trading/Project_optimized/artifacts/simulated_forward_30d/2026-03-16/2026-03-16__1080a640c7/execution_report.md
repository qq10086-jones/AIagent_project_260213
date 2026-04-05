# Execution Report (2026-03-16)

- run_id: `2026-03-16__1080a640c7`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 106,950
- filled notional (fills sum): 106,926
- fee: 107 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 486.557 |         48655.7 | 48.6557 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 582.7   |         58270   | 58.2409 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 5774741caf10f309 | 5401.T   | SELL   |   100 | 582.7   | 58.2409 |     0 | 2026-03-16 15:00:00 | PAPER   | paper::2026-03-16__1080a640c7::2026-03-16__1080a640c7__000 |
| a85a4bd6581e818d | 4005.T   | SELL   |   100 | 486.557 | 48.6557 |     0 | 2026-03-16 15:00:00 | PAPER   | paper::2026-03-16__1080a640c7::2026-03-16__1080a640c7__001 |

## End-of-day Positions

_No positions found._