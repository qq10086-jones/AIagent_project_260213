# Execution Report (2026-03-05)

- run_id: `2026-03-05__361f25bda5`
- orders count: 2 / fills count: 2 / positions count: 0
- expected notional (orders sum): 108,910
- filled notional (fills sum): 108,886
- fee: 109 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | SELL   |   100 |        100 |          0 | 488.955 |         48895.5 | 48.8955 |     0 |         1 | MKT          |               |
| 5401.T   | SELL   |   100 |        100 |          0 | 599.9   |         59990   | 59.96   |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 128adfe656be06c4 | 5401.T   | SELL   |   100 | 599.9   | 59.96   |     0 | 2026-03-05 15:00:00 | PAPER   | paper::2026-03-05__361f25bda5::2026-03-05__361f25bda5__000 |
| b24b834505e6ec78 | 4005.T   | SELL   |   100 | 488.955 | 48.8955 |     0 | 2026-03-05 15:00:00 | PAPER   | paper::2026-03-05__361f25bda5::2026-03-05__361f25bda5__001 |

## End-of-day Positions

_No positions found._