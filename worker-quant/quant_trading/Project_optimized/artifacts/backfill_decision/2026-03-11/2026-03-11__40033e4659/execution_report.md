# Execution Report (2026-03-11)

- run_id: `2026-03-11__40033e4659`
- orders count: 1 / fills count: 1 / positions count: 2
- expected notional (orders sum): 37,670
- filled notional (fills sum): 37,689
- fee: 38 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 7201.T   | BUY    |   100 |        100 |          0 | 376.888 |         37688.8 | 37.6888 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| dac7ba7bdeb6fc11 | 7201.T   | BUY    |   100 | 376.888 | 37.6888 |     0 | 2026-03-11 15:00:00 | PAPER   | paper::2026-03-11__40033e4659::2026-03-11__40033e4659__000 |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 5401.T   |   100 |    626.013 |          611.3 |          61130 |        -1471.29  |
| 7201.T   |   100 |    376.888 |          376.7 |          37670 |          -18.835 |