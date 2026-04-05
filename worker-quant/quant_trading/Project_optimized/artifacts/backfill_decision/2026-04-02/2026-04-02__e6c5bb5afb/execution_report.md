# Execution Report (2026-04-02)

- run_id: `2026-04-02__e6c5bb5afb`
- orders count: 1 / fills count: 1 / positions count: 5
- expected notional (orders sum): 15,700
- filled notional (fills sum): 15,680
- fee: 16 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |   vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|-------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 9432.T   | BUY    |   100 |        100 |          0 |  156.8 |           15680 | 15.6878 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                        | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------------|:--------|:-----------------------------------------------------------|
| 980f13b2d0048304 | 9432.T   | BUY    |   100 |   156.8 | 15.6878 |     0 | 2026-04-02T06:24:00+00:00 | PAPER   | paper::2026-04-02__e6c5bb5afb::2026-04-02__e6c5bb5afb__000 |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 4755.T   |   100 |    756.178 |          736.7 |          73670 |      -1947.79    |
| 5401.T   |   100 |    626.013 |          584   |          58400 |      -4201.29    |
| 7201.T   |   100 |    345.873 |          345.8 |          34580 |         -7.28742 |
| 9432.T   |   100 |    156.8   |          157   |          15700 |         19.9997  |
| 9434.T   |   100 |    212     |          215.4 |          21540 |        339.999   |