# Execution Report (2026-03-02)

- run_id: `2026-03-02__a747ca00de`
- orders count: 1 / fills count: 1 / positions count: 2
- expected notional (orders sum): 55,610
- filled notional (fills sum): 55,638
- fee: 56 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |    vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|--------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 4005.T   | BUY    |   100 |        100 |          0 | 556.378 |         55637.8 | 55.6378 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| 4f14d3e117e5f234 | 4005.T   | BUY    |   100 | 556.378 | 55.6378 |     0 | 2026-03-02 15:00:00 | PAPER   | paper::2026-03-02__a747ca00de::2026-03-02__a747ca00de__000 |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 4005.T   |   100 |    556.378 |          556.1 |          55610 |          -27.805 |
| 5401.T   |   100 |    636.318 |          629.5 |          62950 |         -681.8   |