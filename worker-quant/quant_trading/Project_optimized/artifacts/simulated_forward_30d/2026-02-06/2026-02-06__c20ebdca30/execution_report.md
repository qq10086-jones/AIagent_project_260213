# Execution Report (2026-02-06)

- run_id: `2026-02-06__c20ebdca30`
- orders count: 1 / fills count: 1 / positions count: 1
- expected notional (orders sum): 41,660
- filled notional (fills sum): 41,680
- fee: 42 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |   vwap |   fill_notional |     fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|-------:|----------------:|--------:|------:|----------:|:-------------|:--------------|
| 7201.T   | BUY    |   100 |        100 |          0 |  416.8 |           41680 | 41.6808 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |     fee |   tax | ts                  | venue   | external_ref                                               |
|:-----------------|:---------|:-------|------:|--------:|--------:|------:|:--------------------|:--------|:-----------------------------------------------------------|
| aeb584e16bc49bb7 | 7201.T   | BUY    |   100 |   416.8 | 41.6808 |     0 | 2026-02-06 15:00:00 | PAPER   | paper::2026-02-06__c20ebdca30::2026-02-06__c20ebdca30__000 |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 7201.T   |   100 |      416.8 |          416.6 |          41660 |         -19.9982 |