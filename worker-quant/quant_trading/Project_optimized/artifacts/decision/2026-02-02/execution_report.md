# Execution Report (2026-02-02)

- run_id: `2026-02-02__076c1372ad`
- artifact_dir: `artifacts\decision\2026-02-02`
- orders count: 2 / fills count: 2 / positions count: 2
- expected notional (orders sum): 198,787
- filled notional (fills sum): 948,391
- fee: 0 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |   vwap |   fill_notional |   fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|-------:|----------------:|------:|------:|----------:|:-------------|:--------------|
| 2914.T   | BUY    |    11 |         11 |          0 |   5581 |           61391 |     0 |     0 |         1 | MKT          |               |
| 9432.T   | BUY    |   887 |        887 |          0 |   1000 |          887000 |     0 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |   fee |   tax | ts             | venue   | external_ref   |
|:-----------------|:---------|:-------|------:|--------:|------:|------:|:---------------|:--------|:---------------|
| 32010661acb72ca0 | 9432.T   | BUY    |   887 |    1000 |     0 |     0 | 2/2/2026 14:32 | SBI     | TEST-FILL-001  |
| 00645bfd1e1eb077 | 2914.T   | BUY    |    11 |    5581 |     0 |     0 | 2/2/2026 14:35 | SBI     | TEST-FILL-002  |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |   market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|---------------:|-----------------:|
| 2914.T   |    11 |       5581 |         5581   |          61391 |                0 |
| 9432.T   |   887 |       1000 |          154.9 |         137396 |          -749604 |