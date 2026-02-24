# Execution Report (2026-02-03)

- run_id: `2026-02-03__a49c53984e`
- artifact_dir: `artifacts\decision\2026-02-03\2026-02-03__a49c53984e`
- orders count: 2 / fills count: 8 / positions count: 3
- expected notional (orders sum): 195,168
- filled notional (fills sum): 121,676,271
- fee: 0 / tax: 0

## Orders vs Fills (by symbol & side)

| symbol   | side   |   qty |   fill_qty |   qty_diff |   vwap |    fill_notional |   fee |   tax |   n_fills | order_type   | limit_price   |
|:---------|:-------|------:|-----------:|-----------:|-------:|-----------------:|------:|------:|----------:|:-------------|:--------------|
| 2914.T   | BUY    |    10 |         10 |          0 |  57780 | 577800           |     0 |     0 |         1 | MKT          |               |
| 9432.T   | BUY    |   879 |        929 |         50 | 130294 |      1.21043e+08 |     0 |     0 |         1 | MKT          |               |

## Fills (raw)

| fill_id          | symbol   | side   |   qty |   price |   fee |   tax | ts                  | venue   | external_ref   |
|:-----------------|:---------|:-------|------:|--------:|------:|------:|:--------------------|:--------|:---------------|
| ace9230fbb22e20c | 9432.T   | BUY    |   879 |  137388 |     0 |     0 | 2026-02-04T00:40:21 | SBI     | manual         |
| 78dbaefaf0f0843f | 2914.T   | BUY    |    10 |   57780 |     0 |     0 | 2026-02-04T00:40:56 | SBI     | manual         |
|                  | 9432     | BUY    |    10 |    5578 |     0 |     0 | 2026-02-05T23:52:34 | 0       |                |
|                  | 9432.T   | BUY    |    10 |    5578 |     0 |     0 | 2026-02-06T00:15:40 | SBI     |                |
|                  | 9432.T   | BUY    |    10 |    5578 |     0 |     0 | 2026-02-06T00:20:05 | SBI     |                |
|                  | 9432.T   | BUY    |    10 |    5578 |     0 |     0 | 2026-02-06T00:21:48 | SBI     |                |
|                  | 9432.T   | BUY    |    10 |    5578 |     0 |     0 | 2026-02-06T00:22:35 | SBI     |                |
|                  | 9432.T   | BUY    |    10 |    5578 |     0 |     0 | 2026-02-06T00:29:01 | SBI     |                |

## End-of-day Positions

| symbol   |   qty |   avg_cost |   market_price |     market_value |   unrealized_pnl |
|:---------|------:|-----------:|---------------:|-----------------:|-----------------:|
| 2914.T   |    91 |    51470.2 |         5778   | 525798           |     -4.15799e+06 |
| 9432     |    80 |     5578   |          nan   |    nan           |    nan           |
| 9432.T   |  8089 |   119662   |          156.3 |      1.26431e+06 |     -9.66681e+08 |