# Factor Tearsheet Summary — 2026-04-15

## 每因子概览

| 因子 | 样本 | IC(1D) | IC IR(1D) | t-stat | Q5-Q1 年化 | 单调 | Turnover(1D) | 建议 |
|------|------|--------|-----------|--------|-----------|------|--------------|------|
| kmid | — | — | — | — | — | — | — | insufficient_data |
| klen | — | — | — | — | — | — | — | insufficient_data |
| kmid2 | — | — | — | — | — | — | — | insufficient_data |
| kup | — | — | — | — | — | — | — | insufficient_data |
| klow | — | — | — | — | — | — | — | insufficient_data |
| ksft | — | — | — | — | — | — | — | insufficient_data |
| roc1 | — | — | — | — | — | — | — | insufficient_data |
| roc2 | — | — | — | — | — | — | — | insufficient_data |
| roc3 | — | — | — | — | — | — | — | insufficient_data |
| roc5 | — | — | — | — | — | — | — | insufficient_data |
| roc10 | — | — | — | — | — | — | — | insufficient_data |
| ma_gap_3 | — | — | — | — | — | — | — | insufficient_data |
| ma_gap_5 | — | — | — | — | — | — | — | insufficient_data |
| ma_gap_10 | — | — | — | — | — | — | — | insufficient_data |
| ma_gap_20 | — | — | — | — | — | — | — | insufficient_data |
| std3 | — | — | — | — | — | — | — | insufficient_data |
| std5 | — | — | — | — | — | — | — | insufficient_data |
| std10 | — | — | — | — | — | — | — | insufficient_data |
| sharpe_5 | — | — | — | — | — | — | — | insufficient_data |
| vol_ratio_5_20 | — | — | — | — | — | — | — | insufficient_data |
| resi5 | — | — | — | — | — | — | — | insufficient_data |
| resi10 | — | — | — | — | — | — | — | insufficient_data |
| rsqr5 | — | — | — | — | — | — | — | insufficient_data |
| rsqr10 | — | — | — | — | — | — | — | insufficient_data |
| imax5 | — | — | — | — | — | — | — | insufficient_data |
| imin5 | — | — | — | — | — | — | — | insufficient_data |
| imax10 | — | — | — | — | — | — | — | insufficient_data |
| gap | — | — | — | — | — | — | — | insufficient_data |
| overnight | — | — | — | — | — | — | — | insufficient_data |
| intraday | — | — | — | — | — | — | — | insufficient_data |
| gap_reversal | — | — | — | — | — | — | — | insufficient_data |
| reversal_1d | — | — | — | — | — | — | — | insufficient_data |
| vma3 | — | — | — | — | — | — | — | insufficient_data |
| vma5 | — | — | — | — | — | — | — | insufficient_data |
| vma10 | — | — | — | — | — | — | — | insufficient_data |
| vstd5 | — | — | — | — | — | — | — | insufficient_data |
| corr_pv_5 | — | — | — | — | — | — | — | insufficient_data |
| corr_pv_10 | — | — | — | — | — | — | — | insufficient_data |
| wvma5 | — | — | — | — | — | — | — | insufficient_data |
| obv_slope_5 | — | — | — | — | — | — | — | insufficient_data |
| mom_consist | 504997 | +0.0139 | +0.16 | +2.21 | +0.0% | ✗ | 18.0% | ✓ **KEEP** |
| high52w | 483208 | +0.0281 | +0.22 | +3.02 | +0.0% | ✗ | 19.2% | ✓ **KEEP** |
| vol_z | 498136 | -0.0035 | -0.08 | -1.09 | +0.0% | ✗ | 49.5% | ⚫ **DROP** |

## 多周期 IC 衰减

| 因子 | T+1 | T+5 | T+10 | T+21 |
|------|-----|-----|------|------|
| mom_consist | +0.0139 | +0.0317 | +0.0415 | +0.0563 |
| high52w | +0.0281 | +0.0702 | +0.0933 | +0.1220 |
| vol_z | -0.0035 | -0.0059 | +0.0023 | +0.0054 |

## 因子相关矩阵

```
             mom_consist  high52w  vol_z
mom_consist        1.000    0.495  0.026
high52w            0.495    1.000  0.000
vol_z              0.026    0.000  1.000
```


## 详细说明
### mom_consist
- 无特别告警

### high52w
- 无特别告警

### vol_z
- IC 均值 -0.0035，IR -0.08，几乎无 alpha
