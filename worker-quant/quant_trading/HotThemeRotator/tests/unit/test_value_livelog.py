"""Tests for the retroactive value-on-live-log read (P23-F).

Guards the join logic: value score = per-share value / reference_price, forward
return from adjusted prices with a PIT entry (strictly after trade_date), and
honest maturity accounting (unmatured windows excluded, not zero-filled).
"""
import sys
from collections import namedtuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.backtest_value_on_livelog import _fwd_return, value_livelog_read  # noqa: E402

Rec = namedtuple("Rec", "symbol trade_date reference_price")


def _series(n, start=100.0, step=1.0):
    dates = [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
    return ([d for d in dates], [start + i * step for i in range(n)])


def test_fwd_return_pit_entry_and_maturity():
    s = _series(40, 100.0, 1.0)  # rising 1/day
    # asof at index ~ date; entry is first session STRICTLY after asof
    r = _fwd_return(s, s[0][5], 10)
    # entry = close[6]=106, exit = close[16]=116 -> 116/106-1
    assert abs(r - (116 / 106 - 1)) < 1e-9
    # window not closed -> None (only ~5 sessions left after asof)
    assert _fwd_return(s, s[0][36], 10) is None
    assert _fwd_return(None, "2026-01-01", 10) is None


def test_value_score_is_pit_value_over_reference_price():
    # one date, enough names: cheaper (higher eps/price) should rank with higher return
    series = {}
    recs = []
    n = 8
    for i in range(n):
        sym = f"S{i}.T"
        # cheaper names (higher i -> higher eps) get faster-rising series
        series[sym] = ([f"2026-06-{d:02d}" for d in range(1, 30)],
                       [100.0 + j * (0.2 + 0.15 * i) for j in range(29)])
        recs.append(Rec(sym, "2026-06-01", 1000.0))
    eps = {f"S{i}.T": 10.0 * i for i in range(n)}   # eps/price = 0.01*i (monotone)
    out = value_livelog_read(
        recs, lambda s: series.get(s),
        eps_lookup=lambda s, d: eps.get(s), bps_lookup=lambda s, d: None,
        horizons=(21,), min_names=5)
    ey = out["earnings_yield"][21]
    assert ey["matured"] == n
    # single date -> n_dates 1 -> rank_ic needs >=2 dates -> mean_ic None, but the
    # join happened (matured counted). Add a second date to get an IC.
    assert ey["n_dates"] == 1 and ey["mean_ic"] is None


def test_two_dates_give_positive_ic_when_cheap_outperforms():
    dates = [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(60)]
    series, recs = {}, []
    for di, tdate in enumerate((dates[0], dates[5])):  # both early enough for 21D to close
        for i in range(8):
            sym = f"D{di}_S{i}.T"
            series[sym] = (dates, [100.0 + j * (0.2 + 0.2 * i) for j in range(60)])
            recs.append(Rec(sym, tdate, 1000.0))
    eps = {r.symbol: 10.0 * int(r.symbol.split("_S")[1].split(".")[0]) for r in recs}
    out = value_livelog_read(
        recs, lambda s: series.get(s),
        eps_lookup=lambda s, d: eps.get(s), bps_lookup=lambda s, d: None,
        horizons=(21,), min_names=5)
    ey = out["earnings_yield"][21]
    assert ey["n_dates"] == 2
    assert ey["mean_ic"] > 0.9  # monotone construction -> near-perfect rank IC


def test_missing_reference_price_or_value_skipped():
    series = {"A.T": _series(40)}
    recs = [Rec("A.T", "2026-06-01", None), Rec("A.T", "2026-06-01", 0.0)]
    out = value_livelog_read(
        recs, lambda s: series.get(s),
        eps_lookup=lambda s, d: 10.0, bps_lookup=lambda s, d: None, horizons=(21,))
    assert out["earnings_yield"][21]["matured"] == 0
