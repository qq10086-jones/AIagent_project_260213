"""Tests for the fundamental research-cohort lane (P19-02b, Rule 16.2).

A SEPARATE lane from the §10 decision log: monthly broad-universe cohorts of
{earnings_yield, value_bp} scores, swept at 21D/63D from an injected price
series. Contracts: PIT scoring, idempotent emit, maturity-honest sweep (no
premature returns), IC report shape, and writes confined to
reports/research_cohorts/.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.backtesting.fundamental_cohort import (  # noqa: E402
    build_cohort_rows,
    cohort_report,
    emit_cohort,
    sweep_cohorts,
)


def _series(n=100, start_val=100.0, step=1.0):
    dates = [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
    closes = [start_val + i * step for i in range(n)]
    return dates, closes


def test_build_rows_pit_scores_and_skips_missing():
    eps = {"A.T": 100.0, "B.T": None}
    bps = {"A.T": 1500.0, "B.T": 2000.0}
    px = {"A.T": 2000.0, "B.T": 1000.0, "C.T": 500.0}
    rows = build_cohort_rows(
        "2026-07-05",
        symbols=["A.T", "B.T", "C.T"],
        eps_lookup=lambda s, d: eps.get(s),
        bps_lookup=lambda s, d: bps.get(s),
        price_lookup=lambda s: px.get(s),
    )
    by = {r["symbol"]: r for r in rows}
    assert by["A.T"]["earnings_yield"] == 0.05
    assert by["A.T"]["value_bp"] == 0.75
    assert by["B.T"]["earnings_yield"] is None  # missing eps → honest None
    assert by["B.T"]["value_bp"] == 2.0
    assert "C.T" not in by  # no fundamentals at all → dropped
    assert all(r["asof"] == "2026-07-05" for r in rows)


def test_emit_idempotent(tmp_path):
    rows = [{"symbol": "A.T", "asof": "2026-07-05", "ref_price": 100.0,
             "earnings_yield": 0.05, "value_bp": 0.8}]
    p1 = emit_cohort(tmp_path, "2026-07-05", rows)
    p2 = emit_cohort(tmp_path, "2026-07-05", rows)  # second call: no rewrite
    assert p1 == p2
    assert p1.exists()
    assert str(p1).startswith(str(tmp_path / "reports" / "research_cohorts"))
    lines = p1.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # not duplicated


def test_sweep_matures_only_closed_horizons(tmp_path):
    dates, closes = _series(100)
    emit_cohort(tmp_path, dates[0], [
        {"symbol": "A.T", "asof": dates[0], "ref_price": closes[0],
         "earnings_yield": 0.05, "value_bp": 0.8},
    ])
    series = {"A.T": (dates, closes)}
    lookup = lambda sym: series.get(sym)  # noqa: E731

    # only ~30 sessions elapsed → 21D closes, 63D must stay unmatured
    out = sweep_cohorts(tmp_path, price_series=lookup, today=dates[35])
    rec = json.loads((tmp_path / "reports" / "research_cohorts" / "fundamental"
                      / "outcomes" / f"{dates[0]}.jsonl").read_text().splitlines()[0])
    assert rec["ret_21d"] is not None
    assert rec["ret_63d"] is None
    assert out["swept_rows"] == 1

    # after enough sessions, 63D fills in (re-sweep replaces the outcome file)
    sweep_cohorts(tmp_path, price_series=lookup, today=dates[99])
    rec2 = json.loads((tmp_path / "reports" / "research_cohorts" / "fundamental"
                       / "outcomes" / f"{dates[0]}.jsonl").read_text().splitlines()[0])
    assert rec2["ret_63d"] is not None
    # entry = first date strictly after asof → closes[1]; exit = closes[1+63]
    assert abs(rec2["ret_63d"] - (closes[64] / closes[1] - 1.0)) < 1e-9


def test_cohort_report_rank_ic_shape(tmp_path):
    dates, closes = _series(100)
    # two cohorts, many names: cheap names (high ey) get faster-rising series
    n_names = 12
    series = {}
    rows = []
    for i in range(n_names):
        sym = f"S{i}.T"
        series[sym] = (dates, [100.0 + j * (0.5 + 0.1 * i) for j in range(100)])
        rows.append({"symbol": sym, "asof": dates[0], "ref_price": 100.0,
                     "earnings_yield": 0.01 * i, "value_bp": 0.1 * i})
    emit_cohort(tmp_path, dates[0], rows)
    sweep_cohorts(tmp_path, price_series=lambda s: series.get(s), today=dates[99])
    rep = cohort_report(tmp_path, min_names=5)
    assert rep["n_cohorts"] == 1
    ey63 = rep["signals"]["earnings_yield"]["63"]
    assert ey63["n_cohorts"] == 1
    assert ey63["mean_ic"] > 0.9  # monotone construction → near-perfect rank IC
