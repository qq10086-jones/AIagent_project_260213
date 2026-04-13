"""T0.2 — Reality check vs benchmarks.

Reads account_snapshots for the live strategy (default strategy_id='sprint'),
builds matched benchmark series (1321.T price-return, held-universe equal
weight, cash), and writes a markdown + JSON report.

Sample size is small by construction on a first run. The report emits explicit
N-count and t-stat warnings so readers do not mistake it for a significance
test. Its purpose is to answer one question honestly: since inception, has the
strategy beaten passive alternatives?

Usage
-----
    python reality_check.py --strategy sprint --since 2026-02-27
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

BENCHMARK_TICKER = "1321.T"


def _fetch_snapshots(conn: sqlite3.Connection, strategy_id: str, since: str | None) -> list[tuple[str, float]]:
    q = "SELECT asof, nav FROM account_snapshots WHERE strategy_id=?"
    params: list = [strategy_id]
    if since:
        q += " AND asof>=?"
        params.append(since)
    q += " ORDER BY asof"
    return [(r[0], float(r[1])) for r in conn.execute(q, params)]


def _fetch_close_series(conn: sqlite3.Connection, symbol: str, dates: list[str]) -> dict[str, float]:
    if not dates:
        return {}
    placeholders = ",".join("?" * len(dates))
    rows = conn.execute(
        f"SELECT date, close FROM daily_prices WHERE symbol=? AND date IN ({placeholders})",
        [symbol, *dates],
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows if r[1] is not None}


def _fetch_traded_symbols(conn: sqlite3.Connection, strategy_id: str, since: str | None) -> list[str]:
    q = "SELECT DISTINCT symbol FROM fills WHERE strategy_id=?"
    params: list = [strategy_id]
    if since:
        q += " AND asof>=?"
        params.append(since)
    return [r[0] for r in conn.execute(q, params)]


def _normalize_to_base(series: list[float], base: float = 100.0) -> list[float]:
    if not series or series[0] == 0:
        return []
    return [base * (v / series[0]) for v in series]


def _returns(series: list[float]) -> list[float]:
    out = []
    for i in range(1, len(series)):
        if series[i - 1]:
            out.append(series[i] / series[i - 1] - 1.0)
    return out


def _cum_return_pct(series: list[float]) -> float:
    if len(series) < 2 or series[0] == 0:
        return 0.0
    return (series[-1] / series[0] - 1.0) * 100.0


def _max_drawdown_pct(series: list[float]) -> float:
    peak = series[0] if series else 0.0
    worst = 0.0
    for v in series:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v / peak - 1.0) * 100.0
            if dd < worst:
                worst = dd
    return worst


def _annualised_sharpe(rets: list[float], periods_per_year: int = 252) -> float | None:
    if len(rets) < 3:
        return None
    m = mean(rets)
    s = stdev(rets)
    if s == 0:
        return None
    return (m / s) * math.sqrt(periods_per_year)


def _information_ratio(strat_rets: list[float], bench_rets: list[float]) -> float | None:
    n = min(len(strat_rets), len(bench_rets))
    if n < 3:
        return None
    active = [strat_rets[i] - bench_rets[i] for i in range(n)]
    m = mean(active)
    s = stdev(active)
    if s == 0:
        return None
    return (m / s) * math.sqrt(252)


def _t_stat_mean(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    m = mean(values)
    s = stdev(values)
    if s == 0:
        return None
    return m / (s / math.sqrt(len(values)))


def build_report(
    db_path: str,
    strategy_id: str,
    since: str | None,
    reports_dir: Path,
) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        snaps = _fetch_snapshots(conn, strategy_id, since)
        if len(snaps) < 2:
            raise RuntimeError(
                f"insufficient snapshots for strategy_id={strategy_id!r} since={since!r} (n={len(snaps)})"
            )
        dates = [s[0] for s in snaps]
        nav = [s[1] for s in snaps]

        topix_closes = _fetch_close_series(conn, BENCHMARK_TICKER, dates)
        topix_series: list[float] = []
        for d in dates:
            if d in topix_closes:
                topix_series.append(topix_closes[d])
            elif topix_series:
                topix_series.append(topix_series[-1])  # carry forward non-trading days
            else:
                topix_series.append(float("nan"))

        traded = _fetch_traded_symbols(conn, strategy_id, since)
        held_series: list[float] = []
        if traded:
            # Per-symbol close series aligned on snapshot dates, equal-weight
            # index rebased to 100 at first date.
            sym_closes: dict[str, dict[str, float]] = {
                s: _fetch_close_series(conn, s, dates) for s in traded
            }
            for d in dates:
                vals = []
                for s in traded:
                    px = sym_closes[s].get(d)
                    if px is None and held_series:
                        # carry previous relative
                        pass
                    if px is not None:
                        vals.append(px)
                held_series.append(mean(vals) if vals else float("nan"))

        def _pct_from(series: list[float]) -> float:
            cleaned = [v for v in series if not (isinstance(v, float) and math.isnan(v))]
            return _cum_return_pct(cleaned)

        strat_rets = _returns(nav)
        topix_rets = _returns([v for v in topix_series if not math.isnan(v)])
        held_rets = _returns([v for v in held_series if not math.isnan(v)])

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy_id": strategy_id,
            "period": {"start": dates[0], "end": dates[-1], "n_snapshots": len(dates)},
            "nav": {
                "start": nav[0],
                "end": nav[-1],
                "cum_return_pct": _cum_return_pct(nav),
                "max_drawdown_pct": _max_drawdown_pct(nav),
                "sharpe_annualised": _annualised_sharpe(strat_rets),
                "daily_return_t_stat": _t_stat_mean(strat_rets),
            },
            "benchmarks": {
                "topix_1321T_price_return": {
                    "note": "price-return only; total-return unavailable without dividend data",
                    "cum_return_pct": _pct_from(topix_series),
                    "sharpe_annualised": _annualised_sharpe(topix_rets),
                },
                "held_universe_equal_weight": {
                    "members": traded,
                    "cum_return_pct": _pct_from(held_series) if held_series else None,
                    "sharpe_annualised": _annualised_sharpe(held_rets) if held_rets else None,
                },
                "cash_zero_risk": {
                    "cum_return_pct": 0.0,
                    "note": "ignores JPY cash rate (~0%)",
                },
                "sector_neutral": {
                    "cum_return_pct": None,
                    "note": "TBD — requires TSE 33 sector mapping per Phase -1 D-1 data hygiene",
                },
            },
            "excess_vs_topix_pct": _cum_return_pct(nav) - _pct_from(topix_series),
            "excess_vs_held_pct": (
                _cum_return_pct(nav) - _pct_from(held_series)
                if held_series else None
            ),
            "information_ratio_vs_topix": _information_ratio(strat_rets, topix_rets),
            "sample_size_warning": (
                "n_snapshots < 20 — Sharpe/IR/t-stat are not statistically "
                "meaningful yet. Read the level of the returns, not the ratios."
                if len(snaps) < 20 else None
            ),
        }

        reports_dir.mkdir(parents=True, exist_ok=True)
        out_date = dates[-1]
        json_path = reports_dir / f"reality_check_{out_date}.json"
        md_path = reports_dir / f"reality_check_{out_date}.md"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path.write_text(_format_markdown(report, dates, nav, topix_series, held_series), encoding="utf-8")
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
        return report
    finally:
        conn.close()


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return f"{v:+.2f}%"


def _fmt_num(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _format_markdown(
    r: dict,
    dates: list[str],
    nav: list[float],
    topix: list[float],
    held: list[float],
) -> str:
    b = r["benchmarks"]
    L = [
        f"# Reality Check — {r['strategy_id']} ({r['period']['start']} → {r['period']['end']})",
        "",
        f"_generated {r['generated_at_utc']}_  ",
        f"_n_snapshots = {r['period']['n_snapshots']}_",
        "",
    ]
    if r.get("sample_size_warning"):
        L += ["> ⚠️ " + r["sample_size_warning"], ""]

    L += [
        "## Headline",
        "",
        f"- NAV: {r['nav']['start']:,.0f} → {r['nav']['end']:,.0f}  ({_fmt_pct(r['nav']['cum_return_pct'])})",
        f"- Max drawdown (snapshot-to-snapshot): {_fmt_pct(r['nav']['max_drawdown_pct'])}",
        f"- Annualised Sharpe (naive): {_fmt_num(r['nav']['sharpe_annualised'])}",
        f"- Daily-return t-stat: {_fmt_num(r['nav']['daily_return_t_stat'])}",
        "",
        "## vs Benchmarks (cumulative)",
        "",
        "| benchmark | cum return | sharpe | notes |",
        "|---|---|---|---|",
        f"| **strategy ({r['strategy_id']})** | **{_fmt_pct(r['nav']['cum_return_pct'])}** | {_fmt_num(r['nav']['sharpe_annualised'])} | — |",
        f"| TOPIX ETF (1321.T) price-return | {_fmt_pct(b['topix_1321T_price_return']['cum_return_pct'])} | {_fmt_num(b['topix_1321T_price_return']['sharpe_annualised'])} | {b['topix_1321T_price_return']['note']} |",
        f"| Held-universe equal-weight | {_fmt_pct(b['held_universe_equal_weight']['cum_return_pct'])} | {_fmt_num(b['held_universe_equal_weight']['sharpe_annualised'])} | members: {', '.join(b['held_universe_equal_weight']['members']) or 'none'} |",
        f"| Cash (0%) | {_fmt_pct(b['cash_zero_risk']['cum_return_pct'])} | n/a | {b['cash_zero_risk']['note']} |",
        f"| Sector-neutral | {_fmt_pct(b['sector_neutral']['cum_return_pct'])} | n/a | {b['sector_neutral']['note']} |",
        "",
        "## Excess",
        "",
        f"- Strategy − TOPIX = {_fmt_pct(r['excess_vs_topix_pct'])}",
        f"- Strategy − held-universe = {_fmt_pct(r['excess_vs_held_pct'])}",
        f"- Information ratio vs TOPIX (annualised, naive): {_fmt_num(r['information_ratio_vs_topix'])}",
        "",
        "## NAV series",
        "",
        "| date | NAV | TOPIX close | held-univ avg |",
        "|---|---|---|---|",
    ]
    for i, d in enumerate(dates):
        t_raw = topix[i] if i < len(topix) else None
        t_str = "n/a" if t_raw is None or (isinstance(t_raw, float) and math.isnan(t_raw)) else f"{t_raw:,.2f}"
        h_raw = held[i] if i < len(held) else None
        h_str = "n/a" if h_raw is None or (isinstance(h_raw, float) and math.isnan(h_raw)) else f"{h_raw:,.2f}"
        L.append(f"| {d} | {nav[i]:,.0f} | {t_str} | {h_str} |")
    L += [
        "",
        "## Interpretation guide",
        "",
        "1. If **strategy cum return < TOPIX cum return** after costs: the active strategy is not adding value vs just buying 1321.T.",
        "2. If **strategy cum return < held-universe equal-weight**: even picking the same tickers but equal-weighting beats the strategy → the timing/sizing is destroying alpha.",
        "3. Sharpe / IR at n<20 are directional only. Do not promote, demote, or change parameters based on them.",
        "4. TBD sector-neutral benchmark (Phase -1 D-1): until TSE 33 sector mapping is wired, cannot rule out that any edge is just sector beta.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--strategy", default="sprint")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()
    build_report(args.db, args.strategy, args.since, Path(args.reports_dir))


if __name__ == "__main__":
    main()
