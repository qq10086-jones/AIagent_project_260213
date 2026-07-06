"""Forward shadow-eval report + DSR trace (§16 / ADR-0010, Rule 16.2 live-only).

Runs candidate signals through the live-only forward gate and records, per horizon,
Rank-IC / t-stat / cost-hurdle / net-after-cost / clears?, PLUS the anti-overfit
Deflated Sharpe (overfit_gate.promote_gate) for the price_reversal family — the
metric whose climb toward 0.95 is the promotion milestone.

Read-only research: NEVER touches score_status, calibration, or advice (Rule 16.6
defers promotion to the ADR-0010 gate). Wired into daily_routine afterclose as a
NON-FATAL diagnostic so the trace self-accumulates ("stop waiting, measure").

Artifacts (under reports/observability/):
- forward_signal_eval/{asof}.json     full per-signal + DSR snapshot
- forward_signal_eval_trace.jsonl     one summary row per run (DSR/IC trajectory)

Signals: screener_buy (production score, identity) · reversal_of_score (-buy) ·
price_reversal_{2,3,5,10,20}d (-prior N-session return, PIT from price DB).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from math import sqrt
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.backtesting.signal_library import (  # noqa: E402
    cross_signal_rank_corr,
    evaluate_signal,
    fundamentals_pit_lookup,
    kline_prior_return_lookup,
    make_fundamental_yield_signal,
    make_price_reversal_signal,
    reversal_of_score,
    _load_records_and_returns,
)
from hot_theme_rotator.calibration.overfit_gate import promote_gate  # noqa: E402

LOOKBACK_GRID = (2, 3, 5, 10, 20)
HORIZON_GRID = (1, 3, 5)


def screener_buy(records):
    return {r.prediction_id: r.buy for r in records}


screener_buy.__name__ = "screener_buy"


def _fmt(rec: dict) -> str:
    if rec.get("n_days", 0) == 0:
        return "no usable days"
    net = rec.get("net_ic_after_cost")
    net_s = f"{net:+.4f}" if isinstance(net, (int, float)) else "—"
    return (
        f"n_days={rec['n_days']:>2}  IC={rec['mean_ic']:+.4f}  t={rec.get('t_stat', 0):+.2f}  "
        f"net={net_s}  clears={rec.get('clears', '—')}"
    )


def _price_reversal_dsr(base: str) -> dict:
    """Honest Bailey/Lopez-de-Prado DSR for the BEST price_reversal config, deflated
    against the full (lookback x horizon) search breadth. n_obs is overlap-adjusted
    (n_days // horizon) because forward windows overlap day-to-day."""
    trials = []  # (lookback, horizon, sr, n_days, mean_ic)
    for lb in LOOKBACK_GRID:
        res = evaluate_signal(
            make_price_reversal_signal(lookback=lb, return_lookup=kline_prior_return_lookup(lookback=lb)),
            base,
            horizons=HORIZON_GRID,
        )
        for H, rec in res["horizons"].items():
            nd = rec.get("n_days", 0)
            if nd and nd > 1:
                sr = rec["t_stat"] / sqrt(nd)  # per-observation Sharpe of the daily-IC series
                trials.append({"lookback": lb, "horizon": H, "sr": sr, "n_days": nd, "mean_ic": rec["mean_ic"]})
    if len(trials) < 2:
        return {"available": False, "reason": "need >=2 live trials"}
    sr_std = pstdev([t["sr"] for t in trials])
    best = max(trials, key=lambda t: t["mean_ic"])  # the config we'd actually pick
    n_obs_eff = max(2, best["n_days"] // best["horizon"])  # overlap-adjusted independent obs
    gate = promote_gate(
        best["sr"], n_trials=len(trials), n_obs=n_obs_eff, sr_std=sr_std, min_obs=60, dsr_threshold=0.95
    )
    return {
        "available": True,
        "best_config": f"price_reversal_{best['lookback']}d @ {best['horizon']}D",
        "best_mean_ic": round(best["mean_ic"], 4),
        "best_sr_per_obs": round(best["sr"], 4),
        "sr_std_across_trials": round(sr_std, 4),
        "n_trials": len(trials),
        "n_obs_effective": n_obs_eff,
        "n_days_raw": best["n_days"],
        **gate,  # dsr, expectedMaxSharpe, pass, reasons, ...
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", default=None, help="ISO date stamp for the artifact (default: today).")
    ap.add_argument("--base-dir", default=str(ROOT))
    ap.add_argument("--no-write", action="store_true", help="Print only; do not write artifacts.")
    args = ap.parse_args(argv)
    base = args.base_dir
    asof = args.asof or _dt.date.today().isoformat()

    signals = {"screener_buy": screener_buy, "reversal_of_score": reversal_of_score}
    for lb in (5, 10):
        signals[f"price_reversal_{lb}d"] = make_price_reversal_signal(
            lookback=lb, return_lookup=kline_prior_return_lookup(lookback=lb)
        )
    # P19-02 (2026-07-04): the P23-B gate-passing family enters the live forward
    # track. NOTE: their historical verdict horizon is 63D; the forward log only
    # matures 1/3/5D today, so this accumulates sign-consistency + orthogonality
    # evidence — the 63D forward verdict needs the extended-horizon outcome
    # collection (P19-02b, pending). Never promotes anything by itself.
    for field in ("eps", "bps"):
        fn = make_fundamental_yield_signal(
            field=field, fundamental_lookup=fundamentals_pit_lookup(field)
        )
        signals[fn.__name__] = fn
    table = {name: evaluate_signal(fn, base) for name, fn in signals.items()}
    dsr = _price_reversal_dsr(base)

    recs, _ = _load_records_and_returns(base, 5)
    rho = None
    rho_ey = None
    if recs:
        pr5 = make_price_reversal_signal(lookback=5, return_lookup=kline_prior_return_lookup(lookback=5))
        rho = cross_signal_rank_corr(recs, pr5(recs), screener_buy(recs))
        # Rule 16.3 — fundamentals vs the production momentum score (stacking
        # requires |rho| < 0.5; fundamentals should be near-orthogonal).
        rho_ey = cross_signal_rank_corr(
            recs, signals["earnings_yield"](recs), screener_buy(recs)
        )

    # ---- console ----
    print(f"=== FORWARD SHADOW-EVAL  asof={asof}  (live-only, Rule 16; research-only) ===")
    for name, res in table.items():
        print(f"\n[{name}]")
        for H, rec in res["horizons"].items():
            print(f"  {H}D: {_fmt(rec)}")
    print("\n[price_reversal ANTI-OVERFIT GATE (ADR-0010)]")
    if dsr.get("available"):
        print(f"  best={dsr['best_config']}  IC={dsr['best_mean_ic']:+.4f}  "
              f"DSR={dsr['dsr']:.3f} / 0.95  pass={dsr['pass']}  "
              f"(n_obs_eff={dsr['n_obs_effective']}/60, n_trials={dsr['n_trials']})")
        for r in dsr.get("reasons", []):
            print(f"    - {r}")
    else:
        print(f"  {dsr.get('reason')}")
    rho_s = f"{rho:+.3f}" if rho is not None else "—"
    print(f"\northogonality price_reversal_5d vs screener_buy: rho={rho_s}")
    rho_ey_s = f"{rho_ey:+.3f}" if rho_ey is not None else "—"
    print(f"orthogonality earnings_yield vs screener_buy: rho={rho_ey_s} (Rule 16.3 stack needs |rho|<0.5)")

    # ---- artifacts ----
    if not args.no_write:
        obs = Path(base) / "reports" / "observability"
        (obs / "forward_signal_eval").mkdir(parents=True, exist_ok=True)
        snap = {"asof": asof, "table": table, "price_reversal_gate": dsr,
                "orthogonality_pr5_vs_buy": rho,
                "orthogonality_ey_vs_buy": rho_ey}
        (obs / "forward_signal_eval" / f"{asof}.json").write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        buy5 = table["screener_buy"]["horizons"].get(5, {})
        ey5 = table["earnings_yield"]["horizons"].get(5, {})
        vb5 = table["value_bp"]["horizons"].get(5, {})
        row = {
            "asof": asof,
            "screener_buy_ic_5d": buy5.get("mean_ic"),
            "earnings_yield_ic_5d": ey5.get("mean_ic"),
            "value_bp_ic_5d": vb5.get("mean_ic"),
            "pr_best_config": dsr.get("best_config"),
            "pr_best_ic": dsr.get("best_mean_ic"),
            "pr_dsr": dsr.get("dsr"),
            "pr_dsr_pass": dsr.get("pass"),
            "pr_n_obs_eff": dsr.get("n_obs_effective"),
            "orthogonality_pr5_vs_buy": rho,
            "orthogonality_ey_vs_buy": rho_ey,
        }
        with (obs / "forward_signal_eval_trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {obs / 'forward_signal_eval' / (asof + '.json')}")
        print(f"appended trace row -> {obs / 'forward_signal_eval_trace.jsonl'}")
    print("\nNote: DSR pass is NECESSARY-not-sufficient; promotion also needs PBO + the")
    print("forward log (Rule 16.6). This runner accumulates that log; it never promotes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
