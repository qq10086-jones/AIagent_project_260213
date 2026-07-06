"""Bi-weekly DSR-progress digest (reads the forward-eval trace; no recompute).

The afterclose daily_routine already appends a row to
``reports/observability/forward_signal_eval_trace.jsonl`` each day. This digest
just READS that trace, compares the latest row to the one ~`--lookback-days` ago,
writes a human-readable ``reports/observability/DSR_PROGRESS.txt``, and prints a
one-line ``HEADLINE: ...`` for a desktop alert. Research-only; touches nothing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "reports" / "observability" / "forward_signal_eval_trace.jsonl"
OUT = ROOT / "reports" / "observability" / "DSR_PROGRESS.txt"


def _rows():
    if not TRACE.exists():
        return []
    rows = []
    for ln in TRACE.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    rows.sort(key=lambda r: r.get("asof", ""))
    return rows


def _fnum(x, fmt="{:+.4f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "—"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback-days", type=int, default=14)
    args = ap.parse_args(argv)

    rows = _rows()
    lines = ["=== price_reversal DSR progress (research-only; not tradable until gated) ==="]
    if not rows:
        head = "DSR trace empty — daily afterclose has not appended yet."
        lines.append(head)
    else:
        cur = rows[-1]
        prior = None
        try:
            cut = (_dt.date.fromisoformat(cur["asof"]) - _dt.timedelta(days=args.lookback_days)).isoformat()
            earlier = [r for r in rows if r.get("asof", "") <= cut]
            prior = earlier[-1] if earlier else (rows[0] if len(rows) > 1 else None)
        except (ValueError, KeyError):
            prior = rows[0] if len(rows) > 1 else None

        dsr = cur.get("pr_dsr")
        nobs = cur.get("pr_n_obs_eff")
        ic = cur.get("pr_best_ic")
        cfg = cur.get("pr_best_config")
        gated = cur.get("pr_dsr_pass")
        delta = ""
        if prior and isinstance(dsr, (int, float)) and isinstance(prior.get("pr_dsr"), (int, float)):
            d = dsr - prior["pr_dsr"]
            delta = f" ({d:+.3f} vs {prior['asof']})"

        dsr_s = f"{dsr:.3f}" if isinstance(dsr, (int, float)) else "—"
        head = (
            f"DSR {dsr_s}/0.95{delta} | n_obs {nobs}/60 | {cfg} IC {_fnum(ic)} | "
            f"{'PROMOTABLE — re-run full ADR-0010 gate' if gated and isinstance(nobs,int) and nobs>=60 else 'still accumulating, not tradable'}"
        )
        lines += [
            f"as of {cur.get('asof')}  (trace rows: {len(rows)})",
            "",
            f"  Deflated Sharpe : {dsr_s} / 0.95 target{delta}",
            f"  effective obs   : {nobs} / 60 minimum",
            f"  best config     : {cfg}   forward Rank-IC {_fnum(ic)}",
            f"  screener_buy 5D : IC {_fnum(cur.get('screener_buy_ic_5d'))}  (production score; negative = why calibration stays gated)",
            f"  orthogonality   : rho {_fnum(cur.get('orthogonality_pr5_vs_buy'), '{:+.3f}')}  vs buy score",
            "",
            f"  verdict: {'PROMOTABLE candidate — run full ADR-0010 gate (DSR+PBO+regime)' if (gated and isinstance(nobs,int) and nobs>=60) else 'NOT promotable yet — keep accumulating forward samples'}",
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nHEADLINE: {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
