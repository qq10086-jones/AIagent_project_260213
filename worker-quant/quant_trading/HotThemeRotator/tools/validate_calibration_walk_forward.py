"""Run the purged + embargoed walk-forward validation on real samples (P12-02).

Joins reports/predictions/*.jsonl with complete reports/outcomes/*.jsonl, derives a
binary label (3D realized return > 0), and runs the leak-resistant protocol.

DIAGNOSTIC ONLY for backdated/bootstrap evidence (Rule 8.2.2.2): a result here can
NOT promote any UI label or satisfy the ship gate — backdated evidence is quarantined
(Rule 9.4.2.3) and forward-sample primacy (Rule 8.2.2.2) still applies. Use this on
genuine forward samples once they mature.

Usage::
  python tools/validate_calibration_walk_forward.py [--origin all|backdated|live] [--horizon 3]
      [--leakage-verdict clean|inconclusive|contaminated]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.calibration.purged_walk_forward import (  # noqa: E402
    WFSample, walk_forward_validate,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

PRED_DIR = ROOT / "reports" / "predictions"
OUT_DIR = ROOT / "reports" / "outcomes"


def _load_samples(origin: str, horizon: str) -> list[WFSample]:
    preds = {}
    for f in sorted(glob.glob(str(PRED_DIR / "*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ex = r.get("extra", {}) or {}
            is_backdated = ex.get("backdated") is True
            if origin == "backdated" and not is_backdated:
                continue
            if origin == "live" and is_backdated:
                continue
            preds[r["prediction_id"]] = (float(r.get("buy", 0.0)), str(r.get("trade_date")))
    samples = []
    for f in sorted(glob.glob(str(OUT_DIR / "*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("status") != "complete":
                continue
            pid = o.get("prediction_id")
            if pid not in preds:
                continue
            ret = (o.get("realized_returns") or {}).get(horizon)
            if ret is None:
                continue
            score, td = preds[pid]
            score = max(0.0, min(1.0, score))
            samples.append(WFSample(score, 1 if float(ret) > 0 else 0, td))
    return samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", choices=["all", "backdated", "live"], default="all")
    ap.add_argument("--horizon", default="3D")
    ap.add_argument("--asof", default=None)
    ap.add_argument(
        "--leakage-verdict",
        choices=["clean", "inconclusive", "contaminated"],
        default="clean",
        help="Forward leakage audit verdict; non-clean cannot ship (Rule 8.2.3).",
    )
    args = ap.parse_args(argv)

    horizon_days = int(str(args.horizon).rstrip("Dd")) or 3
    samples = _load_samples(args.origin, args.horizon)
    report = walk_forward_validate(
        samples,
        horizon_days=horizon_days,
        leakage_verdict=args.leakage_verdict,
    )
    report["origin"] = args.origin
    report["label"] = f"{args.horizon} realized return > 0"
    report["diagnostic_only"] = (args.origin != "live")
    report["note"] = (
        "DIAGNOSTIC ONLY (Rule 8.2.2.2): backdated/bootstrap evidence is quarantined "
        "(Rule 9.4.2.3) and cannot un-downgrade the UI; forward primacy applies."
        if args.origin != "live" else
        "Forward-sample validation. Still requires a clean Rule 9.4.2 leakage verdict."
    )
    asof = args.asof or date.today().isoformat()
    out = ROOT / "reports" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / f"walk_forward_{args.origin}_{asof}.json"
    artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nverdict={report['verdict']} (origin={args.origin}) -> {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
