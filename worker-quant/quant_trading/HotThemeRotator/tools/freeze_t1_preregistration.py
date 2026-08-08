"""P34-02 — freeze the T1 buyback-resolution analysis plan and register its family.

    python tools/freeze_t1_preregistration.py --asof 2026-08-08
    python tools/freeze_t1_preregistration.py --asof 2026-08-08 --dry-run

Freezing is a one-way door by construction: re-running with identical content is
idempotent, and any change requires a new version so the original promise stays
on disk. This tool reads NO outcomes and computes NO returns — that is the whole
point of running it before the analysis exists.

Rule 3 / Rule 4: freezing a plan authorizes no capital, promotes no signal, and
changes no config. It only fixes what a later confirmatory run is allowed to do.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.preregistration import (  # noqa: E402
    AnalysisPlan,
    PreregistrationError,
    freeze_plan,
)
from hot_theme_rotator.research.trial_registry import (  # noqa: E402
    DuplicateTrialError,
    register_trial,
)

PLAN_ID = "P34_T1_buyback_resolution"
FAMILY_ID = "P34_T1_v1"
PRIMARY_HORIZON = 20
SECONDARY_HORIZONS = [5, 10, 40, 60]
STRATA = ["auction", "tostnet", "method_unknown"]


def build_plan(frozen_at: str, event_rate: float | None) -> AnalysisPlan:
    return AnalysisPlan(
        plan_id=PLAN_ID,
        version=1,
        frozen_at=frozen_at,
        provenance="prospective",
        hypothesis=(
            "Uncontaminated TDnet buyback RESOLUTIONS (自己株式の取得に係る事項の決定) "
            "are followed by positive cumulative abnormal return versus 1306.T over "
            f"{PRIMARY_HORIZON} trading days. Direction is predicted positive; a null "
            "or negative result is a valid and reportable outcome."
        ),
        event_definition=(
            "A TDnet disclosure whose buyback subtype is `resolution` per "
            "hot_theme_rotator.data.external.buyback_events.classify_buyback_subtype "
            "(parser_version p34-01a-v1), with an empty contamination tuple. "
            "Event time is `published_ts` (PIT: TDnet publication), never "
            "collected_ts and never the trade date."
        ),
        inclusion_criteria=[
            "subtype == 'resolution'",
            "contamination == () (no same-release earnings/dividend/split/cancellation)",
            "ticker resolves to a listed TSE name with price data on the entry day",
            "is_correction == False for the primary sample",
        ],
        exclusion_criteria=[
            "subtype == 'disposal' (処分 is a treasury-share DISPOSAL, opposite sign; "
            "it is the largest treasury subtype and must never enter the sample)",
            "subtype in ('cancellation','execution_report','completion','other_treasury')",
            "same-release earnings or dividend co-announcement (contamination flags)",
            "corrections and modifications enter a separate secondary stratum only",
        ],
        entry_rule=(
            "Enter at the OPEN of the first trading day strictly after published_ts. "
            "A disclosure published intraday is NOT tradable at that day's open, so "
            "same-day entry would be look-ahead."
        ),
        benchmark="1306.T (TOPIX ETF), the same benchmark the existing event-study skeleton uses",
        primary_horizon_days=PRIMARY_HORIZON,
        secondary_horizons_days=SECONDARY_HORIZONS,
        strata=STRATA,
        test_statistic="mean cumulative abnormal return (CAR) vs 1306.T",
        inference_method=(
            "date-cluster bootstrap over event dates (events sharing a publication "
            "date are one cluster); calendar-time portfolio as the overlap-robust "
            "cross-check. Overlapping holding periods are NOT treated as independent."
        ),
        multiple_testing=(
            f"every horizon x stratum combination is a trial registered in {FAMILY_ID}; "
            "the deflation denominator is the registry count, and the P31 frozen "
            "family is cited additively, never merged"
        ),
        stopping_rule=(
            "No interim peeking. The first confirmatory read happens when the primary "
            "horizon has matured for at least 100 uncontaminated resolutions; until "
            "then the lane reports event counts only, never returns."
        ),
        expected_event_rate_per_year=event_rate,
        trial_family_id=FAMILY_ID,
        notes=(
            "Literature support is a HYPOTHESIS, not validation: Japanese buyback "
            "announcement drift is reported in the literature and a 2025 PBFJ "
            "registered report tests it, but a registered report supplies design "
            "credibility rather than a positive result. Nothing here is [V]. "
            "Parser limitation recorded at freeze time: amount/share caps and windows "
            "are absent from RSS titles in ~91% of observed treasury disclosures, so "
            "size-based strata are NOT part of the primary plan."
        ),
    )


def _measured_event_rate(base: Path, asof: str) -> tuple[float | None, dict]:
    """Annualize the measured T1 event rate from the P34-01a summary, if present."""
    path = base / "reports/research/buyback_events" / f"summary_{asof}.json"
    if not path.exists():
        return None, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    n = data.get("t1_primary_events")
    span = data.get("source_date_range")
    if not n or not span:
        return None, data
    d0 = datetime.fromisoformat(span[0]).date()
    d1 = datetime.fromisoformat(span[1]).date()
    cal_days = max((d1 - d0).days, 1)
    # ~245 TSE trading days/yr; the corpus span is in calendar days.
    return round(n * (365.0 / cal_days), 1), data


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rate, summary = _measured_event_rate(base, args.asof)
    plan = build_plan(frozen_at, rate)

    if args.dry_run:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        return 0

    try:
        path = freeze_plan(plan, base_dir=base)
    except PreregistrationError as exc:
        print(f"FREEZE REFUSED: {exc}", file=sys.stderr)
        return 1

    print(f"frozen plan     : {path}")
    print(f"provenance      : {plan.provenance} (confirmatory={plan.is_confirmatory})")
    print(f"primary horizon : {plan.primary_horizon_days}D vs {plan.benchmark}")
    print(f"measured T1 rate: {rate} events/yr"
          + (f"  (from {summary.get('t1_primary_events')} events over "
             f"{summary.get('source_date_range')})" if summary else "  (no summary artifact)"))

    # Register the horizon x stratum grid BEFORE any outcome is read.
    registered, duplicates = 0, 0
    for horizon in [PRIMARY_HORIZON] + SECONDARY_HORIZONS:
        for stratum in STRATA:
            config = {
                "plan_id": PLAN_ID, "plan_version": 1,
                "horizon_days": horizon, "stratum": stratum,
                "benchmark": plan.benchmark, "entry_rule": plan.entry_rule,
                "parser_version": "p34-01a-v1",
            }
            try:
                register_trial(
                    family_id=FAMILY_ID,
                    hypothesis=f"buyback resolution CAR>0 at {horizon}D, stratum={stratum}",
                    config=config,
                    base_dir=base,
                    hypothesis_lineage=["P34-02", "PBFJ2025_registered_report(hypothesis_only)"],
                    horizon_days=horizon,
                    note="registered at freeze time; no outcome read",
                )
                registered += 1
            except DuplicateTrialError:
                duplicates += 1

    print(f"trials registered: {registered} new, {duplicates} already present "
          f"(family {FAMILY_ID})")
    print("\nNo outcomes were read. A confirmatory run requires "
          "assert_outcome_access_allowed() and record_outcome_access().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
