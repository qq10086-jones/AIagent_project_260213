"""P34-07 — feasibility probes for a research data chain.

Why probe before building
--------------------------
T2 (ownership-conditioned PEAD) needs four inputs. Building the estimator first
and discovering afterwards that an input is absent wastes the work AND risks the
worse outcome: an estimator that runs on whatever partial data exists and
produces a number nobody realises is unusable.

So each requirement is probed independently and reported as
``available`` / ``degraded`` / ``absent``, with the evidence attached. A chain is
feasible only if every required link is available — a single ``absent`` link
means the study cannot run, no matter how good the other three are.

Two failure modes this catches specifically
--------------------------------------------
- **Repeated snapshots masquerading as a time series.** A table with N rows per
  symbol looks like history until you count DISTINCT periods. A seasonal SUE
  needs EPS from the same quarter a year earlier; re-fetching one quarter N
  times supplies none of it.
- **Fetch time masquerading as PIT time.** If an ``available_ts`` column records
  when a backfill ran rather than when the market learned the fact, then every
  event built on it is misdated — and misdated in a way that looks tidy, because
  the column is populated and ISO-formatted.

Rule 3: diagnostics only. Nothing here estimates a return.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "LinkStatus",
    "ChainLink",
    "FeasibilityReport",
    "assess_time_series_depth",
    "assess_pit_timestamp",
    "assess_presence",
    "build_chain_report",
]

LinkStatus = str  # "available" | "degraded" | "absent"


@dataclass
class ChainLink:
    name: str
    required: bool
    status: LinkStatus
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_time_series_depth(
    periods_by_key: Mapping[str, Iterable[Any]],
    *,
    min_distinct_periods: int,
    name: str = "time_series_depth",
    remedy: str = "",
) -> ChainLink:
    """Depth measured in DISTINCT periods per key, never in row count.

    Row count is the number that flatters: a table can hold thousands of rows and
    still contain one period per symbol.
    """
    distinct = {k: len(set(v)) for k, v in periods_by_key.items()}
    n_keys = len(distinct)
    qualifying = sum(1 for v in distinct.values() if v >= min_distinct_periods)
    hist: dict[int, int] = {}
    for v in distinct.values():
        hist[v] = hist.get(v, 0) + 1
    if n_keys == 0:
        status, detail = "absent", "no keys present at all"
    elif qualifying == 0:
        status = "absent"
        detail = (f"0 of {n_keys} keys reach {min_distinct_periods} distinct "
                  f"periods (max observed = {max(distinct.values())})")
    elif qualifying < n_keys * 0.5:
        status = "degraded"
        detail = f"only {qualifying}/{n_keys} keys reach {min_distinct_periods} periods"
    else:
        status = "available"
        detail = f"{qualifying}/{n_keys} keys reach {min_distinct_periods} periods"
    return ChainLink(
        name=name, required=True, status=status, detail=detail,
        evidence={"n_keys": n_keys, "qualifying_keys": qualifying,
                  "min_distinct_periods": min_distinct_periods,
                  "distinct_period_histogram": dict(sorted(hist.items())),
                  "median_distinct_periods": (
                      statistics.median(distinct.values()) if distinct else 0)},
        remedy=remedy,
    )


def assess_pit_timestamp(
    rows: Sequence[Mapping[str, Any]],
    *,
    ts_field: str,
    event_field: str,
    name: str = "pit_timestamp",
    max_distinct_ratio: float = 0.05,
    remedy: str = "",
) -> ChainLink:
    """Detect a fetch-time column being used as a point-in-time column.

    The tell: a genuine disclosure timestamp varies with the event, so its
    distinct-value count tracks the number of events. A backfill timestamp
    collapses onto the few days the backfill ran — many events, few timestamps,
    and all of them clustered long after the periods they describe.
    """
    if not rows:
        return ChainLink(name=name, required=True, status="absent",
                         detail="no rows to assess", remedy=remedy)
    ts_days = {str(r.get(ts_field, ""))[:10] for r in rows if r.get(ts_field)}
    events = {str(r.get(event_field, "")) for r in rows if r.get(event_field)}
    # Denominator is RECORDS, not distinct events. A real disclosure timestamp
    # is per-record; a backfill timestamp collapses thousands of records onto the
    # handful of days the script ran. Comparing against distinct events instead
    # lets a table with few periods but many rows pass by arithmetic accident.
    ratio = len(ts_days) / max(len(rows), 1)

    # Second, independent tell: lag from the event the row describes to its
    # timestamp. A genuine disclosure lag is roughly constant (a filing follows
    # its period end by a fairly fixed interval); a backfill lag varies wildly
    # because it is really "period end -> the day the script happened to run".
    lags = []
    for r in rows:
        ts, ev = str(r.get(ts_field, ""))[:10], str(r.get(event_field, ""))[:10]
        if len(ts) == 10 and len(ev) == 10:
            try:
                from datetime import date as _d
                y1, m1, d1 = (int(x) for x in ev.split("-"))
                y2, m2, d2 = (int(x) for x in ts.split("-"))
                lags.append((_d(y2, m2, d2) - _d(y1, m1, d1)).days)
            except ValueError:
                continue
    lag_spread = (max(lags) - min(lags)) if lags else 0
    median_lag = statistics.median(lags) if lags else 0

    status = "available"
    detail = (f"{len(ts_days)} distinct timestamp days across {len(rows)} records "
              f"(median lag {median_lag}d)")
    if ratio <= max_distinct_ratio:
        status = "absent"
        detail = (f"{len(ts_days)} distinct timestamp days for {len(rows)} records "
                  f"(ratio {ratio:.4f} <= {max_distinct_ratio}), median lag "
                  f"{median_lag}d, lag spread {lag_spread}d: this column records "
                  f"when data was FETCHED, not when it became public")
    return ChainLink(
        name=name, required=True, status=status, detail=detail,
        evidence={"distinct_timestamp_days": sorted(ts_days)[:12],
                  "n_distinct_timestamp_days": len(ts_days),
                  "n_records": len(rows),
                  "n_distinct_events": len(events),
                  "distinct_ratio": ratio,
                  "median_lag_days": median_lag,
                  "lag_spread_days": lag_spread},
        remedy=remedy,
    )


def assess_presence(
    present: bool,
    *,
    name: str,
    detail_present: str = "present",
    detail_absent: str = "not found",
    required: bool = True,
    remedy: str = "",
) -> ChainLink:
    return ChainLink(
        name=name, required=required,
        status="available" if present else "absent",
        detail=detail_present if present else detail_absent,
        remedy=remedy,
    )


@dataclass
class FeasibilityReport:
    chain: str
    links: list[ChainLink]

    @property
    def blocking(self) -> list[ChainLink]:
        return [l for l in self.links if l.required and l.status == "absent"]

    @property
    def feasible(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "_kind": "data_feasibility_report",
            "chain": self.chain,
            "feasible": self.feasible,
            "n_blocking": len(self.blocking),
            "blocking_links": [l.name for l in self.blocking],
            "links": [l.to_dict() for l in self.links],
            "note": (
                "A chain is feasible only when every REQUIRED link is available. "
                "One absent link blocks the study regardless of the others — "
                "partial data does not yield a partial answer, it yields a "
                "confident wrong one."
            ),
        }


def build_chain_report(chain: str, links: Sequence[ChainLink]) -> FeasibilityReport:
    return FeasibilityReport(chain=chain, links=list(links))
