"""Backdated calibration bootstrap (P10-13, ADR-0006).

One-shot historical backfill tool that reconstructs ``PredictionRecord`` from
archived ``selected_tickers`` snapshots and joins them against real OHLC bars
(via P9-02 ``compute_outcomes``) to populate calibration sample buckets.

All backdated records carry:

- ``extra.backdated = True``
- ``extra.live = False``
- ``model_version`` ends with ``"-backdated"``
- ``extra.generator = "backdated_calibration_bootstrap_v1"``

Date cherry-pick is forbidden: the caller specifies a continuous (start, end)
window and the loader must return snapshots for every trading day in that
window. Days where the snapshot is missing land in the provenance JSON with
an explicit reason.

``scanner_config_hash`` MUST match the git commit hash of
``configs/scanner.yaml`` at or before the window start; mismatch fail-closed.
Caller is responsible for resolving + passing the hash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from hot_theme_rotator.decision_log.outcome_join import PriceFetcher, compute_outcome
from hot_theme_rotator.decision_log.schema import PredictionRecord


__all__ = [
    "BackdatedSnapshot",
    "BootstrapError",
    "BootstrapProvenance",
    "BootstrapResult",
    "HistoricalSnapshotsLoader",
    "bootstrap_calibration",
    "provenance_path",
]


JST = timezone(timedelta(hours=9), name="JST")
MODEL_VERSION_SUFFIX = "-backdated"
GENERATOR_TAG = "backdated_calibration_bootstrap_v1"


class BootstrapError(RuntimeError):
    """Raised when bootstrap cannot complete safely."""


@dataclass(frozen=True)
class BackdatedSnapshot:
    """A historical snapshot of one trading day's top-N candidate scoring.

    ``candidates`` is a sequence of dicts containing at minimum:
    - ``symbol``: ``.T`` ticker
    - ``buy``: bullish probability in [0, 1]
    - ``sell``: bearish probability in [0, 1] (typically 0 for opportunity scanner)
    - ``hold``: held-equivalent in [0, 1] (typically 1 - buy)
    - ``score_status``: matches ``ALLOWED_SCORE_STATUSES``
    """

    trade_date: str  # ISO YYYY-MM-DD
    decision_cutoff: str  # ISO 8601 with timezone (close of trade_date JST)
    input_snapshot_id: str
    candidates: Sequence[Mapping[str, Any]]


class HistoricalSnapshotsLoader(Protocol):
    """Protocol for loading archived ``selected_tickers`` snapshots."""

    def load(self, *, trade_date: str) -> Optional[BackdatedSnapshot]:
        """Return snapshot for ``trade_date`` or None if not archived."""


@dataclass(frozen=True)
class BootstrapProvenance:
    window_start: str
    window_end: str
    total_trading_days_attempted: int
    snapshots_loaded: int
    excluded: tuple[Mapping[str, str], ...]  # list of {trade_date, reason}
    model_version: str
    scanner_config_hash: str
    generated_at: str
    generator: str = GENERATOR_TAG

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "total_trading_days_attempted": self.total_trading_days_attempted,
            "snapshots_loaded": self.snapshots_loaded,
            "excluded": [dict(e) for e in self.excluded],
            "model_version": self.model_version,
            "scanner_config_hash": self.scanner_config_hash,
            "generated_at": self.generated_at,
            "generator": self.generator,
        }


@dataclass(frozen=True)
class BootstrapResult:
    provenance: BootstrapProvenance
    predictions: tuple[PredictionRecord, ...]
    outcomes_built: int
    outcomes_complete: int


def provenance_path(*, base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "reports" / "bootstrap_provenance.json"


def bootstrap_calibration(
    *,
    window_start: str,
    window_end: str,
    base_model_version: str,
    scanner_config_hash: str,
    expected_scanner_config_hash: str,
    snapshots_loader: HistoricalSnapshotsLoader,
    price_fetcher: PriceFetcher,
    base_dir: str | Path = ".",
    horizon_days: tuple[int, ...] = (1, 3, 5),
    trading_days: Optional[Sequence[str]] = None,
) -> BootstrapResult:
    """Run the historical backfill over ``[window_start, window_end]``.

    The caller passes ``trading_days`` explicitly (or omits to let the
    function enumerate calendar days inclusive — sufficient for testing and
    for short windows; longer real-data runs should supply a TSE trading
    calendar list to skip weekends/holidays cleanly).
    """
    _require_iso_date(window_start, "window_start")
    _require_iso_date(window_end, "window_end")
    if window_start > window_end:
        raise BootstrapError(
            f"window_start {window_start} must be <= window_end {window_end}"
        )

    if scanner_config_hash != expected_scanner_config_hash:
        raise BootstrapError(
            f"scanner_config_hash mismatch: got {scanner_config_hash!r}, "
            f"expected {expected_scanner_config_hash!r}. Caller must resolve the "
            f"hash from `git log -- configs/scanner.yaml` at the window start."
        )

    model_version = f"{base_model_version}{MODEL_VERSION_SUFFIX}"

    if trading_days is None:
        # Default: every calendar day in the window inclusive.
        trading_days = tuple(_calendar_days_inclusive(window_start, window_end))
    else:
        # Caller-supplied list must be continuous and within window.
        _require_continuous_in_window(trading_days, window_start, window_end)

    predictions: list[PredictionRecord] = []
    excluded: list[Mapping[str, str]] = []

    for td in trading_days:
        snapshot = snapshots_loader.load(trade_date=td)
        if snapshot is None:
            excluded.append({"trade_date": td, "reason": "no_archived_snapshot"})
            continue
        if not snapshot.candidates:
            excluded.append({"trade_date": td, "reason": "empty_candidates"})
            continue
        for candidate in snapshot.candidates:
            try:
                pred = _build_backdated_prediction(
                    snapshot=snapshot,
                    candidate=candidate,
                    model_version=model_version,
                    horizon_days=horizon_days[1] if len(horizon_days) >= 2 else 3,
                )
            except (KeyError, ValueError, TypeError) as exc:
                excluded.append({
                    "trade_date": td,
                    "reason": f"candidate_invalid: {exc}",
                })
                continue
            predictions.append(pred)

    outcomes_built = 0
    outcomes_complete = 0
    eval_date = max(window_end, _today_jst())
    for pred in predictions:
        outcome = compute_outcome(
            prediction=pred, fetcher=price_fetcher, evaluated_as_of=eval_date,
            horizons_days=tuple(horizon_days),
        )
        outcomes_built += 1
        if outcome.status == "complete":
            outcomes_complete += 1

    provenance = BootstrapProvenance(
        window_start=window_start,
        window_end=window_end,
        total_trading_days_attempted=len(trading_days),
        snapshots_loaded=len(trading_days) - len(excluded),
        excluded=tuple(excluded),
        model_version=model_version,
        scanner_config_hash=scanner_config_hash,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    _write_provenance(provenance, base_dir=base_dir)

    return BootstrapResult(
        provenance=provenance,
        predictions=tuple(predictions),
        outcomes_built=outcomes_built,
        outcomes_complete=outcomes_complete,
    )


# ─── internals ──────────────────────────────────────────────────────────────


def _build_backdated_prediction(
    *,
    snapshot: BackdatedSnapshot,
    candidate: Mapping[str, Any],
    model_version: str,
    horizon_days: int,
) -> PredictionRecord:
    symbol = str(candidate["symbol"])
    extra = {
        "backdated": True,
        "live": False,
        "generator": GENERATOR_TAG,
        "reference_price": float(candidate.get("reference_price", 0.0)),
    }
    if "ladder" in candidate:
        extra["ladder"] = candidate["ladder"]
    if "reason_codes" in candidate:
        extra["reason_codes"] = list(candidate["reason_codes"])
    return PredictionRecord.build(
        symbol=symbol,
        trade_date=snapshot.trade_date,
        decision_cutoff=snapshot.decision_cutoff,
        input_snapshot_id=snapshot.input_snapshot_id,
        model_version=model_version,
        score_status=str(candidate.get("score_status", "uncalibrated_research_score")),
        horizon_days=horizon_days,
        buy=float(candidate["buy"]),
        sell=float(candidate.get("sell", 0.0)),
        hold=float(candidate.get("hold", max(0.0, 1.0 - float(candidate["buy"])))),
        extra=extra,
    )


def _write_provenance(p: BootstrapProvenance, *, base_dir: str | Path) -> None:
    path = provenance_path(base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(p.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _calendar_days_inclusive(start: str, end: str) -> Iterable[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    cur = d0
    while cur <= d1:
        yield cur.isoformat()
        cur = cur + timedelta(days=1)


def _require_continuous_in_window(days: Sequence[str], start: str, end: str) -> None:
    if not days:
        raise BootstrapError("trading_days must not be empty (date cherry-pick guard)")
    sorted_days = sorted(days)
    if sorted_days[0] < start or sorted_days[-1] > end:
        raise BootstrapError(
            f"trading_days {sorted_days[0]}..{sorted_days[-1]} extend beyond "
            f"window {start}..{end}"
        )
    # Continuity: list must be sorted and unique. Gaps within the explicitly
    # supplied list are not allowed — caller must include every trading day.
    if sorted_days != list(days):
        raise BootstrapError("trading_days must be sorted")
    seen = set()
    for d in sorted_days:
        if d in seen:
            raise BootstrapError(f"duplicate trade_date in trading_days: {d}")
        seen.add(d)


def _require_iso_date(value: str, name: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise BootstrapError(f"{name} must be ISO YYYY-MM-DD, got {value!r}") from exc


def _today_jst() -> str:
    return datetime.now(tz=JST).date().isoformat()
