"""Universal cross-symbol attribution contracts.

This module turns the single-symbol attribution idea into a representative
instrument framework. It does not fetch market data or create live advice; it
validates point-in-time inputs and combines per-symbol decision scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


REQUIRED_ATTRIBUTION_BUCKETS = (
    "japan_equity_beta",
    "rates",
    "fx",
    "external_risk",
    "news",
    "own_trading",
)

ALLOWED_SCORE_STATUSES = {
    "calibrated_probability",
    "uncalibrated_research_score",
    "insufficient_calibration",
}


class AttributionValidationError(ValueError):
    """Raised when attribution inputs would make analysis unsafe."""


@dataclass(frozen=True)
class RepresentativeInstrument:
    symbol: str
    role: str
    market: str
    feature_tags: tuple[str, ...]
    weight: float

    def __post_init__(self) -> None:
        _require_text(self.symbol, "symbol")
        _require_text(self.role, "role")
        _require_text(self.market, "market")
        if not self.feature_tags:
            raise AttributionValidationError("feature_tags must contain at least one value")
        if float(self.weight) <= 0:
            raise AttributionValidationError("instrument weight must be positive")


@dataclass(frozen=True)
class PointInTimeFeature:
    feature_id: str
    symbol: str
    bucket: str
    available_ts: str
    value: float
    source: str

    def __post_init__(self) -> None:
        _require_text(self.feature_id, "feature_id")
        _require_text(self.symbol, "symbol")
        _require_text(self.bucket, "bucket")
        _require_text(self.available_ts, "available_ts")
        _require_text(self.source, "source")


@dataclass(frozen=True)
class SymbolAttributionSnapshot:
    snapshot_id: str
    symbol: str
    trade_date: str
    decision_cutoff: str
    features: tuple[PointInTimeFeature, ...]
    required_buckets: tuple[str, ...] = REQUIRED_ATTRIBUTION_BUCKETS

    @property
    def present_buckets(self) -> tuple[str, ...]:
        return tuple(sorted({feature.bucket for feature in self.features}))

    @property
    def missing_buckets(self) -> tuple[str, ...]:
        present = set(self.present_buckets)
        return tuple(bucket for bucket in self.required_buckets if bucket not in present)

    @property
    def is_complete(self) -> bool:
        return not self.missing_buckets


@dataclass(frozen=True)
class SymbolDecisionScore:
    symbol: str
    horizon_days: int
    buy: float
    sell: float
    hold: float
    status: str
    model_version: str

    def __post_init__(self) -> None:
        _require_text(self.symbol, "symbol")
        _require_text(self.status, "status")
        _require_text(self.model_version, "model_version")
        if self.status not in ALLOWED_SCORE_STATUSES:
            raise AttributionValidationError("status must be a known score status")
        if int(self.horizon_days) <= 0:
            raise AttributionValidationError("horizon_days must be positive")
        for field_name, value in (
            ("buy", self.buy),
            ("sell", self.sell),
            ("hold", self.hold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise AttributionValidationError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True)
class IntegratedDecisionScore:
    horizon_days: int
    buy: float
    sell: float
    hold: float
    status: str
    model_versions: tuple[str, ...]
    contributing_symbols: tuple[str, ...]
    method: str = "role_weighted_average"


def default_japan_representative_universe() -> tuple[RepresentativeInstrument, ...]:
    """Return the first Japan representative set used by the generic framework."""
    return validate_representative_universe(
        [
            RepresentativeInstrument(
                "1306.T",
                "broad_beta",
                "JP",
                ("topix", "index_proxy", "market_beta"),
                2.0,
            ),
            RepresentativeInstrument(
                "7203.T",
                "export_cyclical",
                "JP",
                ("auto", "fx", "global_demand"),
                1.0,
            ),
            RepresentativeInstrument(
                "8306.T",
                "rate_sensitive",
                "JP",
                ("bank", "jgb_yield", "value"),
                1.0,
            ),
            RepresentativeInstrument(
                "8035.T",
                "ai_semiconductor",
                "JP",
                ("semiconductor", "ai", "global_tech"),
                1.0,
            ),
            RepresentativeInstrument(
                "9432.T",
                "domestic_defensive",
                "JP",
                ("telecom", "defensive", "domestic_demand"),
                1.0,
            ),
            RepresentativeInstrument(
                "1605.T",
                "commodity_energy",
                "JP",
                ("energy", "commodity", "inflation"),
                1.0,
            ),
            RepresentativeInstrument(
                "9984.T",
                "risk_growth",
                "JP",
                ("growth", "external_risk", "technology"),
                1.0,
            ),
        ]
    )


def validate_representative_universe(
    instruments: list[RepresentativeInstrument] | tuple[RepresentativeInstrument, ...],
    *,
    min_distinct_roles: int = 3,
) -> tuple[RepresentativeInstrument, ...]:
    """Validate a representative instrument universe for cross-symbol analysis."""
    universe = tuple(instruments)
    if not universe:
        raise AttributionValidationError("representative universe must not be empty")

    symbols = [instrument.symbol for instrument in universe]
    duplicate_symbols = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
    if duplicate_symbols:
        raise AttributionValidationError(
            f"duplicate representative symbols: {', '.join(duplicate_symbols)}"
        )

    roles = {instrument.role for instrument in universe}
    if len(roles) < min_distinct_roles:
        raise AttributionValidationError(
            f"representative universe must include at least {min_distinct_roles} distinct roles"
        )

    total_weight = sum(float(instrument.weight) for instrument in universe)
    if total_weight <= 0:
        raise AttributionValidationError("representative universe total weight must be positive")
    return universe


def build_symbol_snapshot(
    *,
    snapshot_id: str,
    symbol: str,
    trade_date: str,
    decision_cutoff: str,
    features: list[PointInTimeFeature] | tuple[PointInTimeFeature, ...],
    required_buckets: tuple[str, ...] = REQUIRED_ATTRIBUTION_BUCKETS,
) -> SymbolAttributionSnapshot:
    """Build a point-in-time feature snapshot for one target symbol."""
    _require_text(snapshot_id, "snapshot_id")
    _require_text(symbol, "symbol")
    _require_text(trade_date, "trade_date")
    cutoff = _parse_ts(decision_cutoff, "decision_cutoff")

    snapshot_features = tuple(features)
    for feature in snapshot_features:
        available_ts = _parse_ts(feature.available_ts, "available_ts")
        if available_ts > cutoff:
            raise AttributionValidationError(
                f"feature {feature.feature_id} available_ts is later than decision cutoff"
            )

    return SymbolAttributionSnapshot(
        snapshot_id=snapshot_id,
        symbol=symbol,
        trade_date=trade_date,
        decision_cutoff=decision_cutoff,
        features=snapshot_features,
        required_buckets=required_buckets,
    )


def integrate_symbol_decisions(
    decisions: list[SymbolDecisionScore] | tuple[SymbolDecisionScore, ...],
    universe: list[RepresentativeInstrument] | tuple[RepresentativeInstrument, ...],
) -> IntegratedDecisionScore:
    """Combine per-symbol outputs into one role-weighted framework output."""
    validated_universe = validate_representative_universe(universe)
    decision_by_symbol = {decision.symbol: decision for decision in decisions}
    expected_symbols = {instrument.symbol for instrument in validated_universe}
    missing_symbols = sorted(expected_symbols - set(decision_by_symbol))
    extra_symbols = sorted(set(decision_by_symbol) - expected_symbols)
    if missing_symbols:
        raise AttributionValidationError(
            f"missing decisions for symbols: {', '.join(missing_symbols)}"
        )
    if extra_symbols:
        raise AttributionValidationError(
            f"decisions contain symbols outside universe: {', '.join(extra_symbols)}"
        )

    horizons = {decision.horizon_days for decision in decision_by_symbol.values()}
    if len(horizons) != 1:
        raise AttributionValidationError("all symbol decisions must use the same horizon")
    horizon_days = horizons.pop()

    total_weight = sum(float(instrument.weight) for instrument in validated_universe)
    buy = sell = hold = 0.0
    for instrument in validated_universe:
        decision = decision_by_symbol[instrument.symbol]
        weight = float(instrument.weight) / total_weight
        buy += decision.buy * weight
        sell += decision.sell * weight
        hold += decision.hold * weight

    statuses = {decision.status for decision in decision_by_symbol.values()}
    if statuses == {"calibrated_probability"}:
        status = "calibrated_probability"
    elif "insufficient_calibration" in statuses:
        status = "insufficient_calibration"
    else:
        status = "uncalibrated_research_score"

    return IntegratedDecisionScore(
        horizon_days=horizon_days,
        buy=round(buy, 6),
        sell=round(sell, 6),
        hold=round(hold, 6),
        status=status,
        model_versions=tuple(
            sorted({decision.model_version for decision in decision_by_symbol.values()})
        ),
        contributing_symbols=tuple(instrument.symbol for instrument in validated_universe),
    )


def _parse_ts(value: str, field: str) -> datetime:
    _require_text(value, field)
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AttributionValidationError(f"{field} must be an ISO timestamp") from exc


def _require_text(value: str, field: str) -> None:
    if not str(value).strip():
        raise AttributionValidationError(f"{field} must be non-empty")
