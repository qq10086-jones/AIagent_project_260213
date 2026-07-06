"""Strategy synthesis layer — combines ladder + risk + discipline.

ADR-NEW (2026-05-28) + Rule 11.6: this layer composes already-existing
ladder / risk / discipline outputs into a single user-facing Strategy Card.
It does NOT add new signals; it does NOT call brokers; it explicitly
preserves the advice-only boundary (Rule 3) and never returns broker-facing
fields. The endpoint is read-only by contract.
"""
from .strategy_synthesizer import (
    RiskWarning,
    StrategyCard,
    StrategySynthesisError,
    StrategySynthesisInput,
    synthesize_strategy_card,
)

__all__ = [
    "RiskWarning",
    "StrategyCard",
    "StrategySynthesisError",
    "StrategySynthesisInput",
    "synthesize_strategy_card",
]
