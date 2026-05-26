"""Counterfactual validity class — enum + classifier helpers (P11-03, ADR-0007 §5).

Consumers of any policy-replay output MUST phrase conclusions conditional on
the validity class:

- ``exact_replay``: "under reconstructed universe U and config C, this would
  have appeared" — strong language allowed.
- ``partial_replay``: "under reconstructed universe U (partial fidelity) and
  config C, this likely would have appeared" — hedge.
- ``universe_reconstructed``: weaken claims about specific symbols.
- ``price_only_replay``: refuse to claim alert-level behavior.
- ``data_too_stale``: refuse to claim anything — refresh data first.
- ``invalid``: no claim possible.

The enum is sourced from ``observability.schema.VALIDITY_CLASSES`` so the
single source of truth lives at L0 (PIT ledger).
"""
from __future__ import annotations

from hot_theme_rotator.observability.schema import VALIDITY_CLASSES


__all__ = [
    "VALIDITY_CLASSES",
    "conditional_language_prefix",
    "is_stronger_than",
    "is_publishable",
]


# Ordering from strongest to weakest counterfactual claim.
_STRENGTH_ORDER = {
    "exact_replay": 5,
    "partial_replay": 4,
    "universe_reconstructed": 3,
    "price_only_replay": 2,
    "data_too_stale": 1,
    "invalid": 0,
}


def conditional_language_prefix(validity_class: str) -> str:
    """Return the required sentence prefix for any conclusion at this validity.

    Use this verbatim at the start of any user-facing claim derived from a
    policy-replay result. The prefix encodes the level of caveat the claim
    must carry.
    """
    if validity_class == "exact_replay":
        return "Under the reconstructed universe and config, "
    if validity_class == "partial_replay":
        return "Under the partially-reconstructed universe and config, "
    if validity_class == "universe_reconstructed":
        return "Under a synthesized universe (specific-symbol claims weakened), "
    if validity_class == "price_only_replay":
        return "With only price evidence (alert-level behavior unverifiable), "
    if validity_class == "data_too_stale":
        return "REFUSING TO CLAIM: data is too stale to support counterfactual replay; "
    if validity_class == "invalid":
        return "REFUSING TO CLAIM: replay inputs are insufficient; "
    raise ValueError(f"unknown validity_class: {validity_class!r}")


def is_stronger_than(a: str, b: str) -> bool:
    """Return True if validity class ``a`` is strictly stronger than ``b``."""
    if a not in _STRENGTH_ORDER:
        raise ValueError(f"unknown validity_class: {a!r}")
    if b not in _STRENGTH_ORDER:
        raise ValueError(f"unknown validity_class: {b!r}")
    return _STRENGTH_ORDER[a] > _STRENGTH_ORDER[b]


def is_publishable(validity_class: str) -> bool:
    """Whether a result at this validity class may go into user-facing output.

    ``data_too_stale`` and ``invalid`` MUST NOT publish numerical claims;
    they may publish the refusal banner via ``conditional_language_prefix``.
    """
    if validity_class not in _STRENGTH_ORDER:
        raise ValueError(f"unknown validity_class: {validity_class!r}")
    return _STRENGTH_ORDER[validity_class] >= 2  # price_only_replay and above
