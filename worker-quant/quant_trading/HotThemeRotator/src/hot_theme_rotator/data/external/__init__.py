"""External data source adapters (TDnet, Yahoo JP, J-Quants live).

Per P10-14 storage decision (Rule 4 amendment 2026-05-25): HTR-native storage
in `reports/tdnet/` and similar. Project_optimized stays read-only per ADR-0005.
"""

from .tdnet_schema import (
    ALLOWED_TDNET_CATEGORIES,
    TdnetDisclosure,
    TdnetDisclosureValidationError,
    compute_disclosure_id,
)

__all__ = [
    "ALLOWED_TDNET_CATEGORIES",
    "TdnetDisclosure",
    "TdnetDisclosureValidationError",
    "compute_disclosure_id",
]
