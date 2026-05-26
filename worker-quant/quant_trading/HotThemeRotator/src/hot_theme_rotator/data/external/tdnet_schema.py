"""TdnetDisclosure dataclass and disclosure_id helper (P10-14 Cycle 1).

Storage location: HTR-native `reports/tdnet/{trade_date}.jsonl` per Rule 4
amendment 2026-05-25 — Project_optimized is read-only per ADR-0005, so HTR
owns the storage for the external data sources it ingests.

Fail-closed validation: missing or malformed required fields raise
`TdnetDisclosureValidationError`. `disclosure_id` is integrity-checked on
construction to prevent silently storing records whose id does not match the
deterministic hash of (ticker, published_ts, title).
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from typing import Any, Dict, Mapping, Optional


ALLOWED_TDNET_CATEGORIES = frozenset(
    {
        "earnings",     # 業績修正、決算短信、業績予想
        "order",        # 業務提携、大型受注
        "tob",          # TOB / 公開買付
        "dividend",     # 配当予想、配当増額、減配
        "split",        # 株式分割、株式併合
        "suspension",   # 売買停止、上場廃止
        "governance",   # 役員異動、業務改善命令
        "other",        # fallback for unmapped disclosure types
    }
)


class TdnetDisclosureValidationError(ValueError):
    """Raised when a TdnetDisclosure fails fail-closed validation."""


_REQUIRED_STRING_FIELDS = (
    "disclosure_id",
    "ticker",
    "published_ts",
    "collected_ts",
    "title",
    "category",
    "url",
)


@dataclass(frozen=True)
class TdnetDisclosure:
    """TDnet 適時開示 disclosure record.

    Fields:
      disclosure_id: deterministic SHA-256(ticker|published_ts|title)[:16].
      ticker:        normalized "X.T" form (e.g. "6779.T"). Parser is responsible
                     for transforming raw 4- or 5-digit TDnet codes.
      published_ts:  ISO 8601 timestamp when TDnet published the disclosure.
      collected_ts:  ISO 8601 timestamp when this adapter fetched it.
      title:         original Japanese title from TDnet.
      category:      one of ALLOWED_TDNET_CATEGORIES; parser maps unmapped to "other".
      url:           link to the original disclosure document.
      company_name:  optional Japanese company name when available.
      summary:       optional adapter-side extracted summary.
      raw:           optional dict of the original source-format fields.
    """

    disclosure_id: str
    ticker: str
    published_ts: str
    collected_ts: str
    title: str
    category: str
    url: str
    company_name: Optional[str] = None
    summary: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        for field_name in _REQUIRED_STRING_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise TdnetDisclosureValidationError(
                    f"TdnetDisclosure.{field_name} must be a non-empty string, got {value!r}"
                )

        if not _is_valid_ticker(self.ticker):
            raise TdnetDisclosureValidationError(
                f"TdnetDisclosure.ticker must be 'NNNN.T' form, got {self.ticker!r}"
            )

        for ts_field in ("published_ts", "collected_ts"):
            ts_value = getattr(self, ts_field)
            try:
                datetime.fromisoformat(ts_value)
            except ValueError as exc:
                raise TdnetDisclosureValidationError(
                    f"TdnetDisclosure.{ts_field} must be ISO 8601, got {ts_value!r}"
                ) from exc

        if self.category not in ALLOWED_TDNET_CATEGORIES:
            raise TdnetDisclosureValidationError(
                f"TdnetDisclosure.category must be one of "
                f"{sorted(ALLOWED_TDNET_CATEGORIES)}, got {self.category!r}"
            )

        expected_id = compute_disclosure_id(self.ticker, self.published_ts, self.title)
        if self.disclosure_id != expected_id:
            raise TdnetDisclosureValidationError(
                f"TdnetDisclosure.disclosure_id integrity check failed: "
                f"expected {expected_id!r}, got {self.disclosure_id!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TdnetDisclosure":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data.keys()) - known_fields
        if unknown:
            raise TdnetDisclosureValidationError(
                f"TdnetDisclosure.from_dict received unknown keys: {sorted(unknown)}"
            )
        return cls(
            disclosure_id=data["disclosure_id"],
            ticker=data["ticker"],
            published_ts=data["published_ts"],
            collected_ts=data["collected_ts"],
            title=data["title"],
            category=data["category"],
            url=data["url"],
            company_name=data.get("company_name"),
            summary=data.get("summary"),
            raw=data.get("raw"),
        )


def compute_disclosure_id(ticker: str, published_ts: str, title: str) -> str:
    """Deterministic 32-hex-char (128-bit) SHA-256 prefix of (ticker | published_ts | title).

    Per Codex review 2026-05-25: extended from 16 to 32 hex chars (64-bit → 128-bit)
    because the ID is a storage integrity key and 128 bits costs nothing while
    eliminating any practical birthday-collision risk at HTR horizon.

    The `|` delimiter is assumption-pinned: none of ticker / published_ts / title
    naturally contain `|`. Callers must not feed `|`-containing input or the id
    becomes ambiguous.
    """
    payload = f"{ticker}|{published_ts}|{title}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _is_valid_ticker(ticker: str) -> bool:
    """Tokyo Stock Exchange normalized form: 4-digit number + '.T' suffix."""
    if not ticker.endswith(".T"):
        return False
    head = ticker[:-2]
    return len(head) == 4 and head.isdigit()
