"""P34-09 tests — EDGAR read-only replication lane guards."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.edgar_lane import (  # noqa: E402
    SEC_RATE_LIMIT_PER_SEC,
    EdgarFiling,
    EdgarLaneError,
    assert_not_event_source,
    build_headers,
    pit_available_at,
    replication_scope_guard,
)


def _filing(**kw):
    params = dict(accession="0000320193-26-000001", cik="320193", form="8-K",
                  filing_date="2026-08-07",
                  acceptance_datetime="2026-08-07T10:15:00")
    params.update(kw)
    return EdgarFiling(**params)


# --- SEC access policy ------------------------------------------------------

def test_valid_user_agent_builds_headers():
    h = build_headers("Jones jones@example.org")
    assert h["User-Agent"] == "Jones jones@example.org"
    assert h["Host"] == "data.sec.gov"


def test_missing_contact_user_agent_refused():
    with pytest.raises(EdgarLaneError, match="fair-access"):
        build_headers("HotThemeRotator")


def test_placeholder_user_agent_refused():
    with pytest.raises(EdgarLaneError, match="placeholder"):
        build_headers("Your Name your.name@example.com")


def test_empty_user_agent_refused():
    with pytest.raises(EdgarLaneError):
        build_headers("")


def test_rate_limit_constant_matches_sec_guidance():
    assert SEC_RATE_LIMIT_PER_SEC == 10


# --- filing date is not acceptance timestamp --------------------------------

def test_morning_acceptance_is_available_same_day():
    assert pit_available_at(_filing(acceptance_datetime="2026-08-07T10:15:00")) \
        == "2026-08-07"


def test_after_close_acceptance_rolls_to_next_day():
    """16:00 ET is the close; a 18:30 acceptance was not tradable that session."""
    assert pit_available_at(_filing(acceptance_datetime="2026-08-07T18:30:00")) \
        == "2026-08-08"


def test_exactly_at_close_rolls_forward():
    assert pit_available_at(_filing(acceptance_datetime="2026-08-07T16:00:00")) \
        == "2026-08-08"


def test_missing_acceptance_is_refused_by_default():
    with pytest.raises(EdgarLaneError, match="look-ahead"):
        pit_available_at(_filing(acceptance_datetime=None))


def test_missing_acceptance_allowed_only_with_explicit_opt_in():
    assert pit_available_at(_filing(acceptance_datetime=None),
                            require_acceptance=False) == "2026-08-07"


def test_acceptance_can_differ_from_filing_date():
    f = _filing(filing_date="2026-08-07", acceptance_datetime="2026-08-06T17:00:00")
    assert pit_available_at(f) == "2026-08-07"
    assert f.filing_date != f.acceptance_datetime[:10]


def test_malformed_dates_refused():
    with pytest.raises(EdgarLaneError, match="ISO 8601"):
        _filing(filing_date="August 7")
    with pytest.raises(EdgarLaneError, match="ISO 8601"):
        _filing(acceptance_datetime="whenever")


def test_missing_identifiers_refused():
    with pytest.raises(EdgarLaneError):
        _filing(accession="")


# --- companyfacts is a panel, not an event stream ---------------------------

@pytest.mark.parametrize("ds", ["companyfacts", "companyconcept", "frames", "FRAMES"])
def test_panel_datasets_refused_as_event_sources(ds):
    with pytest.raises(EdgarLaneError, match="fiscal PERIOD"):
        assert_not_event_source(ds)


def test_submissions_index_is_an_acceptable_event_source():
    assert_not_event_source("submissions")  # does not raise


# --- replication scope ------------------------------------------------------

def test_replicating_a_jp_signal_is_allowed():
    replication_scope_guard(["earnings_yield", "buyback_resolution_car"],
                            ["buyback_resolution_car"])


def test_novel_us_only_signal_is_refused():
    with pytest.raises(EdgarLaneError, match="replicates; it does not originate"):
        replication_scope_guard(["earnings_yield"], ["net_share_issuance"])


def test_scope_guard_is_case_insensitive():
    replication_scope_guard(["Earnings_Yield"], ["earnings_yield"])


def test_partial_novelty_still_refused():
    with pytest.raises(EdgarLaneError):
        replication_scope_guard(["earnings_yield"],
                                ["earnings_yield", "accruals"])
