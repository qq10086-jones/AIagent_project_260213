from __future__ import annotations

from datetime import date

from trade_schema import connect


def latest_db_date(db_path: str) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def refresh_market_data_if_needed(
    db_path: str,
    *,
    target_date: str | None = None,
    lookback_days: int = 30,
    force: bool = False,
) -> tuple[str | None, str | None, bool]:
    before = latest_db_date(db_path)
    needs_refresh = force or before is None

    if target_date is not None:
        needs_refresh = needs_refresh or str(before or "") < str(target_date)
    else:
        today_str = date.today().strftime("%Y-%m-%d")
        needs_refresh = needs_refresh or str(before or "") < today_str

    if needs_refresh:
        from db_update import update_database

        update_database(db_path=db_path, default_lookback_days=lookback_days, universe_path=None)

    after = latest_db_date(db_path)
    return before, after, needs_refresh


def latest_intraday_timestamp(db_path: str, asof: str | None = None) -> str | None:
    with connect(db_path) as conn:
        if asof:
            row = conn.execute("SELECT MAX(ts) FROM intraday_quotes WHERE asof=?", (asof,)).fetchone()
        else:
            row = conn.execute("SELECT MAX(ts) FROM intraday_quotes").fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def refresh_intraday_if_needed(
    db_path: str,
    *,
    target_date: str | None = None,
    symbols_arg: str | None = None,
    force: bool = False,
) -> tuple[str | None, str | None, bool]:
    before = latest_intraday_timestamp(db_path, target_date)
    today_str = date.today().strftime("%Y-%m-%d")
    needs_refresh = force or before is None

    if target_date is None:
        target_date = today_str

    needs_refresh = needs_refresh or (str(target_date) >= today_str and before is None)

    if needs_refresh:
        from intraday_update import update_intraday_quotes

        update_intraday_quotes(db_path=db_path, symbols_arg=symbols_arg)

    after = latest_intraday_timestamp(db_path, target_date)
    return before, after, needs_refresh
