from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from trade_schema import connect, ensure_trade_tables


SNAPSHOT_NUMERIC_COLUMNS = [
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "book_value_per_share",
    "dividend_per_share",
    "operating_cf",
    "free_cf",
    "total_assets",
    "total_equity",
    "total_debt",
    "shares_outstanding",
    "guidance_revenue",
    "guidance_operating_income",
    "guidance_eps",
]

EVENT_NUMERIC_COLUMNS = [
    "revenue_yoy",
    "operating_income_yoy",
    "eps_yoy",
    "guidance_delta_revenue",
    "guidance_delta_op",
    "guidance_delta_eps",
    "surprise_score",
]


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    try:
        num_f = float(num)
        den_f = float(den)
    except Exception:
        return None
    if abs(den_f) <= 1e-12:
        return None
    return num_f / den_f


def _coerce_float(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _optional_text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _validate_required(frame: pd.DataFrame, required: list[str], fail_closed: bool) -> bool:
    missing = [col for col in required if col not in frame.columns]
    if not missing:
        return True
    message = f"Missing required CSV columns: {', '.join(missing)}"
    if fail_closed:
        raise ValueError(message)
    print(f"[fundamentals] skip import: {message}")
    return False


def _latest_close(conn, symbol: str, asof: str) -> float | None:
    row = conn.execute(
        """
        SELECT close
        FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _build_feature_daily_rows(conn, raw: pd.DataFrame, require_available_ts: bool) -> list[dict]:
    rows: list[dict] = []

    snapshot_mask = raw["fiscal_period_end"].str.strip() != ""
    for rec in raw.loc[snapshot_mask].to_dict(orient="records"):
        available_ts = str(rec.get("available_ts") or "").strip()
        if not available_ts:
            continue
        asof = available_ts[:10]
        symbol = str(rec["symbol"])
        price = _latest_close(conn, symbol, asof)
        features = {
            "value_bp": _safe_div(rec.get("book_value_per_share"), price),
            "quality_roe": _safe_div(rec.get("net_income"), rec.get("total_equity")),
            "quality_cfo": _safe_div(rec.get("operating_cf"), rec.get("net_income")),
            "margin_op": _safe_div(rec.get("operating_income"), rec.get("revenue")),
            "leverage_safety": _safe_div(rec.get("total_equity"), rec.get("total_debt")),
            "dividend_yield": _safe_div(rec.get("dividend_per_share"), price),
        }
        for feature_name, value in features.items():
            if value is None:
                continue
            rows.append(
                {
                    "asof": asof,
                    "symbol": symbol,
                    "feature_name": feature_name,
                    "value": float(value),
                    "source_fact_ids": f"fundamental_snapshots:{symbol}:{rec.get('fiscal_period_end')}:{rec.get('published_ts')}",
                }
            )

    if "event_type" not in raw.columns:
        return rows

    event_mask = raw["event_type"].fillna("").astype(str).str.strip() != ""
    for rec in raw.loc[event_mask].to_dict(orient="records"):
        available_ts = str(rec.get("available_ts") or "").strip()
        if (not available_ts) and (not require_available_ts):
            available_ts = str(rec.get("published_ts") or "").strip()
        if not available_ts:
            continue
        asof = available_ts[:10]
        symbol = str(rec["symbol"])
        event_features = {
            "growth_rev_yoy": rec.get("revenue_yoy"),
            "growth_op_yoy": rec.get("operating_income_yoy"),
            "guidance_delta": rec.get("guidance_delta_eps"),
        }
        for feature_name, value in event_features.items():
            try:
                value_f = float(value)
            except Exception:
                continue
            rows.append(
                {
                    "asof": asof,
                    "symbol": symbol,
                    "feature_name": feature_name,
                    "value": value_f,
                    "source_fact_ids": f"earnings_events:{symbol}:{rec.get('published_ts')}:{rec.get('event_type')}",
                }
            )
    return rows


def import_csv(db_path: str, csv_path: Path, fail_closed: bool, require_available_ts: bool) -> None:
    if not csv_path.exists():
        message = f"CSV not found: {csv_path}"
        if fail_closed:
            raise FileNotFoundError(message)
        print(f"[fundamentals] {message}; skip import.")
        return

    raw = pd.read_csv(csv_path)
    if raw.empty:
        print("[fundamentals] CSV is empty; nothing to import.")
        return
    if not _validate_required(raw, ["symbol", "published_ts"], fail_closed):
        return

    raw = raw.copy()
    raw["symbol"] = raw["symbol"].astype(str)
    raw["published_ts"] = raw["published_ts"].astype(str)
    raw["available_ts"] = _optional_text(raw, "available_ts")
    raw["fiscal_period_end"] = _optional_text(raw, "fiscal_period_end")
    raw["source"] = _optional_text(raw, "source", default="csv")
    raw["currency"] = _optional_text(raw, "currency", default="JPY")
    raw = _coerce_float(raw, SNAPSHOT_NUMERIC_COLUMNS + EVENT_NUMERIC_COLUMNS)

    if fail_closed and (raw["available_ts"].str.strip() == "").any():
        raise ValueError("available_ts is required in fail-closed mode.")

    snapshot_mask = raw["fiscal_period_end"].str.strip() != ""
    event_mask = raw.get("event_type", pd.Series(index=raw.index, dtype=object)).fillna("").astype(str).str.strip() != ""

    with connect(db_path) as conn:
        ensure_trade_tables(conn)

        if snapshot_mask.any():
            snapshots = raw.loc[snapshot_mask].copy()
            conn.executemany(
                """
                INSERT OR REPLACE INTO fundamental_snapshots (
                  symbol, fiscal_period_end, published_ts, available_ts, source, currency,
                  revenue, operating_income, net_income, eps, book_value_per_share, dividend_per_share,
                  operating_cf, free_cf, total_assets, total_equity, total_debt, shares_outstanding,
                  guidance_revenue, guidance_operating_income, guidance_eps
                ) VALUES (
                  :symbol, :fiscal_period_end, :published_ts, :available_ts, :source, :currency,
                  :revenue, :operating_income, :net_income, :eps, :book_value_per_share, :dividend_per_share,
                  :operating_cf, :free_cf, :total_assets, :total_equity, :total_debt, :shares_outstanding,
                  :guidance_revenue, :guidance_operating_income, :guidance_eps
                )
                """,
                snapshots.to_dict(orient="records"),
            )
            print(f"[fundamentals] imported snapshot rows: {int(snapshot_mask.sum())}")

        if event_mask.any():
            events = raw.loc[event_mask].copy()
            events["event_type"] = events["event_type"].fillna("").astype(str)
            events["headline"] = _optional_text(events, "headline")
            conn.executemany(
                """
                INSERT OR REPLACE INTO earnings_events (
                  symbol, published_ts, event_type, headline,
                  revenue_yoy, operating_income_yoy, eps_yoy,
                  guidance_delta_revenue, guidance_delta_op, guidance_delta_eps,
                  surprise_score, source
                ) VALUES (
                  :symbol, :published_ts, :event_type, :headline,
                  :revenue_yoy, :operating_income_yoy, :eps_yoy,
                  :guidance_delta_revenue, :guidance_delta_op, :guidance_delta_eps,
                  :surprise_score, :source
                )
                """,
                events.to_dict(orient="records"),
            )
            print(f"[fundamentals] imported event rows: {int(event_mask.sum())}")

        feature_rows = _build_feature_daily_rows(conn, raw, require_available_ts=require_available_ts)
        if feature_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO feature_daily (
                  asof, symbol, feature_name, value, source_fact_ids
                ) VALUES (
                  :asof, :symbol, :feature_name, :value, :source_fact_ids
                )
                """,
                feature_rows,
            )
            print(f"[fundamentals] upserted feature_daily rows: {len(feature_rows)}")
        conn.commit()


def _jquants_env_credentials() -> tuple[str | None, str | None]:
    mail = os.getenv("JQUANTS_MAIL") or os.getenv("JQUANTS_EMAIL")
    password = os.getenv("JQUANTS_PASSWORD")
    return mail, password


def _normalize_jquants_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    rename_map = {
        "LocalCode": "symbol",
        "DisclosureDate": "published_ts",
        "DisclosedDate": "published_ts",
        "CurrentPeriodEndDate": "fiscal_period_end",
        "FiscalYearEnd": "fiscal_period_end",
        "NetSales": "revenue",
        "Revenue": "revenue",
        "OperatingProfit": "operating_income",
        "OperatingIncome": "operating_income",
        "Profit": "net_income",
        "NetIncome": "net_income",
        "EarningsPerShare": "eps",
        "BookValuePerShare": "book_value_per_share",
        "DividendPerShare": "dividend_per_share",
        "CashFlowsFromOperatingActivities": "operating_cf",
        "CashFlowsFromInvestingActivities": "free_cf",
        "TotalAssets": "total_assets",
        "Equity": "total_equity",
        "TotalEquity": "total_equity",
        "InterestBearingDebt": "total_debt",
        "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock": "shares_outstanding",
        "ForecastNetSales": "guidance_revenue",
        "ForecastOperatingProfit": "guidance_operating_income",
        "ForecastEarningsPerShare": "guidance_eps",
    }
    for src, dst in rename_map.items():
        if src in frame.columns and dst not in frame.columns:
            frame[dst] = frame[src]
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.strip()
        frame.loc[~frame["symbol"].str.endswith(".T"), "symbol"] = frame["symbol"].astype(str) + ".T"
    if "published_ts" in frame.columns:
        frame["published_ts"] = pd.to_datetime(frame["published_ts"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    if "available_ts" not in frame.columns and "published_ts" in frame.columns:
        frame["available_ts"] = frame["published_ts"]
    if "fiscal_period_end" not in frame.columns:
        frame["fiscal_period_end"] = ""
    if "source" not in frame.columns:
        frame["source"] = "jquants"
    if "currency" not in frame.columns:
        frame["currency"] = "JPY"
    return frame


def import_jquants(db_path: str, fail_closed: bool, require_available_ts: bool) -> None:
    mail, password = _jquants_env_credentials()
    if not mail or not password:
        message = "J-Quants credentials are not set. Expected JQUANTS_MAIL/JQUANTS_PASSWORD."
        if fail_closed:
            raise RuntimeError(message)
        print(f"[fundamentals] {message}")
        return
    try:
        import jquantsapi  # type: ignore
    except Exception as exc:
        message = f"jquantsapi is unavailable: {exc}"
        if fail_closed:
            raise RuntimeError(message) from exc
        print(f"[fundamentals] {message}")
        return

    client = jquantsapi.Client(mail_address=mail, password=password)
    statements = None
    for candidate in ["get_statements", "get_fins_statements", "get_fin_statements"]:
        fetcher = getattr(client, candidate, None)
        if fetcher is None:
            continue
        try:
            statements = fetcher()
            break
        except TypeError:
            try:
                statements = fetcher(code=None)
                break
            except Exception:
                continue
        except Exception as exc:
            if fail_closed:
                raise RuntimeError(f"J-Quants statements fetch failed via {candidate}: {exc}") from exc
    if statements is None:
        message = "Unable to fetch statements from J-Quants client."
        if fail_closed:
            raise RuntimeError(message)
        print(f"[fundamentals] {message}")
        return

    raw = statements if isinstance(statements, pd.DataFrame) else pd.DataFrame(statements)
    if raw.empty:
        print("[fundamentals] J-Quants returned no statements.")
        return
    normalized = _normalize_jquants_frame(raw)
    temp_csv = Path(db_path).with_name("_tmp_jquants_fundamentals.csv")
    normalized.to_csv(temp_csv, index=False)
    try:
        import_csv(db_path, temp_csv, fail_closed=fail_closed, require_available_ts=require_available_ts)
    finally:
        try:
            temp_csv.unlink()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Initialize or import point-in-time fundamentals")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--source", default="noop", choices=["noop", "csv", "jquants"])
    ap.add_argument("--csv_path", default=None)
    ap.add_argument("--fail_closed", action="store_true")
    ap.add_argument("--require_available_ts", action="store_true")
    args = ap.parse_args()

    with connect(args.db) as conn:
        ensure_trade_tables(conn)
    print("[fundamentals] PIT tables ensured.")

    if args.source == "csv":
        if not args.csv_path:
            if args.fail_closed:
                raise ValueError("--csv_path is required when --source csv and --fail_closed is set.")
            print("[fundamentals] csv source selected without --csv_path; schema only.")
            return
        import_csv(args.db, Path(args.csv_path), args.fail_closed, args.require_available_ts)
        return

    if args.source == "jquants":
        import_jquants(args.db, fail_closed=args.fail_closed, require_available_ts=args.require_available_ts)
        return

    print("[fundamentals] noop mode; schema only.")


if __name__ == "__main__":
    main()
