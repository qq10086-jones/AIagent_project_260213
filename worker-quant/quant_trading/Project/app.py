import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

DB = "japan_market.db"


def q(conn, sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)


st.set_page_config(page_title="Quant Dashboard (Read-only)", layout="wide")
st.title("Quant Dashboard (Read-only)")

db_path = st.sidebar.text_input("SQLite DB", DB)
conn = sqlite3.connect(db_path)

# ----- Market DB status -----
latest_px = q(conn, "SELECT MAX(date) AS max_date FROM daily_prices")
latest_px_date = latest_px["max_date"].iloc[0] if len(latest_px) else None
st.sidebar.markdown("### Market DB")
st.sidebar.write(f"latest price date: **{latest_px_date}**")

# Latest runs
runs = q(conn, """
SELECT run_id, asof, ts, status, snapshot_path
FROM decision_runs
ORDER BY ts DESC
LIMIT 50
""")
if len(runs) == 0:
    st.warning("No decision_runs found.")
    st.stop()

auto_latest = st.sidebar.checkbox("Auto-select latest run", value=True)
run_list = runs["run_id"].tolist()
run_id = st.sidebar.selectbox("Select run_id", run_list, index=0 if auto_latest else 0)

run_row = runs[runs["run_id"] == run_id].iloc[0]
asof = run_row["asof"]

# Header summary
colA, colB, colC, colD = st.columns(4)
colA.metric("asof", asof)
colB.metric("status", run_row["status"])
colC.metric("ts", run_row["ts"])
colD.write("snapshot_path")
colD.code(str(run_row["snapshot_path"]))

if latest_px_date is not None and str(asof) != str(latest_px_date):
    st.warning(
        f"⚠️ run asof={asof} but market latest date={latest_px_date}. "
        "If you see missing prices/valuation gaps, update DB or use a run aligned to the latest trading day."
    )

orders = q(conn, """
SELECT order_id, symbol, side, qty, order_type, limit_price, expected_value, status, created_ts
FROM orders WHERE run_id=?
ORDER BY side, expected_value DESC
""", (run_id,))

fills = q(conn, """
SELECT fill_id, ts, symbol, side, qty, price, fee, tax, venue, external_ref
FROM fills WHERE run_id=? AND asof=?
ORDER BY ts
""", (run_id, asof))

pos = q(conn, """
SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl
FROM positions WHERE asof=?
ORDER BY market_value DESC
""", (asof,))

# --- Reconciliation (Orders vs Fills) - no groupby.apply (future-proof) ---
recon = None
if len(orders) > 0:
    if len(fills) > 0:
        f = fills.copy()
        f["fee"] = f["fee"].fillna(0.0)
        f["tax"] = f["tax"].fillna(0.0)
        f["notional"] = f["qty"] * f["price"]
        f["px_qty"] = f["price"] * f["qty"]

        agg = f.groupby(["symbol", "side"], as_index=False).agg(
            fill_qty=("qty", "sum"),
            fill_notional=("notional", "sum"),
            px_qty=("px_qty", "sum"),
            fee=("fee", "sum"),
            tax=("tax", "sum"),
            n_fills=("ts", "count"),
        )
        agg["vwap"] = agg["px_qty"] / agg["fill_qty"]
        agg = agg.drop(columns=["px_qty"])
    else:
        agg = pd.DataFrame(columns=["symbol", "side", "fill_qty", "fill_notional", "fee", "tax", "n_fills", "vwap"])

    recon = orders.merge(agg, on=["symbol", "side"], how="left")
    recon["fill_qty"] = recon["fill_qty"].fillna(0.0)
    recon["qty_diff"] = recon["fill_qty"] - recon["qty"]
    recon["fill_notional"] = recon["fill_notional"].fillna(0.0)
    recon["fee"] = recon["fee"].fillna(0.0)
    recon["tax"] = recon["tax"].fillna(0.0)
    recon["n_fills"] = recon["n_fills"].fillna(0).astype(int)

# ----- Summary by side (BUY/SELL) -----
if len(orders) > 0:
    o = orders.copy()
    o["expected_value"] = o["expected_value"].fillna(0.0)
    exp_buy = float(o.loc[o["side"] == "BUY", "expected_value"].sum())
    exp_sell = float(o.loc[o["side"] == "SELL", "expected_value"].sum())
else:
    exp_buy = exp_sell = 0.0

if len(fills) > 0:
    ff = fills.copy()
    ff["fee"] = ff["fee"].fillna(0.0)
    ff["tax"] = ff["tax"].fillna(0.0)
    ff["notional"] = ff["qty"] * ff["price"]
    fill_buy = float(ff.loc[ff["side"] == "BUY", "notional"].sum())
    fill_sell = float(ff.loc[ff["side"] == "SELL", "notional"].sum())
    fees = float(ff["fee"].sum())
    tax = float(ff["tax"].sum())
else:
    fill_buy = fill_sell = fees = tax = 0.0

# Top metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("orders", len(orders))
m2.metric("fills", len(fills))
m3.metric("expected BUY notional", f"{exp_buy:,.0f}")
m4.metric("expected SELL notional", f"{exp_sell:,.0f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("filled BUY notional", f"{fill_buy:,.0f}")
m6.metric("filled SELL notional", f"{fill_sell:,.0f}")
m7.metric("fee", f"{fees:,.0f}")
m8.metric("tax", f"{tax:,.0f}")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Runs", "Orders", "Fills", "Positions", "Reconciliation", "Execution Report", "Account"]
)


with tab1:
    st.dataframe(runs, use_container_width=True)

with tab2:
    st.dataframe(orders, use_container_width=True)
    st.download_button(
        "Download orders.csv",
        orders.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"orders_{asof}_{run_id}.csv",
        mime="text/csv",
    )

with tab3:
    st.dataframe(fills, use_container_width=True)
    st.download_button(
        "Download fills.csv",
        fills.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"fills_{asof}_{run_id}.csv",
        mime="text/csv",
    )

with tab4:
    st.dataframe(pos, use_container_width=True)
    st.download_button(
        "Download positions.csv",
        pos.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"positions_{asof}.csv",
        mime="text/csv",
    )

    # valuation completeness hint
    if len(pos) > 0 and "market_price" in pos.columns:
        missing = pos[pos["market_price"].isna()]["symbol"].tolist()
        if missing:
            st.warning(f"⚠️ Missing market_price for valuation: {missing}")

with tab5:
    st.subheader("Orders vs Fills (reconciliation)")
    if recon is None:
        st.info("No orders found for this run.")
    else:
        st.dataframe(recon, use_container_width=True)
        st.download_button(
            label="Download reconciliation.csv",
            data=recon.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"reconciliation_{asof}_{run_id}.csv",
            mime="text/csv",
        )

with tab6:
    st.subheader("Execution Report")

    report_md = Path(rf"artifacts\decision\{asof}\execution_report.md")
    report_csv = Path(rf"artifacts\decision\{asof}\execution_report.csv")

    if report_md.exists():
        st.markdown(report_md.read_text(encoding="utf-8"))
    else:
        st.info(f"Not found: {report_md}")

    st.divider()

    st.subheader("execution_report.csv")
    if report_csv.exists():
        df_rep = pd.read_csv(report_csv, encoding="utf-8-sig")
        st.dataframe(df_rep, use_container_width=True)
        st.download_button(
            label="Download execution_report.csv",
            data=report_csv.read_bytes(),
            file_name=report_csv.name,
            mime="text/csv",
        )
    else:
        st.info(f"Not found: {report_csv}")

with tab7:
    st.subheader("Account snapshots")

    # try read account_snapshots (table may not exist before migration)
    try:
        acc = q(conn, """
            SELECT asof, cash, positions_value, nav, net_trade_cashflow, fees, tax, run_id
            FROM account_snapshots
            ORDER BY asof
        """)
    except Exception as e:
        st.info("account_snapshots table not found yet. Run migrations/002 and build_account_snapshot.py first.")
        st.stop()

    if len(acc) == 0:
        st.info("No account snapshots yet. Run build_account_snapshot.py to create daily NAV records.")
    else:
        st.dataframe(acc, use_container_width=True)

        # NAV chart
        acc2 = acc.copy()
        acc2["asof"] = pd.to_datetime(acc2["asof"])
        acc2 = acc2.sort_values("asof")
        acc2 = acc2.set_index("asof")

        st.subheader("NAV curve")
        st.line_chart(acc2["nav"])

        # Daily return & drawdown
        nav = acc2["nav"].astype(float)
        ret = nav.pct_change().fillna(0.0)
        peak = nav.cummax()
        dd = (nav / peak) - 1.0

        c1, c2, c3 = st.columns(3)
        c1.metric("latest NAV", f"{nav.iloc[-1]:,.0f}")
        c2.metric("latest daily return", f"{ret.iloc[-1]*100:.2f}%")
        c3.metric("max drawdown", f"{dd.min()*100:.2f}%")

        st.subheader("Daily return")
        st.line_chart(ret)

        st.subheader("Drawdown")
        st.line_chart(dd)


conn.close()
