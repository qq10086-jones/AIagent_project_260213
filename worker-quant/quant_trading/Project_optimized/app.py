from pathlib import Path
import pandas as pd
import streamlit as st

from trade_schema import connect, ensure_trade_tables, get_latest_trading_day, get_run_meta, resolve_run_artifact_dir
from import_fills import read_trade_file, import_fills_df
from build_positions import build_positions
from build_account_snapshot import build_account_snapshot
from execution_report import generate_execution_report
from live_trade_advisor import build_live_trade_advice
from monitor_live_orders import build_order_monitor
from report_utils import build_human_report, load_target_weights, load_weights_history

DB = "japan_market.db"


def q(conn, sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)


def rerun():
    # Streamlit renamed experimental_rerun -> rerun
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


st.set_page_config(page_title="Quant Dashboard", layout="wide")
st.title("Quant Dashboard")

db_path = st.sidebar.text_input("SQLite DB", DB)
conn = connect(db_path)
ensure_trade_tables(conn)

# ----- Market DB status -----
latest_px_date = get_latest_trading_day(conn)
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
artifact_dir = resolve_run_artifact_dir(run_row.get("snapshot_path")) or (Path("artifacts/decision") / str(asof) / str(run_id))

# Header summary
colA, colB, colC, colD = st.columns(4)
colA.metric("asof", asof)
colB.metric("status", run_row["status"])
colC.metric("ts", run_row["ts"])
colD.write("artifact_dir")
colD.code(str(artifact_dir))

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    ["Runs", "Orders", "Fills", "Positions", "Reconciliation", "Execution Report", "Human Report", "Account", "Live Advice", "Order Monitor", "Trading Ops"]
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

    report_md = artifact_dir / "execution_report.md"
    report_csv = artifact_dir / "execution_report.csv"

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
    st.subheader("Human Report")

    reports_dir = Path("reports")
    target_weights = load_target_weights(reports_dir)
    weights_history = load_weights_history(reports_dir)

    try:
        acc = q(conn, """
            SELECT asof, cash, positions_value, nav, net_trade_cashflow, fees, tax, run_id
            FROM account_snapshots
            ORDER BY asof
        """)
    except Exception:
        acc = pd.DataFrame()

    report = build_human_report(
        asof=str(asof),
        run_row=run_row.to_dict(),
        orders=orders,
        fills=fills,
        positions=pos,
        account_snapshots=acc,
        target_weights=target_weights,
        weights_history=weights_history,
    )

    st.markdown(f"### {report.headline}")
    st.markdown("**Highlights**")
    for item in report.highlights:
        st.write(f"- {item}")

    st.markdown("**Warnings**")
    if report.warnings:
        for item in report.warnings:
            st.write(f"- {item}")
    else:
        st.write("- None")

    st.markdown("**Recommended Actions**")
    for item in report.actions:
        st.write(f"- {item}")


with tab8:
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


with tab9:
    st.subheader("Live Trade Advice")
    st.caption("Decision-support only. Use this to compare the latest live portfolio against the latest model target weights.")

    reports_dir = Path("reports")
    target_weights_path = reports_dir / "target_weights.csv"
    target_meta_path = reports_dir / "target_weights_meta.json"
    promotion_path = reports_dir / "promotion_decision.json"

    advice_report, advice_df = build_live_trade_advice(
        conn,
        reports_dir=reports_dir,
        asof=None,
        target_weights_path=target_weights_path if target_weights_path.exists() else None,
        target_meta_path=target_meta_path if target_meta_path.exists() else None,
        promotion_path=promotion_path if promotion_path.exists() else None,
        lot_size=100,
        min_trade_value=5000.0,
    )

    s1, s2, s3, s4 = st.columns(4)
    summary = advice_report.get("summary", {})
    s1.metric("advice asof", advice_report.get("asof") or "-")
    s2.metric("actionable rows", int(summary.get("actionable_count", 0)))
    s3.metric("cash", f"{float(summary.get('cash', 0.0)):,.0f}")
    s4.metric("NAV", f"{float(summary.get('nav', 0.0)):,.0f}")

    st.markdown(f"### {advice_report.get('headline', 'Live advice')}")
    if advice_report.get("warnings"):
        for item in advice_report["warnings"]:
            st.warning(item)
    else:
        st.success("No structural warning flags in the current advice snapshot.")

    st.markdown("**Recommended Actions**")
    for item in advice_report.get("actions", []):
        st.write(f"- {item}")

    actionable_df = advice_df[advice_df["side"].isin(["BUY", "SELL"])].copy() if not advice_df.empty else advice_df
    st.markdown("**Suggested Orders**")
    if actionable_df.empty:
        st.info("No actionable buy/sell rows after lot-size and min-trade filters.")
    else:
        st.dataframe(actionable_df, use_container_width=True)
        st.download_button(
            "Download live_trade_advice.csv",
            actionable_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="live_trade_advice.csv",
            mime="text/csv",
        )

    with st.expander("Show all advice rows"):
        st.dataframe(advice_df, use_container_width=True)


with tab10:
    st.subheader("Order Monitor")
    st.caption("This panel infers whether a limit order may have traded through in market data. It does not broker-confirm execution.")

    monitor_report, monitor_df = build_order_monitor(conn, run_id, str(asof))
    om1, om2, om3 = st.columns(3)
    om1.metric("orders", int(monitor_report.get("summary", {}).get("order_count", 0)))
    om2.metric("suspected fills", int(monitor_report.get("summary", {}).get("suspected_filled", 0)))
    om3.metric("pending/near", int(monitor_report.get("summary", {}).get("pending", 0)))

    st.markdown(f"### {monitor_report.get('headline', 'Order monitor')}")
    if monitor_report.get("warnings"):
        for item in monitor_report["warnings"]:
            st.warning(item)

    if monitor_df.empty:
        st.info("No orders found for this run.")
    else:
        st.dataframe(monitor_df, use_container_width=True)
        st.download_button(
            "Download order_monitor.csv",
            monitor_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"order_monitor_{asof}_{run_id}.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Confirm Fill")
    st.caption("Use this only after you confirm the actual execution. This writes to fills, rebuilds positions, refreshes NAV, and regenerates the execution report.")

    if len(orders) == 0:
        st.info("No proposed orders available to confirm.")
    else:
        order_labels = [
            f"{r.order_id} | {r.symbol} {r.side} qty={float(r.qty):,.0f} limit={float(r.limit_price) if pd.notna(r.limit_price) else 0.0:,.3f}"
            for r in orders.itertuples(index=False)
        ]
        label_to_order = {label: row for label, (_, row) in zip(order_labels, orders.iterrows())}
        selected_label = st.selectbox("Order to confirm", order_labels, key="confirm_fill_order")
        selected_order = label_to_order[selected_label]

        with st.form("confirm_fill_form"):
            cf1, cf2, cf3, cf4 = st.columns(4)
            fill_qty = cf1.number_input("Fill Qty", value=float(selected_order["qty"]), step=1.0)
            fill_price = cf2.number_input(
                "Fill Price",
                value=float(selected_order["limit_price"]) if pd.notna(selected_order["limit_price"]) else 0.0,
                step=0.1,
            )
            fill_fee = cf3.number_input("Fee", value=0.0, step=1.0)
            fill_tax = cf4.number_input("Tax", value=0.0, step=1.0)
            cf5, cf6 = st.columns(2)
            fill_ts = cf5.text_input("Fill TS", value="")
            fill_ref = cf6.text_input("External Ref / Memo", value="manual_confirm")
            initial_cash_cf = st.number_input(
                "Initial cash if first snapshot",
                value=400000.0,
                step=10000.0,
                key="confirm_fill_initial_cash",
            )
            confirm_submitted = st.form_submit_button("Confirm and Write Fill")

        if confirm_submitted:
            try:
                fill_df = pd.DataFrame(
                    [
                        {
                            "ts": fill_ts or pd.Timestamp.now().isoformat(timespec="seconds"),
                            "symbol": str(selected_order["symbol"]),
                            "side": str(selected_order["side"]),
                            "qty": float(fill_qty),
                            "price": float(fill_price),
                            "fee": float(fill_fee),
                            "tax": float(fill_tax),
                            "external_ref": fill_ref,
                            "order_id": str(selected_order["order_id"]),
                        }
                    ]
                )
                n = import_fills_df(conn, run_id, str(asof), fill_df, venue="SBI", force=False, source="manual_confirm")
                _prev, _rows, missing = build_positions(conn, run_id, str(asof))
                snap = build_account_snapshot(conn, run_id, str(asof), initial_cash=float(initial_cash_cf))
                md, _csvp = generate_execution_report(conn, run_id, str(asof), artifact_dir)
                msg = f"Imported {n} fill. NAV={snap['nav']:,.0f}. report={md.name}"
                if missing:
                    msg += f" (missing prices: {missing})"
                st.success(msg)
                rerun()
            except Exception as e:
                st.error(str(e))


with tab11:
    st.subheader("Import fills & refresh reports")
    st.caption("This tab writes to SQLite: fills / positions / account_snapshots / execution_report")

    st.markdown("### 1) Upload broker fills (CSV/XLSX)")
    uploaded = st.file_uploader("Upload fills file", type=["csv", "xlsx", "xls"], key="fills_upload")
    venue = st.text_input("Venue", value="SBI")
    initial_cash = st.number_input("Initial cash (only used if no previous account_snapshot)", value=0.0, step=10000.0)
    force = st.checkbox("Force import even if asof mismatch", value=False)

    df_preview = None
    if uploaded is not None:
        # Keep original suffix so readers that rely on extension still work
        suffix = Path(uploaded.name).suffix or ".csv"
        tmp_path = Path(f".streamlit_tmp_fills{suffix}")
        tmp_path.write_bytes(uploaded.getvalue())
        try:
            df_preview = read_trade_file(str(tmp_path))
            st.dataframe(df_preview.head(50), use_container_width=True)
            st.info("Expected columns: ts, symbol, side, qty, price, (fee,tax,external_ref,order_id)")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    colx, coly = st.columns(2)
    with colx:
        if st.button("Import fills", disabled=(df_preview is None)):
            try:
                n = import_fills_df(conn, run_id, str(asof), df_preview, venue=venue, force=force)
                st.success(f"Imported {n} fills.")
                rerun()
            except Exception as e:
                st.error(str(e))

    with coly:
        if st.button("Rebuild positions + NAV + report"):
            try:
                _prev, _rows, missing = build_positions(conn, run_id, str(asof))
                snap = build_account_snapshot(conn, run_id, str(asof), initial_cash=float(initial_cash))
                md, csvp = generate_execution_report(conn, run_id, str(asof), artifact_dir)
                msg = f"Done. NAV={snap['nav']:,.0f}. report={md.name}"
                if missing:
                    msg += f" (missing prices: {missing})"
                st.success(msg)
                rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("### 2) Manual trade entry (optional)")
    with st.form("manual_trade"):
        c1, c2, c3, c4 = st.columns(4)
        sym = c1.text_input("Symbol")
        side = c2.selectbox("Side", ["BUY", "SELL"])
        qty = c3.number_input("Qty", value=0.0, step=1.0)
        price = c4.number_input("Price", value=0.0, step=0.01)
        c5, c6, c7, c8 = st.columns(4)
        fee = c5.number_input("Fee", value=0.0, step=1.0)
        tax = c6.number_input("Tax", value=0.0, step=1.0)
        ts = c7.text_input("TS", value="")
        memo = c8.text_input("Memo / Ref", value="")
        submitted = st.form_submit_button("Add to staged trades")
        if submitted:
            if "staged" not in st.session_state:
                st.session_state["staged"] = []
            st.session_state["staged"].append({
                "ts": ts or pd.Timestamp.now().isoformat(timespec="seconds"),
                "symbol": sym,
                "side": side,
                "qty": float(qty),
                "price": float(price),
                "fee": float(fee),
                "tax": float(tax),
                "external_ref": memo or "manual",
            })
            st.success("Added.")

    staged = st.session_state.get("staged", [])
    if staged:
        st.dataframe(pd.DataFrame(staged), use_container_width=True)
        if st.button("Import staged trades as fills"):
            try:
                df_s = pd.DataFrame(staged)
                n = import_fills_df(conn, run_id, str(asof), df_s, venue=venue, force=force)
                st.session_state["staged"] = []
                st.success(f"Imported {n} staged fills.")
                rerun()
            except Exception as e:
                st.error(str(e))


conn.close()
