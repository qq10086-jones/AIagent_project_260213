/* S株 overlay card (task#7 / Rule 5.2 / ADR-0011).
 *
 * Deliberately PLAIN JS (React.createElement, no JSX) so it is `node --check`-able
 * and is loaded as a normal <script> — it never enters the babel-standalone bundle,
 * so a problem here cannot white-screen the main dashboard. It renders into its own
 * #skabu-root (a separate ReactDOM root) and fetches /api/dashboard itself, fully
 * decoupled from the main App's state. Fail-soft: any error leaves the card empty.
 *
 * Shows held/watchlist names the 100-share-lot gate excludes but S株 unlocks. This is
 * a tradability hint (execMode=s_kabu), NOT an edge signal.
 */
(function () {
  "use strict";
  var tries = 0;

  function fmt(n) {
    try { return Math.round(n).toLocaleString(); } catch (e) { return String(n); }
  }

  function render() {
    var mount = document.getElementById("skabu-root");
    if (!mount) return;
    if (!window.React || !window.ReactDOM || !window.ReactDOM.createRoot) {
      if (tries++ < 25) setTimeout(render, 400);
      return;
    }
    var e = window.React.createElement;
    fetch("/api/dashboard", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var rows = (d && d.sKabuOverlay) || [];
        var header = e("div", { style: { fontWeight: 700, fontSize: "13px", marginBottom: "4px" } },
          "S株 候选 · 持仓/关注");
        var sub = e("div", { style: { fontSize: "10px", opacity: 0.6, marginBottom: "8px" } },
          "整手不可做但 S株 可做(0 佣 / 0 点差)· 可交易性提示,非 edge");
        var body;
        if (!rows.length) {
          body = e("div", { style: { fontSize: "12px", opacity: 0.7 } },
            "无(无 held/watchlist 被 S株 解锁)");
        } else {
          body = rows.map(function (c) {
            var warn = c.concentrationWarn ? " ⚠>20%NAV" : "";
            return e("div", {
              key: c.symbol,
              style: {
                display: "flex", justifyContent: "space-between", gap: "10px",
                fontSize: "12px", padding: "3px 0",
                borderTop: "1px solid rgba(140,150,180,.18)"
              }
            },
              e("span", { style: { fontWeight: 600 } }, c.symbol + " · S株"),
              e("span", { style: { textAlign: "right", opacity: 0.85 } },
                "¥" + fmt(c.price) + " · 可买" + c.sharesAffordable + "股 · " +
                (Number(c.positionFrac) * 100).toFixed(1) + "%NAV" + warn)
            );
          });
        }
        var card = e("div", {
          style: {
            font: "13px/1.45 system-ui, -apple-system, sans-serif",
            background: "rgba(18,20,27,.93)", color: "#e8eaf0",
            border: "1px solid rgba(120,140,200,.35)", borderRadius: "10px",
            padding: "10px 12px", boxShadow: "0 8px 28px rgba(0,0,0,.4)"
          }
        }, header, sub, body);
        window.ReactDOM.createRoot(mount).render(card);
      })
      .catch(function () { /* fail-soft */ });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { setTimeout(render, 500); });
  } else {
    setTimeout(render, 500);
  }
})();
