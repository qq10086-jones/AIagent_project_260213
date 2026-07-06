// htr-cards.jsx — compact reusable cards to enrich V1 / V2 / V4 rails.
// Reuse the existing governance-safe bodies (FactorBody, OutcomesBody, GateFlow
// from already-loaded files) so labels (uncalibrated, raw frequency, §10) stay
// intact. These are real research context, not filler.

function FactorCard({ candidate }) {
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<Term>因子构成</Term>} sub={`排序信号 · ${candidate.symbol}`} />
      <div style={{ padding: "12px 14px" }}><FactorBody candidate={candidate} /></div>
    </div>
  );
}

function OutcomesCard({ candidate }) {
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title="历史命中" sub="raw frequency · NOT win rate" />
      <div style={{ padding: "12px 14px" }}><OutcomesBody candidate={candidate} /></div>
    </div>
  );
}

// K-fold verdict summary (Rule 9.4) — read-only, compact.
function CalibCard() {
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title="校准状态" sub="K-fold OOS · Rule 9.4" right={<span className="htr-chip bear">downgrade</span>} />
      <div style={{ padding: "12px 14px" }}>
        <CalibBody />
      </div>
    </div>
  );
}
// Rule 11.9.4 — calibration figures are governance-sensitive and MUST come from
// the real backend, never hardcoded. Fetches /api/calibration/reliability (the
// K-fold verdict). Previously shipped wrong hardcoded values (~0.27 / ~0.20);
// the real values are OOS 0.2823 / in-sample 0.2427.
function CalibBody() {
  const [k, setK] = React.useState(null);
  React.useEffect(() => {
    let alive = true;
    fetch("/api/calibration/reliability", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d && d.kfold) setK(d.kfold); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);
  if (!k) return <div style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>校准状态加载中…</div>;
  if (!k.verdict) return <div style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>校准数据未就绪（无 K-fold 报告）</div>;
  const f = (x, d = 4) => (x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d));
  const down = k.verdict !== "ship";
  const tone = down ? "bear" : "bull";
  return (
    <div>
      <div style={{ padding: "8px 11px", border: `1px solid var(--htr-${tone})`, background: `var(--htr-${tone}-bg)`, borderRadius: 6, marginBottom: 10 }}>
        <div style={{ fontSize: 12.5, fontWeight: 700, color: `var(--htr-${tone})` }}>VERDICT: {k.verdict}</div>
        <div style={{ fontSize: 11, color: "var(--htr-ink)", marginTop: 4, lineHeight: 1.5 }}>{k.reason}</div>
      </div>
      <div style={{ display: "flex", gap: 14, fontSize: 11.5, fontFamily: "var(--htr-font-mono)", flexWrap: "wrap" }}>
        <span>OOS mean <b>{f(k.oos_brier_mean)}</b></span><span>in-sample <b>{f(k.in_sample_brier)}</b></span><span>random <b>{f(k.random_baseline, 2)}</b></span>
      </div>
      <div style={{ marginTop: 8, fontSize: 10, color: "var(--htr-ink-3)", fontStyle: "italic" }}>诚实的不确定性：分数仅作 ranking，不可读作胜率。</div>
    </div>
  );
}

// Full-width §10 gate flow strip — governance context that anchors the page bottom.
function GateStripCard({ gates }) {
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<span><Term>§10</Term> 自动化八阶门槛</span>} sub="governance pipeline · Gate 8 券商执行硬锁定" right={<span className="htr-chip bull">{gates.filter((g) => g.status === "done").length}/{gates.length} done</span>} />
      <div style={{ padding: "12px 14px" }}><GateFlow gates={gates} compact /></div>
    </div>
  );
}

Object.assign(window, { FactorCard, OutcomesCard, CalibCard, CalibBody, GateStripCard });
