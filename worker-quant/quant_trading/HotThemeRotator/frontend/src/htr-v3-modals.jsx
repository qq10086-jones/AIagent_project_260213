// htr-v3-modals.jsx — secondary modals + topbar chips for V3.
// Kept governance disclosures intact: watchlist (Rule 11.3 user-state),
// calibration K-fold verdict (Rule 9.4), notifier double-confirm (Rule 12.7),
// proposal inbox full metadata (Rule 13.18), manual entry (Rule 3 carve-out).

const M_inputStyle = {
  fontSize: 12, padding: "5px 9px", border: "1px solid var(--htr-line)", borderRadius: 5,
  background: "var(--htr-surface)", color: "var(--htr-ink)", fontFamily: "var(--htr-font-mono)",
  width: "100%", boxSizing: "border-box",
};
function mBtn(kind) {
  if (kind === "primary") return { fontSize: 12, padding: "6px 15px", background: "var(--htr-accent)", color: "var(--htr-accent-ink)", border: "1px solid var(--htr-accent)", borderRadius: 6, cursor: "pointer", fontFamily: "inherit" };
  return { fontSize: 12, padding: "6px 15px", background: "transparent", color: "var(--htr-ink-2)", border: "1px solid var(--htr-line-2)", borderRadius: 6, cursor: "pointer", fontFamily: "inherit" };
}

function ModalShell({ title, sub, onClose, width = 560, children }) {
  // Rule 13.18 / WCAG 2.4.3 / 4.1.2 — accessible dialog: role+aria-modal, focus the
  // close button on mount, restore focus on unmount, Escape to close, Tab trapped.
  const titleId = React.useId();
  const boxRef = React.useRef(null);
  const closeRef = React.useRef(null);
  React.useEffect(() => {
    const prev = document.activeElement;
    if (closeRef.current) closeRef.current.focus();
    const onKey = (e) => {
      if (e.key === "Escape") { onClose(); return; }
      if (e.key === "Tab" && boxRef.current) {
        const f = boxRef.current.querySelectorAll('button,[href],input,select,textarea,[tabindex]:not([tabindex="-1"])');
        if (!f.length) return;
        const first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("keydown", onKey); if (prev && prev.focus) prev.focus(); };
  }, [onClose]);
  return (
    <div className="htr" style={{ position: "fixed", inset: 0, background: "rgba(20,19,15,0.5)", zIndex: 500, display: "flex", alignItems: "flex-start", justifyContent: "center", paddingTop: "7vh", backdropFilter: "blur(2px)" }}
         onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={boxRef} role="dialog" aria-modal="true" aria-labelledby={titleId} style={{ background: "var(--htr-surface)", borderRadius: 12, padding: "20px 24px", width, maxWidth: "94vw", maxHeight: "84vh", overflow: "auto", boxShadow: "var(--htr-shadow-lg)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14, alignItems: "baseline", gap: 12 }}>
          <div>
            <h3 id={titleId} style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>{title}</h3>
            {sub && <div style={{ fontSize: 11, color: "var(--htr-ink-3)", marginTop: 3 }}>{sub}</div>}
          </div>
          <button ref={closeRef} onClick={onClose} aria-label="关闭" style={{ background: "none", border: "none", fontSize: 20, cursor: "pointer", color: "var(--htr-ink-3)", lineHeight: 1 }}>×</button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ── §10 gates chip + modal ──
function GatesChip({ gates }) {
  const [open, setOpen] = React.useState(false);
  const done = gates.filter((g) => g.status === "done").length;
  const blocked = gates.filter((g) => g.status === "blocked").length;
  return (
    <>
      <button onClick={() => setOpen(true)} className="htr-chip accent" style={{ cursor: "pointer", fontWeight: 700 }}>
        §10 <span style={{ color: "var(--htr-bull)" }}>{done}/{gates.length} ✓</span>{blocked > 0 && <span style={{ color: "var(--htr-bear)" }}>⛔{blocked}</span>}
      </button>
      {open && (
        <ModalShell title="§10 自动化八阶门槛" sub={`governance pipeline · 按规则逐级解锁 · ${done}/${gates.length} done · ${blocked} blocked`} onClose={() => setOpen(false)} width={940}>
          <GateFlow gates={gates} compact={false} />
          <div style={{ marginTop: 14, fontSize: 11.5, color: "var(--htr-ink-3)", lineHeight: 1.6 }}>
            自动化是分阶产品路径，不是通往实盘的捷径。Gate 8（券商执行）当前<b style={{ color: "var(--htr-bear)" }}>硬锁定</b>，需治理评审 + 人工批准方可解锁。在 Gate 3-5 验证完成前，所有界面输出保持 <span className="htr-mono">uncalibrated_research_score</span>。
          </div>
        </ModalShell>
      )}
    </>
  );
}

// ── ★ watchlist chip + modal ──
function WatchlistChip({ onSelectSymbol }) {
  const [open, setOpen] = React.useState(false);
  const wl = useWatchlist();
  return (
    <>
      <button onClick={() => setOpen(true)} className="htr-chip" style={{ cursor: "pointer" }}>★ 关注 <span className="htr-mono" style={{ color: "var(--htr-accent)" }}>{wl.size}</span></button>
      {open && (
        <ModalShell title="关注列表 · Watchlist" sub="Rule 14.9 user_state · 服务端持久 + localStorage 缓存 · 不影响 portfolio / calibration / 自动 cron" onClose={() => setOpen(false)} width={480}>
          {wl.entries.length === 0 ? (
            <div style={{ fontSize: 12.5, color: "var(--htr-ink-3)", fontStyle: "italic", padding: "16px 0", textAlign: "center" }}>关注列表为空。在龙头卡点 ☆ 加关注，或顶部搜索 ticker 后加入。</div>
          ) : wl.entries.map((e) => (
            <div key={e.symbol} style={{ display: "grid", gridTemplateColumns: "90px 1fr 56px 56px", gap: 8, alignItems: "center", padding: "7px 0", borderBottom: "1px solid var(--htr-line-soft)", fontSize: 12.5 }}>
              <span className="htr-mono" style={{ fontWeight: 700 }}>{e.symbol}</span>
              <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>加入 {String(e.added_ts).slice(0, 16).replace("T", " ")}</span>
              <button onClick={() => { onSelectSymbol(e.symbol); setOpen(false); }} style={mBtn("ghost")}>查看</button>
              <button onClick={() => wl.remove(e.symbol)} style={mBtn("ghost")}>移除</button>
            </div>
          ))}
        </ModalShell>
      )}
    </>
  );
}

// ── 📐 calibration K-fold chip + modal (Rule 9.4) ──
// Rule 11.9.4 — K-fold figures are governance-sensitive; fetch the REAL
// /api/calibration/reliability instead of shipping hardcoded literals (which
// were wrong (~0.27 / ~0.20) vs the real OOS 0.2823 / in-sample 0.2427).
function CalibrationChip() {
  const [open, setOpen] = React.useState(false);
  const [k, setK] = React.useState(null);
  React.useEffect(() => {
    if (!open) return;
    let alive = true;
    fetch("/api/calibration/reliability", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d && d.kfold) setK(d.kfold); })
      .catch(() => {});
    return () => { alive = false; };
  }, [open]);
  const folds = (k && k.folds) || [];
  const max = Math.max(0.5, ...folds.map((f) => f.calibrated_brier));
  const baseline = (k && k.random_baseline) || 0.25;
  const f4 = (x) => (x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(4));
  return (
    <>
      <button onClick={() => setOpen(true)} className="htr-chip" style={{ cursor: "pointer" }}>📐 校准</button>
      {open && (
        <ModalShell title="Calibration · K-fold OOS" sub="Rule 9.4 OOS gate · 决定龙头卡是否暴露 calibrated_prob" onClose={() => setOpen(false)} width={680}>
          {(!k || !k.verdict) ? (
            <div style={{ padding: "16px 14px", fontSize: 12, color: "var(--htr-ink-3)" }}>{k ? "校准数据未就绪（无 K-fold 报告）" : "校准数据加载中…（/api/calibration/reliability）"}</div>
          ) : (<>
          <div style={{ padding: "12px 14px", marginBottom: 14, border: `1px solid var(--htr-${k.verdict === "ship" ? "bull" : "bear"})`, borderRadius: 8, background: `var(--htr-${k.verdict === "ship" ? "bull" : "bear"}-bg)` }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: `var(--htr-${k.verdict === "ship" ? "bull" : "bear"})` }}>VERDICT: {k.verdict}</div>
            <div style={{ fontSize: 11.5, marginTop: 5, color: "var(--htr-ink)" }}>{k.reason}</div>
            <div style={{ display: "flex", gap: 16, marginTop: 10, fontSize: 11.5, fontFamily: "var(--htr-font-mono)" }}>
              <span>OOS mean <b>{f4(k.oos_brier_mean)}</b></span><span>in-sample <b>{f4(k.in_sample_brier)}</b></span><span>random <b>{Number(baseline).toFixed(2)}</b></span><span>OOS std <b>{f4(k.oos_brier_std)}</b></span>
            </div>
          </div>
          <div className="htr-eyebrow" style={{ marginBottom: 8 }}>Per-fold OOS Brier（越低越好；橙线 = 随机基线）</div>
          {folds.map((f) => {
            const calPct = (f.calibrated_brier / max) * 100, rawPct = (f.raw_brier / max) * 100, basePct = (baseline / max) * 100;
            const over = f.calibrated_brier >= baseline;
            return (
              <div key={f.fold_idx} style={{ marginBottom: 8, fontSize: 11 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--htr-font-mono)" }}>
                  <span>fold {f.fold_idx} (test n={f.test_n})</span>
                  <span style={{ color: over ? "var(--htr-bear)" : "var(--htr-bull)" }}>cal Brier {f.calibrated_brier.toFixed(4)}</span>
                </div>
                <div style={{ position: "relative", height: 16, background: "var(--htr-surface-2)", borderRadius: 4, marginTop: 3, border: "1px solid var(--htr-line-soft)" }}>
                  <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${rawPct}%`, background: "var(--htr-ink-4)", opacity: 0.45 }} />
                  <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${calPct}%`, background: over ? "var(--htr-bear)" : "var(--htr-bull)", borderRadius: "4px 0 0 4px" }} />
                  <div style={{ position: "absolute", left: `${basePct}%`, top: -2, bottom: -2, width: 2, background: "var(--htr-warn)" }} />
                </div>
              </div>
            );
          })}
          </>)}
        </ModalShell>
      )}
    </>
  );
}

// ── 📢 notifier chip + modal (Rule 12.7 double-confirm; Rule 11.10 — real POST + audit log) ──
function NotifierChip() {
  const [open, setOpen] = React.useState(false);
  const [channels, setChannels] = React.useState({ desktop: false, email: false, telegram: false });
  const [pending, setPending] = React.useState(null);
  const [confirmText, setConfirmText] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [msg, setMsg] = React.useState("");

  const loadState = () => fetch("/api/notifier/state", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d && d.channels) setChannels((c) => ({ ...c, ...d.channels })); })
    .catch(() => {});
  React.useEffect(() => { if (open) loadState(); }, [open]);

  // Rule 11.10: real POST /api/notifier/toggle. Backend enforces the Rule 12.7
  // stage-2 gate + writes the append-only audit log; a refusal returns 422.
  const toggle = (ch, action, dryRun) => {
    setBusy(true); setMsg("");
    return fetch("/api/notifier/toggle", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        channel: ch, action,
        user_confirm_text: action === "enable" ? `我理解这会触发 ${ch} 推送` : "",
        dry_run: !!dryRun,
      }),
    }).then(async (r) => {
      const d = await r.json().catch(() => ({}));
      if (!r.ok) { setMsg(`已拦截 (Rule 12.7): ${d && d.detail ? d.detail : "HTTP " + r.status}`); return; }
      if (dryRun) { setMsg(`dry-run 完成 — 样本告警已走 discipline filter，未真正启用、未写 toggle 日志`); return; }
      setMsg(`${ch} 已${action === "enable" ? "启用" : "禁用"}（已写 audit log）`);
      loadState();
    }).catch((e) => setMsg(String(e))).finally(() => { setBusy(false); setPending(null); setConfirmText(""); });
  };

  return (
    <>
      <button onClick={() => setOpen(true)} className="htr-chip" style={{ cursor: "pointer" }}>📢 推送</button>
      {open && (
        <ModalShell title="Notifier Channels (Rule 12.7)" sub="默认全禁 · 启用需双重确认 · 每个 toggle 写入 append-only audit log" onClose={() => setOpen(false)} width={540}>
          <div style={{ padding: "8px 12px", marginBottom: 14, background: "var(--htr-warn-bg)", border: "1px solid var(--htr-warn)", borderRadius: 6, fontSize: 11.5, color: "var(--htr-warn)", lineHeight: 1.5 }}>
            ⚠ 启用 desktop / email / telegram 推送会<b>主动通知</b>你 — Rule 12.0 stage 2 强制 + Rule 12.1-12.6 anti-FOMO discipline filter 全过才被推。
          </div>
          {msg && <div style={{ padding: "7px 12px", marginBottom: 10, background: "var(--htr-info-bg)", border: "1px solid var(--htr-info)", borderRadius: 6, fontSize: 11, color: "var(--htr-info)" }}>{msg}</div>}
          {Object.entries(channels).map(([ch, on]) => (
            <div key={ch} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 12px", marginBottom: 8, background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 8 }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700 }}>{ch}</div>
                <div style={{ fontSize: 11, color: on ? "var(--htr-bull)" : "var(--htr-ink-3)" }}>{on ? "● 已启用" : "○ 已禁用"}</div>
              </div>
              {on ? (
                <button disabled={busy} onClick={() => toggle(ch, "disable", false)} style={mBtn("ghost")}>禁用</button>
              ) : pending === ch ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 5, width: 320 }}>
                  <input value={confirmText} onChange={(e) => setConfirmText(e.target.value)} placeholder={`输入: 我理解这会触发 ${ch} 推送`} style={M_inputStyle} />
                  <div style={{ display: "flex", gap: 5 }}>
                    <button disabled={busy} onClick={() => toggle(ch, "enable", true)} title="预览：让样本告警走一遍 anti-FOMO discipline filter，不真正启用、不写 toggle 日志 (Rule 12.7.5)" style={mBtn("ghost")}>dry-run</button>
                    <button disabled={busy || confirmText !== `我理解这会触发 ${ch} 推送`} onClick={() => toggle(ch, "enable", false)} style={{ ...mBtn("primary"), opacity: confirmText === `我理解这会触发 ${ch} 推送` ? 1 : 0.45 }}>确认启用</button>
                    <button disabled={busy} style={mBtn("ghost")} onClick={() => { setPending(null); setConfirmText(""); }}>取消</button>
                  </div>
                </div>
              ) : (
                <button disabled={busy} onClick={() => { setPending(ch); setConfirmText(""); setMsg(""); }} style={mBtn("primary")}>启用</button>
              )}
            </div>
          ))}
        </ModalShell>
      )}
    </>
  );
}

// ── 🧠 proposal inbox chip + modal (Rule 11.9.4 real data + plain-language) ──
// Fetches the real /api/proposals (no hardcoded items). Each proposal leads with
// a PLAIN-LANGUAGE summary a non-expert can read; the full Rule 13.6 metadata is
// rendered ALWAYS-VISIBLE on the same screen as the buttons (Rule 13.18.1 — never
// collapsed), with shadow-ack checkbox (13.18.2) + expiry-override gate (13.18.4).
const PROPOSAL_EVIDENCE_CN = {
  chase_filter_overshoot: "追涨过滤器可能设得太松——该拦的追高没拦住",
  cooldown_too_short: "冷却期可能太短——刚加关注就触发了买入提示",
  concentration_drift: "单一标的集中度可能偏高",
  stale_data_leak: "用了过期数据做判断",
};
function proposalStrength(it) {
  const n = it.sample_size || 0;
  const ci = it.confidence_interval;
  const crossesZero = Array.isArray(ci) && ci.length === 2 && ci[0] < 0 && ci[1] > 0;
  if (n < 30 || crossesZero) return { t: "弱", c: "var(--htr-warn)" };
  if (n < 100) return { t: "中", c: "var(--htr-ink-2)" };
  return { t: "强", c: "var(--htr-bull)" };
}
const PROPOSAL_TIER_CN = {
  diagnostic_only: "仅诊断 · 接受后进 shadow 观察，不直接改参数 (Rule 13.14)",
  shadow: "影子 · 并行验证，不影响实盘建议",
  active: "生效 · 接受后改变后续建议",
};
function ProposalInboxChip() {
  const [open, setOpen] = React.useState(false);
  const [data, setData] = React.useState(null);
  // Rule 13.18.2 / 13.18.4 — per-proposal acks: shadow "I understand" + expiry override.
  const [shadowAck, setShadowAck] = React.useState({});
  const [overrideExp, setOverrideExp] = React.useState({});
  const [rejecting, setRejecting] = React.useState(null);   // proposal_id in reject-mode
  const [rejReason, setRejReason] = React.useState("user_disagrees");
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState(null);
  const load = () => fetch("/api/proposals", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null)).then((d) => { if (d) setData(d); }).catch(() => {});
  React.useEffect(() => { if (open) load(); }, [open]);
  // Rule 13.18 — whitelisted POST (Rule 11.5). Accept of a parameter_change
  // requires explicit shadow-disclosure ack (13.18.2); reject takes a bounded reason.
  const post = (id, action, body) => {
    setBusy(true); setErr(null);
    return fetch(`/api/proposals/${encodeURIComponent(id)}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
    }).then((r) => r.json().then((j) => ({ ok: r.ok, j })).catch(() => ({ ok: r.ok, j: null })))
      .then(({ ok, j }) => {
        setBusy(false);
        if (!ok) { setErr((j && j.detail && (j.detail.message || j.detail.reason)) || "操作失败"); return; }
        setRejecting(null); load();
      }).catch((e) => { setBusy(false); setErr(String(e)); });
  };
  // Rule 13.18.2 — the shadow ack is a visible "I understand" checkbox (NOT a
  // browser dialog), and the accept button is gated on it (+ expiry override) below.
  const accept = (it) => {
    const needsShadow = !!(it.requires_shadow_disclosure || it.parameter_change);
    post(it.proposal_id, "accept", { user_confirm_shadow: needsShadow && !!shadowAck[it.proposal_id] });
  };
  const items = (data && data.items) || [];
  const counts = (data && data.counts) || {};
  return (
    <>
      <button onClick={() => setOpen(true)} className="htr-chip warn" style={{ cursor: "pointer" }}>🧠 提案 <span className="htr-mono">{counts.proposals != null ? counts.proposals : items.length}</span></button>
      {open && (
        <ModalShell title="决策提案收件箱 (L6)" sub="系统的自我改进建议 · 你来批准或拒绝 · 永不自动生效 (Rule 13.18)" onClose={() => setOpen(false)} width={680}>
          {!data && <div style={{ padding: "16px 14px", fontSize: 12, color: "var(--htr-ink-3)" }}>加载中…（/api/proposals）</div>}
          {/* Rule 13.18.6 — read-side parity: pending/accepted/rejected/expired counts always visible */}
          {data && (
            <div style={{ display: "flex", gap: 14, padding: "8px 14px", marginBottom: 12, background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 6, fontSize: 11, color: "var(--htr-ink-3)", flexWrap: "wrap" }}>
              <span>待批 <b style={{ color: "var(--htr-ink)" }}>{counts.proposals != null ? counts.proposals : items.length}</b></span>
              <span>已接受 <b style={{ color: "var(--htr-ink)" }}>{counts.accepted || 0}</b></span>
              <span>已拒绝 <b style={{ color: "var(--htr-ink)" }}>{counts.rejected || 0}</b></span>
              <span>已过期 <b style={{ color: "var(--htr-ink)" }}>{counts.expired || 0}</b></span>
              <span style={{ flex: 1 }} />
              <span style={{ fontStyle: "italic" }}>逐条审阅 · 无批量 (Rule 13.18.5)</span>
            </div>
          )}
          {data && items.length === 0 && (
            <div style={{ padding: "22px 16px", textAlign: "center" }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>暂无待批提案</div>
              <div style={{ fontSize: 11.5, color: "var(--htr-ink-3)", lineHeight: 1.6 }}>反思系统要等 forward 样本累积到足够量，才会从复盘里生成改进建议（当前样本不足）。</div>
            </div>
          )}
          {items.map((it) => {
            const s = proposalStrength(it);
            const ciStr = Array.isArray(it.confidence_interval) ? `[${it.confidence_interval.join(", ")}]` : "—";
            const expired = !!(it.is_expired_by_age || it.is_expired);
            const needsShadow = !!(it.requires_shadow_disclosure || it.parameter_change);
            const ackedShadow = !!shadowAck[it.proposal_id];
            const overridden = !!overrideExp[it.proposal_id];
            // Rule 13.18.2/13.18.4 — accept gated on shadow ack (param change) + expiry override.
            const acceptDisabled = busy || (needsShadow && !ackedShadow) || (expired && !overridden);
            return (
              <div key={it.proposal_id} style={{ marginBottom: 14, padding: "12px 14px", border: "1px solid " + (expired ? "var(--htr-bear)" : "var(--htr-line)"), borderRadius: 8, background: "var(--htr-surface-2)" }}>
                {/* plain-language headline */}
                <div style={{ fontSize: 13.5, fontWeight: 700, lineHeight: 1.45 }}>系统发现：{PROPOSAL_EVIDENCE_CN[it.evidence_class] || `一处可能的改进（${it.evidence_class || "未分类"}）`}</div>
                <div style={{ fontSize: 11.5, color: "var(--htr-ink-2)", lineHeight: 1.6, marginTop: 4 }}>
                  证据强度 <b style={{ color: s.c }}>{s.t}</b>（样本 {it.sample_size != null ? it.sample_size : "—"} 次 · 置信区间 <span className="htr-mono">{ciStr}</span>{s.t === "弱" ? "，样本偏少/区间跨 0，仅供参考" : ""}）<br />
                  类型：{PROPOSAL_TIER_CN[it.tier] || it.tier || "—"}
                </div>
                {/* Rule 13.18.4 — expiry banner */}
                {expired && (
                  <div style={{ marginTop: 10, padding: "7px 10px", background: "var(--htr-bear-bg)", border: "1px solid var(--htr-bear)", borderRadius: 6, fontSize: 11, color: "var(--htr-bear)", lineHeight: 1.5 }}>⚠ 已过期（≥7 天，Rule 13.5）— 现在接受需显式 override；接受按钮默认禁用。</div>
                )}
                {/* Rule 13.18.1 — FULL Rule 13.6 metadata, ALWAYS visible on the same screen as the buttons (never collapsed) */}
                <div style={{ display: "grid", gridTemplateColumns: "150px 1fr", gap: "5px 8px", fontSize: 11, lineHeight: 1.6, marginTop: 10, paddingTop: 8, borderTop: "1px dashed var(--htr-line)" }}>
                  <div style={{ gridColumn: "1 / -1", fontSize: 10, fontWeight: 700, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>RULE 13.6 完整元数据（不可折叠 · Rule 13.18.1）</div>
                  {[["proposal_id", it.proposal_id], ["evidence_class", it.evidence_class], ["sample_size", it.sample_size], ["confidence_interval", ciStr], ["counterfactual_validity", it.counterfactual_validity], ["generator", it.generator], ["tier (Rule 13.13)", it.tier], ["source_trace_ids", Array.isArray(it.source_trace_ids) ? it.source_trace_ids.join(", ") : "—"], ["rationale_pointer", it.rationale_pointer], ["created_ts", it.created_ts], ["age", it.age_days != null ? `${it.age_days}d (expires in ${it.expiry_days_remaining}d)` : "—"]].map(([k, v]) => (
                    <React.Fragment key={k}><span style={{ color: "var(--htr-ink-3)" }}>{k}</span><span className="htr-mono" style={{ fontSize: 10, wordBreak: "break-all" }}>{v != null ? String(v) : "—"}</span></React.Fragment>
                  ))}
                </div>
                {/* Rule 13.18.2 — shadow disclosure + separate "I understand" checkbox (no browser dialog) */}
                {needsShadow && (
                  <label style={{ display: "flex", alignItems: "flex-start", gap: 7, marginTop: 10, padding: "7px 10px", background: "var(--htr-warn-bg)", border: "1px solid var(--htr-warn)", borderRadius: 6, fontSize: 11, color: "var(--htr-warn)", lineHeight: 1.5, cursor: "pointer" }}>
                    <input type="checkbox" checked={ackedShadow} onChange={(e) => setShadowAck((m) => ({ ...m, [it.proposal_id]: e.target.checked }))} style={{ marginTop: 2 }} />
                    <span>我已理解：接受后进入 <span className="htr-mono">lifecycle_stage=shadow</span>（并行验证），<b>不进入实盘建议</b> (Rule 13.14)。</span>
                  </label>
                )}
                {/* Rule 13.18.4 — expiry override toggle */}
                {expired && (
                  <label style={{ display: "flex", alignItems: "center", gap: 7, marginTop: 8, fontSize: 11, color: "var(--htr-bear)", cursor: "pointer" }}>
                    <input type="checkbox" checked={overridden} onChange={(e) => setOverrideExp((m) => ({ ...m, [it.proposal_id]: e.target.checked }))} />
                    <span>override expiry — 我确认在过期后仍接受</span>
                  </label>
                )}
                {rejecting === it.proposal_id ? (
                  <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center", flexWrap: "wrap" }}>
                    <span style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>拒绝理由</span>
                    <select value={rejReason} onChange={(e) => setRejReason(e.target.value)} style={{ fontSize: 11.5, padding: "4px 8px", borderRadius: 5, border: "1px solid var(--htr-line-2)", background: "var(--htr-surface)" }}>
                      <option value="user_disagrees">不同意结论</option>
                      <option value="insufficient_evidence">证据不足</option>
                      <option value="needs_more_data">需要更多数据</option>
                      <option value="duplicate_proposal">重复提案</option>
                      <option value="out_of_scope">超出范围</option>
                      <option value="other">其他</option>
                    </select>
                    <button disabled={busy} onClick={() => post(it.proposal_id, "reject", { reason: rejReason, free_text: "" })} style={{ ...mBtn("primary"), background: "var(--htr-bear)", borderColor: "var(--htr-bear)" }}>确认拒绝</button>
                    <button onClick={() => { setRejecting(null); setErr(null); }} style={mBtn("ghost")}>取消</button>
                  </div>
                ) : (
                  <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center" }}>
                    <button disabled={acceptDisabled} title={acceptDisabled && !busy ? "需先勾选上方确认项 (Rule 13.18.2/13.18.4)" : ""} onClick={() => accept(it)} style={{ ...mBtn("primary"), background: "var(--htr-bull)", borderColor: "var(--htr-bull)", opacity: acceptDisabled ? 0.5 : 1, cursor: acceptDisabled ? "not-allowed" : "pointer" }}>接受</button>
                    <button disabled={busy} onClick={() => { setRejecting(it.proposal_id); setErr(null); }} style={mBtn("ghost")}>拒绝</button>
                  </div>
                )}
                {err && rejecting === it.proposal_id && <div style={{ fontSize: 10.5, color: "var(--htr-bear)", marginTop: 6 }}>{err}</div>}
              </div>
            );
          })}
        </ModalShell>
      )}
    </>
  );
}

// ── manual entry modal (Rule 3 carve-out; Rule 11.10 — real POST, preview->commit) ──
function nowJstIso() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  // Local Beta v0 runs on a single JST machine (Rule 15.0); stamp wall-clock as +09:00.
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}+09:00`;
}
function ManualEntryModal({ kind, onClose, onCommitted, defaultSymbol }) {
  const isFill = kind === "fill";
  // P6-E — seed the fill symbol with the symbol the user is currently viewing.
  const [form, setForm] = React.useState(isFill ? { side: "BUY", symbol: defaultSymbol || "", qty: "", price: "", fee: "0", note: "" } : { amount: "", reason: "deposit", symbol: "", note: "" });
  const [preview, setPreview] = React.useState(null);   // real backend preview response (commit:false)
  const [done, setDone] = React.useState(null);          // committed response (commit:true)
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState("");
  const up = (k, v) => { setForm((f) => ({ ...f, [k]: v })); setPreview(null); setErr(""); };
  const valid = isFill
    ? (form.symbol.trim() && Number(form.qty) > 0 && Number(form.price) > 0)
    : (Number(form.amount) !== 0 && !Number.isNaN(Number(form.amount)));

  const body = (commit) => isFill
    ? { side: form.side, symbol: form.symbol.trim(), qty: Number(form.qty), price: Number(form.price), fee: Number(form.fee || 0), note: form.note, ts: nowJstIso(), commit }
    : { amount: Number(form.amount), reason: form.reason, symbol: (form.symbol || "").trim() || null, note: form.note, ts: nowJstIso(), commit };

  // Rule 11.10: real POST. preview click -> commit:false; confirm click -> commit:true.
  const send = (commit) => {
    setBusy(true); setErr("");
    return fetch(isFill ? "/api/portfolio/fill" : "/api/portfolio/cash_event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body(commit)) })
      .then(async (r) => {
        const data = await r.json().catch(() => ({}));
        if (!r.ok) { setErr(data && data.detail ? String(data.detail) : `HTTP ${r.status}`); return null; }
        return data;
      })
      .catch((e) => { setErr(String(e)); return null; })
      .finally(() => setBusy(false));
  };
  const doPreview = () => send(false).then((d) => { if (d) setPreview(d); });
  const doCommit = () => send(true).then((d) => { if (d && d.committed) { setDone(d); if (onCommitted) onCommitted(); } });

  return (
    <ModalShell title={isFill ? "录入已成交交易" : "录入现金事件"} sub="Rule 3 carve-out · 此操作仅记录用户在外部券商已完成的交易/现金事件" onClose={onClose} width={480}>
      <div style={{ padding: "8px 12px", marginBottom: 14, background: "var(--htr-warn-bg)", border: "1px solid var(--htr-warn)", borderRadius: 6, fontSize: 11.5, color: "var(--htr-warn)", lineHeight: 1.5 }}>
        ⚠ <b>不是下单</b>。HTR 永远不向券商发送订单 (Rule 3 + §10 gate 8)。此表单只录入你已经在外部券商执行过的成交。
      </div>
      {isFill ? (
        <>
          <FRow label="方向"><select value={form.side} disabled={!!done} onChange={(e) => up("side", e.target.value)} style={M_inputStyle}><option value="BUY">BUY (买入)</option><option value="SELL">SELL (卖出)</option></select></FRow>
          <FRow label="Symbol (.T)"><input value={form.symbol} disabled={!!done} onChange={(e) => up("symbol", e.target.value)} placeholder="例: 8035.T" style={M_inputStyle} /></FRow>
          <FRow label="数量"><input type="number" value={form.qty} disabled={!!done} onChange={(e) => up("qty", e.target.value)} placeholder="100" style={M_inputStyle} /></FRow>
          <FRow label="成交价 ¥"><input type="number" value={form.price} disabled={!!done} onChange={(e) => up("price", e.target.value)} placeholder="30176" style={M_inputStyle} /></FRow>
          <FRow label="手续费 ¥"><input type="number" value={form.fee} disabled={!!done} onChange={(e) => up("fee", e.target.value)} style={M_inputStyle} /></FRow>
          <FRow label="备注 (可选)"><input value={form.note} disabled={!!done} onChange={(e) => up("note", e.target.value)} style={M_inputStyle} /></FRow>
        </>
      ) : (
        <>
          <FRow label="金额 ¥ (正/负)"><input type="number" value={form.amount} disabled={!!done} onChange={(e) => up("amount", e.target.value)} placeholder="+10000 入金 / -5000 出金" style={M_inputStyle} /></FRow>
          <FRow label="类型"><select value={form.reason} disabled={!!done} onChange={(e) => up("reason", e.target.value)} style={M_inputStyle}><option value="deposit">deposit (入金)</option><option value="withdrawal">withdrawal (出金)</option><option value="dividend">dividend (分红)</option></select></FRow>
          <FRow label="备注 (可选)"><input value={form.note} disabled={!!done} onChange={(e) => up("note", e.target.value)} style={M_inputStyle} /></FRow>
        </>
      )}
      {preview && !done && (
        <div style={{ padding: "8px 12px", marginTop: 10, background: "var(--htr-info-bg)", border: "1px solid var(--htr-info)", borderRadius: 6, fontSize: 11, color: "var(--htr-info)", lineHeight: 1.6 }}>
          预览(未写入) · NAV {fmtMaybe(preview.before && preview.before.nav)} → {fmtMaybe(preview.after && preview.after.nav)}
          {Array.isArray(preview.warnings) && preview.warnings.length > 0 && <div style={{ marginTop: 4, color: "var(--htr-warn)" }}>⚠ {preview.warnings.join(" · ")}</div>}
          <div style={{ marginTop: 4, color: "var(--htr-ink-3)" }}>确认后 POST commit:true 写入 journal (SSoT)。</div>
        </div>
      )}
      {done && (
        <div style={{ padding: "8px 12px", marginTop: 10, background: "var(--htr-bull-bg, var(--htr-info-bg))", border: "1px solid var(--htr-bull)", borderRadius: 6, fontSize: 11, color: "var(--htr-bull)", lineHeight: 1.6 }}>
          ✓ 已写入 journal (SSoT){done.journal_path ? <> · <span className="htr-mono" style={{ fontSize: 10 }}>{String(done.journal_path).split(/[\\/]/).pop()}</span></> : null}
        </div>
      )}
      {err && <div style={{ padding: "8px 12px", marginTop: 10, background: "var(--htr-bear-bg, var(--htr-warn-bg))", border: "1px solid var(--htr-bear)", borderRadius: 6, fontSize: 11, color: "var(--htr-bear)" }}>错误: {err}</div>}
      <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 16 }}>
        {done ? (
          <>
            <button onClick={onClose} style={mBtn("ghost")}>关闭</button>
            <button onClick={() => window.location.reload()} style={mBtn("primary")}>刷新看持仓</button>
          </>
        ) : (
          <>
            <button onClick={onClose} style={mBtn("ghost")}>取消</button>
            {preview
              ? <button disabled={busy} onClick={doCommit} style={mBtn("primary")}>{busy ? "提交中…" : "确认提交"}</button>
              : <button disabled={busy || !valid} onClick={doPreview} style={{ ...mBtn("primary"), opacity: valid ? 1 : 0.45 }}>{busy ? "…" : "预览"}</button>}
          </>
        )}
      </div>
    </ModalShell>
  );
}
function fmtMaybe(v) {
  return (typeof v === "number" && isFinite(v)) ? "¥" + Math.round(v).toLocaleString() : "—";
}
function FRow({ label, children }) {
  return <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: 8, alignItems: "center", marginBottom: 7 }}><label style={{ fontSize: 12, color: "var(--htr-ink-2)" }}>{label}</label>{children}</div>;
}

Object.assign(window, {
  ModalShell, GatesChip, WatchlistChip, CalibrationChip, NotifierChip, ProposalInboxChip, ManualEntryModal,
  M_inputStyle, mBtn,
});
