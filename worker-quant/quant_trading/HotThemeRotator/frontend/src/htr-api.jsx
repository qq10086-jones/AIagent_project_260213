// htr-api.jsx — wires the redesigned (offline-mock) variants to the REAL HTR
// backend. The designer's components read per-symbol detail synchronously off
// `candidate.strategy / .factors / .outcomes / .kline / .aiBrief / .debate`.
// In the live app those come from separate endpoints, so `useEnrichedCandidate`
// fetches them for the selected symbol and OVERLAYS the real values on top of
// the deterministic mock defaults (filled by window.HTR_enrich at boot). The
// mock defaults only ever show for the sub-second before the real fetch lands,
// guaranteeing the cards never crash on undefined while the steady state is the
// project's authoritative data.
//
// Governance: read-only GET only. No POST/write path here (Rule 11.5). No field
// is fabricated — every overlay is verbatim from /api/symbol/{T}/*.

const { useState: apiUseState, useEffect: apiUseEffect } = React;

function useEnrichedCandidate(candidate) {
  const symbol = candidate && candidate.symbol;
  const [detail, setDetail] = apiUseState({});
  // Rule 11.9.1/11.9.5 — per-endpoint arrival status (pending|ok|failed) so the
  // detail bodies can show an honest loading / 示例占位 state instead of leaving
  // the offline mock indistinguishable from real on a failed fetch.
  const [status, setStatus] = apiUseState({});

  apiUseEffect(() => {
    if (!symbol) return;
    let alive = true;
    setDetail({});                       // drop stale overlay on symbol change
    setStatus({ strategy: "pending", profile: "pending", outcomes: "pending", kline: "pending", aiBrief: "pending", debate: "pending" });
    const mark = (key, st) => { if (alive) setStatus((s) => ({ ...s, [key]: st })); };
    const merge = (patch) => { if (alive && patch) setDetail((d) => ({ ...d, ...patch })); };
    const get = (path) =>
      fetch(`/api/symbol/${encodeURIComponent(symbol)}${path}`, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
    // load(): a null fetch OR a payload that fails its shape check = miss → status
    // 'failed' (body keeps the COMPLETE mock and labels it 示例占位, never crashes
    // on a partial/empty real payload); a valid payload = 'ok' + real overlay.
    const load = (key, path, ok, valid) => get(path).then((raw) => {
      if (!alive) return;
      if (raw && (!valid || valid(raw))) { merge(ok(raw)); mark(key, "ok"); }
      else mark(key, "failed");
    });

    // ── fast structural endpoints (overlay near-instantly) ──
    load("strategy", "/strategy", (s) => ({ strategy: s }),
      (s) => Array.isArray(s.risk_warnings) && Array.isArray(s.ladder_tiers));
    load("profile", "/profile", (p) => ({
      factors: p,
      scoreStatus: p.score_status,
      kfold: { status: p.kfold_status, reason: p.kfold_reason, oos: p.kfold_oos_brier, inSample: p.kfold_in_sample_brier },
    }), (p) => p && p.ticker != null);
    load("outcomes", "/outcomes", (o) => ({ outcomes: o }), (o) => Array.isArray(o.items));
    // empty bars = no real series → 'failed' (chart keeps mock, surfaces 示例 badge)
    load("kline", "/kline?sessions=120", (k) => ({ kline: k.bars.map((b) => ({ date: b.date, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.vol })) }),
      (k) => Array.isArray(k.bars) && k.bars.length > 0);

    // ── slow LLM endpoints (Ollama; show loading skeleton until they resolve,
    //    then real narrative — a failure surfaces a 示例占位 banner, never silent) ──
    load("aiBrief", "/llm_brief", (a) => ({ aiBrief: a }), (a) => typeof a.narrative === "string" && Array.isArray(a.factual_grounding));
    load("debate", "/debate_brief", (g) => ({ debate: g }), (g) => g.bull && g.bear && g.judge);

    return () => { alive = false; };
  }, [symbol]);

  // real overlay (detail) wins over the mock defaults carried on `candidate`;
  // `_status` lets the bodies render honest pending/failed states.
  return { ...candidate, ...detail, _status: status };
}

window.useEnrichedCandidate = useEnrichedCandidate;
