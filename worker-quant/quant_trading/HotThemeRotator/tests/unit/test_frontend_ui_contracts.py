"""Frontend governance-contract checks for the redesigned zero-build UI.

The 2026-05-30 designer integration (quant0530) replaced the old
`frontend/{v1..v4,shared}.jsx` with `frontend/src/htr-*.jsx` wired to the real
backend via `htr-api.jsx` + index.html bootWithApi. These tests are intentionally
DESIGN-INDEPENDENT: they assert the governance red-lines that must survive any
visual redesign (Rule 3 / 8.3 / 9.4 / 9.6 / 11.5 / 11.6 / 11.7 / 11.8), not
implementation details of a particular layout.
"""
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = PROJECT_ROOT / "frontend"


def read_frontend_file(rel: str) -> str:
    return (FRONTEND / rel).read_text(encoding="utf-8")


def all_frontend_source() -> dict:
    """Map of relative-path -> text for every shipped frontend source file."""
    out = {}
    for p in sorted(FRONTEND.rglob("*.jsx")) + sorted(FRONTEND.glob("*.html")) + sorted(FRONTEND.rglob("*.js")):
        if "_backup" in str(p):
            continue
        out[str(p.relative_to(FRONTEND))] = p.read_text(encoding="utf-8")
    return out


def test_redesign_files_present_and_real_backend_wired():
    """Integration sanity — the new structure + real-backend wiring exists."""
    names = set(all_frontend_source().keys())
    assert any(n.endswith("htr-v3.jsx") for n in names), "htr-v3.jsx missing"
    assert any(n.endswith("htr-api.jsx") for n in names), "htr-api.jsx (real-backend enrichment) missing"
    index = read_frontend_file("index.html")
    assert "bootWithApi" in index, "index.html must merge real /api/dashboard (bootWithApi)"
    assert "/api/dashboard" in index
    api = read_frontend_file("src/htr-api.jsx")
    # per-symbol detail must come from the real endpoints, not be fabricated
    for ep in ("/strategy", "/profile", "/outcomes", "/kline", "/llm_brief", "/debate_brief"):
        assert ep in api, f"htr-api.jsx must fetch real {ep}"


def test_strategy_card_advice_only_banner_and_rule3_disclaimer():
    """Rule 11.6.1 + 11.6.3 — the banner + Rule 3 manual-execution disclaimer
    literals must exist in the frontend (as the offline-default that mirrors the
    real /api/symbol/{T}/strategy payload), and the Strategy card must render
    them (`s.banner` / `s.rule_3_disclaimer`)."""
    src = "\n".join(all_frontend_source().values())
    assert "由用户在外部券商手动执行" in src, "Rule 11.6.1 advice-only banner literal missing"
    assert "Rule 3 — manual execution outside HTR" in src, "Rule 11.6.3 disclaimer literal missing"
    v3 = read_frontend_file("src/htr-v3.jsx")
    assert "s.banner" in v3 and "rule_3_disclaimer" in v3, "Strategy card must render banner + disclaimer"


def test_strategy_banner_is_non_collapsible():
    """Rule 11.6.1 — the banner must render OUTSIDE the `{expanded && (` collapse
    guard, so collapsing the Strategy card never removes it from the DOM."""
    v3 = read_frontend_file("src/htr-v3.jsx")
    start = v3.find("function V3StrategyCard")
    assert start >= 0, "V3StrategyCard not found"
    end = v3.find("\nfunction ", start + 1)
    body = v3[start:end]
    expanded_idx = body.find("{expanded && (")
    banner_idx = body.find("s.banner")
    disclaimer_idx = body.find("rule_3_disclaimer")
    assert expanded_idx >= 0 and banner_idx >= 0 and disclaimer_idx >= 0
    assert banner_idx < expanded_idx, "Rule 11.6.1 — banner must be before the collapse guard"
    assert disclaimer_idx < expanded_idx, "Rule 11.6.3 — disclaimer must be before the collapse guard"


def test_strategy_risk_warnings_non_collapsible_when_default_collapsed():
    """Rule 11.6.9 — the StrategyCard body may default-collapse for density, but the
    safety triad stays visible: the risk_warnings block MUST render BEFORE the
    {expanded && ( collapse guard (like banner + disclaimer), and the body defaults
    to collapsed."""
    v3 = read_frontend_file("src/htr-v3.jsx")
    start = v3.find("function V3StrategyCard")
    assert start >= 0, "V3StrategyCard not found"
    end = v3.find("\nfunction ", start + 1)
    body = v3[start:end]
    expanded_idx = body.find("{expanded && (")
    risk_idx = body.find("风险标记 (Rule 12)")
    assert expanded_idx >= 0 and risk_idx >= 0
    assert risk_idx < expanded_idx, "Rule 11.6.9 — risk_warnings must render before the collapse guard"
    assert "useState(false)" in body, "Rule 11.6.9 — StrategyCard detail body defaults to collapsed"


def test_seven_tier_ladder_kinds_all_present():
    """Rule 9.6 — all seven ladder tiers must exist; none dropped."""
    src = "\n".join(all_frontend_source().values())
    for kind in ("exit_stretch", "exit_2", "exit_1",
                 "entry_aggressive", "entry_balanced", "entry_conservative", "stop"):
        assert kind in src, f"Rule 9.6 — ladder tier '{kind}' missing from frontend"


def test_uncalibrated_research_score_label_and_rule_9_4_note():
    """Rule 9.4 — score is labeled an uncalibrated research score, never a probability."""
    src = "\n".join(all_frontend_source().values())
    assert "uncalibrated_research_score" in src
    assert "不可视作概率" in src or "非概率" in src or "不是真实胜率" in src


def test_no_probability_or_win_rate_claim_leaks():
    """Rule 8.3 / 9.4 — no positive probability/win-rate claim anywhere. The only
    permitted mentions of 胜率/概率/win rate are NEGATIONS / disclaimers."""
    leak = re.compile(r"\d+\s*%\s*(?:的)?\s*(?:概率|胜率)|(?:概率|胜率)\s*(?:为|是)?\s*\d+\s*%")
    for name, text in all_frontend_source().items():
        m = leak.search(text)
        assert m is None, f"probability/win-rate claim leak in {name}: {m.group(0)!r}"
        # every 'win rate' / '胜率' occurrence must sit in a negated disclaimer
        negations = ("not", "never", "非", "不", "无", "no ")
        for token in ("win rate", "胜率"):
            for idx in (i for i in range(len(text)) if text.startswith(token, i)):
                ctx = text[max(0, idx - 40):idx].lower()
                assert any(neg in ctx for neg in negations), (
                    f"non-negated {token!r} in {name}: ...{text[max(0, idx-40):idx+10]!r}"
                )


def test_no_write_methods_outside_rule_11_5_whitelist():
    """Rule 11.5 — frontend may only POST to a documented whitelist; never
    PUT/DELETE/PATCH. Watchlist is local user-state, not a server write."""
    whitelist = (
        "/api/portfolio/fill", "/api/portfolio/cash_event",
        "/api/notifier/toggle",
        "/api/proposals/", "/api/watchlist/",
    )
    for name, text in all_frontend_source().items():
        for forbidden in ('"PUT"', "'PUT'", '"DELETE"', "'DELETE'", '"PATCH"', "'PATCH'"):
            assert forbidden not in text, f"forbidden write method {forbidden} in {name}"
        # any POST fetch must target a whitelisted path
        for m in re.finditer(r'method\s*:\s*["\']POST["\']', text):
            window = text[max(0, m.start() - 400): m.start() + 400]
            url = re.search(r'/api/[A-Za-z0-9_/{}.$-]+', window)
            assert url and any(url.group(0).startswith(w) for w in whitelist), (
                f"non-whitelisted POST in {name}: {url.group(0) if url else '?'}"
            )


def test_v3_is_document_flow_not_viewport_locked():
    """Rule 11.7 — the default variant (V3) scrolls in document flow."""
    v3 = read_frontend_file("src/htr-v3.jsx")
    assert 'minHeight: "100%"' in v3, "Rule 11.7 — V3 root must use minHeight, not a hard height lock"
    assert 'height: "100vh"' not in v3, "Rule 11.7 — V3 must not viewport-lock content"


def test_v1_risk_sizing_is_advice_only_research_calc():
    """Rule 11.8 (new) — the position-sizing / risk calculator stays advice-only:
    deterministic research math, respects the Rule 12.5 concentration cap, and
    never an order. No imperative execution verbs against the system."""
    v1 = read_frontend_file("src/htr-v1.jsx")
    start = v1.find("function V1RiskSizing")
    assert start >= 0, "V1RiskSizing not found"
    end = v1.find("\nfunction ", start + 1)
    body = v1[start:end]
    assert "advice-only" in body, "Rule 11.8 — risk calc must carry advice-only framing"
    assert "Rule 12.5" in body, "Rule 11.8 — risk calc must honor the concentration cap"
    for verb in ("下单", "提交订单", "auto-trade", "place order", "submit order"):
        assert verb not in body, f"Rule 11.8/Rule 3 — execution verb '{verb}' in risk calc"


def test_user_state_persists_to_localstorage_keys():
    """Rule 11.3 — symbol selection + variant are local user-state."""
    src = "\n".join(all_frontend_source().values())
    assert 'localStorage.getItem("htr_symbol")' in src
    assert 'localStorage.setItem("htr_symbol"' in src
    assert "htr_variant" in src


def test_no_simulated_price_movement():
    """Rule 11.9.1 — the UI must NOT simulate price movement. useTickingPrice
    must return the real backend value statically (no setInterval / random walk)."""
    shared = read_frontend_file("src/htr-shared.jsx")
    start = shared.find("function useTickingPrice")
    assert start >= 0, "useTickingPrice not found"
    end = shared.find("\nfunction ", start + 1)
    body = shared[start:end]
    assert "setInterval" not in body, "Rule 11.9.1 — useTickingPrice must not run a timer (no simulated ticks)"
    assert "Math.random" not in body, "Rule 11.9.1 — useTickingPrice must not random-walk the price"
    assert "return base" in body, "Rule 11.9.1 — useTickingPrice must return the real backend value unchanged"


def test_calibration_figures_come_from_backend_not_hardcoded():
    """Rule 11.9.4 — calibration K-fold figures are governance-sensitive; they
    MUST be fetched from /api/calibration/reliability, never shipped as literals.
    The redesign initially hardcoded WRONG values (OOS 0.2668 / in-sample 0.1982
    vs real 0.2823 / 0.2427)."""
    src = "\n".join(all_frontend_source().values())
    assert "/api/calibration/reliability" in src, "calibration surfaces must fetch the real endpoint"
    for wrong in ("0.2668", "0.1982", "0.0061"):
        assert wrong not in src, f"hardcoded (wrong) calibration literal {wrong!r} still present"


def test_freshness_session_label_present():
    """Rule 11.9.2 — an honest freshness/session label derived from asof vs
    tradeDate must exist and be rendered on the default variant."""
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "function marketFreshness" in shared, "Rule 11.9.2 — marketFreshness helper missing"
    assert "tradeDate" in shared and "asof" in shared
    v3 = read_frontend_file("src/htr-v3.jsx")
    assert "marketFreshness(meta)" in v3, "Rule 11.9.2 — V3 must render the freshness/session label"


def test_manual_entry_modal_really_posts_preview_and_commit():
    """Rule 11.10 — the fill/cash modal must actually POST to its real endpoint
    with a preview(commit:false) -> commit(commit:true) round-trip, not a local
    stub. This is the regression guard for the 2026-05-31 dead-button bug."""
    modals = read_frontend_file("src/htr-v3-modals.jsx")
    for ep in ("/api/portfolio/fill", "/api/portfolio/cash_event"):
        assert ep in modals, f"manual-entry modal must reference {ep}"
    assert 'method: "POST"' in modals, "manual-entry modal must issue a real POST"
    assert re.search(r"commit\b", modals), "must send a commit flag (Rule 11.10.3)"
    assert "send(false)" in modals and "send(true)" in modals, (
        "preview(commit:false) -> commit(commit:true) two-step must hit the backend"
    )
    # the dead demo stub MUST be gone (the bug this rule was written for)
    assert "setPreview(true)" not in modals, "submit button must not be a setPreview-only stub"
    assert "预览 OK（演示）" not in modals and "真实环境将 POST" not in modals, (
        "demo placeholder text must be replaced by a real preview round-trip"
    )


# ── P13-01 Phase 1: content-honesty + dead-interaction sweep (Rule 11.9/11.10/8.3/9.4) ──
# These bind EVERY variant (the Rule 11.7 layout carve-out does NOT relieve content
# honesty). Each asserts a fabricated literal is GONE / real-data derivation is used.


def test_v4_event_counts_not_hardcoded_literals():
    """Rule 11.9.1/11.9.4 — V4 header event-count chips must be DERIVED from the
    real event stream (v4BuildEvents), never the hardcoded [7,6,8,3,0] literals."""
    v4 = read_frontend_file("src/htr-v4.jsx")
    for literal in ('"候选浮现", "6"', '"决策动作", "8"', '"宏观事件", "3"'):
        assert literal not in v4, f"Rule 11.9 — hardcoded V4 count literal {literal!r} still present"


def test_v4_no_fabricated_score_sparkline_or_delta():
    """Rule 11.9.1 / 9.4 — V4 must not render an invented score trajectory. No
    hardcoded rising sparkline array and no '研究分 (+12)' momentum delta (the score
    is an uncalibrated point signal with no time series in the payload)."""
    v4 = read_frontend_file("src/htr-v4.jsx")
    assert "[42, 45, 48, 52, 58, 62, 68, 72, 75, 78]" not in v4, "fabricated V4SpineHeader score spark"
    assert "[42, 48, 52, 58, 62, 68, 72, 75, 78]" not in v4, "fabricated V4LeaderCard score spark"
    assert "研究分 (+12)" not in v4, "Rule 9.4 — fabricated '(+12)' score-momentum delta"


def test_v4_no_dead_action_buttons_with_false_write_claim():
    """Rule 11.10.1 — the V4 research-mode action cluster must not be a dead button
    set whose caption falsely claims a decision-log write (decision-log is not a
    Rule 11.5 whitelisted path). The false write-claim caption must be gone."""
    v4 = read_frontend_file("src/htr-v4.jsx")
    assert "按钮仅写入" not in v4, "Rule 11.10 — false 'buttons write to decision log' claim still present"


def test_v3_news_high_count_derived_not_hardcoded():
    """Rule 11.9.4 — the V3 news 'N high' chip must be derived from items.weight,
    not the hardcoded '2 high' literal."""
    v3 = read_frontend_file("src/htr-v3.jsx")
    assert ">2 high<" not in v3 and '"htr-chip warn">2 high' not in v3, "hardcoded '2 high' news count"


def test_v1_action_zone_caveat_not_hardcoded():
    """Rule 11.9.1 — V1's risk note must not hardcode a per-symbol macro caveat
    (the '跌破 156' literal carried over from the 8035.T mock and showed for every
    candidate). It must derive from real risk data or render nothing."""
    v1 = read_frontend_file("src/htr-v1.jsx")
    assert "跌破 156" not in v1, "Rule 11.9 — hardcoded per-symbol macro caveat still present"


def test_v2_chart_uses_real_kline_overlay():
    """Rule 11.9.5 — V2's price chart must source the enriched real overlay
    (candidate.kline from /api/symbol/{T}/kline), not only the boot mock baseline
    prop. V1/V3 already use the real overlay; V2 was rendering the mock series."""
    v2 = read_frontend_file("src/htr-v2.jsx")
    start = v2.find("function V2ChartCard")
    assert start >= 0, "V2ChartCard not found"
    end = v2.find("\nfunction ", start + 1)
    body = v2[start:end]
    assert "candidate.kline" in body, "Rule 11.9.5 — V2ChartCard must use candidate.kline (real overlay)"


def test_screener_v2_placeholder_guarded_in_v1_v2():
    """Rule 11.9 — the internal placeholder theme 'screener_v2' must not be rendered
    raw as a user-facing chip/label; V1/V2 must guard it (as V3 already does)."""
    for rel in ("src/htr-v1.jsx", "src/htr-v2.jsx"):
        text = read_frontend_file(rel)
        assert "screener_v2" in text, (
            f"Rule 11.9 — {rel} must reference 'screener_v2' to suppress the raw placeholder"
        )


def test_v2_masthead_no_fabricated_issue_number():
    """Rule 11.9 — the V2 masthead must not print a static 'vol. 4 issue 113'
    implying a publication cadence that does not exist."""
    v2 = read_frontend_file("src/htr-v2.jsx")
    assert "vol. 4 issue 113" not in v2, "fabricated static issue number in V2 masthead"


# ── P13-02 Phase 2: async-state honesty + write-path reachability ──


def test_enrichment_tracks_per_endpoint_status():
    """Rule 11.9.1/11.9.5 — useEnrichedCandidate must expose per-endpoint arrival
    status (pending|ok|failed) so a failed overlay fetch is not silently the mock.
    A null fetch must mark 'failed', a success must mark 'ok'."""
    api = read_frontend_file("src/htr-api.jsx")
    assert "_status" in api, "useEnrichedCandidate must return a _status map"
    assert '"failed"' in api or "'failed'" in api, "must mark failed on a missed fetch"
    assert '"pending"' in api or "'pending'" in api, "must mark pending before arrival"


def test_async_bodies_render_honest_loading_and_failure_state():
    """Rule 11.9.4 — the per-symbol detail bodies must drive loading / 示例占位 from
    the real _status, never leave the offline mock indistinguishable from real."""
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "function AsyncBodyState" in shared, "shared AsyncBodyState helper missing"
    assert "生成中" in shared and "示例占位" in shared, "loading + 示例占位 states must exist"
    v3 = read_frontend_file("src/htr-v3.jsx")
    for body in ("function FactorBody", "function OutcomesBody", "function AiBody", "function DebateBody"):
        start = v3.find(body)
        assert start >= 0, f"{body} not found"
        end = v3.find("\nfunction ", start + 1)
        chunk = v3[start:end]
        assert "_status" in chunk and "AsyncBodyState" in chunk, (
            f"{body} must read _status and render AsyncBodyState (honest async state)"
        )


def test_watchlist_add_remove_really_posts_to_server():
    """Rule 11.10.4 — watchlist add/remove must issue real POSTs to the whitelisted
    /api/watchlist/{add,remove} (user-allowed existing POST), reconciling to the
    server payload, not a localStorage-only stub. The hook also hydrates from GET."""
    shared = read_frontend_file("src/htr-shared.jsx")
    start = shared.find("function useWatchlist")
    assert start >= 0, "useWatchlist not found"
    end = shared.find("\nfunction ", start + 1)
    body = shared[start:end]
    assert "/api/watchlist/" in body, "useWatchlist must call the server watchlist endpoint"
    assert 'method: "POST"' in body, "add/remove must POST, not localStorage-only"
    assert 'post("add"' in body and 'post("remove"' in body, "both add and remove must POST"
    assert 'fetch("/api/watchlist"' in body, "must hydrate from GET /api/watchlist (server = source of truth)"
    # the old localStorage-only honesty label must no longer claim local-only
    modals = read_frontend_file("src/htr-v3-modals.jsx")
    assert "仅存 localStorage" not in modals, "stale 'localStorage-only' label must be updated (now server-backed)"


# ── P13-03 Phase 3: shared-component unification (governance-sensitive) ──


def test_calibration_score_wording_single_sourced():
    """P3-C / Rule 8.3-9.4 — the uncalibrated-score qualifier is single-sourced via
    HTR.LABELS so the no-probability wording cannot drift; V3 uses the shared
    CalibPill (not an inline duplicate) and references HTR.LABELS for score labels."""
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "HTR_LABELS" in shared and "scoreUncalibrated" in shared, "canonical HTR.LABELS missing"
    v3 = read_frontend_file("src/htr-v3.jsx")
    assert "<CalibPill" in v3, "V3 must use the shared CalibPill, not an inline calib pill"
    assert "HTR.LABELS" in v3, "V3 score labels must reference the canonical HTR.LABELS"
    v4 = read_frontend_file("src/htr-v4.jsx")
    assert "HTR.LABELS" in v4, "V4 score labels must reference the canonical HTR.LABELS"


def test_heat_hot_distinct_from_loss_red():
    """P3-F — an overheated market temperature (>=70) must use a dedicated heat ramp
    (--htr-heat-hot), never share the loss/stop red (--htr-bear)."""
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "--htr-heat-hot" in shared, "dedicated --htr-heat-hot token missing"
    start = shared.find("function heatColor")
    end = shared.find("\n", start + 1)
    line = shared[start:end]
    assert "--htr-heat-hot" in line and "--htr-bear" not in line, (
        "heatColor(>=70) must map to --htr-heat-hot, not --htr-bear"
    )


def test_v3_outcomes_no_low_scroll_trap():
    """P4-C / Rule 11.7.3 — the V3 outcomes table (bounded primary decision evidence)
    must not be locked behind a tiny inner scroll; the re-opened maxHeight:180 trap
    is removed (a high safety cap is allowed)."""
    v3 = read_frontend_file("src/htr-v3.jsx")
    assert "maxHeight: 180" not in v3, "Rule 11.7.3 — outcomes maxHeight:180 scroll-trap must be removed"


def test_focus_visible_ring_present():
    """P5-A / WCAG 2.4.7 — keyboard focus must be visible on interactive elements;
    a :focus-visible accent ring exists and is not globally suppressed."""
    shared = read_frontend_file("src/htr-shared.jsx")
    assert ":focus-visible" in shared, "P5-A — :focus-visible ring missing"
    assert "[role=\"button\"]:focus-visible" in shared or "[tabindex=\"0\"]:focus-visible" in shared, (
        "focus ring must cover keyboard-focusable div-buttons"
    )


def test_kline_failed_status_surfaced_not_silent_mock():
    """Codex-2026-06-06 / Rule 11.9.4 — when the real /kline overlay misses (empty
    or failed), the chart must surface a 示例 marker instead of showing the offline
    mock series silently. The enrichment marks empty bars as a miss (validator),
    and the chart panels read _status.kline."""
    api = read_frontend_file("src/htr-api.jsx")
    # kline validator must reject empty bars (so empty → failed, not ok)
    assert "bars.length > 0" in api, "kline must be marked failed when bars are empty"
    # every per-endpoint load now has a shape validator (no empty-payload masquerade)
    assert api.count("valid") >= 3, "load() must validate payload shape before overlaying"
    v3 = read_frontend_file("src/htr-v3.jsx")
    assert "_status.kline" in v3 and "示例K线" in v3, "V3 chart must surface a 示例 marker on kline miss"


def test_proposal_inbox_rule_13_18_discipline():
    """Rule 13.18 — L6 accept/reject surface discipline (Codex-flagged 2026-06-06):
    (1) full Rule 13.6 metadata is NEVER behind a collapse; (2) shadow ack is a
    visible checkbox gating accept, not window.confirm; (4) an expired proposal
    disables accept by default behind an override toggle; (5) no batch ops."""
    modals = read_frontend_file("src/htr-v3-modals.jsx")
    # 13.18.1 — metadata always visible (the collapsed "技术细节" expander is gone)
    assert "技术细节" not in modals, "13.18.1 — Rule 13.6 metadata must not be collapsed"
    assert "完整元数据" in modals, "13.18.1 — full metadata block must render"
    # 13.18.2 — no window.confirm; checkbox + user_confirm_shadow gating
    assert "window.confirm" not in modals, "13.18.2 — shadow ack must be a checkbox, not window.confirm"
    assert "user_confirm_shadow" in modals and 'type="checkbox"' in modals
    assert "shadowAck" in modals, "13.18.2 — accept must gate on the shadow checkbox state"
    # 13.18.4 — expired accept disabled by default + explicit override
    assert "is_expired_by_age" in modals, "13.18.4 — must read the real expiry field"
    assert "overrideExp" in modals and "acceptDisabled" in modals, "13.18.4 — expired accept gated behind override"
    # 13.18.5 — no batch operations
    assert "全部接受" not in modals and "accept_all" not in modals


def test_watchlist_shared_store_syncs_instances():
    """Codex-2026-06-06 #5 — useWatchlist must use a module-level shared store so a
    mutation from one instance (leader ☆) re-renders the others (nav chip), not
    per-instance local state that goes stale until remount."""
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "HTR_WL" in shared and "subscribe" in shared, "shared watchlist store missing"
    assert "setLocal" in shared and "listeners" in shared, "store must notify subscribers"
    # add/remove must still POST (Rule 11.10) and reconcile via the shared store
    assert 'post("add"' in shared and 'post("remove"' in shared


def test_v4_surfaces_real_decision_log_events():
    """Codex-2026-06-06 #4 — v4BuildEvents must map the REAL decision-log actions
    (candidate_persisted aggregated, news_overlay_hit), not silently drop them."""
    v4 = read_frontend_file("src/htr-v4.jsx")
    assert "candidate_persisted" in v4, "V4 must surface candidate_persisted events"
    assert "news_overlay_hit" in v4, "V4 must surface news_overlay_hit events"
    assert "写入决策日志" in v4, "candidate_persisted aggregate event must render"


def test_modal_accessible_dialog():
    """P5-C / WCAG 4.1.2 / 2.4.3 — ModalShell is an accessible dialog: role+aria-modal,
    aria-labelledby on the title, focus restore, and an accessible close-button name."""
    modals = read_frontend_file("src/htr-v3-modals.jsx")
    start = modals.find("function ModalShell")
    end = modals.find("\nfunction ", start + 1)
    body = modals[start:end]
    assert 'role="dialog"' in body and 'aria-modal="true"' in body, "modal must be a labelled dialog"
    assert "aria-labelledby" in body and "useId" in body, "title must be programmatically associated"
    assert 'aria-label="关闭"' in body, "close button needs an accessible name"
    assert "prev.focus" in body or "restore" in body.lower(), "focus must restore on close"


def test_no_hardcoded_white_svg_fills():
    """P5-D — accent-filled SVG/badge text must use --htr-accent-ink / theme tokens,
    not a hardcoded #fff/white that breaks dark-mode + custom-accent themes."""
    for rel in ("src/htr-shared.jsx", "src/htr-shared2.jsx", "src/htr-v3.jsx"):
        text = read_frontend_file(rel)
        assert 'fill="#fff"' not in text and 'fill="white"' not in text, f"{rel} has a hardcoded white SVG fill"
        assert 'color: "white"' not in text and 'color:"white"' not in text, f"{rel} has a hardcoded white color"


def test_notifier_toggle_really_posts_and_reads_state():
    """Rule 11.10 + Rule 12.7 — the notifier toggle must POST the real endpoint
    (which writes the audit log + enforces the stage-2 gate) and read real state,
    with a dry-run path. The old local-state-only flip is forbidden."""
    modals = read_frontend_file("src/htr-v3-modals.jsx")
    assert "/api/notifier/toggle" in modals, "notifier toggle must POST the real endpoint"
    assert "/api/notifier/state" in modals, "notifier modal must read real channel state"
    assert "dry_run" in modals, "Rule 12.7.5 — dry-run capability must be wired"
    # the old local-only stub (flip channel state with no fetch) is gone
    assert "setChannels({ ...channels, [ch]: true })" not in modals, (
        "notifier enable must call the backend, not flip local state"
    )


def _cohort_card_body() -> str:
    v3 = read_frontend_file("src/htr-v3.jsx")
    start = v3.find("function V3CandidateHistoryCard")
    assert start >= 0, "V3CandidateHistoryCard (Rule 11.11) not found"
    end = v3.find("\nwindow.V3MarketDashboard", start)
    return v3[start:(end if end >= 0 else len(v3))]


def test_cohort_review_card_is_read_only_and_honest():
    """Rule 11.11 — the historical-candidate review card reads the decision log,
    renders the standing disclosure, walls off backdated rows, presents the
    cohort aggregate before per-candidate detail, and never writes."""
    body = _cohort_card_body()
    # 11.11.7 read-only — reads the history endpoints, issues no POST/write.
    assert "/api/candidates/history" in body, "card must read /api/candidates/history"
    assert '"POST"' not in body and "'POST'" not in body, (
        "Rule 11.11.7 — cohort review card must be read-only (no POST)"
    )
    # 11.11.6 — the standing uncalibrated/no-win-rate disclosure is rendered.
    assert "honestyNote" in body, "Rule 11.11.6 — standing disclosure must render"
    # 11.11.2 cohort-first — the whole-cohort aggregate table precedes per-candidate detail.
    agg = body.find("整批均涨")
    detail = body.find("个股明细")
    assert agg >= 0 and detail >= 0
    assert agg < detail, "Rule 11.11.2 — cohort aggregate must come before per-candidate detail"
    # 11.11.5 — backdated bootstrap is opt-in and labeled synthetic.
    assert "include_backdated" in body, "Rule 11.11.5 — backdated must be an explicit toggle"
    assert "合成样本" in body, "Rule 11.11.5 — backdated region must be labeled synthetic"
    # P14-03 full-series view + no win-rate labeling on the pooled trend.
    assert "全实盘累计" in body and "非胜率" in body


def test_cohort_review_card_present_in_all_four_variants():
    """Rule 11.7 / P14-02 — the governed cohort card binds every variant."""
    for v in ("htr-v1.jsx", "htr-v2.jsx", "htr-v3.jsx", "htr-v4.jsx"):
        src = read_frontend_file("src/" + v)
        assert "V3CandidateHistoryCard" in src, f"{v} missing cohort review card (four-variant parity)"


def _catalyst_badge_body() -> str:
    """Source of the shared CatalystBadges component (ADR-0009 P15 / Rule 11.12)."""
    shared = read_frontend_file("src/htr-shared.jsx")
    start = shared.find("function CatalystBadges")
    assert start >= 0, "CatalystBadges component missing from htr-shared.jsx"
    # bound the body at the next top-level component / the window export
    end = shared.find("Object.assign(window", start)
    return shared[start:end if end > start else len(shared)]


def test_catalyst_leader_badge_is_ordering_signal_not_winrate():
    """Rule 11.12 — the news-catalyst / theme-leader badge surfaces the hybrid-rerank
    ORDERING (today's news theme-heat), never a win-rate. It is derived only from the
    served candidate fields (PIT) and carries the standing order≠score disclosure."""
    body = _catalyst_badge_body()
    # Derived only from served fields — no probability/calibrated number invented.
    assert "newsCatalyzed" in body, "badge must gate on the served newsCatalyzed flag"
    assert "isThemeLeader" in body, "leader designation must read the served isThemeLeader flag"
    # Standing honesty disclosure: ordering signal, not win-rate; score stays uncalibrated.
    assert "CATALYST_BADGE_NOTE" in body, "badge must carry the standing disclosure note"
    note_decl = read_frontend_file("src/htr-shared.jsx")
    assert "排序信号" in note_decl and "非胜率" in note_decl, (
        "Rule 11.12 — disclosure must state the badge is an ordering signal, not a win rate"
    )
    assert "未校准研究分" in note_decl, "Rule 11.12 — disclosure must state the score stays uncalibrated"
    # No probability/calibrated-probability/win-rate claim is presented as a positive number.
    assert "calibrated_probability" not in body and "胜率" not in body.replace("非胜率", ""), (
        "Rule 11.12 — badge must not present a win-rate/probability claim"
    )


def test_catalyst_leader_badge_present_in_all_four_variants():
    """Rule 11.7 / Rule 11.12 — the governed catalyst/leader badge binds every variant."""
    for v in ("htr-v1.jsx", "htr-v2.jsx", "htr-v3.jsx", "htr-v4.jsx"):
        src = read_frontend_file("src/" + v)
        assert "CatalystBadges" in src, f"{v} missing catalyst/leader badge (four-variant parity)"


def _event_desk_card_body() -> str:
    """Source of the shared EventDeskCard component (P16-E4 / Rule 11.13)."""
    v3 = read_frontend_file("src/htr-v3.jsx")
    start = v3.find("function EventDeskCard(")
    assert start >= 0, "EventDeskCard missing from htr-v3.jsx"
    end = v3.find("function V3CandidateHistoryCard(", start)
    return v3[start:end if end > start else len(v3)]


def test_event_desk_card_is_read_only_and_not_a_prediction():
    """Rule 11.13 — the Event Desk surfaces exposure + priced-in read; it reads the
    read-only endpoint, never writes, and never presents an event-outcome probability
    or win-rate. The freshness label is descriptive, with a standing disclosure."""
    body = _event_desk_card_body()
    # 11.13.5 read-only — reads /api/event-desk, issues no POST/write.
    assert "/api/event-desk" in body, "card must read /api/event-desk"
    assert '"POST"' not in body and "'POST'" not in body and "method:" not in body, (
        "Rule 11.13.5 — Event Desk must be read-only (no POST/write)"
    )
    # 11.13.1/.2 — surfaces the standing not-a-prediction disclosure from the payload,
    # and never labels a probability / win-rate.
    assert "data.disclosure" in body, "Rule 11.13.1 — standing disclosure must render"
    assert "胜率" not in body and "概率" not in body and "probability" not in body.lower(), (
        "Rule 11.13.1 — Event Desk must not present a win-rate/probability"
    )
    # 11.13.2 — descriptive freshness labels (how far already moved), not buy/sell calls.
    assert "已大涨" in body and "回吐中" in body, "freshness labels must be descriptive"


def test_event_desk_card_present_in_all_four_variants():
    """Rule 11.7 / Rule 11.13.5 — the Event Desk binds every variant."""
    for v in ("htr-v1.jsx", "htr-v2.jsx", "htr-v3.jsx", "htr-v4.jsx"):
        src = read_frontend_file("src/" + v)
        assert "EventDeskCard" in src, f"{v} missing Event Desk card (four-variant parity)"


def test_skhy_catalyst_indicator_present_and_neutral():
    """P20 / Rule 11.15 — the frontend renders a read-only SKHY ADR catalyst
    indicator on candidate badges, gated on the served `skhyCatalystActive` field,
    with neutral research wording only (no buy/edge/win-rate/expected-return)."""
    src = read_frontend_file("src/htr-shared.jsx")
    # the read-only SKHY annotation fields the API now exposes must be referenced
    assert "skhyCatalystActive" in src, "SKHY indicator must gate on skhyCatalystActive"
    assert "skhyCatalystStatus" in src
    assert "semiSympathyReasons" in src
    assert "relativeStrengthVsSkhy" in src
    assert "SKHY" in src
    # gated → renders nothing when off/stale/pending/unavailable (fail-open display)
    assert "c.skhyCatalystActive &&" in src
    # positively assert the RENDERED title carries a neutral research-only disclaimer
    assert "研究用" in src and "非买入建议" in src and "不预测收益" in src, \
        "SKHY badge must carry a neutral research disclaimer, not buy advice"
    # the RENDERED title (not code comments) must show no buy-now CTA / probability value
    ts = src.find("SK hynix ADR (SKHY)")
    assert ts >= 0
    rendered = src[ts:ts + 400].lower()
    for bad in ("buy now", "guaranteed", "expected return", "%概率", "%胜率", "概率为", "胜率为"):
        assert bad not in rendered, f"forbidden rendered SKHY wording: {bad!r}"
    # whole-frontend probability/win-rate governance is covered by
    # test_no_probability_or_win_rate_claim_leaks (already passing, incl. this badge).


def test_skhy_indicator_does_not_reorder_candidates():
    """Rule 11.15 — the SKHY overlay is display-only; the frontend must NOT sort/
    reorder candidates by any SKHY/sympathy field."""
    for name, text in all_frontend_source().items():
        assert not re.search(
            r"sort[^;\n]*(?:semiSympathy|skhyCatalyst|relativeStrengthVsSkhy)", text
        ), f"{name} must not reorder candidates by a SKHY/sympathy field"


# ---------------------------------------------------------------------------
# P21-05 / Rule 11.16 — Daily Action Board frontend contracts
# ---------------------------------------------------------------------------


def test_action_board_card_present_in_all_four_variants():
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "function ActionBoardCard" in shared, "shared ActionBoardCard missing"
    for v in ("src/htr-v1.jsx", "src/htr-v2.jsx", "src/htr-v3.jsx", "src/htr-v4.jsx"):
        assert "<ActionBoardCard />" in read_frontend_file(v), (
            f"{v} missing ActionBoardCard (Rule 11.7 parity)"
        )


def test_action_board_is_plan_not_prediction_and_fails_open():
    shared = read_frontend_file("src/htr-shared.jsx")
    # Rule 11.16.1 — standing disclosure rendered from the backend field,
    # in a non-collapsible header block, plus the 计划≠预测 framing.
    assert "board.disclosure" in shared
    assert "计划≠预测" in shared
    # Fail-open: renders nothing when the backend omits the board.
    assert "if (!board || !Array.isArray(board.rows) || !board.rows.length) return null" in shared
    # Rule 11.6.5 — no imperative-buy recommendation vocabulary.
    assert "建议买入" not in shared
    # Rule 11.16.5 — sizing labeled arithmetic; never fabricated without NAV.
    assert "算术" in shared
    assert "不虚构" in shared


# ---------------------------------------------------------------------------
# P22-01 / Rule 11.17 — Position Exit Discipline Board frontend contracts
# ---------------------------------------------------------------------------


def test_exit_board_card_present_in_all_four_variants():
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "function ExitBoardCard" in shared, "shared ExitBoardCard missing"
    for v in ("src/htr-v1.jsx", "src/htr-v2.jsx", "src/htr-v3.jsx", "src/htr-v4.jsx"):
        assert "<ExitBoardCard />" in read_frontend_file(v), (
            f"{v} missing ExitBoardCard (Rule 11.7 parity)"
        )


def test_exit_board_is_operator_params_not_prediction():
    shared = read_frontend_file("src/htr-shared.jsx")
    # Rule 11.17.1 — parameters shown as the operator's own declared discipline.
    assert "你声明的" in shared
    assert "board.disclosure" in shared
    # Rule 11.17.2 — breached stop is a first-class surfaced state.
    assert "stop_reference_breached" in shared
    assert "止损参考已破" in shared
    # Fail-open when backend omits the board.
    assert (
        "if (!board || !Array.isArray(board.rows) || !board.rows.length) return null"
        in shared
    )
    # Rule 11.17.3 — no imperative sell vocabulary; rotate x-ref is a fact count.
    assert "建议卖出" not in shared
    assert "非换仓指令" in shared


# ---------------------------------------------------------------------------
# P24 / Rule 11.7.7-8 — design-review remediation (mobile priority, no void,
# no S株 occlusion). Structure asserted on source since this is zero-build.
# ---------------------------------------------------------------------------


def test_v3_bento_interlocks_and_redistributes(_="P24-07"):
    """Rule 11.7.8 — the command bento interlocks the deep-dive column with the
    positions stack (matched height, no void); the wide Action Board and the four
    feed cards stay full-width so no column towers."""
    v3 = read_frontend_file("src/htr-v3.jsx")
    for cls in ('className="v3-hero-bento"', 'className="v3-hero-main"',
                'className="v3-hero-side"', 'className="v3-plan-row"',
                'className="v3-work-bento"', 'className="v3-feeds-row"'):
        assert cls in v3, f"missing bento container: {cls}"
    # command bento: deep-dive column = leader/chart/strategy
    hero_main = v3.split('v3-hero-main"')[1].split('v3-hero-side')[0]
    for card in ("V3LeaderCard", "V3KLinePanel", "V3StrategyCard"):
        assert card in hero_main, f"{card} must be in the hero-main column"
    # positions stack = portfolio + exit
    hero_side = v3.split('v3-hero-side"')[1].split("v3-plan-row")[0]
    for card in ("V3PortfolioCard", "ExitBoardCard"):
        assert card in hero_side, f"{card} must be in the positions stack"
    # Action Board full-width, feeds full-width
    plan_row = v3.split('v3-plan-row"')[1].split("v3-work-bento")[0]
    assert "ActionBoardCard" in plan_row
    feeds_row = v3.split('className="v3-feeds-row"')[1]
    for card in ("V3CandidateHistoryCard", "EventDeskCard", "V3NewsCard", "V3FeedsTabs"):
        assert card in feeds_row, f"{card} must be in the feeds row"


def test_v3_mobile_priority_order_css(_="P24-07"):
    """Rule 11.7.7 — on mobile the decision surfaces (action board + positions
    stack) are ordered before the chart and the workbench."""
    css = read_frontend_file("index.html")
    for rule in (".v3-plan-row    { order:1; }", ".v3-hero-bento  { order:2; }",
                 ".v3-work-bento  { order:3; }", ".v3-feeds-row   { order:4; }",
                 ".v3-hero-side   { order:1; }", ".v3-hero-main   { order:2; }"):
        assert rule in css, f"missing mobile-order rule: {rule}"


def test_skabu_card_not_fixed_overlay():
    """Rule 11.7.2 — the S株 card is docked in flow, no longer a fixed overlay
    occluding the right rail."""
    html = read_frontend_file("index.html")
    root_line = [l for l in html.splitlines() if 'id="skabu-root"' in l][0]
    assert "position:fixed" not in root_line
    assert "z-index:9999" not in root_line


# ── P25-04 / Section 17: Risk Mandate card contracts ─────────────────────

def test_risk_mandate_card_exists_and_mounted_on_all_four_variants():
    """Section 17 — the owner risk-mandate panel is a shared governed card with
    four-variant parity (Rule 11.7), same as the exit board."""
    shared = read_frontend_file("src/htr-shared.jsx")
    assert "function RiskMandateCard()" in shared
    assert "RiskMandateCard," in shared  # exported on window
    for variant in ("src/htr-v1.jsx", "src/htr-v2.jsx", "src/htr-v3.jsx", "src/htr-v4.jsx"):
        assert "<RiskMandateCard />" in read_frontend_file(variant), f"missing mount in {variant}"


def test_risk_mandate_card_honesty_redlines():
    """Rule 17.6 — the card renders the backend disclosure verbatim, labels the
    panel as the owner's own declaration (not a prediction), and fails open."""
    shared = read_frontend_file("src/htr-shared.jsx")
    card = shared[shared.index("function RiskMandateCard()"):]
    assert "board.disclosure" in card              # standing disclosure rendered
    assert "不是预测" in card                       # owner-declaration framing
    assert "非指令" in card                         # deployment gap is not an order
    assert "return null" in card                   # fail-open when backend omits
