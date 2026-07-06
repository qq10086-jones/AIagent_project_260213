// htr-data.js — offline-complete mock data for the V3 redesign.
// Builds on the original data.js baseline, then ENRICHES it so the redesigned
// dashboard renders fully without a backend: portfolio, daily cockpit, and
// per-symbol strategy / factor / outcome / AI-brief / debate synthesis.
//
// Governance note: every synthesized field keeps the red-line labels intact —
// scores are uncalibrated research scores (Rule 9.4), no probability/% language
// in narrative briefs (Rule 8.3.1), strategy banners carry the literal Rule 3
// disclaimer (Rule 11.6). Nothing here implies execution.

// ── seeded deterministic RNG so the mock is stable across reloads ──
function mkRng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; };
}

// Per-symbol candle series ending at the candidate's current price.
function klineFor(symbol, endPrice, sessions = 90) {
  const rng = mkRng([...symbol].reduce((a, c) => a + c.charCodeAt(0), 7));
  const out = [];
  let close = endPrice * (0.62 + rng() * 0.12); // start ~62-74% of current
  for (let i = 0; i < sessions; i++) {
    const trend = (i / sessions) * 0.012;        // gentle uptrend toward today
    const drift = (rng() - 0.46 + trend) * endPrice * 0.022;
    const open = close;
    const next = Math.max(endPrice * 0.4, open + drift);
    const high = Math.max(open, next) + rng() * endPrice * 0.012;
    const low = Math.min(open, next) - rng() * endPrice * 0.012;
    const vol = Math.round(600000 + rng() * 2200000);
    out.push({ open, high, low, close: next, volume: vol });
    close = next;
  }
  // pin the final close to the real current price
  const scale = endPrice / out[out.length - 1].close;
  return out.map((c, i) => ({
    date: `D-${sessions - 1 - i}`,
    open: c.open * scale, high: c.high * scale, low: c.low * scale,
    close: c.close * scale, volume: c.volume,
  }));
}

const BASE = {
  meta: {
    asof: "2026-05-24 10:32:14 JST",
    tradeDate: "2026-05-24",
    refreshLabel: "120s",
    eventTrigger: false,
    calibration: { level: "warning", text: "未校准研究分 · 不是真实胜率", sample: 184, brier: 0.241 },
  },

  gates: [
    { id: "G1", label: "决策日志", status: "done", glyph: "✓", taskId: "T-101", next: "已上线" },
    { id: "G2", label: "反馈采集", status: "done", glyph: "✓", taskId: "T-104", next: "已上线" },
    { id: "G3", label: "分数校准", status: "in_progress", glyph: "●", taskId: "T-108", next: "+12d 样本" },
    { id: "G4", label: "提醒推送", status: "pending", glyph: "○", taskId: "T-112", next: "依赖 G3" },
    { id: "G5", label: "纸面交易", status: "pending", glyph: "○", taskId: "T-118", next: "依赖 G4" },
    { id: "G6", label: "人工批准", status: "pending", glyph: "○", taskId: "T-123", next: "依赖 G5" },
    { id: "G7", label: "半自动下单", status: "blocked", glyph: "—", taskId: "T-130", next: "需治理评审" },
    { id: "G8", label: "全自动", status: "blocked", glyph: "—", taskId: "T-140", next: "需治理评审" },
  ],

  markets: [
    { id: "N225", label: "日经225", sub: "Nikkei 225", price: 38420.18, chg: 1.24, temp: 72, region: "JP", state: "OPEN", spark: [3,4,5,4,6,7,6,8,9,8,9,11,10,12,11,12] },
    { id: "TOPIX", label: "TOPIX", sub: "Tokyo Price Idx", price: 2712.40, chg: 0.83, temp: 64, region: "JP", state: "OPEN", spark: [3,4,4,5,5,6,5,6,7,7,8,8,9,9,10,10] },
    { id: "SOX", label: "SOX 半导体", sub: "PHLX Semi", price: 5210.92, chg: 1.81, temp: 81, region: "US", state: "CLOSED", spark: [2,3,5,4,6,7,9,8,10,11,12,11,13,14,13,15] },
    { id: "SPX", label: "S&P 500", sub: "S&P 500", price: 5821.30, chg: 0.42, temp: 58, region: "US", state: "CLOSED", spark: [5,5,6,5,6,6,7,7,7,8,7,8,8,8,9,9] },
    { id: "USDJPY", label: "USD/JPY", sub: "美元兑日元", price: 156.18, chg: -0.15, temp: 46, region: "FX", state: "LIVE", spark: [10,9,9,8,8,9,8,7,7,8,7,7,6,7,6,6] },
    { id: "SSE", label: "上证", sub: "Shanghai Comp", price: 3420.55, chg: -0.31, temp: 38, region: "CN", state: "OPEN", spark: [8,7,8,7,6,7,6,7,6,5,6,5,6,5,4,5] },
  ],

  themes: [
    { id: "semi", label: "半导体设备", leaders: ["8035.T","6920.T","6857.T"], heat: 86, mom: 1.8 },
    { id: "ai", label: "AI 算力", leaders: ["9984.T","6532.T"], heat: 71, mom: 0.6 },
    { id: "auto", label: "汽车出口", leaders: ["7203.T","7267.T"], heat: 52, mom: -0.2 },
    { id: "bank", label: "金融升息", leaders: ["8306.T","8316.T"], heat: 44, mom: 0.1 },
    { id: "defense", label: "防卫", leaders: ["7011.T","7012.T"], heat: 38, mom: 0.4 },
    { id: "energy", label: "能源贸商", leaders: ["8058.T","8001.T"], heat: 31, mom: -0.5 },
  ],

  candidates: [
    {
      rank: 1, symbol: "8035.T", nameJa: "東京エレクトロン", nameCn: "东京电子",
      theme: "半导体设备", themeId: "semi", priority: "重点关注",
      score: 78, scoreStatus: "uncalibrated_research_score",
      price: 30420, chg: 2.14,
      one_liner: "SOX 隔夜 +1.8%，叠加 ASML 上修 25Q3 指引；开盘后 30 分钟若守住 30,180，进入 ladder 均衡介入区间。",
      reason: "外部温度：SOX +1.81%、费城半导体三日累涨 4.6%。主题强度：半导体设备板块今日热力 86（六板块第一），龙头集中度高。个股：5 日动量 +6.2%，量价配合，催化新闻命中 2 条。",
      risk: "未校准研究分。USD/JPY 跌破 156 会拖累出口估值。若开盘 15 分钟未站稳 30,300，激进档作废。",
      dataQuality: "数据：yfinance 延迟 ≤15s；新闻 overlay 命中 2 条；available_ts 校验通过。",
      ladder: [
        { kind: "exit_stretch", label: "延伸卖出", price: 32246, pct: 6.0 },
        { kind: "exit_2", label: "卖出 2", price: 31636, pct: 4.0 },
        { kind: "exit_1", label: "卖出 1", price: 31180, pct: 2.5 },
        { kind: "entry_aggressive", label: "买入 · 激进", price: 30176, pct: -0.8 },
        { kind: "entry_balanced", label: "买入 · 均衡", price: 29872, pct: -1.8 },
        { kind: "entry_conservative", label: "买入 · 保守", price: 29568, pct: -2.8 },
        { kind: "stop", label: "止损", price: 29051, pct: -4.5 },
      ],
      decisionCutoff: "10:30 JST",
      kfold: { status: "downgraded", reason: "OOS Brier (0.268) ≥ random baseline 0.25，自动降级", oos: 0.268, inSample: 0.198 },
      catalysts: [
        { ts: "前一日 22:14 ET", src: "Bloomberg", text: "ASML 上修 25Q3 出货指引 +8%", weight: 0.42 },
        { ts: "今日 08:55 JST", src: "日経", text: "経产省补贴半导体厂房第二期", weight: 0.31 },
      ],
    },
    {
      rank: 2, symbol: "6920.T", nameJa: "レーザーテック", nameCn: "Lasertec", theme: "半导体设备", themeId: "semi",
      priority: "重点关注", score: 71, scoreStatus: "uncalibrated_research_score",
      price: 18260, chg: 1.62,
      one_liner: "EUV 检测设备龙头，跟随 8035.T 节奏；作为半导体主题的次配进入观察。",
      reason: "主题强度共振；个股 RS 排名行业 #2；外资连续 3 日净买入。",
      risk: "波动率高，单日±5% 常见；流动性次于 8035.T。",
      dataQuality: "数据完整；新闻命中 1 条。",
      decisionCutoff: "10:30 JST",
      kfold: { status: "downgraded", reason: "OOS Brier ≥ baseline，自动降级", oos: 0.271, inSample: 0.205 },
      catalysts: [{ ts: "前一日 22:14 ET", src: "Bloomberg", text: "ASML 上修 25Q3 指引", weight: 0.28 }],
    },
    {
      rank: 3, symbol: "7203.T", nameJa: "トヨタ自動車", nameCn: "丰田汽车", theme: "汽车出口", themeId: "auto",
      priority: "观察", score: 58, scoreStatus: "uncalibrated_research_score",
      price: 2980.5, chg: 0.34,
      one_liner: "USD/JPY 走弱压制估值；今日仅做主题平衡，留在池里跟踪。",
      reason: "防御性配置；龙头流动性极佳。", risk: "汇率敏感；主题热力下行。", dataQuality: "数据完整。",
      decisionCutoff: "10:30 JST", kfold: { status: "no_evidence", reason: "样本不足，无 OOS 证据", oos: null, inSample: null }, catalysts: [],
    },
    {
      rank: 4, symbol: "9984.T", nameJa: "ソフトバンクＧ", nameCn: "软银集团", theme: "AI 算力", themeId: "ai",
      priority: "观察", score: 55, scoreStatus: "uncalibrated_research_score",
      price: 9412, chg: -0.81,
      one_liner: "Arm 财报临近，今日仅观察，留在池里跟踪事件。",
      reason: "AI 主题次热；事件驱动。", risk: "Arm 财报隔夜大波动可能性高。", dataQuality: "数据完整。",
      decisionCutoff: "10:30 JST", kfold: { status: "no_evidence", reason: "样本不足", oos: null, inSample: null }, catalysts: [],
    },
    {
      rank: 5, symbol: "8306.T", nameJa: "三菱ＵＦＪ", nameCn: "三菱UFJ", theme: "金融升息", themeId: "bank",
      priority: "观察", score: 47, scoreStatus: "uncalibrated_research_score",
      price: 1812.0, chg: 0.16,
      one_liner: "无明确催化；保留在池里跟踪日银 6 月会议。",
      reason: "升息主题低热；待催化。", risk: "若日银鸽派则单日 -2~3%。", dataQuality: "数据完整。",
      decisionCutoff: "10:30 JST", kfold: { status: "no_evidence", reason: "样本不足", oos: null, inSample: null }, catalysts: [],
    },
    {
      rank: 6, symbol: "6857.T", nameJa: "アドバンテスト", nameCn: "爱德万", theme: "半导体设备", themeId: "semi",
      priority: "观察", score: 44, scoreStatus: "uncalibrated_research_score",
      price: 7820, chg: 1.05,
      one_liner: "主题受益但 RS 弱于前二名，作为后备留在池里。",
      reason: "测试设备龙头；外资态度中性。", risk: "估值偏高。", dataQuality: "数据完整。",
      decisionCutoff: "10:30 JST", kfold: { status: "no_evidence", reason: "样本不足", oos: null, inSample: null }, catalysts: [],
    },
  ],

  newsTimeline: [
    { ts: "10:28 JST", src: "日経", weight: "high", text: "経産省、半導体製造装置の補助金第2期を 24 日午後発表へ", linkedSymbols: ["8035.T","6920.T","6857.T"] },
    { ts: "10:12 JST", src: "Reuters", weight: "medium", text: "Arm 25Q1 营收前瞻：市场关注度上修", linkedSymbols: ["9984.T"] },
    { ts: "09:43 JST", src: "Bloomberg", weight: "high", text: "USD/JPY 跌破 156.2，市场押注日银 6 月加息", linkedSymbols: ["8306.T","7203.T"] },
    { ts: "09:00 JST", src: "Nikkei QUICK", weight: "low", text: "TOPIX 开盘强势 +0.6%，半导体板块领涨", linkedSymbols: ["TOPIX","8035.T"] },
    { ts: "08:55 JST", src: "日経", weight: "high", text: "経产省补贴半导体厂房第二期", linkedSymbols: ["8035.T"] },
    { ts: "22:14 ET", src: "Bloomberg", weight: "high", text: "ASML 上修 25Q3 出货指引 +8%", linkedSymbols: ["8035.T","6920.T"] },
    { ts: "21:30 ET", src: "WSJ", weight: "medium", text: "费城半导体指数三日累涨 4.6%", linkedSymbols: ["SOX"] },
  ],

  decisionLog: [
    { ts: "10:31:42 JST", symbol: "8035.T", score: 78, action: "candidate_persisted", note: "yfinance 模式落盘成功" },
    { ts: "10:31:42 JST", symbol: "6920.T", score: 71, action: "candidate_persisted", note: "" },
    { ts: "10:31:41 JST", symbol: "7203.T", score: 58, action: "candidate_persisted", note: "" },
    { ts: "10:30:00 JST", symbol: "—", score: null, action: "scan_completed", note: "扫描 6 标的，耗时 1.84s" },
    { ts: "10:28:11 JST", symbol: "8035.T", score: null, action: "news_overlay_hit", note: "経产省补贴 +0.31 权重" },
    { ts: "10:12:03 JST", symbol: "9984.T", score: null, action: "news_overlay_hit", note: "Arm 财报 +0.18 权重" },
    { ts: "09:43:55 JST", symbol: "FX", score: null, action: "macro_change", note: "USD/JPY 跌破 156.2" },
    { ts: "09:00:00 JST", symbol: "—", score: null, action: "session_open", note: "JST 09:00 开盘" },
  ],

  // ── Enrichment: portfolio (Path A etf_buyhold) ──
  positions: {
    available: true,
    strategy_id: "etf_buyhold",
    positions_asof: "2026-05-23",
    asof: "2026-05-24",
    nav: 1000000,
    cash: 151456,
    // portfolio NAV history (last ~12 sessions, ¥) for the sparkline
    navHistory: [948200, 952100, 961400, 957800, 967300, 972500, 980100, 985400, 991200, 996800, 998300, 1000000],
    holdings: [
      { symbol: "1306.T", nameCn: "野村东证ETF", qty: 500, avg_cost: 403, market_price: 418, market_value: 209000, unrealized_pnl: 7450, unrealized_return_pct: 3.7 },
      { symbol: "6920.T", nameCn: "Lasertec", qty: 8, avg_cost: 17200, market_price: 18260, market_value: 146080, unrealized_pnl: 8480, unrealized_return_pct: 6.2 },
      { symbol: "8306.T", nameCn: "三菱UFJ", qty: 60, avg_cost: 1760, market_price: 1812, market_value: 108720, unrealized_pnl: 3120, unrealized_return_pct: 3.0 },
      { symbol: "9984.T", nameCn: "软银集团", qty: 12, avg_cost: 9900, market_price: 9412, market_value: 112944, unrealized_pnl: -5856, unrealized_return_pct: -4.9 },
      { symbol: "7203.T", nameCn: "丰田汽车", qty: 30, avg_cost: 3050, market_price: 2980, market_value: 89400, unrealized_pnl: -2115, unrealized_return_pct: -2.3 },
      { symbol: "6857.T", nameCn: "爱德万", qty: 10, avg_cost: 7500, market_price: 7820, market_value: 78200, unrealized_pnl: 3200, unrealized_return_pct: 4.3 },
      { symbol: "7011.T", nameCn: "三菱重工", qty: 20, avg_cost: 5400, market_price: 5210, market_value: 104200, unrealized_pnl: -3800, unrealized_return_pct: -3.5 },
    ],
  },

  // ── Enrichment: daily cockpit (Rule 12.0 stage 0 pull-only) ──
  dailyCockpit: {
    activationStage: "stage_0_pull_only",
    notificationsInvoked: false,
    execution: { orders: false, broker: false },
    summary: { watchlistCount: 4, symbolsWithQuotes: 0, silentQueueCount: 0 },
    watchlist: [
      { symbol: "6768.T", quoteStatus: "unavailable", tdnetCount: 0 },
      { symbol: "7419.T", quoteStatus: "unavailable", tdnetCount: 0 },
      { symbol: "2938.T", quoteStatus: "unavailable", tdnetCount: 0 },
      { symbol: "6966.T", quoteStatus: "unavailable", tdnetCount: 0 },
    ],
  },
};

// ── Per-symbol synthesis (mimics /api/symbol/{T}/* offline) ──
const FACTORS = {
  "8035.T": { screener_score: 0.781, mom_20: 0.349, mom_60: 0.348, sharpe_20: 1.42, sortino_60: 1.18, vol_z: 1.9, adv: 1331e6, in_screener: true },
  "6920.T": { screener_score: 0.712, mom_20: 0.281, mom_60: 0.262, sharpe_20: 1.21, sortino_60: 0.98, vol_z: 1.4, adv: 642e6, in_screener: true },
  "7203.T": { screener_score: 0.585, mom_20: 0.061, mom_60: 0.092, sharpe_20: 0.41, sortino_60: 0.38, vol_z: 0.3, adv: 980e6, in_screener: true },
  "9984.T": { screener_score: 0.551, mom_20: -0.042, mom_60: 0.121, sharpe_20: -0.18, sortino_60: 0.05, vol_z: 0.9, adv: 1120e6, in_screener: true },
  "8306.T": { screener_score: 0.471, mom_20: 0.018, mom_60: 0.044, sharpe_20: 0.22, sortino_60: 0.20, vol_z: 0.2, adv: 760e6, in_screener: true },
  "6857.T": { screener_score: 0.441, mom_20: 0.112, mom_60: 0.088, sharpe_20: 0.62, sortino_60: 0.51, vol_z: 0.8, adv: 540e6, in_screener: true },
};

function outcomesFor(symbol) {
  const rng = mkRng([...symbol].reduce((a, c) => a + c.charCodeAt(0), 11));
  const n = 14;
  const items = [];
  let hit = 0, complete = 0;
  for (let i = 0; i < n; i++) {
    const settled = i < n - 2;
    const o = settled ? (rng() > 0.46 ? 1 : 0) : null;
    if (settled) { complete++; if (o === 1) hit++; }
    items.push({
      trade_date: `2026-05-${String(9 + i).padStart(2, "0")}`,
      raw_score: 0.4 + rng() * 0.45,
      outcome: o,
      backdated: i < 4,
    });
  }
  items.reverse();
  return {
    horizon: "3D", count: n, n_complete: complete, n_hit: hit,
    hit_rate: complete >= 10 ? hit / complete : null,
    items,
    rule_9_4_note: "原始命中频率 · 非胜率 · 未校准 (Rule 9.4)。样本含 bootstrap，仅作证据透明。",
  };
}

function strategyFor(c) {
  const risk = [];
  if (c.symbol === "8035.T") {
    risk.push({ severity: "warn", message: "日内涨幅接近 +10%，追高需谨慎（Rule 12.3 chase filter）", rule_ref: "Rule 12.3" });
    risk.push({ severity: "info", message: "USD/JPY 跌破 156 将下调出口股温度，影响 ladder 激进档有效性", rule_ref: "Rule 8.5" });
    risk.push({ severity: "info", message: "数据上下文不全（缺日内涨跌或时间戳）", rule_ref: "Rule 12" });
  } else if (c.symbol === "6920.T") {
    risk.push({ severity: "warn", message: "波动率偏高，单日 ±5% 常见", rule_ref: "Rule 12" });
    risk.push({ severity: "info", message: "流动性次于板块龙头，ladder 保守档优先", rule_ref: "Rule 9.3" });
  } else {
    risk.push({ severity: "info", message: "当前无重大风险标记；主题热力中性偏弱", rule_ref: "Rule 12" });
  }
  return {
    banner: "advice-only · 由用户在外部券商手动执行 · Rule 3",
    rule_3_disclaimer: "Rule 3 — manual execution outside HTR",
    catalyst_disclaimer: "结构化日历数据，非建议持有至该日",
    current_price: c.price,
    score_status: c.scoreStatus,
    ladder_tiers: (c.ladder || []).map((t) => ({ kind: t.kind, label: t.label, price: t.price, pct_vs_ref: t.pct })),
    risk_warnings: risk,
    catalyst_calendar: (c.catalysts || []).map((k) => ({ label: `${k.ts} · ${k.text}` })),
  };
}

function aiBriefFor(c) {
  // Rule 8.3.1: narrative only — no probability/%/win-rate language.
  const narrative =
`${c.nameCn}（${c.symbol}）今日处于「${c.theme}」主题领涨位置。外部温度面，隔夜 SOX 与费城半导体走强为日股半导体板块提供顺风；主题层面，板块热力居六板块之首，龙头集中度偏高。

个股层面，价格站在均线之上，近期量价配合，今晨结构化新闻命中其催化主线。需要并列看待的不确定性：当前研究分尚未通过 K-fold 样本外校准，应作为排序信号而非确定结论；汇率方向是本日最大外部变量。

研究视角的处理方式：在 ladder 给出的均衡区间内分批观察介入，止损参考严格执行，延伸卖出档之上不追。所有价位均为研究参考，由用户在外部券商自行判断与执行。`;
  return {
    model_version: "gemma4:e4b",
    generation_ts: "2026-05-24T01:32:10",
    narrative,
    factual_grounding: [
      "SOX overnight close +1.81% (available_ts 2026-05-23T20:00Z)",
      `theme_heat[${c.themeId}] = ${(c.theme === "半导体设备" ? 86 : 60)} (rank 1/6)`,
      "news_overlay: 2 hits, weights 0.42 / 0.31",
      `screener_score = ${(FACTORS[c.symbol]?.screener_score ?? 0.5).toFixed(3)} (uncalibrated)`,
    ],
    score_status: c.scoreStatus,
    advice_only: true,
  };
}

function debateFor(c) {
  // Rule 8.3.1 multi-leg — each leg narrative only, no probability language.
  return {
    model_version: "gemma4:e4b",
    bull: { narrative: `多头看法：板块热力领先且龙头集中，外部半导体顺风延续；价格守在均线上方，结构健康。若守住 ladder 均衡区间，趋势仍偏强。` },
    bear: { narrative: `空头看法：研究分未经校准，K-fold 已降级；日内涨幅偏大有追高风险；USD/JPY 走弱压制出口估值，外部变量可能逆转。` },
    judge: { narrative: `裁决：证据偏向主题强势但确定性不足。以研究观察为主，严格遵守 ladder 止损参考；不在延伸卖出档之上加码。最终执行由用户在外部券商自行决定。` },
  };
}

// Standard 7-tier ladder offsets (% vs current) — used to synthesize a ladder
// for any candidate the mock didn't author one for (offline only #1 has one).
const LADDER_TIERS = [
  { kind: "exit_stretch", label: "延伸卖出", pct: 6.0 },
  { kind: "exit_2", label: "卖出 2", pct: 4.0 },
  { kind: "exit_1", label: "卖出 1", pct: 2.5 },
  { kind: "entry_aggressive", label: "买入 · 激进", pct: -0.8 },
  { kind: "entry_balanced", label: "买入 · 均衡", pct: -1.8 },
  { kind: "entry_conservative", label: "买入 · 保守", pct: -2.8 },
  { kind: "stop", label: "止损", pct: -4.5 },
];
function genLadder(price) {
  return LADDER_TIERS.map((t) => ({ kind: t.kind, label: t.label, price: Math.round(price * (1 + t.pct / 100)), pct: t.pct }));
}

// Attach synthesized blocks + per-symbol kline to a candidate. Idempotent
// (`if (!c.x)`) so it is safe to run on a REAL /api/dashboard candidate too:
// it only fills the per-symbol fields the dashboard endpoint doesn't carry,
// giving the V3 detail cards non-crashing defaults BEFORE the real per-symbol
// endpoints (/strategy /profile /outcomes /kline /llm_brief /debate_brief)
// overlay the authoritative values via useEnrichedCandidate (htr-api.jsx).
function enrichCandidate(c) {
  if (!c.ladder) c.ladder = genLadder(c.price);
  if (!c.factors) c.factors = FACTORS[c.symbol] || null;
  if (!c.outcomes) c.outcomes = outcomesFor(c.symbol);
  if (!c.strategy) c.strategy = strategyFor(c);
  if (!c.aiBrief) c.aiBrief = aiBriefFor(c);
  if (!c.debate) c.debate = debateFor(c);
  if (!c.kline) c.kline = klineFor(c.symbol, c.price, 90);
  return c;
}
BASE.candidates.forEach(enrichCandidate);

BASE.kline = BASE.candidates[0].kline;

window.HTR_DATA = BASE;
window.HTR_klineFor = klineFor;
window.HTR_enrich = enrichCandidate;   // boot uses this on real candidates
