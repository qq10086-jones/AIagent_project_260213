// Mock data for HotThemeRotator redesign.
// All numbers are illustrative — JST market open scenario, news-catalysed semis day.

window.HTR_DATA = {
  meta: {
    asof: "2026-05-24 10:32:14 JST",
    tradeDate: "2026-05-24",
    refreshLabel: "120s",
    eventTrigger: false,
    calibration: {
      level: "warning",
      text: "未校准研究分 · 不是真实胜率",
      sample: 184,
      brier: 0.241,
    },
  },

  // §10 gates
  gates: [
    { id: "G1", label: "决策日志",   status: "done",        glyph: "✓", taskId: "T-101", next: "—" },
    { id: "G2", label: "反馈采集",   status: "done",        glyph: "✓", taskId: "T-104", next: "—" },
    { id: "G3", label: "分数校准",   status: "in_progress", glyph: "●", taskId: "T-108", next: "+12d 样本" },
    { id: "G4", label: "提醒推送",   status: "pending",     glyph: "○", taskId: "T-112", next: "依赖 G3" },
    { id: "G5", label: "纸面交易",   status: "pending",     glyph: "○", taskId: "T-118", next: "依赖 G4" },
    { id: "G6", label: "人工批准",   status: "pending",     glyph: "○", taskId: "T-123", next: "依赖 G5" },
    { id: "G7", label: "半自动下单", status: "blocked",     glyph: "—", taskId: "T-130", next: "需治理评审" },
    { id: "G8", label: "全自动",     status: "blocked",     glyph: "—", taskId: "T-140", next: "需治理评审" },
  ],

  // Market temperature (multi-market temperature factors)
  markets: [
    { id: "N225",   label: "日经225",      sub: "Nikkei 225",      price: 38420.18, chg: 1.24, temp: 72, region: "JP",  state: "OPEN",  spark: [3,4,5,4,6,7,6,8,9,8,9,11,10,12,11,12] },
    { id: "TOPIX",  label: "TOPIX",        sub: "Tokyo Price Idx", price:  2712.40, chg: 0.83, temp: 64, region: "JP",  state: "OPEN",  spark: [3,4,4,5,5,6,5,6,7,7,8,8,9,9,10,10] },
    { id: "SOX",    label: "SOX 半导体",   sub: "PHLX Semi",       price:  5210.92, chg: 1.81, temp: 81, region: "US",  state: "CLOSED",spark: [2,3,5,4,6,7,9,8,10,11,12,11,13,14,13,15] },
    { id: "SPX",    label: "S&P 500",      sub: "S&P 500",         price:  5821.30, chg: 0.42, temp: 58, region: "US",  state: "CLOSED",spark: [5,5,6,5,6,6,7,7,7,8,7,8,8,8,9,9] },
    { id: "USDJPY", label: "USD/JPY",      sub: "美元兑日元",       price:   156.18, chg:-0.15, temp: 46, region: "FX",  state: "LIVE",  spark: [10,9,9,8,8,9,8,7,7,8,7,7,6,7,6,6] },
    { id: "SSE",    label: "上证",         sub: "Shanghai Comp",   price:  3420.55, chg:-0.31, temp: 38, region: "CN",  state: "OPEN",  spark: [8,7,8,7,6,7,6,7,6,5,6,5,6,5,4,5] },
  ],

  // Theme heat (rotating themes)
  themes: [
    { id: "semi",   label: "半导体设备",  leaders: ["8035.T","6920.T","6857.T"], heat: 86, mom: 1.8 },
    { id: "ai",     label: "AI 算力",     leaders: ["9984.T","6532.T"],          heat: 71, mom: 0.6 },
    { id: "auto",   label: "汽车出口",    leaders: ["7203.T","7267.T"],          heat: 52, mom:-0.2 },
    { id: "bank",   label: "金融升息",    leaders: ["8306.T","8316.T"],          heat: 44, mom: 0.1 },
    { id: "defense",label: "防卫",        leaders: ["7011.T","7012.T"],          heat: 38, mom: 0.4 },
    { id: "energy", label: "能源贸商",    leaders: ["8058.T","8001.T"],          heat: 31, mom:-0.5 },
  ],

  // Candidate list — sorted by opportunity score
  candidates: [
    {
      rank: 1,
      symbol: "8035.T",
      nameJa: "東京エレクトロン",
      nameCn: "东京电子",
      theme: "半导体设备",
      themeId: "semi",
      priority: "重点关注",
      score: 78,
      scoreStatus: "warning",
      price: 30420,
      chg: 2.14,
      one_liner: "SOX 隔夜 +1.8%，叠加 ASML 上修 25Q3 指引，开盘后 30 分钟若守住 30,180 即可分批介入。",
      reason: "外部温度：SOX +1.81%、费城半导体三日累涨 4.6%。主题强度：半导体设备板块今日热力 86（六板块第一），龙头集中度高。个股：5 日动量 +6.2%，量价配合，催化新闻命中 2 条。",
      risk: "未校准研究分。USD/JPY 跌破 156 会拖累出口估值。若开盘 15 分钟未站稳 30,300，激进档作废。",
      dataQuality: "数据：yfinance 延迟 ≤15s；新闻 overlay 命中 2 条；available_ts 校验通过。",
      ladder: [
        { kind: "exit_stretch",       label: "延伸卖出",   price: 32246, pct:  6.0 },
        { kind: "exit_2",             label: "卖出 2",     price: 31636, pct:  4.0 },
        { kind: "exit_1",             label: "卖出 1",     price: 31180, pct:  2.5 },
        { kind: "entry_aggressive",   label: "买入 · 激进",price: 30176, pct: -0.8 },
        { kind: "entry_balanced",     label: "买入 · 均衡",price: 29872, pct: -1.8 },
        { kind: "entry_conservative", label: "买入 · 保守",price: 29568, pct: -2.8 },
        { kind: "stop",               label: "止损",       price: 29051, pct: -4.5 },
      ],
      decisionCutoff: "10:30 JST",
      catalysts: [
        { ts: "前一日 22:14 ET", src: "Bloomberg", text: "ASML 上修 25Q3 出货指引 +8%", weight: 0.42 },
        { ts: "今日 08:55 JST",   src: "日経",      text: "经产省补贴半导体厂房第二期", weight: 0.31 },
      ],
    },
    {
      rank: 2, symbol: "6920.T", nameJa: "レーザーテック", nameCn: "Lasertec", theme: "半导体设备", themeId: "semi",
      priority: "重点关注", score: 71, scoreStatus: "warning",
      price: 18260, chg: 1.62,
      one_liner: "EUV 检测设备龙头，跟随 8035.T 节奏；建议作为半导体主题的次配。",
      reason: "主题强度共振；个股 RS 排名行业 #2；外资连续 3 日净买入。",
      risk: "波动率高，单日±5% 常见；流动性次于 8035.T。",
      dataQuality: "数据完整；新闻命中 1 条。",
      decisionCutoff: "10:30 JST",
    },
    {
      rank: 3, symbol: "7203.T", nameJa: "トヨタ自動車", nameCn: "丰田汽车", theme: "汽车出口", themeId: "auto",
      priority: "观察", score: 58, scoreStatus: "warning",
      price: 2980.5, chg: 0.34,
      one_liner: "USD/JPY 走弱压制估值；今日仅做主题平衡，不建议加仓。",
      reason: "防御性配置；龙头流动性极佳。",
      risk: "汇率敏感；主题热力下行。",
      dataQuality: "数据完整。",
    },
    {
      rank: 4, symbol: "9984.T", nameJa: "ソフトバンクＧ", nameCn: "软银集团", theme: "AI 算力", themeId: "ai",
      priority: "观察", score: 55, scoreStatus: "warning",
      price: 9412, chg: -0.81,
      one_liner: "Arm 财报临近，今日仅观察，不建议建仓。",
      reason: "AI 主题次热；事件驱动。",
      risk: "Arm 财报隔夜大波动概率高。",
      dataQuality: "数据完整。",
    },
    {
      rank: 5, symbol: "8306.T", nameJa: "三菱ＵＦＪ", nameCn: "三菱UFJ", theme: "金融升息", themeId: "bank",
      priority: "观察", score: 47, scoreStatus: "warning",
      price: 1812.0, chg: 0.16,
      one_liner: "无明确催化；保留在池里跟踪日银 6 月会议。",
      reason: "升息主题低热；待催化。",
      risk: "若日银鸽派则单日 -2~3%。",
      dataQuality: "数据完整。",
    },
    {
      rank: 6, symbol: "6857.T", nameJa: "アドバンテスト", nameCn: "爱德万", theme: "半导体设备", themeId: "semi",
      priority: "观察", score: 44, scoreStatus: "warning",
      price: 7820, chg: 1.05,
      one_liner: "主题受益但 RS 弱于前二名，作为后备。",
      reason: "测试设备龙头；外资态度中性。",
      risk: "估值偏高。",
      dataQuality: "数据完整。",
    },
  ],

  // News catalyst timeline (most recent first)
  newsTimeline: [
    { ts: "10:28 JST", src: "日経",       weight: "high",   text: "経産省、半導体製造装置の補助金第2期を 24 日午後発表へ", linkedSymbols: ["8035.T","6920.T","6857.T"] },
    { ts: "10:12 JST", src: "Reuters",    weight: "medium", text: "Arm 25Q1 营收前瞻：超市场预期概率上修",                  linkedSymbols: ["9984.T"] },
    { ts: "09:43 JST", src: "Bloomberg",  weight: "high",   text: "USD/JPY 跌破 156.2，市场押注日银 6 月加息",            linkedSymbols: ["8306.T","7203.T"] },
    { ts: "09:00 JST", src: "Nikkei QUICK",weight:"low",    text: "TOPIX 开盘强势 +0.6%，半导体板块领涨",                 linkedSymbols: ["TOPIX","8035.T"] },
    { ts: "08:55 JST", src: "日経",        weight:"high",   text: "経産省补贴半导体厂房第二期",                            linkedSymbols: ["8035.T"] },
    { ts: "22:14 ET",  src: "Bloomberg",  weight: "high",   text: "ASML 上修 25Q3 出货指引 +8%",                            linkedSymbols: ["8035.T","6920.T"] },
    { ts: "21:30 ET",  src: "WSJ",        weight: "medium", text: "费城半导体指数三日累涨 4.6%",                          linkedSymbols: ["SOX"] },
  ],

  // Decision log (§8.6)
  decisionLog: [
    { ts: "10:31:42 JST", symbol: "8035.T", score: 78, action: "candidate_persisted", note: "yfinance 模式落盘成功" },
    { ts: "10:31:42 JST", symbol: "6920.T", score: 71, action: "candidate_persisted", note: "" },
    { ts: "10:31:41 JST", symbol: "7203.T", score: 58, action: "candidate_persisted", note: "" },
    { ts: "10:30:00 JST", symbol: "—",      score: null, action: "scan_completed",     note: "扫描 6 标的，耗时 1.84s" },
    { ts: "10:28:11 JST", symbol: "8035.T", score: null, action: "news_overlay_hit",   note: "经产省补贴 +0.31 权重" },
    { ts: "10:12:03 JST", symbol: "9984.T", score: null, action: "news_overlay_hit",   note: "Arm 财报 +0.18 权重" },
    { ts: "09:43:55 JST", symbol: "FX",     score: null, action: "macro_change",       note: "USD/JPY 跌破 156.2" },
    { ts: "09:00:00 JST", symbol: "—",      score: null, action: "session_open",       note: "JST 09:00 开盘" },
  ],

  // K-line OHLC for the hero (last ~40 sessions, illustrative)
  kline: generateKline(),
};

// Synthetic candle generator — deterministic with seeded RNG.
function generateKline() {
  let seed = 42;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) % 0x100000000; return seed / 0x100000000; };
  const out = [];
  let close = 27800;
  for (let i = 0; i < 40; i++) {
    const drift = (rnd() - 0.45) * 380;
    const open = close;
    const next = open + drift;
    const high = Math.max(open, next) + rnd() * 220;
    const low  = Math.min(open, next) - rnd() * 220;
    out.push({ open, high, low, close: next, vol: 800000 + Math.floor(rnd() * 1800000) });
    close = next;
  }
  // Ensure last close is around 30,420 to match candidate
  const finalClose = 30420;
  const scale = finalClose / out[out.length - 1].close;
  return out.map((c) => ({
    open: c.open * scale, high: c.high * scale, low: c.low * scale, close: c.close * scale, vol: c.vol,
  }));
}
