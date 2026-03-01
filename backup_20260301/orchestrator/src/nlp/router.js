const QWEN_BASE = process.env.QWEN_BASE_URL || "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
export let CURRENT_QWEN_MODEL = process.env.QWEN_MODEL || "qwen-max";

export function setQwenModel(modelName) {
  CURRENT_QWEN_MODEL = modelName;
}

// Define the available tools (Agents) and their capabilities for the LLM Dispatcher
const AGENT_TOOLS_SCHEMA = [
  {
    "tool_name": "quant.deep_analysis",
    "description": "Analyze a specific stock, company, or ticker. Fetches technical indicators, regression models, and news. Can also provide execution/sizing suggestions if capital is provided.",
    "parameters": {
      "symbol": "The stock ticker symbol to analyze (e.g., 'NVDA', 'AAPL', '9432.T').",
      "capital_base_jpy": "Optional: User's total capital in JPY for position sizing suggestions (e.g. 400000).",
      "mode": "Optional: 'buy' or 'sell' to tailor execution advice."
    }
  },
  {
    "tool_name": "quant.calc_limit_price",
    "description": "Calculate specific aggressive/balanced/patient limit prices for a trade based on ATR and volatility.",
    "parameters": {
      "symbol": "The stock ticker symbol.",
      "side": "'BUY' or 'SELL'"
    }
  },
  {
    "tool_name": "quant.discovery_workflow",
    "description": "Recommend stocks or find investment opportunities. Supports capital-constrained filtering, market focus, and goal-oriented position planning.",
    "parameters": {
      "capital_base_jpy": "Optional: Filter stocks that fit within this JPY budget (e.g. 400000).",
      "max_position_pct": "Optional: Max percentage of capital per stock (default 0.25).",
      "market": "Optional: 'US' for USA, 'JP' for Japan, or 'ALL' for both.",
      "goal": "Optional: User objective in plain text (e.g. '6个月稳健增值10%' or '追求成长').",
      "risk_profile": "Optional: 'low' | 'medium' | 'high'."
    }
  },
  {
    "tool_name": "news.daily_report",
    "description": "Provide general market news, daily financial news summaries, or macro intelligence.",
    "parameters": {}
  },
  {
    "tool_name": "news.active_hot_search",
    "description": "主动搜索并聚焦24小时内高热度新闻（Top5-10），并补充当前持仓相关的新闻线索。",
    "parameters": {
      "lookback_hours": "Optional: recency window in hours, default 24.",
      "top_n": "Optional: number of top hot news items, 5-10 recommended.",
      "include_positions": "Optional: whether to include current holdings related news, default true."
    }
  },
  {
    "tool_name": "news.preclose_brief_jp",
    "description": "Provide the Japanese market pre-close brief (盘尾主推/收盘前情报). Use when user asks for pre-close brief or Japanese market pre-close news.",
    "parameters": {}
  },
  {
    "tool_name": "news.tdnet_close_flash",
    "description": "Provide the Japanese market post-close flash focusing on TDnet announcements (盘后闪讯/TDnet公告). Use when user asks for post-close announcements or TDnet flash.",
    "parameters": {}
  },
  {
    "tool_name": "github.skills_daily_report",
    "description": "Scan GitHub for new AI agent skills, projects, or repositories.",
    "parameters": {}
  },
  {
    "tool_name": "portfolio.set_account",
    "description": "Set the starting capital/money for the user's trading account. E.g., 'Set my capital to 20000000 JPY'",
    "parameters": {
      "starting_capital": "The numeric amount of capital (e.g., 20000000)",
      "ccy": "Currency code like 'JPY' or 'USD'"
    }
  },
  {
    "tool_name": "portfolio.record_fill",
    "description": "Record a stock purchase or sale that the user has actually executed. E.g., 'I bought 1000 shares of 9432.T at 150 JPY'.",
    "parameters": {
      "symbol": "The stock ticker symbol (e.g., '9432.T')",
      "side": "'BUY' or 'SELL'",
      "qty": "Number of shares",
      "price": "Execution price per share"
    }
  },
  {
    "tool_name": "web.search_and_browse",
    "description": "当用户询问天气、最新新闻、实时事件、价格/汇率、比赛比分、或不知名概念时调用。可轻轨搜索或重轨浏览器抓取。",
    "parameters": {
      "query": "核心关键字，例如 'Tokyo Kita-ku weather today' 或 'NVIDIA earnings date'",
      "browse_mode": "auto | light | heavy",
      "freshness": "realtime | days | evergreen"
    }
  }
];

export const DISPATCHER_SYSTEM_PROMPT = `
You are the central "Copilot & Task Analyzer" of the Nexus Multi-Agent System.
Your job is to analyze the user's input, determine their intent, and decide whether to just CHAT, PLAN, or RUN tools.

Available Executable Tools:
${JSON.stringify(AGENT_TOOLS_SCHEMA, null, 2)}

Instructions:
1. Understand the user's intent.
2. If the user is asking a general question, seeking an explanation, or chatting without explicitly requesting data fetching, market scanning, or analysis, set "mode_suggested": "chat" and "requires_tools": false.
3. If the user explicitly asks to run an analysis, scan the market, fetch news, or execute a task, set "mode_suggested": "run", "requires_tools": true, and select the EXACT "tool_name" from the available tools.
3.1 If the user asks about geopolitical impact on markets (e.g. Middle East tensions impact on Japan stocks and next trading day suggestions), you MUST set requires_tools=true and choose either "news.active_hot_search" or "quant.discovery_workflow".
4. Extract necessary parameters into "payload".

Output STRICTLY in valid JSON format representing a TaskSpec:
{
  "intent": "qa | quant_research | ops | web_search",
  "mode_suggested": "chat | run",
  "requires_tools": true,
  "tool_name": "selected_tool_name_or_null",
  "payload": {
    "param_name": "param_value"
  },
  "confidence": 0.9,
  "reason": "Explain why this tool was selected or why it's a chat.",
  "language": "zh"
}
`;

function detectLanguageFallback(text) {
  if (/[\u4e00-\u9fff]/.test(text)) return "zh";
  if (/[\u3040-\u30ff]/.test(text)) return "ja";
  return "en";
}

export async function qwenChat(messages, timeoutMs = 15000) {
  const QWEN_KEY = process.env.QWEN_API_KEY;
  if (!QWEN_KEY) return null;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${QWEN_BASE}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${QWEN_KEY}` },
      body: JSON.stringify({ 
        model: CURRENT_QWEN_MODEL, 
        messages: messages,
        temperature: 0.1 // Low temp for more deterministic routing
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      console.error(`Qwen Dispatcher Error: ${response.status} ${response.statusText} ${errText}`);
      return null;
    }
    return await response.json();
  } catch (e) {
    console.error("Qwen Dispatcher Error:", e);
    return null;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function parseIntent(userInput, context = {}) {
  const defaultLang = detectLanguageFallback(userInput || "");

  let systemPrompt = DISPATCHER_SYSTEM_PROMPT;
  if (context.last_symbol) {
    systemPrompt += `\n\nCONTEXT: The user was previously discussing symbol: ${context.last_symbol}. If the current input refers to "it", "this", or doesn't specify a symbol but implies an action on the previously discussed stock, use this symbol.`;
  }

  // Pure LLM Copilot Analyzer
  const data = await qwenChat([
    { role: "system", content: systemPrompt },
    { role: "user", content: userInput },
  ]);

  if (!data) {
    return { mode_suggested: "chat", requires_tools: false, confidence: 0, language: defaultLang };
  }

  try {
    const content = data.choices?.[0]?.message?.content || "";
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    
    if (!jsonMatch) {
      return { mode_suggested: "chat", requires_tools: false, confidence: 0, language: defaultLang };
    }

    const parsed = JSON.parse(jsonMatch[0]);
    
    // Standardize symbol to uppercase if present
    if (parsed.payload && parsed.payload.symbol) {
      parsed.payload.symbol = String(parsed.payload.symbol).toUpperCase();
    }

    return {
      intent: parsed.intent || "qa",
      mode_suggested: parsed.mode_suggested || "chat",
      requires_tools: !!parsed.requires_tools,
      tool_name: parsed.tool_name || null,
      payload: parsed.payload || {},
      confidence: Number(parsed.confidence || 0.8),
      language: parsed.language || defaultLang,
    };
  } catch (err) {
    console.error("[Dispatcher] JSON Parse Error:", err);
    return { mode_suggested: "chat", requires_tools: false, confidence: 0, language: defaultLang };
  }
}


export async function translate(text, targetLang = "zh") {
  const data = await qwenChat(
    [
      { role: "system", content: `Translate to ${targetLang}. ONLY translation output.` },
      { role: "user", content: text },
    ],
    12000
  );
  if (!data) return text;
  return data.choices?.[0]?.message?.content?.trim() || text;
}

