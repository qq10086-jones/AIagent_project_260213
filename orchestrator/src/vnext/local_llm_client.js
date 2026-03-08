/**
 * local_llm_client.js
 *
 * HTTP client wrappers for Qwen cloud LLM, Ollama local LLM, and Brain service relay.
 * Sanitisation helpers for local model outputs.
 * Extracted from index.js as part of WS-11-05 decomposition.
 */

let dynamicNumPredict = Number(process.env.OLLAMA_NUM_PREDICT || 4096);

export function sanitizeLocalAssistantReply(raw) {
  let out = String(raw || "");
  out = out.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  out = out.replace(/<think>[\s\S]*$/gi, "").trim();
  const cotLine = /^(\u5634\uff0c?\u6211\u73b0\u5728\u8981\u5904\u7406\u7528\u6237\u7684\u67e5\u8be2|\u597d\uff0c?\u6211\u73b0\u5728\u9700\u8981|\u9996\u5148\uff0c|\u63a5\u4e0b\u6765\uff0c|\u53e6\u5916\uff0c|\u7528\u6237\u53ef\u80fd|\u4f5c\u4e3aNEXUS\u52a9\u624b|\u6211\u9700\u8981\u7406\u89e3\u8fd9\u4e2a\u95ee\u9898)/;
  const lines = out.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const kept = lines.filter(l => !cotLine.test(l));
  if (kept.length > 0 && kept.length < lines.length) {
    out = kept.join("\n").trim();
  }
  return out;
}

export async function callQwenChat(messages, { qwenBase, qwenModel } = {}) {
  const QWEN_KEY = process.env.QWEN_API_KEY;
  if (!QWEN_KEY) throw new Error("QWEN_API_KEY is not set");
  const base = String(
    qwenBase || process.env.QWEN_BASE_URL || "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  ).replace(/\/+$/, "");
  const model = String(qwenModel || process.env.QWEN_MODEL || "qwen-plus");
  const candidates = [...new Set([
    base,
    base.replace(/\/v1$/i, "/compatible-mode/v1"),
    base.replace(/\/compatible-mode\/v1$/i, "/v1"),
  ])].filter(x => /^https?:\/\//i.test(x));
  let lastErr = "Qwen API error";

  for (const baseUrl of candidates) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${QWEN_KEY}` },
        body: JSON.stringify({ model, messages }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        lastErr = `Qwen API error ${response.status} ${response.statusText} ${errText}`.trim();
        if (response.status === 404) continue;
        throw new Error(lastErr);
      }
      const data = await response.json();
      return data.choices?.[0]?.message?.content?.trim() || "";
    } finally {
      clearTimeout(timeoutId);
    }
  }
  throw new Error(lastErr);
}

export async function callLocalOllamaChat(
  model,
  userInput,
  timeoutMs = Number(process.env.OLLAMA_CHAT_TIMEOUT_MS || 240000),
  extraSystemPrompt = ""
) {
  const base = process.env.OLLAMA_BASE_URL || "http://host.docker.internal:11434";
  const numCtx = Number(process.env.OLLAMA_NUM_CTX || 4096);
  const baseSystemPrompt =
    "\u4f60\u662f NEXUS \u52a9\u624b\u3002\u8bf7\u76f4\u63a5\u56de\u7b54\u7528\u6237\u5f53\u524d\u95ee\u9898\uff0c\u4e0d\u8981\u91cd\u590d\u81ea\u6211\u4ecb\u7ecd\u6216\u901a\u7528\u6b22\u8fce\u8bed\u3002\u4e0d\u8981\u8f93\u51fa\u601d\u8003\u8fc7\u7a0b\u3002\u9ed8\u8ba4\u4f7f\u7528\u7b80\u4f53\u4e2d\u6587\uff0c\u9664\u975e\u7528\u6237\u660e\u786e\u8981\u6c42\u5176\u4ed6\u8bed\u8a00\u3002";
  const systemPrompt = extraSystemPrompt
    ? `${baseSystemPrompt}\n\n[Project Rules & SOPs]:\n${extraSystemPrompt}`
    : baseSystemPrompt;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const startTime = Date.now();

  const formattedMessages = Array.isArray(userInput)
    ? [{ role: "system", content: systemPrompt }, ...userInput]
    : [{ role: "system", content: systemPrompt }, { role: "user", content: userInput }];

  try {
    const chatRes = await fetch(`${base}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: formattedMessages,
        think: false,
        keep_alive: process.env.OLLAMA_KEEP_ALIVE || "30m",
        options: { num_ctx: Number.isFinite(numCtx) ? numCtx : 4096, num_predict: dynamicNumPredict },
        stream: false,
      }),
      signal: controller.signal,
    });
    const elapsed = Date.now() - startTime;
    if (elapsed > 90000 && dynamicNumPredict > 2048) {
      console.log(`[performance] Local LLM took ${elapsed}ms. Downgrading num_predict to 2048 for future queries.`);
      dynamicNumPredict = 2048;
    }
    const chatData = await chatRes.json().catch(() => ({}));
    if (chatRes.ok) {
      return sanitizeLocalAssistantReply(String(chatData.message?.content || ""));
    }
    if (chatRes.status !== 404) {
      throw new Error(chatData?.error || `OLLAMA_HTTP_${chatRes.status}`);
    }

    // Compatibility fallback for Ollama variants without /api/chat.
    const genRes = await fetch(`${base}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt: Array.isArray(userInput)
          ? userInput.map(m => `${m.role}: ${m.content}`).join("\n")
          : userInput,
        think: false,
        keep_alive: process.env.OLLAMA_KEEP_ALIVE || "30m",
        options: { num_ctx: Number.isFinite(numCtx) ? numCtx : 4096, num_predict: dynamicNumPredict },
        stream: false,
      }),
      signal: controller.signal,
    });
    const genData = await genRes.json().catch(() => ({}));
    const elapsed2 = Date.now() - startTime;
    if (elapsed2 > 90000 && dynamicNumPredict > 2048) {
      console.log(`[performance] Local LLM took ${elapsed2}ms. Downgrading num_predict to 2048 for future queries.`);
      dynamicNumPredict = 2048;
    }
    if (!genRes.ok) throw new Error(genData?.error || `OLLAMA_GENERATE_HTTP_${genRes.status}`);
    return sanitizeLocalAssistantReply(genData.response || "");
  } catch (err) {
    if ((err.name === "AbortError" || err.message.includes("timeout")) && dynamicNumPredict > 2048) {
      console.log("[performance] Local LLM timed out. Downgrading num_predict to 2048 for future queries.");
      dynamicNumPredict = 2048;
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function callBrainWithRetry(payload, retries = 2) {
  for (let i = 0; i <= retries; i++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 240000);
      const res = await fetch("http://brain:5000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok) throw new Error(`Brain returned ${res.status}`);
      return await res.json();
    } catch (err) {
      if (i === retries) throw err;
      console.log(`[orchestrator] brain call failed, retrying (${i + 1}/${retries})...`);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}
