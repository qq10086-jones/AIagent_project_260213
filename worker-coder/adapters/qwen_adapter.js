function resolveQwenConfig() {
  const apiKey = String(process.env.QWEN_API_KEY || process.env.DASH_SCOPE_API_KEY || "").trim();
  const baseUrl = String(
    process.env.QWEN_BASE_URL ||
      process.env.DASH_SCOPE_BASE_URL ||
      "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  )
    .trim()
    .replace(/\/+$/, "");
  return { apiKey, baseUrl };
}

export async function runQwenTask({
  taskPrompt,
  model,
  maxRuntimeS = 600,
}) {
  if (!taskPrompt || !String(taskPrompt).trim()) {
    return {
      ok: false,
      provider_used: "qwen",
      model_used: model || null,
      stdout: "",
      stderr: "",
      diagnostics: { error_code: "E_INVALID_INPUT", exit_code: null, timeout: false },
      error: "task_prompt is required for coding.delegate",
    };
  }

  const { apiKey, baseUrl } = resolveQwenConfig();
  if (!apiKey) {
    return {
      ok: false,
      provider_used: "qwen",
      model_used: model || null,
      stdout: "",
      stderr: "",
      diagnostics: { error_code: "E_PROVIDER_UNAVAILABLE", exit_code: null, timeout: false },
      error: "Qwen auth missing: set QWEN_API_KEY (or DASH_SCOPE_API_KEY) in worker-coder env",
    };
  }

  const controller = new AbortController();
  const timeoutMs = Math.max(1, Number(maxRuntimeS || 600)) * 1000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const endpoint = `${baseUrl}/chat/completions`;
  const chosenModel = String(model || process.env.QWEN_MODEL || "qwen-plus");

  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: chosenModel,
        messages: [
          {
            role: "system",
            content:
              "You are a coding assistant. Return concise, practical implementation guidance. If no code change is requested, return a short plan.",
          },
          { role: "user", content: String(taskPrompt) },
        ],
        temperature: 0.2,
      }),
      signal: controller.signal,
    });

    const raw = await resp.text();
    if (!resp.ok) {
      return {
        ok: false,
        provider_used: "qwen",
        model_used: chosenModel,
        stdout: "",
        stderr: raw,
        diagnostics: { error_code: "E_EXEC_FAILED", exit_code: Number(resp.status), timeout: false },
        error: `Qwen API error ${resp.status}`,
      };
    }

    let parsed = {};
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = {};
    }
    const text =
      parsed?.choices?.[0]?.message?.content ||
      parsed?.output?.text ||
      raw ||
      "";

    return {
      ok: true,
      provider_used: "qwen",
      model_used: chosenModel,
      command_used: `qwen_api ${chosenModel}`,
      command_source: "qwen_api",
      stdout: String(text),
      stderr: "",
      diagnostics: { error_code: null, exit_code: 0, timeout: false },
      error: null,
    };
  } catch (err) {
    const timedOut = controller.signal.aborted;
    return {
      ok: false,
      provider_used: "qwen",
      model_used: chosenModel,
      stdout: "",
      stderr: String(err?.message || err || ""),
      diagnostics: {
        error_code: timedOut ? "E_TIMEOUT" : "E_INTERNAL",
        exit_code: null,
        timeout: timedOut,
      },
      error: timedOut ? "Qwen API request timed out" : `Qwen adapter error: ${err?.message || err}`,
    };
  } finally {
    clearTimeout(timer);
  }
}
