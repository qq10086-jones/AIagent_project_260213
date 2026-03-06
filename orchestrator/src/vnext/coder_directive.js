export const DEFAULT_CODER_MODEL = "qwen-coder-next";

export function parseCoderDirectiveOptions(rawText, defaults = {}) {
  const text = String(rawText || "");
  const defaultProvider = String(defaults.provider || "opencode").trim().toLowerCase();
  const defaultModel = String(defaults.model || DEFAULT_CODER_MODEL).trim();

  let cleaned = text;
  let model = "";
  let provider = defaultProvider;

  const explicitModel = cleaned.match(/(?:^|\s)@model=([^\s]+)|\bmodel\s*=\s*([^\s]+)/i);
  if (explicitModel) {
    model = String(explicitModel[1] || explicitModel[2] || "").trim();
    cleaned = cleaned.replace(/(?:^|\s)@model=[^\s]+/gi, " ").replace(/\bmodel\s*=\s*[^\s]+/gi, " ");
  }

  const explicitProvider = cleaned.match(/(?:^|\s)@provider=([^\s]+)|\bprovider\s*=\s*([^\s]+)/i);
  if (explicitProvider) {
    provider = String(explicitProvider[1] || explicitProvider[2] || provider).trim().toLowerCase();
    cleaned = cleaned.replace(/(?:^|\s)@provider=[^\s]+/gi, " ").replace(/\bprovider\s*=\s*[^\s]+/gi, " ");
  }

  if (!model) {
    if (/@gpt-5\.3\b/i.test(cleaned)) {
      model = "gpt-5.3";
      cleaned = cleaned.replace(/@gpt-5\.3\b/gi, " ");
    } else if (/@minimax\b/i.test(cleaned)) {
      model = "minimax-m2.5";
      cleaned = cleaned.replace(/@minimax\b/gi, " ");
    } else if (/@qwen-coder-next\b/i.test(cleaned) || /@qwen\b/i.test(cleaned)) {
      model = DEFAULT_CODER_MODEL;
      cleaned = cleaned.replace(/@qwen-coder-next\b/gi, " ").replace(/@qwen\b/gi, " ");
    }
  }

  return {
    taskPrompt: cleaned.replace(/\s+/g, " ").trim(),
    provider,
    model: model || defaultModel || DEFAULT_CODER_MODEL,
  };
}
