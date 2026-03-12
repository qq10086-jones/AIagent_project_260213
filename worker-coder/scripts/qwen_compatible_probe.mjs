const baseUrl = String(process.env.QWEN_BASE_URL || "").trim();
const apiKey = String(process.env.QWEN_API_KEY || "").trim();
const model = String(process.argv[2] || "qwen3-coder-plus-2025-07-22").trim();
const prompt = String(process.argv[3] || "Reply with exactly OK").trim();

if (!baseUrl) {
  console.error("QWEN_BASE_URL is required");
  process.exit(1);
}

if (!apiKey) {
  console.error("QWEN_API_KEY is required");
  process.exit(1);
}

const endpoint = `${baseUrl.replace(/\/$/, "")}/chat/completions`;

const response = await fetch(endpoint, {
  method: "POST",
  headers: {
    "content-type": "application/json",
    authorization: `Bearer ${apiKey}`,
  },
  body: JSON.stringify({
    model,
    messages: [{ role: "user", content: prompt }],
    max_tokens: 16,
    temperature: 0,
  }),
});

const bodyText = await response.text();
console.log(`STATUS=${response.status}`);
console.log(bodyText);

if (!response.ok) {
  process.exit(2);
}
