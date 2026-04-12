/**
 * http_client.mjs — tiny fetch wrapper with timeout and body parsing.
 */

export async function httpRequest({ method = "GET", url, body = null, headers = {}, timeoutMs = 5000 }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const init = {
    method,
    headers: { "Content-Type": "application/json", ...headers },
    signal: controller.signal,
  };
  if (body !== null && body !== undefined) {
    init.body = typeof body === "string" ? body : JSON.stringify(body);
  }
  try {
    const res = await fetch(url, init);
    let text = "";
    try { text = await res.text(); } catch { /* ignore */ }
    let json = null;
    try { json = text ? JSON.parse(text) : null; } catch { /* not json */ }
    return { ok: res.ok, status: res.status, text, json };
  } catch (err) {
    return { ok: false, status: 0, text: "", json: null, error: String(err?.message || err) };
  } finally {
    clearTimeout(timer);
  }
}

export async function waitForServer(url, { timeoutMs = 15000, intervalMs = 250 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const res = await httpRequest({ url, timeoutMs: 1000 });
    if (res.status > 0) return true;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return false;
}
