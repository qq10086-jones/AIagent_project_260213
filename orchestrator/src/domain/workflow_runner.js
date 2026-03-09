/**
 * workflow_runner.js
 *
 * Token signing / verification utilities and safe JSON parsing helper.
 * Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import crypto from "crypto";

export function parseJsonSafe(raw, fallback = {}) {
  if (!raw || typeof raw !== "string") return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function base64UrlEncode(input) {
  return Buffer.from(input).toString("base64url");
}

export function base64UrlDecode(input) {
  return Buffer.from(String(input || ""), "base64url").toString("utf8");
}

export function signResumePayload(payload, secret) {
  const body = base64UrlEncode(JSON.stringify(payload));
  const sig = crypto.createHmac("sha256", secret).update(body).digest("base64url");
  return `${body}.${sig}`;
}

export function verifyResumePayload(token, secret) {
  const tokenText = String(token || "");
  const [body, sig] = tokenText.split(".");
  if (!body || !sig) return { ok: false, error: "RESUME_INVALID: malformed token" };
  const expected = crypto.createHmac("sha256", secret).update(body).digest("base64url");
  if (expected !== sig) return { ok: false, error: "RESUME_INVALID: bad signature" };
  let payload;
  try {
    payload = JSON.parse(base64UrlDecode(body));
  } catch {
    return { ok: false, error: "RESUME_INVALID: bad payload" };
  }
  const exp = Number(payload?.exp || 0);
  const now = Math.floor(Date.now() / 1000);
  if (!Number.isFinite(exp) || exp <= now) {
    return { ok: false, error: "RESUME_INVALID: expired token" };
  }
  return { ok: true, payload };
}
