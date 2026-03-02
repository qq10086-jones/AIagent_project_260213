import fs from "fs";
import path from "path";

function loadJsonIfExists(p) {
  try {
    if (!p) return null;
    if (!fs.existsSync(p)) return null;
    const raw = fs.readFileSync(p, "utf-8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

export function loadRuntimeConfig() {
  const candidates = [
    String(process.env.RUNTIME_CONFIG_PATH || "").trim(),
    path.resolve("configs/runtime/runtime_defaults.json"),
    path.resolve("../configs/runtime/runtime_defaults.json"),
    "/workspace/configs/runtime/runtime_defaults.json",
  ].filter(Boolean);

  for (const p of candidates) {
    const cfg = loadJsonIfExists(p);
    if (cfg) return { config: cfg, path: p };
  }
  return { config: {}, path: null };
}
