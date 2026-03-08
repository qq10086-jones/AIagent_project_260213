import fs from "fs";
import path from "path";

export function getMemoryRoot() {
  return process.env.MEMORY_ROOT || path.resolve("artifacts", "memory");
}

function memoryPath(project_id, ...segments) {
  return path.join(getMemoryRoot(), String(project_id || "default"), ...segments);
}

function readJsonFile(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

export function getProjectContext(project_id) {
  return readJsonFile(memoryPath(project_id, "project_context.json"));
}

export function getPriorADRs(project_id) {
  const dir = memoryPath(project_id, "adrs");
  if (!fs.existsSync(dir)) return [];
  try {
    return fs.readdirSync(dir)
      .filter((f) => f.endsWith(".json") || f.endsWith(".md"))
      .map((f) => {
        const filePath = path.join(dir, f);
        if (f.endsWith(".json")) return readJsonFile(filePath);
        try {
          const raw = fs.readFileSync(filePath, "utf8");
          const titleLine = raw
            .split(/\r?\n/)
            .map((line) => line.trim())
            .find((line) => /^#{1,6}\s+/.test(line));
          return {
            adr_id: path.basename(f, ".md"),
            title: titleLine ? titleLine.replace(/^#{1,6}\s+/, "").trim() : path.basename(f, ".md"),
            status: "accepted",
            content: raw,
          };
        } catch {
          return null;
        }
      })
      .filter(Boolean);
  } catch {
    return [];
  }
}

export function getTaskHistory(project_id, limit = 10) {
  const filePath = memoryPath(project_id, "task_history.json");
  const history = readJsonFile(filePath);
  if (!Array.isArray(history)) return [];
  return history.slice(-limit);
}
