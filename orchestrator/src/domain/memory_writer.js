import fs from "fs";
import path from "path";

function getMemoryRoot() {
  return process.env.MEMORY_ROOT || path.resolve("artifacts", "memory");
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function memoryPath(project_id, ...segments) {
  return path.join(getMemoryRoot(), String(project_id || "default"), ...segments);
}

export function writeTaskHistoryEntry(project_id, entry) {
  const filePath = memoryPath(project_id, "task_history.json");
  ensureDir(path.dirname(filePath));
  let history = [];
  try {
    const existing = JSON.parse(fs.readFileSync(filePath, "utf8"));
    if (Array.isArray(existing)) history = existing;
  } catch { /* ignore: file may not exist yet */ }
  history.push({ ...entry, created_at: entry.created_at || new Date().toISOString() });
  fs.writeFileSync(filePath, JSON.stringify(history, null, 2));
}

export function writeAdrRecord(project_id, adr) {
  const dir = memoryPath(project_id, "adrs");
  ensureDir(dir);
  const filePath = path.join(dir, `${adr.adr_id}.json`);
  fs.writeFileSync(filePath, JSON.stringify({ ...adr, created_at: adr.created_at || new Date().toISOString() }, null, 2));
}

function copyAdrMarkdownFiles(project_id, releaseRoot) {
  const adrSourceDir = path.join(releaseRoot, "plan", "adr");
  if (!fs.existsSync(adrSourceDir)) return [];
  const targetDir = memoryPath(project_id, "adrs");
  ensureDir(targetDir);
  const copied = [];
  for (const entry of fs.readdirSync(adrSourceDir, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith(".md")) continue;
    const sourcePath = path.join(adrSourceDir, entry.name);
    const targetPath = path.join(targetDir, entry.name);
    fs.copyFileSync(sourcePath, targetPath);
    copied.push(targetPath);
  }
  return copied;
}

function appendJsonlFile(sourcePath, targetPath) {
  if (!sourcePath || !fs.existsSync(sourcePath)) return false;
  ensureDir(path.dirname(targetPath));
  const content = fs.readFileSync(sourcePath, "utf8");
  if (!content.trim()) return false;
  fs.appendFileSync(targetPath, content.endsWith("\n") ? content : `${content}\n`, "utf8");
  return true;
}

function copyCodingFailureMemory(project_id, runtimeRoot) {
  if (!runtimeRoot) {
    return {
      copied: false,
      task_failure_jsonl_path: null,
      task_failure_latest_path: null,
    };
  }
  const sourceJsonl = path.join(runtimeRoot, "memory", "coding_failures.jsonl");
  const sourceLatest = path.join(runtimeRoot, "memory", "coding_failure_latest.json");
  const targetDir = memoryPath(project_id, "coding_failures");
  ensureDir(targetDir);

  const targetJsonl = path.join(targetDir, "coding_failures.jsonl");
  const copiedJsonl = appendJsonlFile(sourceJsonl, targetJsonl);

  let copiedLatest = false;
  const targetLatest = path.join(targetDir, "coding_failure_latest.json");
  if (fs.existsSync(sourceLatest)) {
    fs.copyFileSync(sourceLatest, targetLatest);
    copiedLatest = true;
  }

  return {
    copied: copiedJsonl || copiedLatest,
    task_failure_jsonl_path: copiedJsonl ? targetJsonl.replace(/\\/g, "/") : null,
    task_failure_latest_path: copiedLatest ? targetLatest.replace(/\\/g, "/") : null,
  };
}

export function persistWorkflowMemory({ run, releaseRoot, runtimeRoot = "" }) {
  const projectId = String(run?.run_id || run?.workflow_run_id || "default");
  const historyEntry = {
    workflow_run_id: String(run?.workflow_run_id || ""),
    run_id: String(run?.run_id || ""),
    workflow_id: String(run?.workflow_id || ""),
    project_type: String(run?.project_type || ""),
    step_id: "workflow_complete",
    status: "succeeded",
    summary: `Workflow ${String(run?.workflow_id || "")} completed successfully`,
    release_root: String(releaseRoot || "").replace(/\\/g, "/"),
  };
  writeTaskHistoryEntry(projectId, historyEntry);
  const copiedAdrPaths = copyAdrMarkdownFiles(projectId, releaseRoot);
  const codingFailureMemory = copyCodingFailureMemory(projectId, runtimeRoot);
  return {
    ok: true,
    project_id: projectId,
    task_history_path: memoryPath(projectId, "task_history.json").replace(/\\/g, "/"),
    copied_adr_paths: copiedAdrPaths.map((item) => item.replace(/\\/g, "/")),
    coding_failure_memory: codingFailureMemory,
  };
}
