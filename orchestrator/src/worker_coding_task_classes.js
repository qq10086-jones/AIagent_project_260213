import fs from "fs";
import path from "path";

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

export function getDefaultWorkerCodingTaskClassesPath() {
  const candidates = [
    path.resolve(process.cwd(), "configs", "registry", "worker_coding_task_classes.json"),
    path.resolve(process.cwd(), "..", "configs", "registry", "worker_coding_task_classes.json"),
  ];
  for (const item of candidates) {
    if (fs.existsSync(item)) return item;
  }
  return candidates[0];
}

export function loadWorkerCodingTaskClassesOrThrow(filePath = getDefaultWorkerCodingTaskClassesPath()) {
  const parsed = readJson(filePath);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("worker coding task classes file must be an object");
  }
  if (!Array.isArray(parsed.task_classes)) {
    throw new Error("worker coding task classes file must define task_classes[]");
  }
  const taskClasses = parsed.task_classes.map((item) => String(item || "").trim()).filter(Boolean);
  if (taskClasses.length === 0) {
    throw new Error("worker coding task classes file must not be empty");
  }
  return {
    path: filePath,
    version: String(parsed.version || ""),
    taskClasses,
    taskClassSet: new Set(taskClasses),
  };
}
