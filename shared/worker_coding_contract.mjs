import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const AUTHORITY_PATH = path.resolve(MODULE_DIR, "..", "configs", "registry", "worker_coding_task_classes.json");
const parsed = JSON.parse(fs.readFileSync(AUTHORITY_PATH, "utf8"));

export const WORKER_CODING_TASK_CLASSES = Array.isArray(parsed?.task_classes)
  ? parsed.task_classes.map((item) => String(item || "").trim()).filter(Boolean)
  : [];

export const WORKER_CODING_TASK_CLASS_SET = new Set(WORKER_CODING_TASK_CLASSES);
