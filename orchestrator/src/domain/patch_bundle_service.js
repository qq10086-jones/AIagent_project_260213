import fs from "fs";
import path from "path";

function buildPatchError(code, message, operationIndex = null, extra = {}) {
  const err = new Error(message);
  err.code = code;
  if (Number.isInteger(operationIndex)) err.operation_index = operationIndex;
  Object.assign(err, extra);
  return err;
}

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });
}

function normalizeWorkspaceRoot(workspaceRoot) {
  return path.resolve(String(workspaceRoot || "."));
}

function resolveTargetPath(workspaceRoot, targetFile, operationIndex) {
  const normalized = String(targetFile || "").replace(/\\/g, "/");
  if (!normalized || normalized.startsWith("/")) {
    throw buildPatchError(
      "PATCH_PATH_INVALID",
      `operation ${operationIndex}: target_file must be a repo-relative path`,
      operationIndex
    );
  }
  const rootAbs = normalizeWorkspaceRoot(workspaceRoot);
  const targetAbs = path.resolve(rootAbs, normalized);
  const rel = path.relative(rootAbs, targetAbs);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    throw buildPatchError(
      "PATCH_PATH_TRAVERSAL",
      `operation ${operationIndex}: target_file escapes workspace root`,
      operationIndex,
      { target_file: normalized }
    );
  }
  return { rootAbs, targetAbs, relativePath: rel.replace(/\\/g, "/") };
}

function readTargetFile(targetAbs, operationIndex, targetFile) {
  if (!fs.existsSync(targetAbs)) {
    throw buildPatchError(
      "PATCH_TARGET_MISSING",
      `operation ${operationIndex}: target file '${targetFile}' not found`,
      operationIndex,
      { target_file: targetFile }
    );
  }
  return fs.readFileSync(targetAbs, "utf8");
}

function resolveAnchorIndex(content, anchor, anchorContextBefore, operationIndex, targetFile, label) {
  if (!String(anchor || "").length) {
    throw buildPatchError(
      "PATCH_ANCHOR_INVALID",
      `operation ${operationIndex}: ${label} anchor must be a non-empty string`,
      operationIndex,
      { target_file: targetFile }
    );
  }

  if (String(anchorContextBefore || "").length > 0) {
    const composite = `${anchorContextBefore}${anchor}`;
    const compositeIndex = content.indexOf(composite);
    if (compositeIndex === -1) {
      throw buildPatchError(
        "PATCH_ANCHOR_NOT_FOUND",
        `operation ${operationIndex}: ${label} anchor not found with context in '${targetFile}'`,
        operationIndex,
        { target_file: targetFile, anchor }
      );
    }
    return compositeIndex + anchorContextBefore.length;
  }

  const firstIndex = content.indexOf(anchor);
  if (firstIndex === -1) {
    throw buildPatchError(
      "PATCH_ANCHOR_NOT_FOUND",
      `operation ${operationIndex}: ${label} anchor not found in '${targetFile}'`,
      operationIndex,
      { target_file: targetFile, anchor }
    );
  }
  const secondIndex = content.indexOf(anchor, firstIndex + anchor.length);
  if (secondIndex !== -1) {
    throw buildPatchError(
      "PATCH_ANCHOR_AMBIGUOUS",
      `operation ${operationIndex}: ${label} anchor is ambiguous in '${targetFile}'`,
      operationIndex,
      { target_file: targetFile, anchor }
    );
  }
  return firstIndex;
}

function applyOperationToContent(content, operation, operationIndex) {
  const targetFile = String(operation.target_file || "");
  const type = String(operation.type || "");

  if (type === "insert_after_anchor") {
    const anchorIndex = resolveAnchorIndex(
      content,
      operation.anchor,
      operation.anchor_context_before,
      operationIndex,
      targetFile,
      "insert_after_anchor"
    );
    const insertOffset = anchorIndex + String(operation.anchor).length;
    return `${content.slice(0, insertOffset)}${String(operation.content || "")}${content.slice(insertOffset)}`;
  }

  if (type === "replace_range" || type === "delete_range") {
    const startIndex = resolveAnchorIndex(
      content,
      operation.anchor_start,
      operation.anchor_context_before,
      operationIndex,
      targetFile,
      `${type}.anchor_start`
    );
    const searchFrom = startIndex + String(operation.anchor_start).length;
    const endRelativeIndex = content.indexOf(String(operation.anchor_end || ""), searchFrom);
    if (endRelativeIndex === -1) {
      throw buildPatchError(
        "PATCH_ANCHOR_NOT_FOUND",
        `operation ${operationIndex}: ${type}.anchor_end not found in '${targetFile}'`,
        operationIndex,
        { target_file: targetFile, anchor_end: operation.anchor_end }
      );
    }
    const endIndex = endRelativeIndex + String(operation.anchor_end).length;
    const replacement = type === "replace_range" ? String(operation.content || "") : "";
    return `${content.slice(0, startIndex)}${replacement}${content.slice(endIndex)}`;
  }

  throw buildPatchError(
    "PATCH_OPERATION_UNSUPPORTED",
    `operation ${operationIndex}: unsupported operation type '${type}'`,
    operationIndex,
    { target_file: targetFile }
  );
}

export function createPatchBundleService({ workspaceRoot }) {
  const rootAbs = normalizeWorkspaceRoot(workspaceRoot);

  function applyPatchBundleFromFile(bundlePath) {
    const absoluteBundlePath = path.resolve(String(bundlePath || ""));
    if (!fs.existsSync(absoluteBundlePath)) {
      throw buildPatchError("PATCH_BUNDLE_FILE_MISSING", `patch bundle file not found: ${absoluteBundlePath}`);
    }
    const bundle = JSON.parse(fs.readFileSync(absoluteBundlePath, "utf8"));
    if (String(bundle?.mode || "") === "full_file_fallback") {
      return {
        ok: true,
        bundle_id: String(bundle?.bundle_id || ""),
        step_id: String(bundle?.step_id || ""),
        mode: "full_file_fallback",
        written_files: [],
        operation_count: Array.isArray(bundle?.operations) ? bundle.operations.length : 0,
        summary: String(bundle?.summary || ""),
        bundle_path: absoluteBundlePath,
        bundle,
      };
    }
    const result = applyPatchBundle(bundle);
    return { ...result, bundle_path: absoluteBundlePath, bundle };
  }

  function applyPatchBundle(bundle) {
    const operations = Array.isArray(bundle?.operations) ? bundle.operations : [];
    if (operations.length === 0) {
      throw buildPatchError("PATCH_BUNDLE_INVALID", "patch bundle must include at least one operation");
    }

    const fileContents = new Map();
    const writtenFiles = new Set();

    for (let index = 0; index < operations.length; index += 1) {
      const operation = operations[index] || {};
      const type = String(operation.type || "");
      const { targetAbs, relativePath } = resolveTargetPath(rootAbs, operation.target_file, index);

      if (type === "create_file") {
        if (fileContents.has(relativePath) || fs.existsSync(targetAbs)) {
          throw buildPatchError(
            "PATCH_CREATE_CONFLICT",
            `operation ${index}: target file '${relativePath}' already exists`,
            index,
            { target_file: relativePath }
          );
        }
        fileContents.set(relativePath, String(operation.file_content || ""));
        writtenFiles.add(relativePath);
        continue;
      }

      let currentContent = fileContents.get(relativePath);
      if (currentContent === undefined) {
        currentContent = readTargetFile(targetAbs, index, relativePath);
      }
      const nextContent = applyOperationToContent(currentContent, { ...operation, target_file: relativePath }, index);
      fileContents.set(relativePath, nextContent);
      writtenFiles.add(relativePath);
    }

    for (const relativePath of writtenFiles) {
      const targetAbs = path.resolve(rootAbs, relativePath);
      ensureDir(path.dirname(targetAbs));
      fs.writeFileSync(targetAbs, fileContents.get(relativePath) || "", "utf8");
    }

    return {
      ok: true,
      bundle_id: String(bundle?.bundle_id || ""),
      step_id: String(bundle?.step_id || ""),
      mode: String(bundle?.mode || ""),
      written_files: Array.from(writtenFiles),
      operation_count: operations.length,
      summary: String(bundle?.summary || ""),
    };
  }

  return {
    applyPatchBundle,
    applyPatchBundleFromFile,
  };
}
