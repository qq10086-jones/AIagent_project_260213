import fs from "fs";
import path from "path";
import { execFile } from "child_process";

function execFileCapture(command, args, cwd) {
  return new Promise((resolve) => {
    execFile(command, args, { cwd, timeout: 20000 }, (error, stdout, stderr) => {
      resolve({
        ok: !error,
        stdout: String(stdout || ""),
        stderr: String(stderr || ""),
        exitCode: error?.code ?? 0,
      });
    });
  });
}

function writeAutoCommitArtifact(taskDir, payload) {
  let logPath = null;
  try {
    logPath = path.join(taskDir, `auto_commit_${Date.now()}.json`);
    fs.writeFileSync(logPath, JSON.stringify({
      generated_at: new Date().toISOString(),
      ...payload,
    }, null, 2), "utf8");
  } catch {
    logPath = null;
  }
  return {
    ...payload,
    logPath,
  };
}

export async function runPatchAutoCommit({
  workspaceRoot,
  filePath,
  taskId,
  taskDir,
  execFileImpl = execFileCapture,
}) {
  const gitDir = path.join(workspaceRoot, ".git");
  if (!fs.existsSync(gitDir)) {
    return writeAutoCommitArtifact(taskDir, {
      attempted: false,
      committed: false,
      status: "skipped",
      reason: "no_git_repo",
      file_path: filePath,
    });
  }

  const unmerged = await execFileImpl("git", ["diff", "--name-only", "--diff-filter=U"], workspaceRoot);
  if (!unmerged.ok) {
    return writeAutoCommitArtifact(taskDir, {
      attempted: true,
      committed: false,
      status: "failed",
      reason: "git_diff_failed",
      file_path: filePath,
      stderr: unmerged.stderr.trim(),
      stdout: unmerged.stdout.trim(),
    });
  }
  if (unmerged.stdout.trim()) {
    return writeAutoCommitArtifact(taskDir, {
      attempted: false,
      committed: false,
      status: "skipped",
      reason: "unresolved_merge_conflicts",
      file_path: filePath,
    });
  }

  const addResult = await execFileImpl("git", ["add", "--", filePath], workspaceRoot);
  if (!addResult.ok) {
    return writeAutoCommitArtifact(taskDir, {
      attempted: true,
      committed: false,
      status: "failed",
      reason: "git_add_failed",
      file_path: filePath,
      stderr: addResult.stderr.trim(),
      stdout: addResult.stdout.trim(),
    });
  }

  const diffCached = await execFileImpl("git", ["diff", "--cached", "--quiet", "--", filePath], workspaceRoot);
  if (diffCached.ok) {
    return writeAutoCommitArtifact(taskDir, {
      attempted: true,
      committed: false,
      status: "skipped",
      reason: "no_staged_changes",
      file_path: filePath,
    });
  }

  const commitMessage = `coding_agent: task ${taskId || "unknown"} - updated ${filePath}`;
  const commitResult = await execFileImpl("git", ["commit", "-m", commitMessage], workspaceRoot);
  if (!commitResult.ok) {
    return writeAutoCommitArtifact(taskDir, {
      attempted: true,
      committed: false,
      status: "failed",
      reason: "git_commit_failed",
      file_path: filePath,
      stderr: commitResult.stderr.trim(),
      stdout: commitResult.stdout.trim(),
    });
  }

  return writeAutoCommitArtifact(taskDir, {
    attempted: true,
    committed: true,
    status: "committed",
    reason: null,
    file_path: filePath,
    commit_message: commitMessage,
    stdout: commitResult.stdout.trim(),
    stderr: commitResult.stderr.trim(),
  });
}
