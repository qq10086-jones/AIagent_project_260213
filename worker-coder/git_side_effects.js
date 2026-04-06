import fs from "fs";
import path from "path";
import { execFile } from "child_process";
import https from "https";

import { GIT_EXEC_TIMEOUT_MS } from './constants.js';

function execFileCapture(command, args, cwd) {
  return new Promise((resolve) => {
    execFile(command, args, { cwd, timeout: GIT_EXEC_TIMEOUT_MS }, (error, stdout, stderr) => {
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
  } catch (err) {
    console.warn("[git_side_effects] Failed to write auto-commit artifact:", err.message);
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

// ─── GitHub 交付：push 分支 + 创建 PR ──────────────────────────────────────

function githubApiRequest(method, path, body, token) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const req = https.request({
      hostname: "api.github.com",
      path,
      method,
      headers: {
        "Authorization": `Bearer ${token}`,
        "Accept": "application/vnd.github+json",
        "User-Agent": "nexus-coding-agent/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
        ...(data ? { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) } : {}),
      },
    }, (res) => {
      let raw = "";
      res.on("data", (c) => raw += c);
      res.on("end", () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(raw) }); }
        catch (_parseErr) { resolve({ status: res.statusCode, body: raw }); }
      });
    });
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

export async function pushBranchToGitHub({
  workspaceRoot,
  runId,
  goal = "",
  githubRepo,   // "owner/repo"
  githubToken,
  branchPrefix = "nexus/",
  execFileImpl = execFileCapture,
}) {
  const repo = String(githubRepo || process.env.GITHUB_REPO || "").trim();
  const token = String(githubToken || process.env.GITHUB_TOKEN || "").trim();

  if (!repo || !token) {
    return {
      ok: false,
      reason: "missing_github_config",
      message: "GITHUB_REPO 或 GITHUB_TOKEN 未配置，跳过 GitHub 交付",
    };
  }

  const branch = `${branchPrefix}run-${String(runId || "unknown").slice(0, 10)}`;
  const remoteUrl = `https://x-access-token:${token}@github.com/${repo}.git`;

  // 确保 git repo 存在
  if (!fs.existsSync(path.join(workspaceRoot, ".git"))) {
    const initRes = await execFileImpl("git", ["init"], workspaceRoot);
    if (!initRes.ok) return { ok: false, reason: "git_init_failed", stderr: initRes.stderr };
    await execFileImpl("git", ["config", "user.email", "nexus-agent@nexus.local"], workspaceRoot);
    await execFileImpl("git", ["config", "user.name", "Nexus Coding Agent"], workspaceRoot);
  }

  // 设置 / 更新 remote
  const remoteCheck = await execFileImpl("git", ["remote", "get-url", "origin"], workspaceRoot);
  if (remoteCheck.ok) {
    await execFileImpl("git", ["remote", "set-url", "origin", remoteUrl], workspaceRoot);
  } else {
    await execFileImpl("git", ["remote", "add", "origin", remoteUrl], workspaceRoot);
  }

  // 创建并切换到交付分支
  await execFileImpl("git", ["checkout", "-B", branch], workspaceRoot);

  // 提交所有 impl 产物
  await execFileImpl("git", ["add", "-A"], workspaceRoot);
  const diffCheck = await execFileImpl("git", ["diff", "--cached", "--quiet"], workspaceRoot);
  if (diffCheck.ok) {
    // 没有变更，仍然推送（可能是之前已提交的）
  } else {
    const shortGoal = goal.slice(0, 60).replace(/[^\w\s-]/g, "");
    const commitMsg = `nexus: deliver coding task - ${shortGoal || runId}`;
    const commitRes = await execFileImpl("git", ["commit", "-m", commitMsg], workspaceRoot);
    if (!commitRes.ok) {
      return { ok: false, reason: "git_commit_failed", stderr: commitRes.stderr };
    }
  }

  // Push
  const pushRes = await execFileImpl(
    "git", ["push", "origin", branch, "--force"],
    workspaceRoot,
  );
  if (!pushRes.ok) {
    return { ok: false, reason: "git_push_failed", stderr: pushRes.stderr, stdout: pushRes.stdout };
  }

  const branchUrl = `https://github.com/${repo}/tree/${branch}`;

  // 创建 PR（如果失败不阻断主流程）
  let prUrl = null;
  try {
    const [owner, repoName] = repo.split("/");
    // 获取默认分支
    const repoInfo = await githubApiRequest("GET", `/repos/${owner}/${repoName}`, null, token);
    const baseBranch = repoInfo.body?.default_branch || "main";

    const prRes = await githubApiRequest(
      "POST",
      `/repos/${owner}/${repoName}/pulls`,
      {
        title: `[Nexus] ${goal.slice(0, 72) || `Run ${runId}`}`,
        body: `自动生成的编码交付\n\n**Run ID**: \`${runId}\`\n\n由 Nexus Coding Team 自动创建。`,
        head: branch,
        base: baseBranch,
      },
      token,
    );
    if (prRes.status === 201 && prRes.body?.html_url) {
      prUrl = prRes.body.html_url;
    }
  } catch (prErr) {
    console.warn("[git_side_effects] PR creation failed (non-blocking):", prErr.message);
  }

  const summaryLines = [
    `✅ 代码已推送至 GitHub`,
    `📦 分支：${branch}`,
    `🔗 ${branchUrl}`,
    ...(prUrl ? [`📋 Pull Request：${prUrl}`] : []),
  ];

  return {
    ok: true,
    branch,
    branch_url: branchUrl,
    pr_url: prUrl,
    github_repo: repo,
    summary: summaryLines.join("\n"),
    artifacts: [
      { name: "GitHub 分支", url: branchUrl },
      ...(prUrl ? [{ name: "Pull Request", url: prUrl }] : []),
    ],
  };
}
