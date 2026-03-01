import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { applyEditBlocks } from './patch_manager.js';
import { v4 as uuidv4 } from 'uuid';
import { runCodexTask } from './adapters/codex_adapter.js';
import { runOpenCodeTask } from './adapters/opencode_adapter.js';

/**
 * Service to handle all coding-related business logic.
 */
export const CodingService = {
    /**
     * Applies a patch and records artifacts.
     */
    applyPatch: async (params) => {
        const { workspaceRoot, file_path, edit_block, task_id, run_id } = params;

        const result = applyEditBlocks(workspaceRoot, file_path, edit_block);

        if (result.success) {
            try {
                const runDir = path.join(workspaceRoot, 'artifacts', 'runs', run_id || 'default');
                if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });

                // 1. Record in timeline
                const timelinePath = path.join(runDir, 'timeline.md');
                const timestamp = new Date().toISOString();
                const logEntry = `- ${timestamp} | task: ${task_id || 'unknown'} | File patched: ${file_path} | STATUS: PASS | Detail: ${result.detail || 'Applied'}
`;
                fs.appendFileSync(timelinePath, logEntry);

                // 2. Save raw patch artifact
                const taskDir = path.join(runDir, `task_${task_id || 'unknown'}`);
                if (!fs.existsSync(taskDir)) fs.mkdirSync(taskDir, { recursive: true });
                fs.writeFileSync(path.join(taskDir, `patch_${Date.now()}.diff`), edit_block);

                // 3. Optional Auto-commit
                if (fs.existsSync(path.join(workspaceRoot, '.git'))) {
                    exec(`git add "${file_path}" && git commit -m "coding_agent: task ${task_id || 'unknown'} - updated ${file_path}"`, 
                         { cwd: workspaceRoot }, (err) => {
                        if (err) console.warn("[CodingService] Auto-commit skipped/failed:", err.message);
                    });
                }
            } catch (fsErr) {
                console.error("[CodingService] Artifact logging failed:", fsErr.message);
            }
        }
        return result;
    },

    /**
     * Executes a command and records timeline results.
     */
    executeCommand: async (params) => {
        const { workspaceRoot, command, run_id, task_id } = params;

        return new Promise((resolve) => {
            exec(command, { cwd: workspaceRoot, timeout: 30000 }, (error, stdout, stderr) => {
                const status = error ? "FAIL" : "PASS";
                
                try {
                    const runDir = path.join(workspaceRoot, 'artifacts', 'runs', run_id || 'default');
                    if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
                    
                    const timelinePath = path.join(runDir, 'timeline.md');
                    const timestamp = new Date().toISOString();
                    const logEntry = `- ${timestamp} | task: ${task_id || 'unknown'} | Executed: \\\`${command}\\\` | STATUS: ${status}\n`;
                    fs.appendFileSync(timelinePath, logEntry);
                } catch (fsErr) {
                    console.warn("[CodingService] Failed to write timeline log:", fsErr.message);
                }

                if (error) {
                    resolve({ 
                        ok: false, 
                        exit_code: error.code,
                        stdout: stdout.toString(), 
                        stderr: stderr.toString(),
                        error: error.message
                    });
                } else {
                    resolve({ 
                        ok: true, 
                        exit_code: 0,
                        stdout: stdout.toString(), 
                        stderr: stderr.toString() 
                    });
                }
            });
        });
    },

    /**
     * Delegates coding tasks to Codex-compatible CLI adapter and records artifacts.
     */
    delegateTask: async (params) => {
        const {
            workspaceRoot,
            task_prompt,
            provider = "auto",
            model = null,
            run_id,
            task_id,
            max_runtime_s = 600,
            codex_command = null,
            opencode_command = null,
        } = params;

        const providerRequested = String(provider || "auto").toLowerCase();
        const supportedProviders = new Set(["auto", "opencode", "codex"]);
        if (!supportedProviders.has(providerRequested)) {
            return {
                ok: false,
                error: `Unsupported provider '${providerRequested}'. Use auto/opencode/codex.`,
                diagnostics: { error_code: "E_PROVIDER_UNAVAILABLE", provider_requested: providerRequested }
            };
        }
        const preferredProvider = providerRequested === "auto" ? "opencode" : providerRequested;

        const runDir = path.join(workspaceRoot, 'artifacts', 'runs', run_id || 'default');
        const taskDir = path.join(runDir, `task_${task_id || 'unknown'}`);
        try {
            if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
            if (!fs.existsSync(taskDir)) fs.mkdirSync(taskDir, { recursive: true });
        } catch (e) {
            return { ok: false, error: `Failed to prepare artifacts dir: ${e.message}` };
        }

        const baselineFiles = await getGitStatusFiles(workspaceRoot);
        const started = new Date().toISOString();
        const runByProvider = async (providerName) => {
            if (providerName === "opencode") {
                return runOpenCodeTask({
                    workspaceRoot,
                    taskPrompt: task_prompt,
                    model,
                    maxRuntimeS: max_runtime_s,
                    opencodeCommand: opencode_command,
                });
            }
            return runCodexTask({
                workspaceRoot,
                taskPrompt: task_prompt,
                model,
                maxRuntimeS: max_runtime_s,
                codexCommand: codex_command,
            });
        };

        let result = await runByProvider(preferredProvider);
        let fallbackFrom = null;
        if (
            preferredProvider === "opencode" &&
            String(result?.diagnostics?.error_code || "") === "E_PROVIDER_UNAVAILABLE"
        ) {
            fallbackFrom = "opencode";
            result = await runByProvider("codex");
        }

        const stdoutPath = path.join(taskDir, `delegate_stdout_${Date.now()}.log`);
        const stderrPath = path.join(taskDir, `delegate_stderr_${Date.now()}.log`);
        const redactedStdout = redactSensitiveText(result.stdout || "");
        const redactedStderr = redactSensitiveText(result.stderr || "");
        try {
            fs.writeFileSync(stdoutPath, redactedStdout, "utf8");
            fs.writeFileSync(stderrPath, redactedStderr, "utf8");
        } catch {}

        const gitSummary = await gatherGitSummary(workspaceRoot, taskDir, baselineFiles);
        const summary = {
            ok: !!result.ok,
            provider_used: result.provider_used || preferredProvider,
            model_used: result.model_used || null,
            summary: result.ok
                ? `${result.provider_used || preferredProvider} delegation finished.`
                : `${result.provider_used || preferredProvider} delegation failed: ${result.error || "unknown error"}`,
            files_changed: gitSummary.filesChanged,
            diff_stats: gitSummary.diffStats,
            test_result: "skipped",
            git: gitSummary.git,
            rollback_performed: false,
            artifacts: {
                diff_bundle: gitSummary.diffPath,
                patch_file: null,
                test_log: null,
                raw_stdout: stdoutPath,
                raw_stderr: stderrPath,
            },
            diagnostics: {
                ...(result.diagnostics || {}),
                provider_requested: providerRequested,
                fallback_from: fallbackFrom,
                parse_error: false,
                truncated: false,
            },
            error: redactSensitiveText(result.error || "") || null,
            command_used: result.command_used || null,
            command_source: result.command_source || "unknown",
            started_at: started,
            finished_at: new Date().toISOString(),
        };

        try {
            const timelinePath = path.join(runDir, 'timeline.md');
            const status = summary.ok ? "PASS" : "FAIL";
            const line = `- ${new Date().toISOString()} | task: ${task_id || 'unknown'} | Delegated: ${summary.provider_used} (requested=${providerRequested}${fallbackFrom ? `,fallback_from=${fallbackFrom}` : ""}) | STATUS: ${status}\n`;
            fs.appendFileSync(timelinePath, line);
        } catch {}

        return summary;
    },

    /**
     * Placeholder for starting a task.
     */
    startTask: async (task_prompt, workspaceRoot) => {
        const run_id = uuidv4();
        const runDir = path.join(workspaceRoot, 'artifacts', 'runs', run_id);

        try {
            if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
            
            const initialState = {
                run_id,
                status: "INIT",
                task_prompt,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString()
            };
            
            fs.writeFileSync(path.join(runDir, 'state.json'), JSON.stringify(initialState, null, 2));
            fs.writeFileSync(path.join(runDir, 'timeline.md'), `# Timeline for Run ${run_id}

`);

            return { ok: true, run_id, state: initialState };
        } catch (err) {
            return { ok: false, error: err.message };
        }
    }
};

async function execCapture(command, cwd) {
    return new Promise((resolve) => {
        exec(command, { cwd, timeout: 20000 }, (error, stdout, stderr) => {
            resolve({
                ok: !error,
                stdout: (stdout || "").toString(),
                stderr: (stderr || "").toString(),
            });
        });
    });
}

async function gatherGitSummary(workspaceRoot, taskDir, baselineFiles = new Set()) {
    const inRepo = await execCapture("git rev-parse --is-inside-work-tree", workspaceRoot);
    if (!inRepo.ok || !inRepo.stdout.trim().includes("true")) {
        return {
            filesChanged: [],
            diffStats: { added: 0, deleted: 0, files: 0 },
            diffPath: null,
            git: { base_ref: "unknown", branch: "unknown", commit_sha: null, dirty: false },
        };
    }

    const branch = await execCapture("git rev-parse --abbrev-ref HEAD", workspaceRoot);
    const commit = await execCapture("git rev-parse HEAD", workspaceRoot);
    const status = await execCapture("git status --porcelain -uall", workspaceRoot);
    const numstat = await execCapture("git diff --numstat", workspaceRoot);
    const diff = await execCapture("git diff", workspaceRoot);
    const currentFiles = await getGitStatusFiles(workspaceRoot);

    const filesChangedRaw = [...currentFiles].filter((f) => !baselineFiles.has(f));
    const filesChanged = filesChangedRaw.filter((f) => !String(f).startsWith("artifacts/runs/"));
    const filesChangedSet = new Set(filesChanged);

    let added = 0;
    let deleted = 0;
    for (const line of (numstat.stdout || "").split(/\r?\n/)) {
        const s = line.trim();
        if (!s) continue;
        const parts = s.split(/\s+/);
        if (parts.length >= 3) {
            const filePath = parts.slice(2).join(" ");
            if (!filesChangedSet.has(filePath)) {
                continue;
            }
            const a = Number(parts[0]);
            const d = Number(parts[1]);
            if (Number.isFinite(a)) added += a;
            if (Number.isFinite(d)) deleted += d;
        }
    }

    let diffPath = null;
    try {
        diffPath = path.join(taskDir, `delegate_diff_${Date.now()}.patch`);
        if (filesChanged.length > 0) {
            const fileArgs = filesChanged.map((f) => `"${f.replace(/"/g, '\\"')}"`).join(" ");
            const scopedDiff = await execCapture(`git diff -- ${fileArgs}`, workspaceRoot);
            fs.writeFileSync(diffPath, scopedDiff.stdout || "", "utf8");
        } else {
            fs.writeFileSync(diffPath, diff.stdout || "", "utf8");
        }
    } catch {
        diffPath = null;
    }

    return {
        filesChanged,
        diffStats: { added, deleted, files: filesChanged.length },
        diffPath,
        git: {
            base_ref: "HEAD",
            branch: branch.stdout.trim() || "unknown",
            commit_sha: commit.stdout.trim() || null,
            dirty: !!status.stdout.trim(),
        },
    };
}

async function getGitStatusFiles(workspaceRoot) {
    const status = await execCapture("git status --porcelain -uall", workspaceRoot);
    if (!status.ok) return new Set();
    const files = new Set();
    for (const line of (status.stdout || "").split(/\r?\n/)) {
        const s = line.trim();
        if (!s) continue;
        const m = s.match(/^[A-Z? ]{2}\s+(.+)$/);
        if (m && m[1]) {
            files.add(m[1].trim());
        }
    }
    return files;
}

function redactSensitiveText(value) {
    let text = String(value || "");
    if (!text) return text;
    const rules = [
        { re: /\bsk-[A-Za-z0-9_-]{20,}\b/g, to: "[REDACTED_OPENAI_KEY]" },
        { re: /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, to: "[REDACTED_GITHUB_TOKEN]" },
        { re: /\bAIza[0-9A-Za-z\-_]{20,}\b/g, to: "[REDACTED_GOOGLE_API_KEY]" },
        { re: /\bAKIA[0-9A-Z]{16}\b/g, to: "[REDACTED_AWS_ACCESS_KEY]" },
        { re: /\b(?:xoxb|xoxp|xoxa|xoxr)-[A-Za-z0-9-]{10,}\b/g, to: "[REDACTED_SLACK_TOKEN]" },
        { re: /\b(token|api[_-]?key|secret|password)\s*[:=]\s*['"]?[A-Za-z0-9_\-\/+=.]{8,}['"]?/gi, to: "$1=[REDACTED]" },
    ];
    for (const rule of rules) {
        text = text.replace(rule.re, rule.to);
    }
    return text;
}
