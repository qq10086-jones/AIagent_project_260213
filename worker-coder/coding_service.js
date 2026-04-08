import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import crypto from 'crypto';
import { applyEditBlocks } from './patch_manager.js';
import { executeCodingAdapter } from './coding_executor_runtime.js';
import { runPatchAutoCommit } from './git_side_effects.js';
import {
    buildDelegateFailureSummary,
    persistCodingFailureMemory,
    summarizeTerminalText,
} from './failure_memory.js';
import {
    ensureExpectedArtifacts,
} from './artifact_scaffold.js';
import {
    captureScopedSnapshot,
    ensureImplementationDelta,
    gatherGitSummary,
} from './scoped_delta.js';
import {
    materializeIsolationWorkspace,
    cleanupIsolationWorkspace,
} from './isolation_workspace.js';
import {
    promoteIsolatedChanges,
} from './promotion_workspace.js';
import {
    clampInt,
    runStaticChecks,
} from './static_checks.js';
import {
    buildAutoFixPrompt,
    writePromptContractArtifact,
} from './prompt_contract.js';
import {
    buildAttemptedFixes,
    buildFinalFailureSummary,
    shouldRetryAutoFix,
} from './retry_policy.js';
import {
    redactSensitiveText,
    runVerificationPlan,
    validateSafeCommand,
} from './verification_runner.js';
import {
    normalizeRelPath,
    validateAllowedTargetPaths,
    validateChangedFilesWithinScope,
    validateRequestedWrite,
} from './scope_guard.js';
import { buildTaskContractMetadata, normalizeLineage } from './task_contract.js';
import { canFix, applySurgicalPatches } from './surgical_patch.js';
import { buildRefinementContext } from './refinement_context_builder.js';

import {
    getWorkflowStepHandoff,
    validateWorkflowStepArtifacts,
    validateCodingTeamHandoff,
} from './step_artifact_contract.js';

import {
    EXEC_TIMEOUT_MS,
    ARTIFACTS_DIR,
    RUNS_DIR,
    SUPPORTED_PROVIDERS,
    DEFAULT_PROVIDER_RESOLVED,
    MIN_WALL_CLOCK_S,
    MAX_WALL_CLOCK_S,
    FALLBACK_WALL_CLOCK_S,
    ErrorCode,
} from './constants.js';

// --- Request ID tracing ---
function makeRequestId(taskId, runId) {
    return `[req:${String(runId || "?").slice(0, 8)}/${String(taskId || "?")}]`;
}

function buildRunDir(workspaceRoot, runId) {
    return path.join(workspaceRoot, ARTIFACTS_DIR, RUNS_DIR, runId || 'default');
}

export function salvageWorkflowArtifactFailure({ workspaceRoot, artifactRoot, expectedArtifacts, stepId, taskPrompt, result }) {
    const safeStepId = String(stepId || "");
    const timeoutOnlySteps = new Set(["pm_spec", "arch_design", "release_pack"]);
    const implementationSteps = new Set(["impl_be", "impl_fe"]);
    if (!timeoutOnlySteps.has(safeStepId) && !implementationSteps.has(safeStepId)) return null;
    if (timeoutOnlySteps.has(safeStepId) && !result?.diagnostics?.timeout) return null;
    const handoff = getWorkflowStepHandoff(safeStepId);
    const scaffoldExpectedArtifacts = buildStepContractExpectedArtifacts({
        stepId: safeStepId,
        expectedArtifacts,
    });
    const scaffold = ensureExpectedArtifacts({
        workspaceRoot,
        artifactRoot,
        expectedArtifacts: scaffoldExpectedArtifacts,
        stepId,
        taskPrompt,
    });
    const roleValidation = validateWorkflowStepArtifacts({ workspaceRoot, artifactRoot, stepId: safeStepId });
    const handoffValidation = handoff
        ? validateCodingTeamHandoff({
            workspaceRoot,
            artifactRoot,
            handoff,
        })
        : { checked: false, ok: true };
    if (!roleValidation?.ok || !handoffValidation?.ok) {
        return null;
    }
    return {
        scaffold,
        roleValidation,
        handoffValidation,
        salvageReason: implementationSteps.has(safeStepId) ? "artifact_contract" : "timeout",
    };
}

export function buildStepContractExpectedArtifacts({ stepId, expectedArtifacts }) {
    const handoff = getWorkflowStepHandoff(stepId);
    return Array.from(new Set([
        ...(Array.isArray(expectedArtifacts) ? expectedArtifacts : []),
        ...(Array.isArray(handoff?.required_artifacts) ? handoff.required_artifacts : []),
        String(handoff?.typed_handoff?.file || "").trim(),
    ].filter(Boolean)));
}


export function payloadToAdapterRequest({
    provider,
    task_prompt,
    artifact_root,
    expected_artifacts,
    step_id,
    target_paths,
    verification_command,
    verification_plan,
    wall_clock_timeout_s,
    execution_adapter_packet,
    context_packet,
    repo_map,
    model,
    execution_lane,
    allow_provider_fallback,
    runtime_coder_config,
    run_id,
    task_id,
    artifact_workspace_root,
}) {
    return {
        adapter_type: "coding_executor",
        provider: String(provider || "opencode"),
        task_type: "coding_execution",
        payload: {
            step_id: String(step_id || ""),
            task_prompt: String(task_prompt || ""),
            artifact_root: String(artifact_root || ""),
            expected_artifacts: Array.isArray(expected_artifacts) ? expected_artifacts : [],
            target_paths: Array.isArray(target_paths) ? target_paths : [],
            verification_command: String(verification_command || ""),
            verification_plan: Array.isArray(verification_plan) ? verification_plan : [],
            wall_clock_timeout_s: Number(wall_clock_timeout_s || 0),
            execution_adapter_packet: execution_adapter_packet || null,
            context_packet: context_packet || null,
            repo_map: repo_map || null,
            model_hint: String(model || ""),
            execution_lane: String(execution_lane || ""),
            allow_provider_fallback: Boolean(allow_provider_fallback),
            runtime_coder_config: runtime_coder_config || null,
        },
        context: {
            run_id: String(run_id || ""),
            task_id: String(task_id || ""),
            artifact_workspace_root: String(artifact_workspace_root || ""),
        },
    };
}

export function requiresScopedTargetPaths(stepId, targetPaths) {
    const safeStepId = String(stepId || "").trim().toLowerCase();
    const safeTargetPaths = Array.isArray(targetPaths) ? targetPaths.filter(Boolean) : [];
    if (safeTargetPaths.length > 0) return true;
    return safeStepId === "impl_be" || safeStepId === "impl_fe";
}

/**
 * Service to handle all coding-related business logic.
 */
export const CodingService = {
    /**
     * Applies a patch and records artifacts.
     */
    applyPatch: async (params) => {
        const { workspaceRoot, file_path, edit_block, task_id, run_id, target_paths = [] } = params;
        const reqId = makeRequestId(task_id, run_id);

        const scopeCheck = validateRequestedWrite({
            workspaceRoot,
            targetPath: file_path,
            allowedTargetPaths: target_paths,
        });
        if (!scopeCheck.ok) {
            return {
                success: false,
                message: scopeCheck.error,
                error_code: ErrorCode.UNAUTHORIZED_WRITE,
            };
        }

        const result = applyEditBlocks(workspaceRoot, file_path, edit_block);

        if (result.success) {
            try {
                const runDir = buildRunDir(workspaceRoot, run_id);
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
                result.diagnostics = result.diagnostics || {};
                result.diagnostics.auto_commit = await runPatchAutoCommit({
                    workspaceRoot,
                    filePath: file_path,
                    taskId: task_id,
                    taskDir,
                });
            } catch (fsErr) {
                console.error(`${reqId} [CodingService] Artifact logging failed:`, fsErr.message, fsErr.stack);
            }
        }
        return result;
    },

    /**
     * Executes a command and records timeline results.
     */
    executeCommand: async (params) => {
        const { workspaceRoot, command, artifact_root = "", expected_artifacts = [], step_id = "", task_prompt = "", run_id, task_id } = params;
        const reqId = makeRequestId(task_id, run_id);

        const cmdCheck = validateSafeCommand(command);
        if (!cmdCheck.ok) {
            return {
                ok: false,
                error: cmdCheck.error,
                error_code: ErrorCode.COMMAND_VALIDATION_FAILED,
            };
        }

        return new Promise((resolve) => {
            exec(command, { cwd: workspaceRoot, timeout: EXEC_TIMEOUT_MS }, (error, stdout, stderr) => {
                const status = error ? "FAIL" : "PASS";
                let scaffold = null;
                if (!error) {
                    scaffold = ensureExpectedArtifacts({
                        workspaceRoot,
                        artifactRoot: artifact_root,
                        expectedArtifacts: expected_artifacts,
                        stepId: step_id,
                        taskPrompt: task_prompt,
                    });
                }
                
                try {
                    const runDir = buildRunDir(workspaceRoot, run_id);
                    if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });

                    const timelinePath = path.join(runDir, 'timeline.md');
                    const timestamp = new Date().toISOString();
                    const logEntry = `- ${timestamp} | task: ${task_id || 'unknown'} | Executed: \\\`${command}\\\` | STATUS: ${status}\n`;
                    fs.appendFileSync(timelinePath, logEntry);
                } catch (fsErr) {
                    console.warn(`${reqId} [CodingService] Failed to write timeline log:`, fsErr.message);
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
                        stderr: stderr.toString(),
                        diagnostics: {
                            artifact_scaffold: scaffold || null,
                        },
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
            artifact_root = "",
            expected_artifacts = [],
            step_id = "",
            target_paths = [],
            provider = "auto",
            model = null,
            execution_lane = null,
            allow_provider_fallback = false,
            runtime_coder_config = null,
            run_id,
            task_id,
            max_runtime_s = 600,
            max_attempts = 1,
            same_error_repeat_limit = 2,
            wall_clock_timeout_s = 0,
            codex_command = null,
            opencode_command = null,
            verification_command = "",
            verification_plan = null,
            execution_adapter_packet = null,
            context_packet = null,
            repo_map = null,
            task_class = null,
            beta_template_id = null,
            context_envelope = null,
            lineage = null,
            on_runtime_event = null,
            redis = null,
        } = params;
        const reqId = makeRequestId(task_id, run_id);
        const taskContract = buildTaskContractMetadata({
            taskClass: task_class,
            betaTemplateId: beta_template_id,
            contextEnvelope: context_envelope,
        });
        const refinementReentryEnabled = !!(runtime_coder_config?.refinement_reentry_enabled);
        const normalizedLineage = refinementReentryEnabled ? normalizeLineage(lineage) : null;

        const providerRequested = String(provider || "auto").toLowerCase();
        if (!SUPPORTED_PROVIDERS.has(providerRequested)) {
            const failureMemory = persistCodingFailureMemory({
                workspaceRoot,
                runId: run_id,
                taskId: task_id,
                stepId: step_id,
                providerRequested: providerRequested,
                providerUsed: null,
                targetPaths: target_paths,
                failedPhase: "provider_validation",
                terminalOutcome: "failed",
                errorCode: ErrorCode.PROVIDER_UNAVAILABLE,
                error: `Unsupported provider '${providerRequested}'. Use auto/opencode/codex.`,
                attemptedFixes: [],
                filesChanged: [],
                artifactPaths: {},
                taskClass: taskContract.task_class,
                betaTemplateId: taskContract.beta_template_id,
                contextEnvelope: taskContract.context_envelope,
                lineage: normalizedLineage,
            });
            return {
                ok: false,
                error: `Unsupported provider '${providerRequested}'. Use auto/opencode/codex.`,
                diagnostics: {
                    error_code: ErrorCode.PROVIDER_UNAVAILABLE,
                    provider_requested: providerRequested,
                    task_contract: taskContract,
                    coding_failure_memory: failureMemory,
                }
            };
        }
        const preferredProvider = providerRequested === "auto" ? DEFAULT_PROVIDER_RESOLVED : providerRequested;
        const effectiveTargetPaths = Array.isArray(execution_adapter_packet?.target_paths) && execution_adapter_packet.target_paths.length > 0
            ? execution_adapter_packet.target_paths
            : target_paths;
        const targetPathsRequired = requiresScopedTargetPaths(step_id, effectiveTargetPaths);
        if (targetPathsRequired) {
            const scopeRootsCheck = validateAllowedTargetPaths({
                workspaceRoot,
                allowedTargetPaths: effectiveTargetPaths,
            });
            if (!scopeRootsCheck.ok) {
                const failureMemory = persistCodingFailureMemory({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested: providerRequested,
                    providerUsed: null,
                    targetPaths: effectiveTargetPaths,
                    failedPhase: "scope_guard",
                    terminalOutcome: "failed",
                    errorCode: ErrorCode.UNAUTHORIZED_WRITE,
                    error: scopeRootsCheck.error,
                    attemptedFixes: [],
                    filesChanged: [],
                    artifactPaths: {},
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
                return {
                    ok: false,
                    error: scopeRootsCheck.error,
                    diagnostics: {
                        error_code: ErrorCode.UNAUTHORIZED_WRITE,
                        provider_requested: providerRequested,
                        target_paths: effectiveTargetPaths,
                        task_contract: taskContract,
                        coding_failure_memory: failureMemory,
                    },
                };
            }
        }

        const runDir = buildRunDir(workspaceRoot, run_id);
        const taskDir = path.join(runDir, `task_${task_id || 'unknown'}`);
        try {
            if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
            if (!fs.existsSync(taskDir)) fs.mkdirSync(taskDir, { recursive: true });
        } catch (e) {
            console.error(`${reqId} Failed to prepare artifacts dir:`, e.message, e.stack);
            return { ok: false, error: `Failed to prepare artifacts dir: ${e.message}` };
        }
        let isolationScaffold = null;
        try {
            isolationScaffold = materializeIsolationWorkspace({
                workspaceRoot,
                taskDir,
                targetPaths: effectiveTargetPaths,
            });
        } catch (err) {
            console.error(`${reqId} Isolation workspace materialization failed:`, err.message, err.stack);
            cleanupIsolationWorkspace(taskDir);
            isolationScaffold = {
                enabled: false,
                mode: "error",
                isolatedWorkspaceRoot: null,
                artifacts: {
                    execution_mode: null,
                    baseline_manifest: null,
                    isolated_workspace_manifest: null,
                },
                diagnostics: {
                    isolation_mode: "error",
                    enabled: false,
                    error: String(err?.message || err || "unknown isolation scaffold error"),
                },
            };
        }

        const baselineSnapshot = captureScopedSnapshot(workspaceRoot, effectiveTargetPaths);
        const executionWorkspaceRoot = isolationScaffold?.enabled && isolationScaffold?.isolatedWorkspaceRoot
            ? isolationScaffold.isolatedWorkspaceRoot
            : workspaceRoot;
        const isolationMode = String(isolationScaffold?.mode || "disabled");
        const executionBaselineSnapshot = isolationScaffold?.enabled
            ? captureScopedSnapshot(executionWorkspaceRoot, effectiveTargetPaths)
            : baselineSnapshot;
        const started = new Date().toISOString();
        const startedMs = Date.now();
        const maxAttemptsSafe = clampInt(max_attempts, 1, 3, 1);
        const sameErrorRepeatLimitSafe = clampInt(same_error_repeat_limit, 1, 3, 2);
        const rawWallClock = Number(wall_clock_timeout_s || 0);
        const wallClockTimeoutSafe = rawWallClock > 0
            ? Math.max(MIN_WALL_CLOCK_S, Math.min(MAX_WALL_CLOCK_S, Math.trunc(rawWallClock)))
            : Math.max(max_runtime_s || FALLBACK_WALL_CLOCK_S, FALLBACK_WALL_CLOCK_S);
        const attemptRecords = [];
        const errorCounts = new Map();
        let finalSummary = null;
        let finalFallbackFrom = null;
        let terminalRetryReason = "completed";
        const surgicalPatchEnabled = !!(runtime_coder_config?.surgical_patch_enabled);
        let surgicalPatchAttempted = false;
        let refinementDiagnostics = null;

        for (let attemptIndex = 1; attemptIndex <= maxAttemptsSafe; attemptIndex++) {
            const remainingMs = (startedMs + (wallClockTimeoutSafe * 1000)) - Date.now();
            if (remainingMs <= 0) {
                terminalRetryReason = "wall_clock_timeout_exhausted";
                finalSummary = buildDelegateFailureSummary({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested,
                    providerUsed: preferredProvider,
                    targetPaths: effectiveTargetPaths,
                    finalGitSummary: await gatherGitSummary(workspaceRoot, taskDir, baselineSnapshot, effectiveTargetPaths),
                    stdoutPath: null,
                    stderrPath: null,
                    staticCheck: null,
                    verification: null,
                    adapterResult: null,
                    verificationCommand: verification_command,
                    promptContract: null,
                    redactedStdout: "",
                    redactedStderr: "",
                    startedAt: started,
                    phase: "retry_budget",
                    summaryText: "wall clock timeout reached before another repair attempt could start",
                    testResult: "skipped",
                    errorCode: ErrorCode.WALL_CLOCK_TIMEOUT,
                    error: `wall_clock_timeout_s budget exhausted after ${Date.now() - startedMs}ms`,
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
                break;
            }
            const priorFailure = attemptRecords.length > 0 ? attemptRecords[attemptRecords.length - 1] : null;
            let promptSeed = attemptIndex === 1
                ? task_prompt
                : buildAutoFixPrompt({
                    originalPrompt: task_prompt,
                    previousFailure: priorFailure,
                    attemptIndex,
                });

            // --- Phase 1.5: Refinement Re-entry context injection ---
            if (normalizedLineage && attemptIndex === 1) {
                const refinementCtx = buildRefinementContext({
                    lineage: normalizedLineage,
                    workspaceRoot: executionWorkspaceRoot,
                    targetPaths: effectiveTargetPaths,
                });
                if (refinementCtx.contextBlock) {
                    promptSeed = `${refinementCtx.contextBlock}\n\n${promptSeed}`;
                }
                refinementDiagnostics = refinementCtx.diagnostics;
            }

            const promptContract = writePromptContractArtifact({
                workspaceRoot,
                taskDir,
                attemptIndex,
                stepId: step_id,
                basePrompt: promptSeed,
                targetPaths: effectiveTargetPaths,
                expectedArtifacts: expected_artifacts,
                verificationCommand: verification_command,
                verificationPlan: verification_plan,
                executionAdapterPacket: execution_adapter_packet,
                contextPacket: context_packet,
                repoMap: repo_map,
                contextEnvelope: context_envelope,
            });
            const effectivePrompt = promptContract.prompt;
            const adapterRequest = payloadToAdapterRequest({
                provider: preferredProvider,
                task_prompt: effectivePrompt,
                artifact_root,
                expected_artifacts,
                step_id,
                target_paths: effectiveTargetPaths,
                verification_command,
                verification_plan,
                wall_clock_timeout_s: wallClockTimeoutSafe,
                execution_adapter_packet,
                context_packet,
                repo_map,
                model,
                execution_lane,
                allow_provider_fallback,
                runtime_coder_config,
                run_id,
                task_id,
                artifact_workspace_root: workspaceRoot,
            });
            let result = await executeCodingAdapter({
                workspaceRoot: executionWorkspaceRoot,
                artifactWorkspaceRoot: workspaceRoot,
                adapterRequest,
                provider: preferredProvider,
                model,
                maxRuntimeS: Math.max(1, Math.min(
                    Number(max_runtime_s || 600),
                    Math.max(1, Math.floor(remainingMs / 1000)),
                )),
                codexCommand: codex_command,
                opencodeCommand: opencode_command,
                allowProviderFallback: allow_provider_fallback,
                runtimeCoderConfig: runtime_coder_config || {},
                onRuntimeEvent: on_runtime_event,
            });
            finalFallbackFrom = finalFallbackFrom || result?.diagnostics?.fallback_from || null;

            const scaffoldExpectedArtifacts = buildStepContractExpectedArtifacts({
                stepId: step_id,
                expectedArtifacts: expected_artifacts,
            });
            let artifactScaffold = null;
            if (result?.ok) {
                artifactScaffold = ensureExpectedArtifacts({
                    workspaceRoot,
                    artifactRoot: artifact_root,
                    expectedArtifacts: scaffoldExpectedArtifacts,
                    stepId: step_id,
                    taskPrompt: effectivePrompt,
                });
            }

            const stdoutPath = path.join(taskDir, `delegate_stdout_attempt${attemptIndex}_${Date.now()}.log`);
            const stderrPath = path.join(taskDir, `delegate_stderr_attempt${attemptIndex}_${Date.now()}.log`);
            const redactedStdout = redactSensitiveText(result.stdout || "");
            const redactedStderr = redactSensitiveText(result.stderr || "");
            try {
                fs.writeFileSync(stdoutPath, redactedStdout, "utf8");
                fs.writeFileSync(stderrPath, redactedStderr, "utf8");
            } catch (logErr) {
                console.warn(`${reqId} Failed to write delegate stdout/stderr logs:`, logErr.message);
            }

            const gitSummary = await gatherGitSummary(executionWorkspaceRoot, taskDir, executionBaselineSnapshot, effectiveTargetPaths);
            const finalGitSummary = result?.ok
                ? await ensureImplementationDelta({
                    workspaceRoot: executionWorkspaceRoot,
                    stepId: step_id,
                    taskId: task_id,
                    executionAdapterPacket: execution_adapter_packet,
                    targetPaths: effectiveTargetPaths,
                    taskPrompt: effectivePrompt,
                    taskDir,
                    baselineSnapshot: executionBaselineSnapshot,
                    current: gitSummary,
                })
                : gitSummary;
            const changedScopeCheck = validateChangedFilesWithinScope({
                filesChanged: finalGitSummary?.filesChanged,
                allowedTargetPaths: effectiveTargetPaths,
            });
            if (result?.ok && !changedScopeCheck.ok) {
                finalSummary = buildDelegateFailureSummary({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested,
                    providerUsed: result.provider_used || preferredProvider,
                    targetPaths: effectiveTargetPaths,
                    finalGitSummary,
                    stdoutPath,
                    stderrPath,
                    staticCheck: null,
                    verification: null,
                    adapterResult: result,
                    verificationCommand: verification_command,
                    promptContract,
                    redactedStdout,
                    redactedStderr,
                    startedAt: started,
                    phase: "scope_guard",
                    summaryText: "blocked unauthorized write outside target_paths",
                    testResult: "skipped",
                    errorCode: ErrorCode.UNAUTHORIZED_WRITE,
                    error: changedScopeCheck.error,
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
                break;
            }

            const artifactSalvage = !result?.ok
                ? salvageWorkflowArtifactFailure({
                    workspaceRoot,
                    artifactRoot: artifact_root,
                    expectedArtifacts: expected_artifacts,
                    stepId: step_id,
                    taskPrompt: effectivePrompt,
                    result,
                })
                : null;
            if (artifactSalvage) {
                artifactScaffold = artifactSalvage.scaffold || artifactScaffold;
                result = {
                    ...result,
                    ok: true,
                    error: null,
                    diagnostics: {
                        ...(result.diagnostics || {}),
                        artifact_salvaged: true,
                        artifact_salvage_step: step_id,
                        artifact_salvage_reason: artifactSalvage.salvageReason,
                        artifact_salvage_validations: {
                            role_output: artifactSalvage.roleValidation,
                            handoff: artifactSalvage.handoffValidation,
                        },
                        ...(result?.diagnostics?.timeout
                            ? {
                                timeout_salvaged: true,
                                timeout_salvage_step: step_id,
                                timeout_salvage_validations: {
                                    role_output: artifactSalvage.roleValidation,
                                    handoff: artifactSalvage.handoffValidation,
                                },
                            }
                            : {}),
                    },
                };
            }

            const handoff = getWorkflowStepHandoff(step_id);
            const roleValidation = result?.ok
                ? validateWorkflowStepArtifacts({
                    workspaceRoot,
                    artifactRoot: artifact_root,
                    stepId: step_id,
                })
                : { checked: false, ok: true };
            const handoffValidation = result?.ok && handoff
                ? validateCodingTeamHandoff({
                    workspaceRoot,
                    artifactRoot: artifact_root,
                    handoff,
                })
                : { checked: false, ok: true };

            let staticCheck = result?.ok
                ? await runStaticChecks({
                    workspaceRoot: executionWorkspaceRoot,
                    filesChanged: finalGitSummary?.filesChanged || [],
                    taskDir,
                    redis,
                })
                : { checked: false, ok: true, commands: [], logPath: null };

            const verification = result?.ok
                ? await runVerificationPlan({
                    workspaceRoot: executionWorkspaceRoot,
                    verificationPlan: verification_plan,
                    verificationCommand: verification_command,
                    taskDir,
                })
                : { checked: false, ok: true, command: "", commands: [], records: [], achieved_tiers: [], unresolved_tiers: [], logPath: null };

            // --- Phase 1: Surgical Patch Pre-processor ---
            // Attempt deterministic micro-fix before entering the decision tree.
            // If successful, staticCheck is overridden to the passing recheck result,
            // and the existing if/else chain naturally falls through to the success path.
            if (result?.ok && staticCheck.checked && !staticCheck.ok
                && surgicalPatchEnabled && !surgicalPatchAttempted) {
                const fixability = canFix({ records: staticCheck.records || [] });
                if (fixability.fixable) {
                    const patchResult = applySurgicalPatches({
                        executionWorkspaceRoot,
                        records: staticCheck.records || [],
                        allowedTargetPaths: effectiveTargetPaths,
                        workspaceRoot,
                    });
                    surgicalPatchAttempted = true;
                    if (patchResult.success) {
                        const recheck = await runStaticChecks({
                            workspaceRoot: executionWorkspaceRoot,
                            filesChanged: patchResult.patches_applied.map((p) => p.file),
                            taskDir,
                            redis,
                        });
                        if (recheck.ok) {
                            staticCheck = { ...recheck, surgical_patch: {
                                attempted: true, success: true,
                                patches_applied: patchResult.patches_applied,
                            }};
                        }
                    }
                    if (!staticCheck.ok) {
                        staticCheck.surgical_patch = {
                            attempted: true, success: false,
                            patches_applied: patchResult.patches_applied,
                            failure_reason: patchResult.failure_reason,
                        };
                    }
                }
            }

            if (result?.ok && roleValidation.checked && !roleValidation.ok) {
                finalSummary = buildDelegateFailureSummary({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested,
                    providerUsed: result.provider_used || preferredProvider,
                    targetPaths: effectiveTargetPaths,
                    finalGitSummary,
                    stdoutPath,
                    stderrPath,
                    staticCheck,
                    verification,
                    adapterResult: result,
                    verificationCommand: verification_command,
                    promptContract,
                    redactedStdout,
                    redactedStderr,
                    startedAt: started,
                    phase: "artifact_contract",
                    summaryText: `step artifact validation failed: ${roleValidation.detail || roleValidation.code || "unknown artifact validation error"}`,
                    testResult: "failed",
                    errorCode: roleValidation.code || ErrorCode.WORKFLOW_ARTIFACT_INVALID,
                    error: roleValidation.detail || "workflow step artifact validation failed",
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
                finalSummary.diagnostics = {
                    ...(finalSummary.diagnostics || {}),
                    role_validation: roleValidation,
                    handoff_validation: handoffValidation,
                };
            } else if (result?.ok && handoffValidation.checked && !handoffValidation.ok) {
                finalSummary = buildDelegateFailureSummary({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested,
                    providerUsed: result.provider_used || preferredProvider,
                    targetPaths: effectiveTargetPaths,
                    finalGitSummary,
                    stdoutPath,
                    stderrPath,
                    staticCheck,
                    verification,
                    adapterResult: result,
                    verificationCommand: verification_command,
                    promptContract,
                    redactedStdout,
                    redactedStderr,
                    startedAt: started,
                    phase: "artifact_contract",
                    summaryText: `typed handoff validation failed: ${handoffValidation.detail || handoffValidation.code || "unknown handoff validation error"}`,
                    testResult: "failed",
                    errorCode: handoffValidation.code || ErrorCode.HANDOFF_INVALID,
                    error: handoffValidation.detail || "workflow handoff validation failed",
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
                finalSummary.diagnostics = {
                    ...(finalSummary.diagnostics || {}),
                    role_validation: roleValidation,
                    handoff_validation: handoffValidation,
                };
            } else if (result?.ok && staticCheck.checked && !staticCheck.ok) {
                finalSummary = buildDelegateFailureSummary({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested,
                    providerUsed: result.provider_used || preferredProvider,
                    targetPaths: effectiveTargetPaths,
                    finalGitSummary,
                    stdoutPath,
                    stderrPath,
                    staticCheck,
                    verification,
                    adapterResult: result,
                    verificationCommand: verification_command,
                    redactedStdout,
                    redactedStderr,
                    startedAt: started,
                    phase: "static_check",
                    summaryText: "static checks failed after delegation",
                    testResult: "failed",
                    errorCode: ErrorCode.STATIC_CHECK_FAILED,
                    error: staticCheck.error || "static check failed",
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
            } else if (result?.ok && verification.checked && !verification.ok) {
                finalSummary = buildDelegateFailureSummary({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested,
                    providerUsed: result.provider_used || preferredProvider,
                    targetPaths: effectiveTargetPaths,
                    finalGitSummary,
                    stdoutPath,
                    stderrPath,
                    staticCheck,
                    verification,
                    adapterResult: result,
                    verificationCommand: verification.command || verification_command,
                    promptContract,
                    redactedStdout,
                    redactedStderr: `${redactedStderr}\n${String(verification.stderr || "")}`.trim(),
                    startedAt: started,
                    phase: "verification",
                    summaryText: "verification command failed after delegation",
                    testResult: "failed",
                    errorCode: verification.errorCode || ErrorCode.VERIFICATION_FAILED,
                    error: verification.error || "verification command failed",
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
            } else if (!result?.ok) {
                finalSummary = buildDelegateFailureSummary({
                    workspaceRoot,
                    runId: run_id,
                    taskId: task_id,
                    stepId: step_id,
                    providerRequested,
                    providerUsed: result.provider_used || preferredProvider,
                    targetPaths: effectiveTargetPaths,
                    finalGitSummary,
                    stdoutPath,
                    stderrPath,
                    staticCheck,
                    verification,
                    adapterResult: result,
                    verificationCommand: verification_command,
                    promptContract,
                    redactedStdout,
                    redactedStderr,
                    startedAt: started,
                    phase: "delegate",
                    summaryText: `${result.provider_used || preferredProvider} delegation failed: ${result.error || "unknown error"}`,
                    testResult: "skipped",
                    errorCode: result?.diagnostics?.error_code || ErrorCode.DELEGATE_FAILED,
                    error: redactSensitiveText(result.error || "") || `${result.provider_used || preferredProvider} delegation failed`,
                    taskClass: taskContract.task_class,
                    betaTemplateId: taskContract.beta_template_id,
                    contextEnvelope: taskContract.context_envelope,
                    lineage: normalizedLineage,
                });
            } else {
                const promotion = isolationScaffold?.enabled
                    ? promoteIsolatedChanges({
                        workspaceRoot,
                        isolatedWorkspaceRoot: executionWorkspaceRoot,
                        taskDir,
                        filesChanged: finalGitSummary.filesChanged,
                        allowedTargetPaths: effectiveTargetPaths,
                        mode: isolationMode,
                        baselineSnapshot,
                    })
                    : {
                        ok: true,
                        applied: false,
                        rollbackPerformed: false,
                        error: null,
                        artifacts: {
                            promotion_preflight: null,
                            promotion_result: null,
                        },
                    };
                if (!promotion.ok) {
                    finalSummary = buildDelegateFailureSummary({
                        workspaceRoot,
                        runId: run_id,
                        taskId: task_id,
                        stepId: step_id,
                        providerRequested,
                        providerUsed: result.provider_used || preferredProvider,
                        targetPaths: effectiveTargetPaths,
                        finalGitSummary,
                        stdoutPath,
                        stderrPath,
                        staticCheck,
                        verification,
                        adapterResult: result,
                        verificationCommand: verification.command || verification_command,
                        promptContract,
                        redactedStdout,
                        redactedStderr,
                        startedAt: started,
                        phase: "promotion",
                        summaryText: "isolated execution succeeded but promotion into main workspace failed",
                        testResult: "passed",
                        errorCode: ErrorCode.PROMOTION_FAILED,
                        error: promotion.error || "promotion failed",
                        taskClass: taskContract.task_class,
                        betaTemplateId: taskContract.beta_template_id,
                        contextEnvelope: taskContract.context_envelope,
                        lineage: normalizedLineage,
                    });
                    finalSummary.artifacts = {
                        ...(finalSummary.artifacts || {}),
                        isolation_execution_mode: isolationScaffold?.artifacts?.execution_mode || null,
                        isolation_baseline_manifest: isolationScaffold?.artifacts?.baseline_manifest || null,
                        isolation_workspace_manifest: isolationScaffold?.artifacts?.isolated_workspace_manifest || null,
                        promotion_preflight: promotion.artifacts?.promotion_preflight || null,
                        promotion_result: promotion.artifacts?.promotion_result || null,
                    };
                    finalSummary.diagnostics = {
                        ...(finalSummary.diagnostics || {}),
                        isolation: isolationScaffold?.diagnostics || null,
                        execution_workspace_root: executionWorkspaceRoot,
                        promotion: {
                            ok: promotion.ok,
                            applied: promotion.applied,
                            rollback_performed: promotion.rollbackPerformed,
                        },
                    };
                    break;
                }
                finalSummary = {
                    ok: true,
                    provider_used: result.provider_used || preferredProvider,
                    model_used: result.model_used || null,
                    execution_lane: result.execution_lane || execution_lane || null,
                    model_provider: result?.diagnostics?.model_provider || result.provider_used || preferredProvider,
                    model_name: result?.diagnostics?.model_name || result.model_used || null,
                    summary: `${result.provider_used || preferredProvider} delegation finished.`,
                    files_changed: finalGitSummary.filesChanged,
                    diff_stats: finalGitSummary.diffStats,
                    test_result: verification.checked ? "passed" : "skipped",
                    git: finalGitSummary.git,
                    rollback_performed: Boolean(promotion.rollbackPerformed),
                    artifacts: {
                        diff_bundle: finalGitSummary.diffPath,
                        patch_file: null,
                        test_log: verification.logPath || staticCheck.logPath || null,
                        prompt_contract: promptContract.path || null,
                        isolation_execution_mode: isolationScaffold?.artifacts?.execution_mode || null,
                        isolation_baseline_manifest: isolationScaffold?.artifacts?.baseline_manifest || null,
                        isolation_workspace_manifest: isolationScaffold?.artifacts?.isolated_workspace_manifest || null,
                        promotion_preflight: promotion.artifacts?.promotion_preflight || null,
                        promotion_result: promotion.artifacts?.promotion_result || null,
                        raw_stdout: stdoutPath,
                        raw_stderr: stderrPath,
                    },
                    diagnostics: {
                        ...(result.diagnostics || {}),
                        provider_requested: providerRequested,
                        task_contract: taskContract,
                        failure_attribution: null,
                        artifact_scaffold: artifactScaffold || null,
                        role_validation: roleValidation,
                        handoff_validation: handoffValidation,
                        execution_adapter_packet: execution_adapter_packet || null,
                        tool_adapter_request: adapterRequest,
                        prompt_contract: promptContract.diagnostics,
                        static_check: staticCheck,
                        verification,
                        isolation: isolationScaffold?.diagnostics || null,
                        execution_workspace_root: executionWorkspaceRoot,
                        promotion: {
                            ok: promotion.ok,
                            applied: promotion.applied,
                            rollback_performed: promotion.rollbackPerformed,
                        },
                        verification_plan: Array.isArray(verification_plan) ? verification_plan : [],
                        retry_summary: {
                            attempts_used: attemptIndex,
                            max_attempts: maxAttemptsSafe,
                            same_error_repeat_limit: sameErrorRepeatLimitSafe,
                            wall_clock_timeout_s: wallClockTimeoutSafe,
                            repairs_attempted: Math.max(0, attemptIndex - 1),
                            repaired_after_retry: attemptIndex > 1,
                        },
                        parse_error: false,
                        truncated: false,
                        surgical_patch: staticCheck.surgical_patch || null,
                        refinement: normalizedLineage ? {
                            lineage: normalizedLineage,
                            context_diagnostics: refinementDiagnostics,
                        } : null,
                    },
                    error: redactSensitiveText(result.error || "") || null,
                    command_used: result.command_used || null,
                    command_source: result.command_source || "unknown",
                    started_at: started,
                    finished_at: new Date().toISOString(),
                };
                break;
            }

            finalSummary.diagnostics.retry_summary = {
                attempts_used: attemptIndex,
                max_attempts: maxAttemptsSafe,
                same_error_repeat_limit: sameErrorRepeatLimitSafe,
                repairs_attempted: Math.max(0, attemptIndex - 1),
                repaired_after_retry: false,
            };

            const retryDecision = shouldRetryAutoFix({
                summary: finalSummary,
                attemptIndex,
                maxAttempts: maxAttemptsSafe,
                sameErrorRepeatLimit: sameErrorRepeatLimitSafe,
                attemptRecords,
                errorCounts,
            });
                    attemptRecords.push({
                        attempt: attemptIndex,
                        phase: finalSummary?.diagnostics?.failed_phase || "delegate",
                error_code: finalSummary?.diagnostics?.error_code || ErrorCode.DELEGATE_FAILED,
                error: String(finalSummary?.error || ""),
                        command_used: finalSummary?.command_used || null,
                        verification_command: verification_command || null,
                        verification_plan: Array.isArray(verification_plan) ? verification_plan : [],
                    });
            if (!retryDecision.retry) {
                terminalRetryReason = retryDecision.reason;
                break;
            }
        }

        if (finalSummary && !finalSummary.ok) {
            finalSummary.artifacts = {
                ...(finalSummary.artifacts || {}),
                isolation_execution_mode: isolationScaffold?.artifacts?.execution_mode || null,
                isolation_baseline_manifest: isolationScaffold?.artifacts?.baseline_manifest || null,
                isolation_workspace_manifest: isolationScaffold?.artifacts?.isolated_workspace_manifest || null,
            };
            finalSummary.diagnostics = {
                ...(finalSummary.diagnostics || {}),
                isolation: isolationScaffold?.diagnostics || null,
                execution_workspace_root: executionWorkspaceRoot,
            };
            finalSummary.diagnostics.retry_summary = {
                ...(finalSummary.diagnostics.retry_summary || {}),
                attempts_used: attemptRecords.length || Number(finalSummary?.diagnostics?.retry_summary?.attempts_used || 1),
                max_attempts: maxAttemptsSafe,
                same_error_repeat_limit: sameErrorRepeatLimitSafe,
                wall_clock_timeout_s: wallClockTimeoutSafe,
                repairs_attempted: Math.max(0, (attemptRecords.length || 1) - 1),
                repaired_after_retry: false,
                terminal_reason: terminalRetryReason,
            };
            finalSummary.diagnostics.final_failure_summary = buildFinalFailureSummary({
                summary: finalSummary,
                attemptRecords,
                maxAttempts: maxAttemptsSafe,
                sameErrorRepeatLimit: sameErrorRepeatLimitSafe,
                wallClockTimeoutS: wallClockTimeoutSafe,
                terminalReason: terminalRetryReason,
                startedAt: started,
            });
        }

        try {
            const timelinePath = path.join(runDir, 'timeline.md');
            const status = finalSummary?.ok ? "PASS" : "FAIL";
            const line = `- ${new Date().toISOString()} | task: ${task_id || 'unknown'} | Delegated: ${finalSummary?.provider_used || preferredProvider} (requested=${providerRequested}${finalFallbackFrom ? `,fallback_from=${finalFallbackFrom}` : ""}) | STATUS: ${status}\n`;
            fs.appendFileSync(timelinePath, line);
        } catch (tlErr) {
            console.warn(`${reqId} Failed to write delegate timeline:`, tlErr.message);
        }

        // Cleanup isolation workspace on completion (success or failure)
        if (isolationScaffold?.enabled) {
            cleanupIsolationWorkspace(taskDir);
        }

        return finalSummary;
    },

    /**
     * Placeholder for starting a task.
     */
    startTask: async (task_prompt, workspaceRoot) => {
        const run_id = crypto.randomUUID();
        const runDir = buildRunDir(workspaceRoot, run_id);

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









