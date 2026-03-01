import fs from 'fs';
import path from 'path';

/**
 * Parses multiple search/replace blocks based on Aider's protocol.
 * Expected Format:
 * <<<<<<< SEARCH
 * [exact lines to find]
 * =======
 * [new lines to replace with]
 * >>>>>>> REPLACE
 * 
 * @param {string} editBlocksStr - The raw string containing one or more SEARCH/REPLACE blocks.
 * @returns {Array<{searchStr: string, replaceStr: string}>} Array of parsed blocks.
 */
export function parseEditBlocks(editBlocksStr) {
    const searchMarker = '<<<<<<< SEARCH';
    const dividerMarker = '=======';
    const replaceMarker = '>>>>>>> REPLACE';

    const lines = editBlocksStr.split(/\r?\n/);
    let state = 'out';
    let currentSearchLines = [];
    let currentReplaceLines = [];
    const blocks = [];

    for (const line of lines) {
        if (line.trim() === searchMarker) {
            state = 'search';
            currentSearchLines = [];
            currentReplaceLines = [];
            continue;
        } else if (line.trim() === dividerMarker) {
            state = 'replace';
            continue;
        } else if (line.trim() === replaceMarker) {
            state = 'out';
            if (currentSearchLines.length > 0 || currentReplaceLines.length > 0) {
                blocks.push({
                    searchStr: currentSearchLines.join('\n'),
                    replaceStr: currentReplaceLines.join('\n')
                });
            }
            continue;
        }

        if (state === 'search') {
            currentSearchLines.push(line);
        } else if (state === 'replace') {
            currentReplaceLines.push(line);
        }
    }

    if (blocks.length === 0) {
        throw new Error('Invalid edit block format. No valid SEARCH/REPLACE sections found.');
    }

    return blocks;
}

/**
 * Applies multiple parsed edit blocks to the specified file.
 * 
 * @param {string} workspaceRoot - The absolute path to the project root for safety.
 * @param {string} relativeFilePath - Path to the file to modify, relative to workspaceRoot.
 * @param {string} editBlocksStr - The raw string containing one or more Search/Replace blocks.
 * @returns {object} { success: boolean, message: string, detail?: string }
 */
export function applyEditBlocks(workspaceRoot, relativeFilePath, editBlocksStr) {
    try {
        const workspaceAbsRoot = path.resolve(workspaceRoot);
        const resolvedPath = path.resolve(workspaceAbsRoot, relativeFilePath);
        
        console.log(`[patch_manager] Workspace: ${workspaceAbsRoot}`);
        console.log(`[patch_manager] Target: ${resolvedPath}`);

        if (!resolvedPath.startsWith(workspaceAbsRoot)) {
            return { success: false, message: `Security Error: Target path ${resolvedPath} is outside the workspace root ${workspaceAbsRoot}.` };
        }

        if (!fs.existsSync(resolvedPath)) {
            return { success: false, message: `File not found: ${relativeFilePath}` };
        }

        let content = fs.readFileSync(resolvedPath, 'utf8').replace(/\r\n/g, '\n');
        const blocks = parseEditBlocks(editBlocksStr);
        const results = [];

        for (let i = 0; i < blocks.length; i++) {
            const { searchStr, replaceStr } = blocks[i];
            
            if (!searchStr) {
                results.push(`Block ${i + 1}: Empty SEARCH block skipped.`);
                continue;
            }

            // Extremely robust matching: trim everything
            const searchLines = searchStr.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
            const contentLines = content.split(/\r?\n/);
            
            let foundIndex = -1;
            let occurrences = 0;

            for (let j = 0; j <= contentLines.length - searchLines.length; j++) {
                let match = true;
                for (let k = 0; k < searchLines.length; k++) {
                    const cLine = contentLines[j + k].trim();
                    const sLine = searchLines[k];
                    if (cLine !== sLine) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    occurrences++;
                    foundIndex = j;
                }
            }

            if (occurrences === 0 && searchLines.length === 1) {
                const needle = searchLines[0];
                const containsMatches = [];
                for (let j = 0; j < contentLines.length; j++) {
                    if (contentLines[j].includes(needle)) {
                        containsMatches.push(j);
                    }
                }
                if (containsMatches.length === 1) {
                    const idx = containsMatches[0];
                    contentLines[idx] = contentLines[idx].replace(needle, replaceStr.trim());
                    content = contentLines.join('\n');
                    results.push(`Block ${i + 1}: Applied via single-line contains fallback.`);
                    continue;
                }
            }

            if (occurrences === 0) {
                return { 
                    success: false, 
                    message: `Block ${i + 1}: SEARCH block not found.`,
                    detail: `Failed to find:\n${searchStr}`
                };
            } else if (occurrences > 1) {
                return { 
                    success: false, 
                    message: `Block ${i + 1}: SEARCH block is ambiguous (found ${occurrences} times).`,
                    detail: `Found multiple matches for:\n${searchStr}`
                };
            }

            // Apply replacement by re-assembling the lines
            const replaceLines = replaceStr.split(/\r?\n/);
            const newLines = [
                ...contentLines.slice(0, foundIndex),
                ...replaceLines,
                ...contentLines.slice(foundIndex + searchLines.length)
            ];
            content = newLines.join('\n');
            results.push(`Block ${i + 1}: Applied.`);
        }

        fs.writeFileSync(resolvedPath, content, 'utf8');
        return { success: true, message: `Successfully updated ${relativeFilePath}`, detail: results.join(' ') };
    } catch (err) {
        return { success: false, message: `Failed to apply edit blocks: ${err.message}` };
    }
}
