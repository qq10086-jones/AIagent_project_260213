import fs from "fs";
import path from "path";
import { applyEditBlocks } from "./orchestrator/src/patch_manager.js";

console.log("--- [Edge Case Test] Patch Manager Robustness ---");

const workspaceRoot = process.cwd();
const testFile = "edge_case_test.txt";

function setup(content) {
    fs.writeFileSync(testFile, content);
}

function verify(testName, result, expectedSuccess, expectedContentSnippet = null) {
    if (result.success === expectedSuccess) {
        if (expectedContentSnippet) {
            const content = fs.readFileSync(testFile, "utf8");
            if (content.includes(expectedContentSnippet)) {
                console.log(`✅ ${testName}: Passed (Content matches)`);
            } else {
                console.log(`❌ ${testName}: Failed (Content mismatch)`);
                console.log("Actual Content: [" + content + "]");
            }
        } else {
            console.log(`✅ ${testName}: Passed (Result matches)`);
        }
    } else {
        console.log(`❌ ${testName}: Failed (Expected ${expectedSuccess}, got ${result.success})`);
        console.log("Message:", result.message);
    }
}

// 1. Test: Trailing spaces normalization
setup("Line with space \nLine stable\n");
let patch = "<<<<<<< SEARCH\nLine with space\n=======\nLine fixed\n>>>>>>> REPLACE";
verify("Trailing Space Normalization", applyEditBlocks(workspaceRoot, testFile, patch), true, "Line fixed");

// 2. Test: Ambiguous block
setup("Duplicate\nDuplicate\nUnique\n");
patch = "<<<<<<< SEARCH\nDuplicate\n=======\nNew\n>>>>>>> REPLACE";
verify("Ambiguity Detection", applyEditBlocks(workspaceRoot, testFile, patch), false);

// 3. Test: Out of workspace security
const outsidePath = "../outside.txt";
verify("Security Sandbox", applyEditBlocks(workspaceRoot, outsidePath, patch), false);

// 4. Test: Multiple blocks in one go
setup("A\nB\nC\n");
patch = "<<<<<<< SEARCH\nA\n=======\nA1\n>>>>>>> REPLACE\n\n<<<<<<< SEARCH\nC\n=======\nC1\n>>>>>>> REPLACE";
verify("Multi-block Logic", applyEditBlocks(workspaceRoot, testFile, patch), true, "A1\nB\nC1");

// Cleanup
if (fs.existsSync(testFile)) fs.unlinkSync(testFile);
console.log("--- Edge Case Testing Complete ---");
