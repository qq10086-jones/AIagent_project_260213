import fs from "fs";
import { applyEditBlock } from "./orchestrator/src/patch_manager.js";

console.log("Testing Coding Agent Patch Manager...");

// 1. Setup a test file
const testFilePath = "__write_test__.txt";
fs.writeFileSync(testFilePath, "Hello World\nThis is a test file.\n");

// 2. Define an edit block
const editBlock = `<<<<<<< SEARCH
Hello World
=======
Hello Coding Agent
>>>>>>> REPLACE`;

// 3. Apply the patch
const workspaceRoot = process.cwd();
console.log("Applying patch...");
const result = applyEditBlock(workspaceRoot, testFilePath, editBlock);

console.log("Result:", result);

if (result.success) {
    const updatedContent = fs.readFileSync(testFilePath, "utf8");
    console.log("\\nUpdated File Content:\\n" + updatedContent);
    if (updatedContent.includes("Hello Coding Agent")) {
        console.log("✅ Patch applied successfully!");
    } else {
        console.log("❌ Patch did not modify the text correctly.");
    }
} else {
    console.log("❌ Patch failed.");
}

// 4. Test security check (outside workspace)
console.log("\\nTesting Security (Outside Workspace)...");
const secResult = applyEditBlock(workspaceRoot, "../outside.txt", editBlock);
console.log("Security Result:", secResult);
if (!secResult.success) {
    console.log("✅ Security check passed (blocked outside access).");
}
