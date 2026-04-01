import fs from "fs";
import { applyEditBlocks } from "./orchestrator/src/patch_manager.js";

console.log("Testing Coding Agent Multi-Block Patching...");

// 1. Setup a test file
const testFilePath = "__write_test_multi__.txt";
fs.writeFileSync(testFilePath, "Line 1: Alpha\nLine 2: Beta\nLine 3: Gamma\nLine 4: Delta\n");

// 2. Define multiple edit blocks
const editBlocks = `<<<<<<< SEARCH
Line 1: Alpha
=======
Line 1: Alpha Modified
>>>>>>> REPLACE

<<<<<<< SEARCH
Line 3: Gamma
=======
Line 3: Gamma Modified
>>>>>>> REPLACE`;

// 3. Apply the patch
const workspaceRoot = process.cwd();
console.log("Applying multi-block patch...");
const result = applyEditBlocks(workspaceRoot, testFilePath, editBlocks);

console.log("Result:", result);

if (result.success) {
    const updatedContent = fs.readFileSync(testFilePath, "utf8");
    console.log("\nUpdated File Content:\n" + updatedContent);
    if (updatedContent.includes("Alpha Modified") && updatedContent.includes("Gamma Modified")) {
        console.log("✅ Multi-block patch applied successfully!");
    } else {
        console.log("❌ Multi-block patch did not modify the text correctly.");
    }
} else {
    console.log("❌ Multi-block patch failed.");
}
