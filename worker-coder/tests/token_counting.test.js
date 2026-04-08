import assert from "node:assert/strict";
import { countTokens, validateContextTokenBudget, truncateContextForTokenBudget } from "../task_contract.js";

function testCountTokensBasic() {
  const count = countTokens("Hello, world!");
  assert.ok(count > 0, "should return positive token count");
  assert.ok(count < 20, "simple phrase should be under 20 tokens");
}

function testCountTokensEmpty() {
  assert.equal(countTokens(""), 0);
  assert.equal(countTokens(null), 0);
  assert.equal(countTokens(undefined), 0);
}

function testBudgetWithinLimit() {
  const result = validateContextTokenBudget({ contextText: "short text", maxTokens: 1000 });
  assert.equal(result.ok, true);
  assert.ok(result.usage > 0);
  assert.equal(result.overflow, 0);
  assert.equal(result.max, 1000);
}

function testBudgetOverflow() {
  const longText = "token ".repeat(500);
  const result = validateContextTokenBudget({ contextText: longText, maxTokens: 10 });
  assert.equal(result.ok, false);
  assert.ok(result.overflow > 0);
  assert.ok(result.usage > 10);
}

function testBudgetZeroMax() {
  const result = validateContextTokenBudget({ contextText: "anything", maxTokens: 0 });
  assert.equal(result.ok, true);
  assert.equal(result.usage, 0);
}

function testTruncateNoTruncationNeeded() {
  const cp = { memory_hints: ["hint1"], entrypoints: ["a.js"] };
  const rm = { symbol_hints: [{ file: "a", symbols: [] }], candidate_files: ["a.js"] };
  const result = truncateContextForTokenBudget({ contextPacket: cp, repoMap: rm, maxTokens: 100000 });
  assert.equal(result.truncated, false);
  assert.deepEqual(result.contextPacket.memory_hints, ["hint1"]);
}

function testTruncateRemovesLowPriority() {
  // Build context large enough to exceed a small budget
  const cp = {
    memory_hints: Array.from({ length: 50 }, (_, i) => `memory hint number ${i} with some extra text to increase size`),
    entrypoints: ["src/index.js"],
  };
  const rm = {
    symbol_hints: Array.from({ length: 50 }, (_, i) => ({ file: `file${i}.js`, symbols: [`export function fn${i}()`] })),
    toolchain_facts: [{ file: "package.json", type: "config", snippet: "x".repeat(200) }],
    candidate_files: Array.from({ length: 30 }, (_, i) => `src/file${i}.js`),
  };
  const fullTokens = countTokens(JSON.stringify(cp) + JSON.stringify(rm));
  // Set budget to half of full size
  const budget = Math.floor(fullTokens / 2);
  const result = truncateContextForTokenBudget({ contextPacket: cp, repoMap: rm, maxTokens: budget });
  assert.equal(result.truncated, true);
  assert.ok(result.tokenUsage <= budget || result.tokenUsage < fullTokens, "token usage should be reduced");
  // memory_hints should be cleared first
  assert.deepEqual(result.contextPacket.memory_hints, []);
}

function testTruncateZeroMax() {
  const result = truncateContextForTokenBudget({ contextPacket: {}, repoMap: {}, maxTokens: 0 });
  assert.equal(result.truncated, false);
}

async function main() {
  testCountTokensBasic();
  testCountTokensEmpty();
  testBudgetWithinLimit();
  testBudgetOverflow();
  testBudgetZeroMax();
  testTruncateNoTruncationNeeded();
  testTruncateRemovesLowPriority();
  testTruncateZeroMax();
  console.log("token_counting.test.js: all tests passed");
}

main().catch((err) => {
  console.error("token_counting.test.js: failed");
  console.error(err);
  process.exit(1);
});
