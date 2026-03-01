function sanitizeLocalAssistantReply(raw) {
  let out = String(raw || "");
  out = out.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  out = out.replace(/<think>[\s\S]*$/gi, "").trim();

  const cotLine = /^(嗯，?我现在要处理用户的查询|好，?我现在需要|首先，|接下来，|另外，|用户可能|作为NEXUS助手|我需要理解这个问题)/;
  const lines = out
    .split(/?
/)
    .map(s => s.trim())
    .filter(Boolean);
  const kept = lines.filter(l => !cotLine.test(l));
  if (kept.length > 0 && kept.length < lines.length) {
    out = kept.join("
").trim();
  }
  return out;
}

const res1 = sanitizeLocalAssistantReply("<think>

</think>

Hello! How can I assist you today?");
console.log("res1:", res1);
const res2 = sanitizeLocalAssistantReply("好，我现在需要回答用户问题。
1+1=2。");
console.log("res2:", res2);
