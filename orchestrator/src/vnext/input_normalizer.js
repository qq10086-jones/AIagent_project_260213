function normalizeAttachments(input) {
  const items = Array.isArray(input) ? input : [];
  return items
    .map((item, index) => {
      if (!item || typeof item !== "object") return null;
      const url = String(item.url || item.proxy_url || item.href || "").trim();
      const name = String(item.filename || item.name || `attachment_${index + 1}`).trim();
      const mime = String(item.content_type || item.mime || "application/octet-stream").trim();
      if (!url && !name) return null;
      return {
        name,
        url,
        mime,
      };
    })
    .filter(Boolean);
}

function normalizeDiscordEvent(event = {}) {
  const attachments = normalizeAttachments(event.attachments);
  const content = String(event.content || event.message || "").trim();
  return {
    source: "discord",
    raw_input: content,
    normalized_input: {
      text: content,
      attachments,
      metadata: {
        user_id: String(event.author?.id || event.user_id || "").trim(),
        username: String(event.author?.username || event.username || "").trim(),
        channel_id: String(event.channel_id || "").trim(),
        thread_id: String(event.thread_id || event.id || "").trim(),
        guild_id: String(event.guild_id || "").trim(),
      },
    },
    context: {
      channel_id: String(event.channel_id || "").trim(),
      thread_id: String(event.thread_id || event.id || "").trim(),
      user_id: String(event.author?.id || event.user_id || "").trim(),
      attachments,
      raw_event: event,
    },
  };
}

export function normalizeInputRequest(body = {}) {
  const source = String(body.source || "").trim().toLowerCase();
  if (body.discord_event && typeof body.discord_event === "object") {
    return normalizeDiscordEvent(body.discord_event);
  }

  const text = String(body.raw_input || body.message || body.text || "").trim();
  const attachments = normalizeAttachments(body.attachments);
  const resolvedSource = source || "api";

  return {
    source: resolvedSource,
    raw_input: text,
    normalized_input: {
      text,
      attachments,
      metadata: {
        user_id: String(body.user_id || "").trim(),
        channel_id: String(body.channel_id || "").trim(),
        thread_id: String(body.thread_id || "").trim(),
      },
    },
    context: {
      channel_id: String(body.channel_id || "").trim(),
      thread_id: String(body.thread_id || "").trim(),
      user_id: String(body.user_id || "").trim(),
      attachments,
      raw_event: body.raw_event && typeof body.raw_event === "object" ? body.raw_event : null,
    },
  };
}
