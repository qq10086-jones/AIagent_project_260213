/**
 * stream_repository.js
 *
 * Thin helpers for Redis Streams operations.
 * All exported functions accept `redis` (ioredis instance) as their first argument
 * so they can be unit-tested with a stub.
 */

/**
 * Ensure a consumer group exists for `stream`, then append `payload` as a
 * flat key/value sequence to the stream.
 *
 * The group-creation attempt is fire-and-forget: errors are swallowed because
 * the group will already exist after the first successful call.
 *
 * @param {import('ioredis').Redis} redis
 * @param {string} stream   - Redis stream key
 * @param {string} group    - Consumer group name to ensure
 * @param {Record<string, string>} payload - Flat object; keys and values are
 *   passed to XADD as alternating field/value pairs.
 * @returns {Promise<string>} The Redis message ID returned by XADD.
 */
export async function enqueueToStream(redis, stream, group, payload) {
  // Best-effort group creation; BUSYGROUP error is expected after first call.
  try {
    await redis.xgroup("CREATE", stream, group, "$", "MKSTREAM");
  } catch (_) {
    // group already exists — ignore
  }

  const args = [];
  for (const [k, v] of Object.entries(payload)) {
    args.push(k, String(v ?? ""));
  }
  return redis.xadd(stream, "*", ...args);
}
