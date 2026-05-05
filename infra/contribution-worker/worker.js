/**
 * Reflex contribution-worker — Cloudflare Worker for the Curate wedge.
 *
 * Endpoints:
 *   GET  /healthz                                → health probe
 *   POST /admin/init-bucket                      → confirm R2 bucket exists (admin auth)
 *   GET  /admin/contributors                     → list contributors + stats (admin auth)
 *   POST /admin/manual-purge                     → trigger cascade purge for contributor_id (admin auth)
 *   POST /v1/uploads/sign                        → issue signed PUT URL for an upload
 *   POST /v1/uploads/complete                    → record successful upload, update stats
 *   POST /v1/revoke/cascade                      → mark contributor for purge (30-day SLA)
 *   GET  /v1/contributors/:id/stats              → return contribution stats for `reflex contribute --status`
 *
 * Auth (Phase 1):
 *   - Admin endpoints: Authorization: Bearer <ADMIN_TOKEN>
 *   - Customer endpoints: contributor_id + opted_in_at as soft-proof.
 *     Phase 1.5 will add Ed25519 challenge-response signed by the
 *     consent receipt's user-side key.
 *
 * Storage:
 *   - D1 binding `DB`: contributors, uploads, daily_uploads, revoke_requests
 *   - R2 binding `CURATE_BUCKET`: object payloads under `free-contributors/<id>/`
 *     or `pro-contributors/<id>/` paths
 *
 * Rate limiting:
 *   - 10 GB/day per contributor (configurable via DAILY_BYTES_LIMIT env var)
 *   - 1000 uploads/day per contributor (configurable via DAILY_UPLOADS_LIMIT)
 *   - Cloudflare's built-in DDoS protection on the public endpoints
 */

const ADMIN_TOKEN_HEADER = "Authorization";

const DEFAULT_DAILY_BYTES_LIMIT = 10 * 1024 * 1024 * 1024; // 10 GB
const DEFAULT_DAILY_UPLOADS_LIMIT = 1000;
const SIGNED_URL_TTL_SECONDS = 15 * 60;                    // 15 minutes
const REVOKE_SLA_DAYS = 30;

// ---------- request router ----------

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const method = request.method;
    const path = url.pathname;

    try {
      if (method === "GET" && path === "/healthz") return healthz();
      if (method === "POST" && path === "/admin/init-bucket")
        return await adminAuth(request, env, () => adminInitBucket(env));
      if (method === "GET" && path === "/admin/contributors")
        return await adminAuth(request, env, () => adminListContributors(env));
      if (method === "POST" && path === "/admin/manual-purge")
        return await adminAuth(request, env, () => adminManualPurge(request, env));
      if (method === "POST" && path === "/v1/uploads/sign")
        return await postUploadsSign(request, env);
      if (method === "POST" && path === "/v1/uploads/complete")
        return await postUploadsComplete(request, env);
      if (method === "POST" && path === "/v1/revoke/cascade")
        return await postRevokeCascade(request, env);
      if (method === "GET" && path.startsWith("/v1/contributors/")) {
        const parts = path.split("/").filter(Boolean);
        // /v1/contributors/<id>/stats
        if (parts.length === 4 && parts[3] === "stats") {
          return await getContributorStats(parts[2], env);
        }
      }
      return jsonResponse(404, { error: "not_found", path });
    } catch (e) {
      console.error("Worker error:", e.message, e.stack);
      return jsonResponse(500, { error: "internal_error", message: e.message });
    }
  },
};

// ---------- middleware ----------

async function adminAuth(request, env, handler) {
  const auth = request.headers.get(ADMIN_TOKEN_HEADER) || "";
  const expected = `Bearer ${env.ADMIN_TOKEN}`;
  if (!env.ADMIN_TOKEN || auth !== expected) {
    return jsonResponse(401, { error: "unauthorized" });
  }
  return await handler();
}

// ---------- handlers: health + admin ----------

function healthz() {
  return jsonResponse(200, { status: "ok", service: "reflex-contribution-worker" });
}

async function adminInitBucket(env) {
  if (!env.CURATE_BUCKET) {
    return jsonResponse(500, {
      error: "bucket_not_bound",
      hint: "Run `wrangler r2 bucket create reflex-curate` then redeploy.",
    });
  }
  // Cheap probe: list with limit=1.
  const list = await env.CURATE_BUCKET.list({ limit: 1 });
  return jsonResponse(200, {
    status: "ok",
    bucket_name: "reflex-curate",
    objects_present: list.objects.length > 0,
  });
}

async function adminListContributors(env) {
  const rows = await env.DB.prepare(
    `SELECT contributor_id, tier, first_seen_at, last_active_at,
            total_episodes, total_bytes, total_uploads, revoked_at
       FROM contributors
       ORDER BY total_bytes DESC
       LIMIT 500`
  ).all();
  return jsonResponse(200, { contributors: rows.results || [] });
}

async function adminManualPurge(request, env) {
  const body = await request.json().catch(() => ({}));
  const contributorId = body.contributor_id;
  if (!contributorId) {
    return jsonResponse(400, { error: "missing_contributor_id" });
  }
  return await initiateRevoke(env, contributorId, "all", "admin");
}

// ---------- handlers: customer-facing ----------

/**
 * POST /v1/uploads/sign
 *
 * Body: {
 *   contributor_id: string,
 *   tier: "free" | "pro",
 *   opted_in_at: ISO8601,
 *   file_name: string,         // e.g. "2026-05-05-sess-abcdef.jsonl"
 *   byte_size: number,
 *   episode_count: number,
 *   privacy_mode: "hash_only" | "raw_opt_in",
 * }
 *
 * Returns: {
 *   upload_id: string,
 *   r2_key: string,
 *   put_url: string,           // signed; PUT raw bytes here
 *   expires_at: ISO8601,
 * }
 *
 * Auth (Phase 1): caller asserts contributor_id + opted_in_at; the worker
 * trusts the client. Phase 1.5 adds Ed25519 challenge-response.
 */
async function postUploadsSign(request, env) {
  const body = await request.json().catch(() => null);
  if (!body) return jsonResponse(400, { error: "invalid_json" });

  const contributorId = body.contributor_id;
  const tier = body.tier;
  const optedInAt = body.opted_in_at;
  const fileName = body.file_name;
  const byteSize = Number(body.byte_size);

  if (!contributorId || !tier || !optedInAt || !fileName || !Number.isFinite(byteSize)) {
    return jsonResponse(400, {
      error: "missing_fields",
      required: ["contributor_id", "tier", "opted_in_at", "file_name", "byte_size"],
    });
  }
  if (!["free", "pro", "enterprise"].includes(tier)) {
    return jsonResponse(400, { error: "invalid_tier", got: tier });
  }
  if (!isSafeFileName(fileName)) {
    return jsonResponse(400, { error: "invalid_file_name", message: "no slashes / nulls / leading dots" });
  }
  if (byteSize <= 0 || byteSize > 1 * 1024 * 1024 * 1024) {
    return jsonResponse(400, { error: "byte_size_out_of_range", limit: "0 < size <= 1 GB per upload" });
  }

  // Refuse if contributor is already revoked.
  const existing = await env.DB.prepare(
    `SELECT revoked_at FROM contributors WHERE contributor_id = ?`
  ).bind(contributorId).first();
  if (existing && existing.revoked_at) {
    return jsonResponse(403, {
      error: "contributor_revoked",
      revoked_at: existing.revoked_at,
      message: "Contributor previously revoked. New uploads will not be accepted.",
    });
  }

  // Rate limit: daily bytes + uploads.
  const dailyBytesLimit = Number(env.DAILY_BYTES_LIMIT) || DEFAULT_DAILY_BYTES_LIMIT;
  const dailyUploadsLimit = Number(env.DAILY_UPLOADS_LIMIT) || DEFAULT_DAILY_UPLOADS_LIMIT;
  const utcDate = new Date().toISOString().slice(0, 10);

  const daily = await env.DB.prepare(
    `SELECT bytes_uploaded, uploads_count FROM daily_uploads
       WHERE contributor_id = ? AND utc_date = ?`
  ).bind(contributorId, utcDate).first();
  const usedBytes = daily?.bytes_uploaded || 0;
  const usedUploads = daily?.uploads_count || 0;
  if (usedBytes + byteSize > dailyBytesLimit) {
    return jsonResponse(429, {
      error: "daily_byte_limit_exceeded",
      used_today: usedBytes,
      limit: dailyBytesLimit,
      retry_after_utc: `${utcDate}T23:59:59Z`,
    });
  }
  if (usedUploads >= dailyUploadsLimit) {
    return jsonResponse(429, {
      error: "daily_upload_count_exceeded",
      used_today: usedUploads,
      limit: dailyUploadsLimit,
    });
  }

  // Ensure contributor row exists.
  const nowIso = new Date().toISOString();
  await env.DB.prepare(
    `INSERT INTO contributors (contributor_id, tier, first_seen_at, last_active_at)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(contributor_id) DO UPDATE SET last_active_at = excluded.last_active_at`
  ).bind(contributorId, tier, nowIso, nowIso).run();

  // Build R2 key + upload_id.
  const subdir = tier === "free" ? "free-contributors" : `${tier}-contributors`;
  const r2Key = `${subdir}/${contributorId}/${utcDate}/${fileName}`;
  const uploadId = `upl_${randomHex(16)}`;
  const expiresAt = new Date(Date.now() + SIGNED_URL_TTL_SECONDS * 1000).toISOString();

  await env.DB.prepare(
    `INSERT INTO uploads (upload_id, contributor_id, r2_key, byte_size, status, signed_at, user_agent, source_ip)
       VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)`
  ).bind(
    uploadId, contributorId, r2Key, byteSize, nowIso,
    request.headers.get("User-Agent") || "",
    request.headers.get("CF-Connecting-IP") || "",
  ).run();

  // Phase 1: Cloudflare R2's S3-compat signed-URL story for Workers is to
  // either (a) use the worker as the upload proxy (receive bytes, write to
  // R2 binding directly), or (b) use AWS SigV4 signed PUTs. We do (a) here
  // — simpler, single auth path. Returns a put_url that points to THIS
  // worker's /v1/uploads/put/<upload_id> endpoint.
  const putUrl = `${new URL(request.url).origin}/v1/uploads/put/${uploadId}`;

  return jsonResponse(200, {
    upload_id: uploadId,
    r2_key: r2Key,
    put_url: putUrl,
    expires_at: expiresAt,
    note: "PUT raw bytes to put_url. Then POST /v1/uploads/complete.",
  });
}

/**
 * POST /v1/uploads/complete
 *
 * Body: { upload_id: string, episode_count: number }
 *
 * Records the upload as completed, increments stats + daily counters.
 * Idempotent — calling twice on the same upload_id is a no-op.
 */
async function postUploadsComplete(request, env) {
  const body = await request.json().catch(() => null);
  if (!body) return jsonResponse(400, { error: "invalid_json" });

  const uploadId = body.upload_id;
  const episodeCount = Number(body.episode_count) || 0;
  if (!uploadId) return jsonResponse(400, { error: "missing_upload_id" });

  const upload = await env.DB.prepare(
    `SELECT upload_id, contributor_id, byte_size, status, signed_at FROM uploads WHERE upload_id = ?`
  ).bind(uploadId).first();
  if (!upload) return jsonResponse(404, { error: "upload_not_found" });
  if (upload.status !== "pending") {
    return jsonResponse(200, { status: upload.status, idempotent: true });
  }

  const nowIso = new Date().toISOString();
  const utcDate = nowIso.slice(0, 10);

  // Mark the upload completed + update aggregate counters atomically.
  await env.DB.batch([
    env.DB.prepare(
      `UPDATE uploads SET status = 'completed', completed_at = ? WHERE upload_id = ?`
    ).bind(nowIso, uploadId),
    env.DB.prepare(
      `UPDATE contributors
         SET total_episodes = total_episodes + ?,
             total_bytes = total_bytes + ?,
             total_uploads = total_uploads + 1,
             last_active_at = ?
         WHERE contributor_id = ?`
    ).bind(episodeCount, upload.byte_size, nowIso, upload.contributor_id),
    env.DB.prepare(
      `INSERT INTO daily_uploads (contributor_id, utc_date, bytes_uploaded, uploads_count)
         VALUES (?, ?, ?, 1)
         ON CONFLICT(contributor_id, utc_date) DO UPDATE SET
           bytes_uploaded = bytes_uploaded + excluded.bytes_uploaded,
           uploads_count = uploads_count + 1`
    ).bind(upload.contributor_id, utcDate, upload.byte_size),
  ]);

  return jsonResponse(200, { status: "completed", upload_id: uploadId });
}

/**
 * POST /v1/revoke/cascade
 *
 * Body: { contributor_id: string, scope: "all" | "future_only" }
 *
 * Marks the contributor for revoke. The actual purge cascade (delete R2
 * objects + rebuild derived datasets + email buyers) runs as a separate
 * background job; this endpoint only enqueues the request and updates
 * the contributor's revoked_at marker so future signs are refused.
 */
async function postRevokeCascade(request, env) {
  const body = await request.json().catch(() => null);
  if (!body) return jsonResponse(400, { error: "invalid_json" });
  const contributorId = body.contributor_id;
  const scope = body.scope || "all";
  if (!contributorId) return jsonResponse(400, { error: "missing_contributor_id" });
  if (!["all", "future_only"].includes(scope)) {
    return jsonResponse(400, { error: "invalid_scope", got: scope });
  }
  return await initiateRevoke(env, contributorId, scope, "customer");
}

/**
 * GET /v1/contributors/:id/stats
 *
 * Returns the contributor's running totals. Used by `reflex contribute --status`.
 */
async function getContributorStats(contributorId, env) {
  if (!contributorId) return jsonResponse(400, { error: "missing_contributor_id" });
  const row = await env.DB.prepare(
    `SELECT contributor_id, tier, first_seen_at, last_active_at,
            total_episodes, total_bytes, total_uploads, revoked_at
       FROM contributors WHERE contributor_id = ?`
  ).bind(contributorId).first();
  if (!row) return jsonResponse(404, { error: "not_found" });
  return jsonResponse(200, row);
}

// ---------- helpers ----------

async function initiateRevoke(env, contributorId, scope, source) {
  const requestId = `rev_${randomHex(16)}`;
  const nowIso = new Date().toISOString();

  // Mark contributor revoked. Idempotent; first call wins.
  await env.DB.prepare(
    `INSERT INTO contributors (contributor_id, tier, first_seen_at, last_active_at, revoked_at)
       VALUES (?, 'unknown', ?, ?, ?)
       ON CONFLICT(contributor_id) DO UPDATE SET
         revoked_at = COALESCE(revoked_at, excluded.revoked_at),
         last_active_at = excluded.last_active_at`
  ).bind(contributorId, nowIso, nowIso, nowIso).run();

  await env.DB.prepare(
    `INSERT INTO revoke_requests (request_id, contributor_id, requested_at, scope, status, notes)
       VALUES (?, ?, ?, ?, 'pending', ?)`
  ).bind(requestId, contributorId, nowIso, scope, `source=${source}`).run();

  // Optional: alert via Slack so an operator can run the cascade job.
  if (env.SLACK_WEBHOOK_URL) {
    await postSlack(env.SLACK_WEBHOOK_URL, {
      text: `Curate revoke requested — contributor_id=${contributorId} scope=${scope} source=${source}. Cascade SLA: ${REVOKE_SLA_DAYS} days.`,
    }).catch((e) => console.error("Slack post failed:", e.message));
  }

  return jsonResponse(200, {
    request_id: requestId,
    contributor_id: contributorId,
    sla_days: REVOKE_SLA_DAYS,
    note: "Cascade purge runs as a background job. Status: GET /v1/contributors/:id/stats.",
  });
}

function isSafeFileName(name) {
  if (typeof name !== "string") return false;
  if (name.length === 0 || name.length > 255) return false;
  if (name.includes("/") || name.includes("\\") || name.includes("\x00")) return false;
  if (name.startsWith(".")) return false;
  return /^[A-Za-z0-9._-]+$/.test(name);
}

function randomHex(byteLen) {
  const bytes = new Uint8Array(byteLen);
  crypto.getRandomValues(bytes);
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function jsonResponse(status, obj) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function postSlack(url, payload) {
  await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}
