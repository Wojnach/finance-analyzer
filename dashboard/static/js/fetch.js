/*
 * fetch.js — fetch wrapper with retry, ttl cache, and error surface.
 *
 * Ported from the legacy `fj()` (index.html ~880) but simplified — no
 * static-fallback fetch (we're behind CF Tunnel + cookie auth, the legacy
 * fallback to /static/api-data/ is closed since 2026-04-30).
 *
 * `credentials: 'same-origin'` is the browser default and what we want;
 * specify it explicitly so a future change can't accidentally break
 * cookie-bearing requests.
 */

import * as state from "./state.js";

const _ttlCache = new Map(); // key -> { at, value }
const DEFAULT_RETRY = { attempts: 2, baseDelayMs: 250 };

/**
 * Fetch JSON with retry. Returns the parsed JSON on success or null on failure.
 * On failure, sets state.ui.error so views can display a banner.
 */
export async function fj(url, opts = {}) {
  const { retry = DEFAULT_RETRY, signal, ttl = 0, cacheKey = null } = opts;

  if (ttl > 0) {
    const key = cacheKey || url;
    const cached = _ttlCache.get(key);
    if (cached && (Date.now() - cached.at) < ttl) {
      return cached.value;
    }
  }

  let lastErr = null;
  for (let attempt = 0; attempt <= retry.attempts; attempt++) {
    try {
      const r = await fetch(url, {
        credentials: "same-origin",
        signal,
        headers: { Accept: "application/json" },
      });
      if (!r.ok) {
        if (r.status === 401 || r.status === 403) {
          // Auth failed — let the caller handle (UI may redirect to /legacy
          // or show a re-auth hint).
          state.set(state.Slots.ERROR, `Auth failed (${r.status}). Try /legacy?token=...`);
          return null;
        }
        throw new Error(`HTTP ${r.status} ${r.statusText}`);
      }
      const data = await r.json();
      if (ttl > 0) _ttlCache.set(cacheKey || url, { at: Date.now(), value: data });
      // Clear any prior error on first successful call
      if (state.get(state.Slots.ERROR)) state.set(state.Slots.ERROR, null);
      return data;
    } catch (err) {
      lastErr = err;
      if (signal && signal.aborted) throw err;
      if (attempt < retry.attempts) {
        await _sleep(retry.baseDelayMs * Math.pow(2, attempt));
        continue;
      }
    }
  }
  // All attempts exhausted
  console.warn("fetch failed:", url, lastErr);
  state.set(state.Slots.ERROR, `Fetch failed: ${url.replace(/\?.*$/, "")}`);
  return null;
}

/** POST JSON convenience wrapper. */
export async function fpost(url, body, opts = {}) {
  try {
    const r = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      signal: opts.signal,
    });
    if (!r.ok) return null;
    return await r.json();
  } catch (err) {
    console.warn("post failed:", url, err);
    return null;
  }
}

/** Drop the entire TTL cache (useful from settings "force refresh"). */
export function dropCache() {
  _ttlCache.clear();
}

function _sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}
