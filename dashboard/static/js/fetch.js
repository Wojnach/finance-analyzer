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

// Errors keyed by url so one endpoint's success can't wipe another
// endpoint's still-outstanding failure out of the banner (state.Slots.ERROR
// stays a single aggregate slot — the banner UX doesn't change, only what
// feeds it).
const _errorsByUrl = new Map(); // url -> message

function _setUrlError(url, message) {
  _errorsByUrl.set(url, message);
  state.set(state.Slots.ERROR, message);
}

function _clearUrlError(url) {
  if (!_errorsByUrl.delete(url)) return;
  const remaining = [..._errorsByUrl.values()].pop();
  state.set(state.Slots.ERROR, remaining ?? null);
}

/**
 * Fetch JSON with retry. Returns the parsed JSON on success or null on failure.
 * On failure, sets state.ui.error so views can display a banner.
 */
export async function fj(url, opts = {}) {
  const { retry = DEFAULT_RETRY, signal, ttl = 0, cacheKey = null } = opts;

  if (ttl > 0) {
    const key = cacheKey || url;
    const cached = _ttlCache.get(key);
    if (cached && Date.now() - cached.at < ttl) {
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
          // Auth failed — let the caller handle (show a re-auth hint).
          _setUrlError(url, `Auth failed (${r.status}). Try /?token=...`);
          return null;
        }
        throw new Error(`HTTP ${r.status} ${r.statusText}`);
      }
      const data = await r.json();
      if (ttl > 0)
        _ttlCache.set(cacheKey || url, { at: Date.now(), value: data });
      // Clear only this url's error — other endpoints' outstanding
      // failures must keep the banner up.
      _clearUrlError(url);
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
  _setUrlError(url, `Fetch failed: ${url.replace(/\?.*$/, "")}`);
  return null;
}

/** POST JSON convenience wrapper. */
export async function fpost(url, body, opts = {}) {
  try {
    const r = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
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
