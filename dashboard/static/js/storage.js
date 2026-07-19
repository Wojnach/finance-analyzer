/*
 * storage.js — Web Storage wrappers that never throw.
 *
 * localStorage/sessionStorage access can throw SecurityError (private
 * browsing with storage disabled, some mobile Safari configs, embedded
 * webviews). Uncaught, that aborts whatever caller ran first — boot
 * (theme.js/desktop-mode.js run at import time) or a render pass
 * (freshness-banner.js). Every read/write goes through here so a blocked
 * store degrades to an in-memory default instead of blanking the SPA.
 */

const _mem = new Map(); // fallback store when Storage is unavailable

export function lsGet(key) {
  try {
    return localStorage.getItem(key);
  } catch (_) {
    return _mem.has(`ls:${key}`) ? _mem.get(`ls:${key}`) : null;
  }
}

export function lsSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_) {
    _mem.set(`ls:${key}`, value);
  }
}

export function ssGet(key) {
  try {
    return sessionStorage.getItem(key);
  } catch (_) {
    return _mem.has(`ss:${key}`) ? _mem.get(`ss:${key}`) : null;
  }
}

export function ssSet(key, value) {
  try {
    sessionStorage.setItem(key, value);
  } catch (_) {
    _mem.set(`ss:${key}`, value);
  }
}
