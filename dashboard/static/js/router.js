/*
 * router.js — hash-based router with mount/unmount lifecycle.
 *
 * Routes look like `#home`, `#decisions`, `#decisions/<id>`, `#more/health`.
 * Each registered view exports `{mount(root, params), unmount?()}`.
 *
 * This module is route-only — it doesn't know about polling or fetching.
 * Views are responsible for registering polling tasks on mount and
 * unregistering on unmount.
 *
 * Security: route segment + params come from URL hash and are user-
 * controllable. Never write them via innerHTML; always use DOM API
 * (textContent / append) to keep XSS impossible.
 */

import * as state from "./state.js";

const _views = new Map(); // name -> {mount, unmount}
let _root = null;
let _current = null;       // {name, params, view}
let _onChange = null;

/**
 * Register a view by name. Idempotent.
 * `view` shape: { mount(root, params): void, unmount?(): void }
 */
export function register(name, view) {
  if (!view || typeof view.mount !== "function") {
    throw new Error(`router: view "${name}" must export mount()`);
  }
  _views.set(name, view);
}

/** Set the DOM mount point. Call once at boot. */
export function init(rootEl, opts = {}) {
  _root = rootEl;
  _onChange = opts.onChange || null;
  window.addEventListener("hashchange", _handleChange);
  window.addEventListener("popstate",   _handleChange);
  // Initial mount
  _handleChange();
}

/** Programmatically navigate to a route. */
export function navigate(name, params = null) {
  const hash = params ? `#${name}/${_encodeParams(params)}` : `#${name}`;
  if (location.hash !== hash) {
    location.hash = hash;
  } else {
    // Same hash — manually re-fire so views can re-mount on user-tap-same-tab.
    _handleChange();
  }
}

/** Read current parsed route. */
export function current() {
  return _current ? { name: _current.name, params: _current.params } : null;
}

// ---------------------------------------------------------------------------
// Internal — DOM-safe rendering
// ---------------------------------------------------------------------------

function _handleChange() {
  if (!_root) return;
  const parsed = _parseHash(location.hash);
  // Unmount previous
  if (_current && _current.view && typeof _current.view.unmount === "function") {
    try { _current.view.unmount(); } catch (e) { console.error("unmount error", e); }
  }
  _clearRoot();

  const view = _views.get(parsed.name);
  if (!view) {
    _renderFallback(`View "${parsed.name}" not yet implemented.`);
    _current = { name: parsed.name, params: parsed.params, view: null };
  } else {
    try {
      view.mount(_root, parsed.params);
      _current = { name: parsed.name, params: parsed.params, view };
    } catch (e) {
      console.error(`mount error: ${parsed.name}`, e);
      _renderError(`Failed to mount "${parsed.name}": ${e?.message || "unknown error"}`);
      _current = { name: parsed.name, params: parsed.params, view: null };
    }
  }

  state.set(state.Slots.ROUTE, parsed);
  if (_onChange) _onChange(parsed);
}

function _clearRoot() {
  while (_root.firstChild) _root.removeChild(_root.firstChild);
}

/** Build the "view not yet implemented" message via DOM API (XSS-safe). */
function _renderFallback(message) {
  const wrap = document.createElement("div");
  wrap.className = "empty";
  const p1 = document.createElement("p");
  p1.textContent = message;
  const p2 = document.createElement("p");
  p2.append("Visit the ");
  const a = document.createElement("a");
  a.href = "/legacy";
  a.textContent = "legacy view";
  p2.append(a, ".");
  wrap.append(p1, p2);
  _root.append(wrap);
}

function _renderError(message) {
  const banner = document.createElement("div");
  banner.className = "banner banner--error";
  banner.textContent = message;
  _root.append(banner);
}

function _parseHash(hash) {
  const raw = (hash || "").replace(/^#/, "").trim();
  if (!raw) return { name: "home", params: null, raw: "" };
  const slash = raw.indexOf("/");
  if (slash === -1) return { name: raw, params: null, raw };
  const name = raw.slice(0, slash);
  const tail = raw.slice(slash + 1);
  return { name, params: _decodeParams(tail), raw };
}

function _encodeParams(p) {
  if (typeof p === "string") return encodeURIComponent(p);
  if (p && typeof p === "object") {
    return Object.keys(p)
      .map((k) => encodeURIComponent(k) + "=" + encodeURIComponent(p[k]))
      .join("&");
  }
  return "";
}

function _decodeParams(tail) {
  if (!tail) return null;
  if (!tail.includes("=")) return decodeURIComponent(tail);
  const out = Object.create(null);
  for (const pair of tail.split("&")) {
    const [k, v = ""] = pair.split("=");
    out[decodeURIComponent(k)] = decodeURIComponent(v);
  }
  return out;
}
