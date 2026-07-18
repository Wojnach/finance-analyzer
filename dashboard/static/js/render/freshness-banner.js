/*
 * render/freshness-banner.js — "the data on this page might be stale" banner.
 *
 * Fires when Layer 1 (pf-dataloop) is inactive OR its signal-critical
 * backing files have gone frozen (see system_status.py's `layer1` /
 * `sources` sections). Without this, the home page kept rendering the
 * last-known signal/health numbers as if they were live — the exact
 * failure mode that let the 2026-07-17 pf-dataloop pause go unnoticed
 * for a full day.
 *
 * Dismissible per-session (sessionStorage) — reappears on the next new
 * session (tab reload after the browser fully closes it, or a fresh
 * tab) while the underlying condition still holds. Tap navigates to
 * /health, which renders the same heartbeat/errors detail (there's no
 * #control view yet — Phase 3 — so /health is the closest existing
 * drill-down).
 */

import * as router from "../router.js";

const DISMISS_KEY = "pi-freshness-banner-dismissed";

/** @returns {HTMLElement|null} null when nothing to show, or dismissed. */
export function freshnessBanner(sys) {
  if (!sys) return null;

  const layer1 = sys.layer1;
  const sources = sys.sources || {};
  const layer1Down = layer1 && layer1.active === false;
  const signalFrozen = !!(sources["signal_log.jsonl"]?.frozen || sources["health_state.json"]?.frozen);
  if (!layer1Down && !signalFrozen) return null;

  if (sessionStorage.getItem(DISMISS_KEY) === "1") return null;

  const banner = document.createElement("div");
  banner.className = "banner banner--warn banner--sticky";
  banner.style.cursor = "pointer";
  banner.addEventListener("click", () => router.navigate("health"));

  const text = document.createElement("span");
  text.style.flex = "1";
  text.textContent = _message(layer1Down, sources);
  banner.append(text);

  const dismiss = document.createElement("button");
  dismiss.type = "button";
  dismiss.className = "icon-btn";
  dismiss.setAttribute("aria-label", "Dismiss");
  dismiss.textContent = "×";
  dismiss.style.minWidth = "auto";
  dismiss.style.minHeight = "auto";
  dismiss.addEventListener("click", (e) => {
    e.stopPropagation();
    sessionStorage.setItem(DISMISS_KEY, "1");
    banner.remove();
  });
  banner.append(dismiss);

  return banner;
}

function _message(layer1Down, sources) {
  const since = sources["signal_log.jsonl"]?.mtime ?? sources["health_state.json"]?.mtime;
  const sinceStr = since ? _relativeAge(since) : "an unknown time";
  if (layer1Down) return `Layer 1 paused — signal data frozen since ${sinceStr}`;
  return `Signal data frozen since ${sinceStr} — Layer 1 may be stuck`;
}

function _relativeAge(mtimeSeconds) {
  const ms = Date.now() - mtimeSeconds * 1000;
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
