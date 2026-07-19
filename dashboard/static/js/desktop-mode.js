/*
 * desktop-mode.js — layout override: Auto / Desktop / Mobile.
 *
 * Mobile-first remains the default ("Auto" — viewport media queries
 * decide, exactly like before this file grew a third state). The header
 * toggle cycles Auto -> Desktop -> Mobile -> Auto:
 *   - Desktop: `html.desktop-mode` forces the wide layout regardless of
 *     viewport (pre-2026-07-18 behaviour, unchanged).
 *   - Mobile (new, 2026-07-18): `html.mobile-mode` forces the phone
 *     layout even when the real viewport is >=1024px — lets the user
 *     check the mobile experience without shrinking a desktop browser
 *     window. CSS neutralizes the native @media(min-width:1024px)
 *     promotions under `:root.mobile-mode` (see layout.css/responsive.css).
 *
 * Persists choice to localStorage("pi-desktop-mode") = "auto" | "desktop"
 * | "mobile". Back-compat: pre-2026-07-18 binary values ("on"/"off") map
 * to "desktop"/"auto" on first read.
 */

import { lsGet, lsSet } from "./storage.js";

const KEY = "pi-desktop-mode";
const MODES = ["auto", "desktop", "mobile"];

let _current = "auto";
const _listeners = new Set();

/** Init from localStorage. Idempotent. Returns "auto" | "desktop" | "mobile". */
export function initDesktopMode() {
  const saved = lsGet(KEY);
  if (saved === "on") _current = "desktop";
  else if (saved === "off") _current = "auto";
  else if (MODES.includes(saved)) _current = saved;
  else _current = "auto";
  _apply(_current);
  return _current;
}

/** Cycle Auto -> Desktop -> Mobile -> Auto, persist, notify, return new mode. */
export function cycleDesktopMode() {
  const idx = MODES.indexOf(_current);
  const next = MODES[(idx + 1) % MODES.length];
  _current = next;
  lsSet(KEY, next);
  _apply(next);
  _listeners.forEach((fn) => { try { fn(next); } catch (e) { console.warn(e); } });
  return next;
}

export function getDesktopMode() { return _current; }

export function subscribeDesktopMode(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

function _apply(mode) {
  const html = document.documentElement;
  html.classList.toggle("desktop-mode", mode === "desktop");
  html.classList.toggle("mobile-mode", mode === "mobile");
}
