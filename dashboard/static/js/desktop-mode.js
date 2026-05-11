/*
 * desktop-mode.js — opt-in wide layout for desktop browsers.
 *
 * Mobile-first remains the default. When the user clicks the "Desktop"
 * header button, we add `html.desktop-mode`. CSS mirrors all
 * `@media (min-width: 1024px)` rules under `:root.desktop-mode { ... }`
 * so the toggle promotes the layout regardless of viewport width.
 *
 * Persists choice to localStorage("pi-desktop-mode") = "on" | "off".
 * Default = "off" (mobile-first stays).
 */

import * as state from "./state.js";

const KEY = "pi-desktop-mode";

// New state slot. Not part of state.Slots enum because the slot list is
// a frozen object; we expose only a getter/setter to keep this scoped.
let _current = "off";
const _listeners = new Set();

/** Init from localStorage. Idempotent. Returns "on" | "off". */
export function initDesktopMode() {
  const saved = localStorage.getItem(KEY);
  _current = saved === "on" ? "on" : "off";
  _apply(_current);
  return _current;
}

/** Toggle between on/off, persist, notify listeners, return new mode. */
export function toggleDesktopMode() {
  const next = _current === "on" ? "off" : "on";
  _current = next;
  localStorage.setItem(KEY, next);
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
  if (mode === "on") html.classList.add("desktop-mode");
  else html.classList.remove("desktop-mode");
}
