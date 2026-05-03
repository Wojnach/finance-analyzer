/*
 * theme.js — dark/light theme toggle.
 *
 * Ported from legacy `initTheme` / `toggleTheme` / `getChartColors`
 * (index.html 927-944, 2941-2963). Persists to localStorage("pi-theme").
 * Toggles `html.light` class which CSS variables key off (tokens.css).
 */

import * as state from "./state.js";

const KEY = "pi-theme";

/**
 * Init from localStorage or prefers-color-scheme. Idempotent.
 * Returns the current theme: "light" | "dark".
 */
export function initTheme() {
  const saved = localStorage.getItem(KEY);
  let theme;
  if (saved === "light" || saved === "dark") {
    theme = saved;
  } else {
    // First visit: respect OS preference.
    theme = window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light" : "dark";
  }
  _applyTheme(theme);
  state.set(state.Slots.THEME, theme);
  return theme;
}

/** Toggle between light/dark, persist, return new theme. */
export function toggleTheme() {
  const next = state.get(state.Slots.THEME) === "light" ? "dark" : "light";
  _applyTheme(next);
  localStorage.setItem(KEY, next);
  state.set(state.Slots.THEME, next);
  return next;
}

/** Apply the theme to the document. */
function _applyTheme(theme) {
  const html = document.documentElement;
  if (theme === "light") html.classList.add("light");
  else html.classList.remove("light");
}

/**
 * Color tokens for charts. Reads computed CSS variables so chart colors
 * always match the active theme. Call inside chart options/datasets.
 */
export function getChartColors() {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  return {
    text:   v("--tx"),
    dim:    v("--txd"),
    muted:  v("--txm"),
    grid:   v("--bdr"),
    bg:     v("--bg"),
    card:   v("--card"),
    green:  v("--grn"),
    red:    v("--red"),
    cyan:   v("--cyn"),
    blue:   v("--blu"),
    orange: v("--org"),
    yellow: v("--yel"),
    gray:   v("--gry"),
  };
}
