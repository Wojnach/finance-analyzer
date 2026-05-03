/*
 * views/settings.js — settings: theme, polling, /legacy link, logout.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { toggleTheme } from "../theme.js";
import { dropCache } from "../fetch.js";

export const view = {
  mount(rootEl) {
    while (rootEl.firstChild) rootEl.removeChild(rootEl.firstChild);

    const v = document.createElement("section");
    v.className = "view view--settings";

    const t = document.createElement("h1");
    t.className = "section-title";
    t.textContent = "Settings";
    v.append(t);

    v.append(_themeRow());
    v.append(_pauseRow());
    v.append(_refreshRow());
    v.append(_legacyLink());
    v.append(_logoutRow());

    rootEl.append(v);
  },
  unmount() {},
};
router.register("settings", view);

// ---------------------------------------------------------------------------

function _row(label, control, hint = "") {
  const card = document.createElement("article");
  card.className = "card";
  card.style.display = "flex";
  card.style.alignItems = "center";
  card.style.justifyContent = "space-between";
  card.style.gap = "var(--sp-3)";
  card.style.marginBottom = "var(--sp-2)";

  const meta = document.createElement("div");
  const lbl = document.createElement("div");
  lbl.className = "card__title";
  lbl.textContent = label;
  meta.append(lbl);
  if (hint) {
    const h = document.createElement("div");
    h.className = "card__subtitle";
    h.textContent = hint;
    meta.append(h);
  }
  card.append(meta, control);
  return card;
}

function _themeRow() {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "icon-btn";
  btn.style.minWidth = "auto";
  btn.style.padding = "var(--sp-2) var(--sp-3)";
  btn.textContent = state.get(state.Slots.THEME) === "light" ? "→ Dark" : "→ Light";
  btn.addEventListener("click", () => {
    toggleTheme();
    btn.textContent = state.get(state.Slots.THEME) === "light" ? "→ Dark" : "→ Light";
  });
  return _row("Theme", btn, "Toggle dark/light. Persists per device.");
}

function _pauseRow() {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "icon-btn";
  btn.style.minWidth = "auto";
  btn.style.padding = "var(--sp-2) var(--sp-3)";
  btn.textContent = polling.isPaused() ? "Resume" : "Pause";
  btn.addEventListener("click", () => {
    polling.setPaused(!polling.isPaused());
    btn.textContent = polling.isPaused() ? "Resume" : "Pause";
  });
  return _row("Auto-refresh", btn,
    "Pause to stop polling everywhere. Visibility-aware regardless.");
}

function _refreshRow() {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "icon-btn";
  btn.style.minWidth = "auto";
  btn.style.padding = "var(--sp-2) var(--sp-3)";
  btn.textContent = "Refresh now";
  let resetTimer = null;
  btn.addEventListener("click", () => {
    dropCache();
    polling.fireAll();
    // Visual confirmation that the click registered: flash to green +
    // checkmark, revert ~1.4s later. Without this the button looks like
    // it did nothing because the network round-trip is async.
    btn.style.color = "var(--grn)";
    btn.style.borderColor = "var(--grn)";
    btn.textContent = "Refreshed ✓";
    if (resetTimer) clearTimeout(resetTimer);
    resetTimer = setTimeout(() => {
      btn.style.color = "";
      btn.style.borderColor = "";
      btn.textContent = "Refresh now";
      resetTimer = null;
    }, 1400);
  });
  return _row("Force refresh", btn,
    "Drops the in-memory ttl cache and refires every active polling task.");
}

function _legacyLink() {
  const a = document.createElement("a");
  a.href = "/legacy";
  a.className = "icon-btn";
  a.style.textDecoration = "none";
  a.style.minWidth = "auto";
  a.style.padding = "var(--sp-2) var(--sp-3)";
  a.textContent = "Open →";
  return _row("Legacy view", a,
    "Pre-redesign single-file dashboard. Useful if something on mobile breaks.");
}

function _logoutRow() {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "icon-btn";
  btn.style.color = "var(--red)";
  btn.style.borderColor = "var(--red)";
  btn.style.padding = "var(--sp-2) var(--sp-3)";
  btn.style.minWidth = "auto";
  btn.textContent = "Sign out";
  btn.addEventListener("click", () => {
    // The auth cookie is HttpOnly, so the server has to expire it. Navigate
    // to /logout which sends Set-Cookie: pf_dashboard_token=; Max-Age=0
    // and redirects to /. Pure JS document.cookie cannot do this.
    location.href = "/logout";
  });
  return _row("Sign out", btn,
    "Server-side logout: clears the HttpOnly auth cookie and redirects to /. CF Access still applies.");
}
