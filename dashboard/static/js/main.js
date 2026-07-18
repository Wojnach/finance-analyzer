/*
 * main.js — module entry. Bootstraps the mobile dashboard.
 *
 * Responsibilities:
 *  - Init theme.
 *  - Wire up bottom-nav clicks → router.navigate.
 *  - Init router (hashchange listener + initial mount).
 *  - Subscribe to error state → render banner.
 *  - Register service worker (gracefully — SW is optional in batch 9+).
 *
 * Views register themselves with the router (see js/views/*). Views are
 * imported here rather than by lazy dynamic import to keep the dependency
 * graph explicit. Each subsequent batch in PLAN.md adds a new view import.
 */

import * as state from "./state.js";
import * as router from "./router.js";
import * as polling from "./polling.js";
import { initTheme, toggleTheme } from "./theme.js";
import {
  initDesktopMode, cycleDesktopMode, getDesktopMode, subscribeDesktopMode,
} from "./desktop-mode.js";
import { fAgo } from "./format.js";
import { showToast } from "./components/toast.js";

// ---- View imports ---------------------------------------------------------
// Each view module self-registers with router.register() on import.
// Adding a view to the bottom-nav routes is just a one-line import here.
//
// Batch 4: views/home.js
// Batch 5+: decisions, signals, more, etc.
// Until a route is implemented, router.js renders a fallback that links to /legacy.

import "./views/home.js";
import "./views/decisions.js"; // also imports decision-detail.js internally
import "./views/signals.js";
import "./views/more.js";
import "./views/health.js";
import "./views/cost.js";
import "./views/messages.js";
import "./views/settings.js";
import "./views/equity.js";
import "./views/metals.js";
import "./views/golddigger.js";
import "./views/avanza.js";
import "./views/assets.js";
import "./views/prices.js";
import "./views/portfolio.js";
import "./views/llm_leaderboard.js";
import "./views/loop_processes.js";
import "./views/pickups.js";
import "./views/control.js";

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initDesktopMode();

  // Wire desktop-mode toggle button (header). Mobile ("Auto") remains the
  // default; tapping cycles Auto -> Desktop -> Mobile -> Auto. Choice
  // persists in localStorage. The visible label shows the CURRENT mode
  // (not the mode a tap would switch to) — matches how the glyph/active
  // state already worked pre-2026-07-18 and avoids "guess what tapping
  // does" ambiguity; the title tooltip spells out what a tap does next.
  const dmBtn = document.getElementById("desktop-mode-toggle");
  if (dmBtn) {
    const GLYPH = { auto: "⊞", desktop: "▤", mobile: "▥" };
    const LABEL = { auto: "Auto", desktop: "Desktop", mobile: "Mobile" };
    const NEXT_LABEL = { auto: "Desktop", desktop: "Mobile", mobile: "Auto" };
    const syncBtn = (mode) => {
      dmBtn.classList.toggle("active", mode !== "auto");
      const glyph = dmBtn.querySelector(".glyph");
      if (glyph) glyph.textContent = GLYPH[mode] || GLYPH.auto;
      let label = dmBtn.querySelector(".label");
      if (!label) {
        label = document.createElement("span");
        label.className = "label";
        dmBtn.append(label);
      }
      label.textContent = LABEL[mode] || LABEL.auto;
      dmBtn.title = `Layout: ${LABEL[mode] || LABEL.auto} — tap for ${NEXT_LABEL[mode] || NEXT_LABEL.auto}`;
    };
    syncBtn(getDesktopMode());
    subscribeDesktopMode(syncBtn);
    dmBtn.addEventListener("click", () => cycleDesktopMode());
  }

  // Wire refresh-dot (header). Tooltip shows "last refresh Xs ago"; tap
  // shows the same as a toast (useful on touch devices where hover/title
  // never fires). Reflects state.Slots.LAST_REFRESH, which every view's
  // polling task bumps on each completed fetch (js/polling.js) — global
  // chrome, so wired once here rather than per-view.
  const refreshDot = document.getElementById("refresh-dot");
  if (refreshDot) {
    const tooltipText = () => {
      const last = state.get(state.Slots.LAST_REFRESH);
      return last ? `last refresh ${fAgo(last)}` : "auto-refresh — no refresh yet";
    };
    const updateTooltip = () => { refreshDot.title = tooltipText(); };
    updateTooltip();
    state.subscribe(state.Slots.LAST_REFRESH, updateTooltip);
    const showRefreshToast = () => showToast(tooltipText());
    refreshDot.addEventListener("click", showRefreshToast);
    refreshDot.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showRefreshToast(); }
    });
  }

  // Wire bottom-nav buttons → hash routes.
  document.querySelectorAll(".bottom-nav__item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const route = btn.dataset.route || "home";
      router.navigate(route);
    });
  });

  // Update active state on bottom-nav as the route changes. Sub-routes
  // under "More" (health, messages, metals, golddigger, equity, settings)
  // keep the More tab highlighted.
  const MORE_SUB_ROUTES = new Set([
    "avanza", "assets", "prices", "portfolio", "health", "messages", "metals", "golddigger", "equity", "settings", "control",
  ]);
  state.subscribe(state.Slots.ROUTE, (parsed) => {
    const route = parsed?.name || "home";
    document.querySelectorAll(".bottom-nav__item").forEach((btn) => {
      const r = btn.dataset.route;
      let isActive = r === route;
      if (!isActive && r === "more") isActive = MORE_SUB_ROUTES.has(route);
      btn.classList.toggle("active", isActive);
    });
  });

  // Surface fetch/auth errors as a banner above the view.
  state.subscribe(state.Slots.ERROR, (msg) => {
    const root = document.getElementById("root");
    if (!root) return;
    const existing = root.querySelector(".banner--error.global-error");
    if (msg) {
      if (existing) {
        existing.textContent = msg;
      } else {
        const b = document.createElement("div");
        b.className = "banner banner--error global-error";
        b.textContent = msg;
        root.prepend(b);
      }
    } else if (existing) {
      existing.remove();
    }
  });

  // Init the router last so views see a fully-built shell.
  const root = document.getElementById("root");
  router.init(root);

  // Service-worker registration is deferred to batch 9. When sw.js exists
  // the registration code below activates it without blocking first paint.
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker
        .register("/static/sw.js")
        .catch(() => {/* sw not yet shipped — ignore */});
    });
  }
});

// Expose a tiny debug surface for the browser console only.
// (Intentional global; safe because it only re-exports read APIs.)
window.__pi = Object.freeze({
  state,
  router,
  polling,
  toggleTheme,
  cycleDesktopMode,
  getDesktopMode,
});
