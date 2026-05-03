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
import "./views/messages.js";
import "./views/settings.js";
import "./views/equity.js";
import "./views/metals.js";
import "./views/golddigger.js";
import "./views/avanza.js";
import "./views/assets.js";
import "./views/prices.js";

document.addEventListener("DOMContentLoaded", () => {
  initTheme();

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
    "avanza", "assets", "prices", "health", "messages", "metals", "golddigger", "equity", "settings",
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
});
