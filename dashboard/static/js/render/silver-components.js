/*
 * render/silver-components.js — #silver page component-health pill grid
 * (Phase 6). One pill per XAG-applicable-or-not signal, from the registry
 * snapshot (/api/control/registry?ticker=XAG-USD); tap a pill to reveal its
 * disabled reason (mobile — no hover tooltips).
 */

import { emptyState } from "../components/empty-state.js";

/**
 * @param {{registryTicker: object|null}} props
 *   registryTicker: the `registry` object from /api/control/registry?ticker=...
 *   ({applicable, disabled, signals}), or null while loading.
 * @returns {HTMLElement}
 */
export function silverComponentGrid({ registryTicker } = {}) {
  const card = document.createElement("article");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  if (!registryTicker) {
    card.append(emptyState("Loading registry…"));
    return card;
  }

  const grid = document.createElement("div");
  grid.style.display = "flex";
  grid.style.flexWrap = "wrap";
  grid.style.gap = "var(--sp-1)";

  const signals = registryTicker.signals || {};
  const names = Object.keys(signals).sort((a, b) => {
    const ea = signals[a].enabled_default,
      eb = signals[b].enabled_default;
    if (ea !== eb) return ea ? -1 : 1;
    return a.localeCompare(b);
  });
  for (const name of names) grid.append(_signalPill(name, signals[name]));
  card.append(grid);
  return card;
}

function _signalPill(name, info) {
  const wrap = document.createElement("div");
  const enabled = !!info.enabled_default;
  const color = enabled ? "var(--grn)" : "var(--gry)";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.style.fontSize = "var(--ty-xs)";
  btn.style.fontWeight = "600";
  btn.style.padding = "2px 6px";
  btn.style.borderRadius = "999px";
  btn.style.border = `1px solid ${color}`;
  btn.style.color = color;
  btn.style.background = "transparent";
  btn.style.cursor = "pointer";
  btn.textContent = name;
  wrap.append(btn);

  const reason = document.createElement("div");
  reason.style.fontSize = "var(--ty-xs)";
  reason.style.color = "var(--txm)";
  reason.style.maxWidth = "220px";
  reason.style.display = "none";
  reason.textContent =
    info.disabled_reason ||
    (enabled ? "enabled — voting for XAG-USD" : "no reason recorded");
  wrap.append(reason);

  btn.addEventListener("click", () => {
    reason.style.display = reason.style.display === "none" ? "block" : "none";
  });

  return wrap;
}
