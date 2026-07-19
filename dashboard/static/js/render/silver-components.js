/*
 * render/silver-components.js — #silver page component-health pill grid
 * (Phase 6). One pill per XAG-applicable-or-not signal, from the registry
 * snapshot (/api/control/registry?ticker=XAG-USD); tap a pill to reveal its
 * disabled reason (mobile — no hover tooltips).
 *
 * Colored/labeled from each signal's `voter_state` (registry already
 * carries this — {state, reason} per signal, same VOTING/SHADOW/DISABLED/
 * PAUSED_* vocabulary as voters-card.js), not `enabled_default`. A signal
 * enabled_default=true but shadow-throttled (e.g. claude_fundamental) is
 * NOT voting — labeling it "enabled — voting" was the bug.
 */

import { emptyState } from "../components/empty-state.js";

const STATE_COLOR = {
  VOTING: "var(--grn)",
  SHADOW: "var(--blu)",
  DISABLED: "var(--gry)",
  GATED_REMOTE_DOWN: "var(--yel)",
  PAUSED_LLM_FLAG: "var(--yel)",
  PAUSED_LOOP_DOWN: "var(--yel)",
};

const STATE_LABEL = {
  VOTING: "voting",
  SHADOW: "shadow (tracked, not voting)",
  DISABLED: "disabled",
  GATED_REMOTE_DOWN: "paused — remote down",
  PAUSED_LLM_FLAG: "paused — LLM off",
  PAUSED_LOOP_DOWN: "paused — loop down",
};

// Interesting-first ordering, matching voters-card.js.
const STATE_ORDER = [
  "VOTING", "SHADOW", "GATED_REMOTE_DOWN", "PAUSED_LLM_FLAG", "PAUSED_LOOP_DOWN", "DISABLED",
];

function _voterState(info) {
  // voter_state comes from the same system_status._voter_state machinery
  // voters-card.js reads (/api/control/registry already embeds it per
  // signal). Fall back to enabled_default only if a future registry
  // response ever drops the field.
  const s = info.voter_state?.state;
  return (s || (info.enabled_default ? "VOTING" : "DISABLED")).toUpperCase();
}

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
    const ra = STATE_ORDER.indexOf(_voterState(signals[a]));
    const rb = STATE_ORDER.indexOf(_voterState(signals[b]));
    const oa = ra === -1 ? STATE_ORDER.length : ra;
    const ob = rb === -1 ? STATE_ORDER.length : rb;
    if (oa !== ob) return oa - ob;
    return a.localeCompare(b);
  });
  for (const name of names) grid.append(_signalPill(name, signals[name]));
  card.append(grid);
  return card;
}

function _signalPill(name, info) {
  const wrap = document.createElement("div");
  const vState = _voterState(info);
  const color = STATE_COLOR[vState] || "var(--gry)";
  const label = STATE_LABEL[vState] || vState.toLowerCase();

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
  const why = info.voter_state?.reason || info.disabled_reason;
  reason.textContent = why ? `${label} — ${why}` : label;
  wrap.append(reason);

  btn.addEventListener("click", () => {
    reason.style.display = reason.style.display === "none" ? "block" : "none";
  });

  return wrap;
}
