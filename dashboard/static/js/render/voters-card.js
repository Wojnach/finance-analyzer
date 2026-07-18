/*
 * render/voters-card.js — per-signal voting truth ("Voters" card).
 *
 * Replaces the old LLM Inference Health card (2026-07-18). That card
 * showed lifetime call-success counters as if they meant "currently
 * voting" — the "100% green but force-HOLD" trap documented in
 * system_status.py's `_voters` docstring (claude_fundamental and forecast
 * both sat at ~100% success while permanently DISABLED). This card
 * renders the curated `voters` payload instead: one row per signal, a
 * state pill, and the reason it's (not) voting.
 *
 * Reads `/api/system_status` payload's `voters` section. Used to also
 * render an `llm_inference` footer (chronos/kronos lifetime call-success
 * counters) — removed 2026-07-18 as the same lifetime-counter-as-health
 * trap this card exists to fix (see the docstring above and
 * system_status.py's `_voters` docstring). The Phase 4 component
 * registry is expected to replace this whole card.
 */

import { fAgo } from "../format.js";
import { sectionErrorChip } from "../components/section-error-chip.js";

const STATE_COLOR = {
  VOTING: "var(--grn)",
  SHADOW: "var(--blu)",
  DISABLED: "var(--gry)",
  GATED_REMOTE_DOWN: "var(--yel)",
  PAUSED_LLM_FLAG: "var(--yel)",
  PAUSED_LOOP_DOWN: "var(--yel)",
};

// Interesting-first ordering: "why is this NOT voting" outranks the
// boring "disabled, working as intended" rows.
const STATE_ORDER = [
  "VOTING", "SHADOW", "GATED_REMOTE_DOWN", "PAUSED_LLM_FLAG", "PAUSED_LOOP_DOWN", "DISABLED",
];

/** @returns {HTMLElement} */
export function votersCard(votersPayload) {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0 0 var(--sp-2) 0";
  title.textContent = "Voters";
  card.append(title);

  if (votersPayload?.error) card.append(sectionErrorChip(votersPayload.error));

  const entries = votersPayload && !votersPayload.error ? Object.entries(votersPayload) : [];
  if (!entries.length) {
    if (!votersPayload?.error) {
      const empty = document.createElement("div");
      empty.style.color = "var(--txm)";
      empty.style.fontSize = "var(--ty-sm)";
      empty.textContent = "no voter data yet";
      card.append(empty);
    }
    return card;
  }

  entries.sort((a, b) => {
    const ra = STATE_ORDER.indexOf(a[1]?.state);
    const rb = STATE_ORDER.indexOf(b[1]?.state);
    return (ra === -1 ? STATE_ORDER.length : ra) - (rb === -1 ? STATE_ORDER.length : rb);
  });

  for (const [name, v] of entries) card.append(_voterRow(name, v));

  return card;
}

function _voterRow(name, v) {
  const row = document.createElement("div");
  row.style.padding = "var(--sp-2) 0";
  row.style.borderTop = "1px solid var(--bdr)";
  row.style.minHeight = "var(--tap-min)";
  row.style.cursor = "pointer";

  const top = document.createElement("div");
  top.style.display = "flex";
  top.style.justifyContent = "space-between";
  top.style.alignItems = "center";
  top.style.gap = "var(--sp-2)";

  const label = document.createElement("span");
  label.style.fontWeight = "600";
  label.style.fontSize = "var(--ty-sm)";
  label.style.color = "var(--tx)";
  label.textContent = name;
  top.append(label);

  const state = (v?.state || "?").toUpperCase();
  const pill = document.createElement("span");
  pill.textContent = state;
  pill.style.fontSize = "var(--ty-xs)";
  pill.style.fontWeight = "700";
  pill.style.padding = "1px 6px";
  pill.style.borderRadius = "999px";
  pill.style.whiteSpace = "nowrap";
  const color = STATE_COLOR[state] || "var(--txm)";
  pill.style.color = color;
  pill.style.border = `1px solid ${color}`;
  top.append(pill);
  row.append(top);

  const reasonWrap = document.createElement("div");
  reasonWrap.style.fontSize = "var(--ty-xs)";
  reasonWrap.style.color = "var(--txm)";
  reasonWrap.style.marginTop = "2px";
  reasonWrap.style.whiteSpace = "nowrap";
  reasonWrap.style.overflow = "hidden";
  reasonWrap.style.textOverflow = "ellipsis";
  reasonWrap.textContent = v?.reason || "";
  row.append(reasonWrap);

  const activityWrap = document.createElement("div");
  activityWrap.style.fontSize = "var(--ty-xs)";
  activityWrap.style.color = "var(--txd)";
  activityWrap.textContent = v?.last_activity_ts ? `active ${fAgo(v.last_activity_ts)}` : "no recent activity";
  row.append(activityWrap);

  // Tap to expand the (often long) reason — truncated by default so rows
  // stay scannable.
  let expanded = false;
  row.addEventListener("click", () => {
    expanded = !expanded;
    reasonWrap.style.whiteSpace = expanded ? "normal" : "nowrap";
  });

  return row;
}
