/*
 * render/trading-status-card.js — per-bot Avanza trading status.
 *
 * Reads `/api/trading_status` payload (see dashboard/trading_status.py)
 * and renders one row per bot with:
 *   - Bot label
 *   - State badge (color-coded by state)
 *   - Reason text (why no trade right now, or what just happened)
 *
 * Tapping a row navigates to that bot's existing detail view.
 */

import * as router from "../router.js";

const STATE_COLORS = {
  TRADING:       "var(--grn)",
  SCANNING:      "var(--cyn)",
  COOLDOWN:      "var(--yel)",
  OUTSIDE_HOURS: "var(--txm)",
  HALTED:        "var(--red)",
  UNKNOWN:       "var(--txm)",
};

const BOT_ROUTES = {
  golddigger: "golddigger",
  metals:     "metals",
  elongir:    "elongir",
  fishing:    "fish",
};

/** @returns {HTMLElement} */
export function tradingStatusCard(payload) {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const titleRow = document.createElement("div");
  titleRow.style.display = "flex";
  titleRow.style.justifyContent = "space-between";
  titleRow.style.alignItems = "baseline";
  titleRow.style.marginBottom = "var(--sp-2)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0";
  title.textContent = "Trading status";
  titleRow.append(title);

  const sessionPill = document.createElement("span");
  sessionPill.style.fontSize = "var(--ty-xs)";
  sessionPill.style.color = payload?.session_open ? "var(--grn)" : "var(--txm)";
  sessionPill.textContent = payload?.session_open ? "session open" : "session closed";
  titleRow.append(sessionPill);

  card.append(titleRow);

  const bots = Array.isArray(payload?.bots) ? payload.bots : [];
  if (!bots.length) {
    const empty = document.createElement("div");
    empty.style.color = "var(--txm)";
    empty.style.fontSize = "var(--ty-sm)";
    empty.textContent = "no bot state available";
    card.append(empty);
    return card;
  }

  for (const bot of bots) {
    card.append(_botRow(bot));
  }
  return card;
}

function _botRow(bot) {
  const route = BOT_ROUTES[bot.bot] || "health";
  const row = document.createElement("button");
  row.type = "button";
  row.className = "card--tap";
  row.style.display = "flex";
  row.style.justifyContent = "space-between";
  row.style.alignItems = "center";
  row.style.gap = "var(--sp-2)";
  row.style.width = "100%";
  row.style.padding = "var(--sp-2) 0";
  row.style.background = "transparent";
  row.style.border = "0";
  row.style.borderTop = "1px solid var(--bd)";
  row.style.minHeight = "var(--tap-min)";
  row.style.textAlign = "left";
  row.style.cursor = "pointer";
  row.addEventListener("click", () => router.navigate(route));

  const left = document.createElement("div");
  left.style.flex = "1";
  left.style.minWidth = "0";

  const label = document.createElement("div");
  label.style.fontWeight = "600";
  label.style.color = "var(--tx)";
  label.textContent = bot.label || bot.bot;
  left.append(label);

  const reason = document.createElement("div");
  reason.style.fontSize = "var(--ty-sm)";
  reason.style.color = "var(--txm)";
  reason.style.overflow = "hidden";
  reason.style.textOverflow = "ellipsis";
  reason.style.whiteSpace = "nowrap";
  reason.textContent = bot.reason || "—";
  left.append(reason);
  row.append(left);

  const stateBadge = document.createElement("span");
  stateBadge.textContent = (bot.state || "UNKNOWN").replace("_", " ");
  stateBadge.style.fontSize = "var(--ty-xs)";
  stateBadge.style.fontWeight = "600";
  stateBadge.style.padding = "2px var(--sp-2)";
  stateBadge.style.borderRadius = "4px";
  stateBadge.style.background = STATE_COLORS[bot.state] || "var(--txm)";
  stateBadge.style.color = "var(--bg)";
  stateBadge.style.whiteSpace = "nowrap";
  row.append(stateBadge);

  return row;
}
