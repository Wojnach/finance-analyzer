/*
 * render/system-status-hero.js — the home page's GREEN/YELLOW/RED hero.
 *
 * Reads the `/api/system_status` payload (see dashboard/system_status.py)
 * and produces a single tap-target card showing:
 *   - Big colored status badge
 *   - Up to 3 prominent reasons
 *   - Footer line with cycle count + heartbeat age
 *
 * Tap → /health for full drill-down.
 */

import * as router from "../router.js";

const STATE_TO_COLOR = {
  GREEN:  "var(--grn)",
  YELLOW: "var(--yel)",
  RED:    "var(--red)",
};

/** @returns {HTMLElement} */
export function systemStatusHero(payload) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "card card--tap system-status-hero";
  card.style.display = "block";
  card.style.width = "100%";
  card.style.textAlign = "left";
  card.style.minHeight = "var(--tap-min)";
  card.addEventListener("click", () => router.navigate("health"));

  const overall = (payload?.overall || "RED").toUpperCase();
  const reasons = Array.isArray(payload?.reasons) ? payload.reasons : ["loading…"];
  const color = STATE_TO_COLOR[overall] || "var(--txm)";

  // Top row: status badge + reasons summary
  const top = document.createElement("div");
  top.style.display = "flex";
  top.style.alignItems = "center";
  top.style.gap = "var(--sp-3)";

  const badge = document.createElement("span");
  badge.textContent = overall;
  badge.style.fontWeight = "700";
  badge.style.fontSize = "var(--ty-xl)";
  badge.style.padding = "var(--sp-1) var(--sp-2)";
  badge.style.borderRadius = "6px";
  badge.style.background = color;
  badge.style.color = "var(--bg)";
  badge.style.minWidth = "76px";
  badge.style.textAlign = "center";
  top.append(badge);

  const summary = document.createElement("div");
  summary.style.flex = "1";
  summary.style.display = "flex";
  summary.style.flexDirection = "column";
  summary.style.gap = "2px";
  for (const r of reasons.slice(0, 3)) {
    const line = document.createElement("div");
    line.style.color = "var(--tx)";
    line.style.fontSize = "var(--ty-md)";
    line.textContent = "• " + r;
    summary.append(line);
  }
  top.append(summary);
  card.append(top);

  // Footer: heartbeat + cycle count
  const hb = payload?.heartbeat || {};
  const ageS = Number.isFinite(hb.age_seconds) ? Math.round(hb.age_seconds) : null;
  const ageStr = ageS == null ? "?"
               : ageS < 60   ? `${ageS}s`
               : ageS < 3600 ? `${Math.floor(ageS / 60)}m ${ageS % 60}s`
               :               `${Math.floor(ageS / 3600)}h ${Math.floor((ageS % 3600) / 60)}m`;

  const foot = document.createElement("div");
  foot.style.marginTop = "var(--sp-2)";
  foot.style.color = "var(--txm)";
  foot.style.fontSize = "var(--ty-sm)";
  const exp = payload?.heartbeat?.expected_heartbeat_seconds;
  foot.append(document.createTextNode(`loop ${ageStr} ago`));
  if (Number.isFinite(exp)) {
    const sep = document.createElement("span");
    sep.style.color = "var(--txm)";
    sep.style.opacity = "0.6";
    sep.textContent = " / ";
    foot.append(sep);
    foot.append(document.createTextNode(`expected ~${exp}s`));
  }
  foot.append(document.createTextNode(` · cycle #${hb.cycle_count ?? "?"} · ${hb.error_count ?? 0} errors recorded`));
  card.append(foot);

  return card;
}
