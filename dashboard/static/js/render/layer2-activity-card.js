/*
 * render/layer2-activity-card.js — Layer 2 trigger activity over 24h.
 *
 * Reads `/api/system_status` payload's `layer2` section. Shows:
 *   - Triggers count + success_pct headline
 *   - 24-hour mini-bar histogram of triggers (oldest left → newest right)
 *   - Latest invocation: timestamp, caller, status
 *
 * Tap navigates to `/decisions`.
 */

import * as router from "../router.js";
import { ft } from "../format.js";

/** @returns {HTMLElement} */
export function layer2ActivityCard(layer2Payload) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "card card--tap";
  card.style.display = "block";
  card.style.width = "100%";
  card.style.textAlign = "left";
  card.style.padding = "var(--sp-3)";
  card.style.minHeight = "var(--tap-min)";
  card.addEventListener("click", () => router.navigate("decisions"));

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.alignItems = "baseline";
  header.style.marginBottom = "var(--sp-2)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0";
  title.textContent = "Layer 2 activity (24h)";
  header.append(title);

  const triggers = layer2Payload?.triggers_24h ?? 0;
  const pct = layer2Payload?.success_pct;
  const headline = document.createElement("span");
  headline.style.fontSize = "var(--ty-sm)";
  headline.style.fontWeight = "600";
  headline.style.color = _colorForPct(pct, triggers);
  headline.textContent = pct == null
    ? `${triggers} triggers`
    : `${triggers} triggers · ${pct.toFixed(0)}%`;
  header.append(headline);
  card.append(header);

  card.append(_sparkBars(layer2Payload?.spark_24h));

  const latest = layer2Payload?.latest;
  if (latest) {
    const summary = document.createElement("div");
    summary.style.marginTop = "var(--sp-2)";
    summary.style.fontSize = "var(--ty-sm)";
    summary.style.color = "var(--txm)";
    summary.style.overflow = "hidden";
    summary.style.textOverflow = "ellipsis";
    summary.style.whiteSpace = "nowrap";
    const status = (latest.status || "?").toUpperCase();
    const dur = Number.isFinite(latest.duration_seconds)
      ? `${latest.duration_seconds.toFixed(0)}s`
      : "?";
    summary.textContent = `last: ${ft(latest.ts)} · ${latest.caller || "?"} · ${status} · ${dur}`;
    card.append(summary);
  }
  return card;
}

function _sparkBars(values) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.alignItems = "flex-end";
  wrap.style.gap = "1px";
  wrap.style.height = "24px";
  wrap.style.padding = "2px 0";

  const arr = Array.isArray(values) && values.length === 24 ? values : new Array(24).fill(0);
  const max = Math.max(...arr, 1);
  for (let i = 0; i < arr.length; i++) {
    const v = Math.max(0, Number(arr[i]) || 0);
    const bar = document.createElement("span");
    bar.style.flex = "1";
    bar.style.height = `${(v / max) * 100}%`;
    bar.style.minHeight = "2px";
    bar.style.background = v ? "var(--cyn)" : "var(--bd)";
    bar.style.borderRadius = "1px";
    wrap.append(bar);
  }
  return wrap;
}

function _colorForPct(pct, triggers) {
  if (pct == null || triggers < 3) return "var(--txm)";
  if (pct < 60) return "var(--red)";
  if (pct < 85) return "var(--yel)";
  return "var(--grn)";
}
