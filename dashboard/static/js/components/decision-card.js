/*
 * decision-card.js — single Layer 2 decision summary card.
 *
 * Track-5 pattern: card-row with ticker + action chip + 1-line reason +
 * relative timestamp. Tap to open the full detail view; long-press to
 * open the bottom-sheet drill.
 */

import { fAgo } from "../format.js";

/**
 * @param {{
 *   ts: string|number|Date,
 *   ticker?: string,
 *   trigger?: string,
 *   regime?: string,
 *   patient?: { action: string, reasoning?: string },
 *   bold?:    { action: string, reasoning?: string },
 *   onTap?: () => void,
 *   onLongPress?: () => void,
 * }} props
 * @returns {HTMLElement}
 */
export function decisionCard({
  ts,
  ticker = "",
  trigger = "",
  regime = "",
  patient,
  bold,
  onTap = null,
} = {}) {
  const card = document.createElement("article");
  card.className = "card decision-card" + (onTap ? " card--tap" : "");
  if (onTap) card.addEventListener("click", onTap);

  // Header row: ticker, time, regime
  const hdr = document.createElement("div");
  hdr.className = "card__header";
  const t = document.createElement("div");
  t.className = "card__title";
  t.textContent = ticker || trigger || "decision";
  hdr.append(t);

  const meta = document.createElement("div");
  meta.className = "card__meta";
  meta.textContent = fAgo(ts);
  hdr.append(meta);
  card.append(hdr);

  // Trigger / regime line
  if (trigger || regime) {
    const sub = document.createElement("div");
    sub.className = "card__subtitle";
    sub.textContent = [trigger, regime].filter(Boolean).join(" · ");
    card.append(sub);
  }

  // Action badges (Patient + Bold)
  const acts = document.createElement("div");
  acts.style.display = "flex";
  acts.style.gap = "var(--sp-2)";
  acts.style.marginTop = "var(--sp-2)";

  if (patient) {
    const lbl = document.createElement("span");
    lbl.style.fontSize = "var(--ty-xs)";
    lbl.style.color = "var(--cyn)";
    lbl.style.fontWeight = "600";
    lbl.textContent = "P:";
    const a = document.createElement("span");
    a.className = "badge " + _badgeClass(patient.action);
    a.textContent = patient.action || "?";
    acts.append(lbl, a);
  }
  if (bold) {
    const lbl = document.createElement("span");
    lbl.style.fontSize = "var(--ty-xs)";
    lbl.style.color = "var(--org)";
    lbl.style.fontWeight = "600";
    lbl.style.marginLeft = "var(--sp-2)";
    lbl.textContent = "B:";
    const a = document.createElement("span");
    a.className = "badge " + _badgeClass(bold.action);
    a.textContent = bold.action || "?";
    acts.append(lbl, a);
  }
  card.append(acts);

  // Reason preview (clamped to 2 lines via CSS)
  const reason = patient?.reasoning || bold?.reasoning;
  if (reason) {
    const r = document.createElement("div");
    r.className = "decision-card__reason";
    r.style.marginTop = "var(--sp-2)";
    r.textContent = reason;
    card.append(r);
  }

  return card;
}

function _badgeClass(action) {
  switch ((action || "").toUpperCase()) {
    case "BUY":         return "badge--BUY";
    case "STRONG_BUY":  return "badge--BUY";
    case "SELL":        return "badge--SELL";
    case "STRONG_SELL": return "badge--SELL";
    default:            return "badge--HOLD";
  }
}
