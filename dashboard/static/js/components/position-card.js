/*
 * position-card.js — single open-position card for the home strip.
 *
 * Compact: ticker / side badge / P&L% / distance-to-stop bar.
 * The sparkline area is reserved (callers can mount a mini-chart there).
 */

import { fpct, fp } from "../format.js";

/**
 * @param {{
 *   ticker: string,
 *   side?: "LONG"|"SHORT",
 *   pnlPct?: number|null,
 *   pricePerUnit?: number|null,
 *   stopPrice?: number|null,
 *   stopDistancePct?: number|null,
 *   sparkline?: HTMLElement|null,
 *   onTap?: () => void,
 *   onLongPress?: () => void,
 * }} props
 * @returns {HTMLElement}
 */
export function positionCard({
  ticker = "",
  side = "LONG",
  pnlPct = null,
  pricePerUnit = null,
  stopPrice = null,
  stopDistancePct = null,
  sparkline = null,
  onTap = null,
  onLongPress = null,
} = {}) {
  const card = document.createElement("article");
  card.className = "card" + (onTap ? " card--tap" : "");
  card.style.minWidth = "180px";
  card.style.maxWidth = "220px";
  if (onTap) card.addEventListener("click", onTap);

  // Header row: ticker + side
  const hdr = document.createElement("div");
  hdr.className = "card__header";
  const t = document.createElement("div");
  t.className = "card__title";
  t.textContent = ticker;
  hdr.append(t);

  const badge = document.createElement("span");
  badge.className = "badge " + (side === "SHORT" ? "badge--SELL" : "badge--BUY");
  badge.textContent = side === "SHORT" ? "SHORT" : "LONG";
  hdr.append(badge);
  card.append(hdr);

  // P&L %
  if (pnlPct != null) {
    const pnl = document.createElement("div");
    pnl.className = "num num--md";
    pnl.classList.add(pnlPct > 0 ? "pos" : pnlPct < 0 ? "neg" : "flat");
    pnl.textContent = fpct(pnlPct, 2);
    card.append(pnl);
  }

  // Price
  if (pricePerUnit != null) {
    const price = document.createElement("div");
    price.style.fontSize = "var(--ty-sm)";
    price.style.color = "var(--txd)";
    price.textContent = "@ " + fp(pricePerUnit);
    card.append(price);
  }

  // Sparkline
  if (sparkline instanceof Node) {
    const sw = document.createElement("div");
    sw.className = "sparkline";
    sw.style.marginTop = "var(--sp-2)";
    sw.append(sparkline);
    card.append(sw);
  }

  // Stop distance bar
  if (stopDistancePct != null) {
    const wrap = document.createElement("div");
    wrap.style.marginTop = "var(--sp-2)";

    const lbl = document.createElement("div");
    lbl.style.display = "flex";
    lbl.style.justifyContent = "space-between";
    lbl.style.fontSize = "var(--ty-xs)";
    lbl.style.color = "var(--txm)";
    const lblL = document.createElement("span");
    lblL.textContent = "Stop";
    const lblR = document.createElement("span");
    lblR.textContent = fpct(stopDistancePct, 1) + (stopPrice != null ? " @" + fp(stopPrice) : "");
    lbl.append(lblL, lblR);

    const bar = document.createElement("div");
    bar.style.height = "4px";
    bar.style.background = "var(--bdr)";
    bar.style.borderRadius = "var(--rad-pill)";
    bar.style.overflow = "hidden";
    bar.style.marginTop = "2px";
    const fill = document.createElement("div");
    // Distance maps 0 (at stop) → red, 1 (far above) → green-ish.
    const clamped = Math.max(0, Math.min(1, Math.abs(stopDistancePct) / 15));
    fill.style.height = "100%";
    fill.style.width = (clamped * 100).toFixed(0) + "%";
    fill.style.background = stopDistancePct < 0
      ? "var(--red)"
      : stopDistancePct < 5 ? "var(--org)" : "var(--grn)";
    bar.append(fill);
    wrap.append(lbl, bar);
    card.append(wrap);
  }

  // Long-press wiring (consumer attaches via bottom-sheet.bindLongPress in
  // their mount code; this prop is recorded as a hint for them).
  if (onLongPress) card.dataset.hasLongPress = "1";

  return card;
}
