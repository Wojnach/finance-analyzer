/*
 * signal-row.js — per-signal row used in the Signals "Per-signal accuracy" tab.
 *
 * Layout: name + B/S/H badge + horizontal accuracy bar + percentage + sample
 * count. Sortable by accuracy at the consumer side.
 */

import { fpct } from "../format.js";

/**
 * @param {{
 *   name: string,
 *   action?: "BUY"|"SELL"|"HOLD",
 *   accuracyPct?: number|null,   // 0..100
 *   sampleSize?: number|null,
 *   threshold?: number,          // shows a marker line at this % (default 47)
 *   onTap?: () => void,
 * }} props
 * @returns {HTMLElement}
 */
export function signalRow({
  name = "",
  action = null,
  accuracyPct = null,
  sampleSize = null,
  threshold = 47,
  onTap = null,
} = {}) {
  const row = document.createElement("div");
  row.style.display = "grid";
  row.style.gridTemplateColumns = "minmax(80px, 130px) auto 1fr 50px 50px";
  row.style.gap = "var(--sp-2)";
  row.style.alignItems = "center";
  row.style.padding = "var(--sp-2) var(--sp-1)";
  row.style.borderBottom = "1px solid var(--bdr)";
  row.style.minHeight = "var(--tap-min)";
  if (onTap) {
    row.style.cursor = "pointer";
    row.addEventListener("click", onTap);
  }

  // 1. Name
  const n = document.createElement("div");
  n.style.fontSize = "var(--ty-sm)";
  n.style.fontWeight = "600";
  n.style.overflow = "hidden";
  n.style.textOverflow = "ellipsis";
  n.style.whiteSpace = "nowrap";
  n.title = name;
  n.textContent = name;
  row.append(n);

  // 2. Action badge (or empty placeholder)
  if (action) {
    const a = document.createElement("span");
    a.className = "badge " + (
      action === "BUY"  ? "badge--BUY"  :
      action === "SELL" ? "badge--SELL" : "badge--HOLD"
    );
    a.textContent = action;
    row.append(a);
  } else {
    row.append(document.createElement("span"));
  }

  // 3. Accuracy bar
  const barWrap = document.createElement("div");
  barWrap.style.position = "relative";
  barWrap.style.height = "8px";
  barWrap.style.background = "var(--bdr)";
  barWrap.style.borderRadius = "var(--rad-pill)";
  barWrap.style.overflow = "hidden";

  const fill = document.createElement("div");
  const pct = accuracyPct == null ? 0 : Math.max(0, Math.min(100, accuracyPct));
  fill.style.height = "100%";
  fill.style.width = pct + "%";
  fill.style.background = pct >= threshold ? "var(--grn)" : "var(--red)";
  barWrap.append(fill);

  // Threshold marker
  const marker = document.createElement("div");
  marker.style.position = "absolute";
  marker.style.top = "0";
  marker.style.bottom = "0";
  marker.style.left = threshold + "%";
  marker.style.width = "1px";
  marker.style.background = "var(--txm)";
  marker.style.opacity = "0.6";
  barWrap.append(marker);

  row.append(barWrap);

  // 4. Percentage
  const pctEl = document.createElement("div");
  pctEl.style.fontSize = "var(--ty-sm)";
  pctEl.style.fontWeight = "600";
  pctEl.style.textAlign = "right";
  pctEl.style.color = pct >= threshold ? "var(--grn)" : "var(--red)";
  pctEl.textContent = accuracyPct == null ? "—" : pct.toFixed(0) + "%";
  row.append(pctEl);

  // 5. Sample size
  const n2 = document.createElement("div");
  n2.style.fontSize = "var(--ty-xs)";
  n2.style.color = "var(--txm)";
  n2.style.textAlign = "right";
  n2.textContent = sampleSize == null ? "" : "n=" + sampleSize;
  row.append(n2);

  return row;
}
