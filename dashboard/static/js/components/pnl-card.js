/*
 * pnl-card.js — Net P&L summary (Patient / Bold / Warrants).
 *
 * Three numbers + delta vs previous close. Optional sparkline slot
 * (a child Node passed in) for the 24h equity curve mini-chart.
 */

import { fs, fpct } from "../format.js";

/**
 * @param {{
 *   patient?:  { value: number, deltaPct?: number, label?: string },
 *   bold?:     { value: number, deltaPct?: number, label?: string },
 *   warrants?: { value: number, deltaPct?: number, label?: string },
 *   sparkline?: HTMLElement|null,
 *   title?: string,
 *   onTap?: () => void,
 * }} props
 * @returns {HTMLElement}
 */
export function pnlCard({ patient, bold, warrants, sparkline = null, title = "Portfolio", onTap = null } = {}) {
  const card = document.createElement("article");
  card.className = "card" + (onTap ? " card--tap" : "");
  if (onTap) card.addEventListener("click", onTap);

  // Header
  const header = document.createElement("header");
  header.className = "card__header";
  const t = document.createElement("div");
  t.className = "card__title";
  t.textContent = title;
  const meta = document.createElement("div");
  meta.className = "card__subtitle";
  meta.textContent = "P&L";
  header.append(t, meta);
  card.append(header);

  // Three columns
  const cols = document.createElement("div");
  cols.style.display = "grid";
  cols.style.gridTemplateColumns = "1fr 1fr 1fr";
  cols.style.gap = "var(--sp-3)";

  cols.append(
    _stratColumn(patient,  patient?.label  || "Patient",  "var(--cyn)"),
    _stratColumn(bold,     bold?.label     || "Bold",     "var(--org)"),
    _stratColumn(warrants, warrants?.label || "Warrants", "var(--blu)"),
  );
  card.append(cols);

  if (sparkline instanceof Node) {
    const sw = document.createElement("div");
    sw.className = "sparkline";
    sw.style.marginTop = "var(--sp-2)";
    sw.append(sparkline);
    card.append(sw);
  }

  return card;
}

function _stratColumn(data, label, accentColor) {
  const col = document.createElement("div");
  col.style.textAlign = "left";

  const lbl = document.createElement("div");
  lbl.style.fontSize = "var(--ty-xs)";
  lbl.style.color = accentColor;
  lbl.style.fontWeight = "600";
  lbl.style.textTransform = "uppercase";
  lbl.style.letterSpacing = "0.5px";
  lbl.textContent = label;

  const v = document.createElement("div");
  const value = data?.value;
  v.className = "num num--md";
  v.classList.add(value > 0 ? "pos" : value < 0 ? "neg" : "flat");
  v.textContent = data ? fs(value) : "--";

  const d = document.createElement("div");
  d.style.fontSize = "var(--ty-xs)";
  const dpct = data?.deltaPct;
  if (dpct == null) {
    d.style.color = "var(--txm)";
    d.textContent = "—";
  } else {
    d.style.color = dpct >= 0 ? "var(--grn)" : "var(--red)";
    d.textContent = fpct(dpct, 2);
  }

  col.append(lbl, v, d);
  return col;
}
