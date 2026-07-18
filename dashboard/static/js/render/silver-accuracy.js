/*
 * render/silver-accuracy.js — #silver page accuracy matrix (Phase 6).
 *
 * Rows = signals, cols = horizons (3h/1d/3d/5d), from the new
 * /api/silver/accuracy route. A Consensus row on top comes from the
 * existing /api/accuracy's per_ticker block — that endpoint only computes
 * consensus at 1d/3d/5d (no 3h), so the 3h consensus cell is always "—".
 * Cell color: n<30 grey (too few samples to trust), pct>=60 green (matches
 * signal_engine's accuracy-tier confidence boost), pct<47 red (matches the
 * force-HOLD gate), else neutral.
 */

import { fAgo } from "../format.js";
import { emptyState } from "../components/empty-state.js";

const TICKER = "XAG-USD";
const HORIZONS = ["3h", "1d", "3d", "5d"]; // matches dashboard/silver.py
const CONSENSUS_HORIZONS = ["1d", "3d", "5d"]; // /api/accuracy has no 3h consensus
const KEEP_BAR = 60;
const FAIL_BAR = 47;
const MIN_SAMPLES = 30;

/**
 * @param {{silverAcc: object|null, accGlobal: object|null}} props
 *   silverAcc: /api/silver/accuracy?ticker=XAG-USD response.
 *   accGlobal: /api/accuracy response (for the Consensus row).
 * @returns {HTMLElement}
 */
export function silverAccuracyMatrix({ silverAcc, accGlobal } = {}) {
  const root = document.createElement("div");
  if (!silverAcc) {
    root.append(emptyState("Loading accuracy…"));
    return root;
  }

  const ageChip = _ageChip(silverAcc.updated_ts);
  if (ageChip) root.append(ageChip);

  const nameSet = new Set();
  for (const h of HORIZONS) {
    for (const name of Object.keys(silverAcc.horizons?.[h]?.signals || {}))
      nameSet.add(name);
  }
  const names = [...nameSet].sort((a, b) => {
    const pa = silverAcc.horizons?.["1d"]?.signals?.[a]?.pct ?? -1;
    const pb = silverAcc.horizons?.["1d"]?.signals?.[b]?.pct ?? -1;
    return pb - pa;
  });

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";
  wrap.style.overflowX = "auto";

  const table = document.createElement("table");
  table.style.width = "100%";
  table.style.borderCollapse = "collapse";

  const thead = document.createElement("thead");
  const trH = document.createElement("tr");
  trH.append(_th("Signal"));
  for (const h of HORIZONS) trH.append(_th(h));
  thead.append(trH);
  table.append(thead);

  const tbody = document.createElement("tbody");

  const consensusRow = document.createElement("tr");
  consensusRow.style.borderTop = "1px solid var(--bdr)";
  const consensusName = document.createElement("td");
  consensusName.textContent = "Consensus";
  consensusName.style.padding = "var(--sp-2)";
  consensusName.style.fontWeight = "700";
  consensusName.style.fontSize = "var(--ty-sm)";
  consensusRow.append(consensusName);
  for (const h of HORIZONS) {
    if (!CONSENSUS_HORIZONS.includes(h)) {
      consensusRow.append(_dashCell());
      continue;
    }
    consensusRow.append(_accCell(accGlobal?.[h]?.per_ticker?.[TICKER]));
  }
  tbody.append(consensusRow);

  if (!names.length) {
    const empty = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = HORIZONS.length + 1;
    td.style.padding = "var(--sp-3)";
    td.style.color = "var(--txm)";
    td.style.fontSize = "var(--ty-sm)";
    td.textContent = "No per-signal directional accuracy for XAG-USD yet.";
    empty.append(td);
    tbody.append(empty);
  }

  for (const name of names) {
    const tr = document.createElement("tr");
    tr.style.borderTop = "1px solid var(--bdr)";
    const tdName = document.createElement("td");
    tdName.textContent = name;
    tdName.style.padding = "var(--sp-2)";
    tdName.style.fontSize = "var(--ty-sm)";
    tr.append(tdName);
    for (const h of HORIZONS)
      tr.append(_accCell(silverAcc.horizons?.[h]?.signals?.[name]));
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  root.append(wrap);
  return root;
}

function _th(text) {
  const th = document.createElement("th");
  th.textContent = text;
  th.style.padding = "var(--sp-2)";
  th.style.fontSize = "var(--ty-xs)";
  th.style.color = "var(--txm)";
  th.style.textAlign = "center";
  return th;
}

function _dashCell() {
  const td = document.createElement("td");
  td.style.textAlign = "center";
  td.style.padding = "var(--sp-2)";
  td.style.color = "var(--txm)";
  td.style.fontSize = "var(--ty-sm)";
  td.textContent = "—";
  return td;
}

function _accCell(info) {
  const total = Number(info?.total ?? 0);
  if (!info || !total) return _dashCell();

  const pct = Number(info.pct ?? Number(info.accuracy) * 100);
  const color =
    total < MIN_SAMPLES
      ? "var(--txm)"
      : pct >= KEEP_BAR
        ? "var(--grn)"
        : pct < FAIL_BAR
          ? "var(--red)"
          : "var(--tx)";

  const td = document.createElement("td");
  td.style.textAlign = "center";
  td.style.padding = "var(--sp-2)";

  const pctEl = document.createElement("div");
  pctEl.style.fontWeight = "600";
  pctEl.style.fontSize = "var(--ty-sm)";
  pctEl.style.color = color;
  pctEl.textContent = pct.toFixed(1) + "%";
  td.append(pctEl);

  const nEl = document.createElement("div");
  nEl.style.fontSize = "var(--ty-xs)";
  nEl.style.color = "var(--txm)";
  nEl.textContent = "n=" + total;
  td.append(nEl);
  return td;
}

function _ageChip(updatedTs) {
  if (updatedTs == null) return null;
  const ageSec = Math.max(0, Date.now() / 1000 - Number(updatedTs));
  const color =
    ageSec > 7200 ? "var(--red)" : ageSec > 1800 ? "var(--yel)" : "var(--txm)";
  const chip = document.createElement("div");
  chip.style.fontSize = "var(--ty-xs)";
  chip.style.color = color;
  chip.style.marginBottom = "var(--sp-2)";
  chip.textContent = `updated ${fAgo(new Date(updatedTs * 1000))}`;
  return chip;
}
