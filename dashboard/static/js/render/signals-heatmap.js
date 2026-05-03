/*
 * render/signals-heatmap.js — Track-5 transposed heatmap.
 *
 * Rows = signals, columns = timeframes, one ticker at a time. Sticky
 * leftmost column (signal name) and sticky top row (timeframe header).
 * Cells are color-only with a 5-class scale (strong-buy → strong-sell).
 * Long-press a cell → bottom sheet with detail.
 */

import { open as openSheet, bindLongPress } from "../components/bottom-sheet.js";
import { fpct } from "../format.js";

/**
 * @param {{
 *   ticker: string,
 *   data: object,                 // /api/signal-heatmap response slice for this ticker
 *   timeframes?: string[],
 *   accuracy?: Record<string, number>, // signal -> accuracy %
 *   disabled?: Set<string>,            // disabled signal names (force-HOLD)
 * }} props
 * @returns {HTMLElement}
 */
export function renderHeatmap({ ticker, data, timeframes = null, accuracy = {}, disabled = new Set() }) {
  const wrap = document.createElement("div");
  wrap.className = "heatmap-wrap";

  const tfs = timeframes || _defaultTimeframes(data);
  const signals = _signalsFor(data, ticker);

  if (!signals.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No heatmap data for " + ticker;
    wrap.append(empty);
    return wrap;
  }

  const table = document.createElement("table");
  table.className = "heatmap";

  // Header
  const thead = document.createElement("thead");
  const trH = document.createElement("tr");
  const cornerTh = document.createElement("th");
  cornerTh.textContent = "Signal";
  trH.append(cornerTh);
  for (const tf of tfs) {
    const th = document.createElement("th");
    th.textContent = tf;
    trH.append(th);
  }
  thead.append(trH);
  table.append(thead);

  // Body
  const tbody = document.createElement("tbody");
  for (const sigName of signals) {
    const tr = document.createElement("tr");
    const tdName = document.createElement("td");
    tdName.textContent = _truncate(sigName, 14);
    tdName.title = sigName;
    tr.append(tdName);

    for (const tf of tfs) {
      const cell = document.createElement("td");
      const cellData = _cellData(data, ticker, sigName, tf);
      const klass = _classForCell(cellData, disabled.has(sigName));
      cell.className = klass;
      cell.dataset.signal = sigName;
      cell.dataset.tf = tf;
      // Long-press / tap drill
      bindLongPress(cell, () => ({
        title: `${sigName} @ ${tf} — ${ticker}`,
        content: _detailNode(sigName, tf, ticker, cellData, accuracy[sigName]),
      }));
      // Plain tap also opens the sheet (Track 5: long-press preferred,
      // but tap is more discoverable on phones).
      cell.addEventListener("click", () => {
        openSheet({
          title: `${sigName} @ ${tf} — ${ticker}`,
          content: _detailNode(sigName, tf, ticker, cellData, accuracy[sigName]),
        });
      });
      tr.append(cell);
    }
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  return wrap;
}

// ---------------------------------------------------------------------------

function _defaultTimeframes(data) {
  // Walk the heatmap to find timeframes present in any cell.
  if (data && Array.isArray(data.timeframes)) return data.timeframes;
  return ["now", "12h", "2d", "7d", "1mo", "3mo", "6mo"];
}

function _signalsFor(data, _ticker) {
  // Two shapes are common in /api/signal-heatmap:
  //  A) { signals: [...], heatmap: { ticker: { signal: { tf: action } } } }
  //  B) { tickers: [...], rows: [{ signal, ticker, tf, action }, ...] }
  if (Array.isArray(data?.signals)) return data.signals;
  if (Array.isArray(data?.core_signals) || Array.isArray(data?.enhanced_signals)) {
    return [...(data.core_signals || []), ...(data.enhanced_signals || [])];
  }
  // Fallback: derive from heatmap object keys
  const ticker = _ticker;
  const heat = data?.heatmap?.[ticker];
  if (heat && typeof heat === "object") return Object.keys(heat);
  return [];
}

function _cellData(data, ticker, signal, tf) {
  // Try several shapes
  const heat = data?.heatmap?.[ticker]?.[signal];
  if (heat) {
    const v = heat[tf] || heat[`${tf}`];
    if (v && typeof v === "object") return v;
    if (typeof v === "string") return { action: v };
  }
  // rows shape
  if (Array.isArray(data?.rows)) {
    const row = data.rows.find((r) =>
      r.ticker === ticker && r.signal === signal && r.tf === tf);
    if (row) return row;
  }
  return null;
}

function _classForCell(cellData, isDisabled) {
  if (isDisabled) return "cell--disabled";
  const action = (cellData?.action || cellData?.consensus || "").toUpperCase();
  const conf = Number(cellData?.confidence || cellData?.weight || 0);
  if (action === "BUY")  return conf >= 0.6 ? "cell--strong-buy"  : "cell--buy";
  if (action === "SELL") return conf >= 0.6 ? "cell--strong-sell" : "cell--sell";
  if (action === "HOLD") return "cell--hold";
  return "cell--hold"; // unknown
}

function _truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function _detailNode(signal, tf, ticker, cellData, accPct) {
  const wrap = document.createElement("div");
  const meta = document.createElement("div");
  meta.style.fontSize = "var(--ty-sm)";
  meta.style.color = "var(--txd)";
  meta.textContent =
    `${ticker} · ${tf} · vote: ${cellData?.action || "—"}`
    + (cellData?.confidence != null ? ` · conf ${(cellData.confidence * 100).toFixed(0)}%` : "");
  wrap.append(meta);

  if (accPct != null) {
    const acc = document.createElement("div");
    acc.style.fontSize = "var(--ty-sm)";
    acc.style.marginTop = "var(--sp-2)";
    acc.style.color = accPct >= 47 ? "var(--grn)" : "var(--red)";
    acc.textContent = `Recent accuracy: ${accPct.toFixed(0)}%`;
    wrap.append(acc);
  }

  if (cellData?.rationale || cellData?.reasoning) {
    const r = document.createElement("p");
    r.style.fontSize = "var(--ty-sm)";
    r.style.lineHeight = "1.6";
    r.style.marginTop = "var(--sp-2)";
    r.style.color = "var(--tx)";
    r.textContent = cellData.rationale || cellData.reasoning;
    wrap.append(r);
  }
  return wrap;
}
