/*
 * render/signals-heatmap.js — signal × ticker heatmap.
 *
 * Adapts to the actual /api/signal-heatmap shape, which is a flat
 * {heatmap: {ticker: {signal: "BUY"|"SELL"|"HOLD"}}} matrix without a
 * timeframe dimension. We render rows=signals × cols=tickers, color-only
 * cells per Track-5. Long-press / tap a cell opens a bottom-sheet with
 * the signal name, ticker, vote, and recent accuracy.
 *
 * Earlier (uncommitted) drafts assumed per-timeframe nested cell objects,
 * which left every cell as cell--hold against the live data.  Codex P1
 * finding 2026-05-03 — fixed here.
 */

import { open as openSheet, bindLongPress } from "../components/bottom-sheet.js";
import { fDurationShort } from "../format.js";

/**
 * @param {{
 *   data: object,                       // /api/signal-heatmap full response
 *   tickers?: string[],                 // override column order
 *   accuracy?: Record<string, number>,  // signal -> accuracy %
 *   disabled?: Set<string>,             // disabled signal names (force-HOLD)
 *   disabledReasons?: Record<string, string>, // signal -> short reason text
 * }} props
 * @returns {HTMLElement}
 */
export function renderHeatmap({ data, tickers = null, accuracy = {}, disabled = new Set(), disabledReasons = {} }) {
  const wrap = document.createElement("div");
  wrap.className = "heatmap-wrap";

  if (!data || typeof data !== "object" || !data.heatmap) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No heatmap data.";
    wrap.append(empty);
    return wrap;
  }

  const colTickers = (Array.isArray(tickers) && tickers.length)
    ? tickers
    : (Array.isArray(data.tickers) && data.tickers.length)
      ? data.tickers
      : Object.keys(data.heatmap);

  const allSignals = _signalOrder(data);
  if (!allSignals.length || !colTickers.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No heatmap rows / columns.";
    wrap.append(empty);
    return wrap;
  }

  const table = document.createElement("table");
  table.className = "heatmap";

  const thead = document.createElement("thead");
  const trH = document.createElement("tr");
  const corner = document.createElement("th");
  corner.textContent = "Signal";
  trH.append(corner);
  for (const t of colTickers) {
    const th = document.createElement("th");
    th.textContent = t.replace(/-USD$/, "");
    th.title = t;
    trH.append(th);
  }
  thead.append(trH);
  table.append(thead);

  const tbody = document.createElement("tbody");
  for (const sigName of allSignals) {
    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    tdName.textContent = _truncate(sigName, 14);
    tdName.title = sigName;
    tr.append(tdName);

    for (const ticker of colTickers) {
      const cell = document.createElement("td");
      const value = data.heatmap?.[ticker]?.[sigName];
      const isDisabled = disabled.has(sigName) || _looksDisabled(value);
      cell.className = _classForValue(value, isDisabled);
      cell.dataset.signal = sigName;
      cell.dataset.ticker = ticker;

      const sinceTs = data.since?.[ticker]?.[sigName];
      const durLabel = !isDisabled ? fDurationShort(sinceTs) : "";
      cell.title = durLabel
        ? `${sigName} · ${ticker}: ${value || "—"} · ${durLabel} in state`
        : `${sigName} · ${ticker}: ${value || "—"}`;
      if (durLabel) {
        const since = document.createElement("span");
        since.className = "cell-since";
        since.textContent = durLabel;
        cell.append(since);
      }

      bindLongPress(cell, () => ({
        title: `${sigName} — ${ticker}`,
        content: _detailNode(sigName, ticker, value, accuracy[sigName], sinceTs, isDisabled, disabledReasons[sigName]),
      }));
      cell.addEventListener("click", () => {
        openSheet({
          title: `${sigName} — ${ticker}`,
          content: _detailNode(sigName, ticker, value, accuracy[sigName], sinceTs, isDisabled, disabledReasons[sigName]),
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

function _signalOrder(data) {
  if (Array.isArray(data?.signals) && data.signals.length) return data.signals;
  const core = Array.isArray(data?.core_signals) ? data.core_signals : [];
  const enh  = Array.isArray(data?.enhanced_signals) ? data.enhanced_signals : [];
  if (core.length || enh.length) return [...core, ...enh];

  // Fallback: union of signal keys across tickers.
  const seen = new Set();
  for (const ticker of Object.keys(data?.heatmap || {})) {
    for (const s of Object.keys(data.heatmap[ticker] || {})) seen.add(s);
  }
  return [...seen];
}

function _classForValue(value, isDisabled) {
  if (isDisabled) return "cell--disabled";
  const v = String(value || "").toUpperCase();
  if (v === "STRONG_BUY")  return "cell--strong-buy";
  if (v === "BUY")          return "cell--buy";
  if (v === "SELL")         return "cell--sell";
  if (v === "STRONG_SELL")  return "cell--strong-sell";
  if (v === "HOLD")         return "cell--hold";
  return "cell--hold";
}

function _looksDisabled(value) {
  return value === "DISABLED" || value === "N/A" || value === "n/a";
}

function _truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function _detailNode(signal, ticker, value, accPct, sinceTs, isDisabled = false, disabledReason = null) {
  const wrap = document.createElement("div");
  const meta = document.createElement("div");
  meta.style.fontSize = "var(--ty-sm)";
  meta.style.color = "var(--txd)";
  meta.textContent = `${ticker} · vote: ${value || "—"}`;
  wrap.append(meta);

  // For disabled signals, suppress the time-in-state and accuracy lines
  // and instead show the disable reason. Time-in-state is meaningless on
  // a force-HOLD'd signal, and the 0% accuracy is a counter-init artifact
  // (see /api/accuracy enabled flag, dashboard/app.py:_enrich_signals).
  if (isDisabled) {
    const tag = document.createElement("div");
    tag.style.fontSize = "var(--ty-sm)";
    tag.style.fontWeight = "600";
    tag.style.color = "var(--txm)";
    tag.style.marginTop = "var(--sp-2)";
    tag.textContent = "Disabled (force-HOLD)";
    wrap.append(tag);
    if (disabledReason) {
      const why = document.createElement("div");
      why.style.fontSize = "var(--ty-xs)";
      why.style.color = "var(--txm)";
      why.style.marginTop = "var(--sp-1)";
      why.style.lineHeight = "1.5";
      why.textContent = disabledReason;
      wrap.append(why);
    }
  } else {
    const durLabel = fDurationShort(sinceTs);
    if (durLabel) {
      const dur = document.createElement("div");
      dur.style.fontSize = "var(--ty-sm)";
      dur.style.marginTop = "var(--sp-1)";
      dur.style.color = "var(--txm)";
      dur.textContent = `In this state for: ${durLabel}`;
      wrap.append(dur);
    }

    if (accPct != null && Number.isFinite(Number(accPct))) {
      const acc = document.createElement("div");
      acc.style.fontSize = "var(--ty-sm)";
      acc.style.marginTop = "var(--sp-2)";
      acc.style.color = accPct >= 47 ? "var(--grn)" : "var(--red)";
      acc.textContent = `Recent accuracy: ${Number(accPct).toFixed(0)}%`;
      wrap.append(acc);
    }
  }

  const note = document.createElement("p");
  note.style.fontSize = "var(--ty-sm)";
  note.style.color = "var(--txm)";
  note.style.lineHeight = "1.5";
  note.style.marginTop = "var(--sp-2)";
  note.textContent =
    "Tip: drill into Per-signal accuracy (sub-tab above) for the full sample size + " +
    "calibration. Underlying timeframe-by-timeframe alignment is on /legacy.";
  wrap.append(note);
  return wrap;
}
