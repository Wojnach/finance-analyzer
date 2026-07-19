/*
 * render/accuracy.js — per-signal accuracy list, per-ticker matrix, and
 * history-chart helpers.
 */

import { signalRow } from "../components/signal-row.js";
import { emptyState } from "../components/empty-state.js";
import { fAgo } from "../format.js";

const KEEP_BAR = 60; // matches the 60%+ accuracy-tier confidence boost (signal_engine.py)

/**
 * Renders the accuracy panel for a given horizon.
 * @param {{ horizon: "1d"|"3d"|"5d"|"10d", data: object, threshold?: number }} props
 * @returns {HTMLElement}
 */
export function renderAccuracyPanel({
  horizon = "1d",
  data,
  threshold = 47,
} = {}) {
  const root = document.createElement("div");
  if (!data) {
    root.append(emptyState(`No accuracy data for ${horizon}.`));
    return root;
  }

  const horizonData = data[horizon] || data;
  const signals = horizonData?.signals || horizonData;
  if (!signals || typeof signals !== "object") {
    root.append(emptyState(`No accuracy data for ${horizon}.`));
    return root;
  }

  // Freshness + sample-count summary (Phase 1 `meta` block: {updated_ts,
  // age_sec}). Cache staleness used to be invisible — a horizon panel
  // looked identical whether it was rebuilt seconds or days ago.
  const summary = _summaryBar(horizonData);
  if (summary) root.append(summary);

  // Convert to array. The backend response shape uses `total` for the
  // sample count (see portfolio/accuracy_stats.py:signal_accuracy); older
  // mappings to `samples`/`n`/`sample_size` are kept for forward-compat.
  // Without the `total` fallback every row used to render n=0 because
  // Number(null) coerces to 0 — fixed 2026-05-05.
  const rows = Object.entries(signals)
    .map(([name, info]) => ({
      name,
      pct: Number(info?.pct ?? info?.accuracy ?? info?.accuracy_pct ?? null),
      samples: Number(
        info?.samples ?? info?.n ?? info?.sample_size ?? info?.total ?? null,
      ),
      enabled: info?.enabled !== false,
      disabledReason: info?.disabled_reason ?? null,
    }))
    .filter((r) => r.pct != null && Number.isFinite(r.pct))
    // Disabled rows sink to the bottom regardless of pct — a force-HOLD'd
    // signal with 0 samples isn't "0% accurate", it's off, and lumping it
    // beneath active signals avoids visually competing with real data.
    .sort((a, b) => {
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;
      return b.pct - a.pct;
    });

  if (!rows.length) {
    // 10d (and any other horizon with 0 outcome rows) used to render as a
    // blank "No accuracy data" box with no explanation. The backend now
    // ships `unavailable_reason` for exactly this case — show it instead.
    if (horizonData?.unavailable_reason) {
      const info = document.createElement("div");
      info.className = "banner banner--info";
      info.textContent = horizonData.unavailable_reason;
      root.append(info);
    } else {
      root.append(emptyState(`No accuracy data for ${horizon}.`));
    }
    return root;
  }

  const list = document.createElement("div");
  list.className = "accuracy-list";
  list.style.background = "var(--card)";
  list.style.border = "1px solid var(--bdr)";
  list.style.borderRadius = "var(--rad-md)";

  rows.forEach((r) => {
    list.append(
      signalRow({
        name: r.name,
        accuracyPct: r.pct,
        sampleSize: Number.isFinite(r.samples) ? r.samples : null,
        threshold,
        disabled: !r.enabled,
        disabledReason: r.disabledReason,
      }),
    );
  });
  root.append(list);
  return root;
}

/**
 * Per-ticker accuracy matrix: rows = tickers, cols = horizons, cells =
 * consensus accuracy% + sample count from each horizon's `per_ticker`
 * block. Complements the per-signal list above — that view answers "which
 * signal is working", this one answers "which instrument is the consensus
 * actually calling correctly".
 * @param {{ data: object, horizons?: string[] }} props
 *   data: full /api/accuracy response, keyed by horizon.
 * @returns {HTMLElement}
 */
export function renderPerTickerTable({
  data,
  horizons = ["1d", "3d", "5d", "10d"],
} = {}) {
  const root = document.createElement("div");
  if (!data) {
    root.append(emptyState("No per-ticker accuracy data yet."));
    return root;
  }

  const tickerSet = new Set();
  for (const h of horizons) {
    const pt = data?.[h]?.per_ticker;
    if (pt) for (const t of Object.keys(pt)) tickerSet.add(t);
  }
  const tickers = [...tickerSet].sort();
  if (!tickers.length) {
    root.append(emptyState("No per-ticker accuracy data yet."));
    return root;
  }

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
  const corner = document.createElement("th");
  corner.textContent = "Ticker";
  corner.style.textAlign = "left";
  corner.style.padding = "var(--sp-2)";
  corner.style.fontSize = "var(--ty-xs)";
  corner.style.color = "var(--txm)";
  trH.append(corner);
  for (const h of horizons) {
    const th = document.createElement("th");
    th.textContent = h;
    th.style.padding = "var(--sp-2)";
    th.style.fontSize = "var(--ty-xs)";
    th.style.color = "var(--txm)";
    // Horizons with no outcome rows yet (e.g. 10d) carry an
    // unavailable_reason — surface it as a header tooltip so an empty
    // column reads as "not ready yet", not "broken".
    const reason = data?.[h]?.unavailable_reason;
    if (reason) th.title = reason;
    trH.append(th);
  }
  thead.append(trH);
  table.append(thead);

  const tbody = document.createElement("tbody");
  for (const ticker of tickers) {
    const tr = document.createElement("tr");
    tr.style.borderTop = "1px solid var(--bdr)";

    const tdName = document.createElement("td");
    tdName.textContent = ticker.replace(/-USD$/, "");
    tdName.title = ticker;
    tdName.style.padding = "var(--sp-2)";
    tdName.style.fontWeight = "600";
    tdName.style.fontSize = "var(--ty-sm)";
    tr.append(tdName);

    for (const h of horizons) {
      tr.append(_tickerCell(data?.[h]?.per_ticker?.[ticker]));
    }
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  root.append(wrap);
  return root;
}

// ---------------------------------------------------------------------------

function _summaryBar(horizonData) {
  const meta = horizonData?.meta;
  const consensusN = Number(horizonData?.consensus?.total ?? 0);
  const signalsTotal = _sumSignalTotals(horizonData?.signals);
  if (!meta?.updated_ts && !consensusN && !signalsTotal) return null;

  const bar = document.createElement("div");
  bar.style.display = "flex";
  bar.style.flexWrap = "wrap";
  bar.style.gap = "var(--sp-2)";
  bar.style.alignItems = "center";
  bar.style.marginBottom = "var(--sp-2)";
  bar.style.fontSize = "var(--ty-xs)";
  bar.style.color = "var(--txm)";

  const ageChip = _ageChip(meta);
  if (ageChip) bar.append(ageChip);

  const n = document.createElement("span");
  n.textContent = `consensus n=${consensusN} · signals total n=${signalsTotal}`;
  bar.append(n);

  return bar;
}

/** Age chip: "data 10h ago" — amber past 30m, red past 2h.
 * Uses data_ts (newest underlying signal) when older than the cache
 * rebuild ts: a recompute over frozen data stamps "now" while the
 * signals are hours old (loops stopped — 2026-07-19). */
function _ageChip(meta) {
  if (!meta || (meta.updated_ts == null && meta.data_ts == null)) return null;
  const candidates = [meta.updated_ts, meta.data_ts]
    .filter((t) => t != null)
    .map(Number);
  const effTs = Math.min(...candidates);
  const isDataAge = meta.data_ts != null && Number(meta.data_ts) === effTs;
  const ageSec = Math.max(0, Date.now() / 1000 - effTs);
  const color =
    ageSec > 7200 ? "var(--red)" : ageSec > 1800 ? "var(--yel)" : "var(--txm)";

  const chip = document.createElement("span");
  chip.textContent = `${isDataAge ? "data" : "updated"} ${fAgo(new Date(effTs * 1000))}`;
  chip.style.padding = "1px 6px";
  chip.style.borderRadius = "999px";
  chip.style.fontWeight = "700";
  chip.style.color = color;
  chip.style.border = `1px solid ${color}`;
  return chip;
}

function _sumSignalTotals(signals) {
  if (!signals || typeof signals !== "object") return 0;
  let sum = 0;
  for (const info of Object.values(signals)) {
    const n = Number(info?.total ?? info?.samples ?? info?.n ?? 0);
    if (Number.isFinite(n)) sum += n;
  }
  return sum;
}

function _tickerCell(info) {
  const td = document.createElement("td");
  td.style.textAlign = "center";
  td.style.padding = "var(--sp-2)";

  const total = Number(info?.total ?? 0);
  if (!info || !total) {
    td.style.color = "var(--txm)";
    td.style.fontSize = "var(--ty-sm)";
    td.textContent = "—";
    return td;
  }

  const pct = Number(info.pct ?? Number(info.accuracy) * 100);
  const color =
    pct >= KEEP_BAR ? "var(--grn)" : pct >= 47 ? "var(--tx)" : "var(--red)";

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
