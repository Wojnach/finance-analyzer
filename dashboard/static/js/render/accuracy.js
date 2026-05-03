/*
 * render/accuracy.js — per-signal accuracy list and history-chart helpers.
 */

import { signalRow } from "../components/signal-row.js";
import { emptyState } from "../components/empty-state.js";

/**
 * Renders the accuracy panel for a given horizon.
 * @param {{ horizon: "1d"|"3d"|"5d"|"10d", data: object, threshold?: number }} props
 * @returns {HTMLElement}
 */
export function renderAccuracyPanel({ horizon = "1d", data, threshold = 47 } = {}) {
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

  // Convert to array, sort by accuracy desc
  const rows = Object.entries(signals)
    .map(([name, info]) => ({
      name,
      pct: Number(info?.pct ?? info?.accuracy ?? info?.accuracy_pct ?? null),
      samples: Number(info?.samples ?? info?.n ?? info?.sample_size ?? null),
    }))
    .filter((r) => r.pct != null && Number.isFinite(r.pct))
    .sort((a, b) => (b.pct - a.pct));

  if (!rows.length) {
    root.append(emptyState(`No accuracy data for ${horizon}.`));
    return root;
  }

  const list = document.createElement("div");
  list.className = "accuracy-list";
  list.style.background = "var(--card)";
  list.style.border = "1px solid var(--bdr)";
  list.style.borderRadius = "var(--rad-md)";

  rows.forEach((r) => {
    list.append(signalRow({
      name: r.name,
      accuracyPct: r.pct,
      sampleSize: Number.isFinite(r.samples) ? r.samples : null,
      threshold,
    }));
  });
  root.append(list);
  return root;
}
