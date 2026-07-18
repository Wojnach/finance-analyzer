/*
 * charts/accuracy-chart.js — accuracy-history line charts.
 *
 * Used by the Signals "History" sub-tab. Split into small-multiples
 * (2026-07-18): consensusHistoryChart is a single always-shown line;
 * accuracyChart is the per-signal chart, capped at ~6 lines and driven by
 * a caller-selected signal list (views/signals.js selector chips) instead
 * of a hardcoded topN — mashing 90 signals + consensus into one chart was
 * unreadable on phone.
 */

import { miniChart } from "../components/mini-chart.js";
import { getChartColors } from "../theme.js";

/**
 * Ranks signals by total sample count across all snapshots (descending).
 * Shared by the view for both the default per-signal selection and the
 * candidate chip list.
 * @param {object[]} history
 * @returns {{name: string, total: number}[]}
 */
export function rankSignalsByVolume(history = []) {
  // The snapshot writer uses `total` for sample count; older drafts
  // assumed `samples` / `n`, which never matched the live data and
  // produced an empty chart.
  const counts = Object.create(null);
  for (const snap of history) {
    const sigs = snap?.signals || {};
    for (const [name, info] of Object.entries(sigs)) {
      const n = Number(info?.total ?? info?.samples ?? info?.n ?? 0);
      if (!Number.isFinite(n)) continue;
      counts[name] = (counts[name] || 0) + n;
    }
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, total]) => ({ name, total }));
}

/**
 * Single-line consensus accuracy over time — kept separate from the
 * per-signal chart so it isn't lost in the 90-line spaghetti.
 * @param {{ history: object[], height?: number }} props
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function consensusHistoryChart({ history = [], height = 200 } = {}) {
  const c = getChartColors();
  const sorted = _sortByTs(history);
  const labels = sorted.map((h) => h.date || h.ts || "");

  const datasets = [
    {
      label: "Consensus",
      data: sorted.map((h) => _toPct(h?.consensus)),
      borderColor: c.green,
      backgroundColor: "rgba(0,255,136,0.08)",
      borderWidth: 1.8,
      fill: true,
      pointRadius: sorted.length < 6 ? 3 : 0,
      tension: 0.2,
      spanGaps: true,
    },
    // 50% baseline
    {
      label: "50%",
      data: labels.map(() => 50),
      borderColor: c.muted,
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
    },
  ];

  return miniChart({
    type: "line",
    data: { labels, datasets },
    options: {
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: { color: c.dim, font: { size: 10 }, boxWidth: 8 },
        },
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { stepSize: 25 },
          title: { display: true, text: "Accuracy %", color: c.muted },
        },
        x: {
          ticks: { maxTicksLimit: 6 },
          title: { display: true, text: "Time", color: c.muted },
        },
      },
    },
    height,
  });
}

/**
 * Per-signal accuracy-history small-multiple.
 * @param {{ history: object[], signals?: string[], topN?: number, height?: number }} props
 *   history: array of snapshots written by the daily accuracy job. Each
 *   snapshot has a `signals: { signal_name: { accuracy: 0..1, total: int, pct?: 0..100 } }`
 *   subdocument plus a `ts` timestamp.
 *   signals: explicit signal names to plot (caller-selected, capped at 6).
 *   Falls back to the top-N by sample volume when omitted.
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function accuracyChart({
  history = [],
  signals = null,
  topN = 6,
  height = 240,
} = {}) {
  const c = getChartColors();
  const palette = [
    c.cyan,
    c.green,
    c.orange,
    c.yellow,
    c.blue,
    c.red,
    "#a855f7",
    "#ec4899",
  ];

  const sorted = _sortByTs(history);
  const top =
    Array.isArray(signals) && signals.length
      ? signals.slice(0, 6)
      : rankSignalsByVolume(sorted)
          .slice(0, topN)
          .map((r) => r.name);

  // Snapshots store accuracy as either `pct` (0..100) or `accuracy` (0..1
  // fraction). Normalize to a 0..100 percentage so the chart's y-axis is
  // honest. Showing 0.55 on a 0..100 scale was the pre-fix bug.
  const labels = sorted.map((h) => h.date || h.ts || "");
  const datasets = top.map((name, i) => ({
    label: name,
    data: sorted.map((h) => _toPct(h?.signals?.[name])),
    borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length],
    borderWidth: 1.4,
    pointRadius: sorted.length < 6 ? 3 : 0, // show dots when sparse
    tension: 0.2,
    spanGaps: true,
  }));
  // 50% baseline
  datasets.push({
    label: "50%",
    data: labels.map(() => 50),
    borderColor: c.muted,
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
  });

  return miniChart({
    type: "line",
    data: { labels, datasets },
    options: {
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: { color: c.dim, font: { size: 10 }, boxWidth: 8 },
        },
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { stepSize: 25 },
          title: { display: true, text: "Accuracy %", color: c.muted },
        },
        x: {
          ticks: { maxTicksLimit: 6 },
          title: { display: true, text: "Time", color: c.muted },
        },
      },
    },
    height,
  });
}

// ---------------------------------------------------------------------------

// Snapshots may arrive out of order; sort ascending by timestamp so the
// line goes left-to-right.
function _sortByTs(history) {
  return [...history].sort((a, b) => {
    const av = String(a?.ts || a?.date || "");
    const bv = String(b?.ts || b?.date || "");
    return av < bv ? -1 : av > bv ? 1 : 0;
  });
}

function _toPct(info) {
  if (!info) return null;
  if (Number.isFinite(Number(info.pct))) return Number(info.pct);
  if (Number.isFinite(Number(info.accuracy_pct)))
    return Number(info.accuracy_pct);
  if (Number.isFinite(Number(info.accuracy))) {
    const a = Number(info.accuracy);
    return a <= 1.5 ? a * 100 : a; // tolerate either scale
  }
  return null;
}
