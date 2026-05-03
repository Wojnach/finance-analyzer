/*
 * charts/accuracy-chart.js — top-N accuracy-history line chart.
 *
 * Used by the Signals "History" sub-tab. Chart.js multi-line, single
 * series default-active to avoid line-spaghetti on phone (Track 5
 * recommended).
 */

import { miniChart } from "../components/mini-chart.js";
import { mobileDefaults } from "./chart-config.js";
import { getChartColors } from "../theme.js";

/**
 * @param {{ history: object[], topN?: number, height?: number }} props
 *   history: array of snapshots written by the daily accuracy job. Each
 *   snapshot has a `signals: { signal_name: { accuracy: 0..1, total: int, pct?: 0..100 } }`
 *   subdocument plus a `ts` timestamp.
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function accuracyChart({ history = [], topN = 8, height = 240 } = {}) {
  const c = getChartColors();
  const palette = [c.cyan, c.green, c.orange, c.yellow, c.blue, c.red, "#a855f7", "#ec4899"];

  // Snapshots may arrive out of order; sort ascending by timestamp so the
  // line goes left-to-right.
  const sorted = [...history].sort((a, b) => {
    const av = String(a?.ts || a?.date || "");
    const bv = String(b?.ts || b?.date || "");
    return av < bv ? -1 : av > bv ? 1 : 0;
  });

  // Aggregate signals by sample count to pick top-N. The snapshot writer
  // uses `total` for sample count; older drafts assumed `samples` / `n`,
  // which never matched the live data and produced an empty chart.
  const counts = Object.create(null);
  for (const snap of sorted) {
    const sigs = snap?.signals || {};
    for (const [name, info] of Object.entries(sigs)) {
      const n = Number(info?.total ?? info?.samples ?? info?.n ?? 0);
      if (!Number.isFinite(n)) continue;
      counts[name] = (counts[name] || 0) + n;
    }
  }
  const top = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([name]) => name);

  // Snapshots store accuracy as either `pct` (0..100) or `accuracy` (0..1
  // fraction). Normalize to a 0..100 percentage so the chart's y-axis is
  // honest. Showing 0.55 on a 0..100 scale was the pre-fix bug.
  function _toPct(info) {
    if (!info) return null;
    if (Number.isFinite(Number(info.pct))) return Number(info.pct);
    if (Number.isFinite(Number(info.accuracy_pct))) return Number(info.accuracy_pct);
    if (Number.isFinite(Number(info.accuracy))) {
      const a = Number(info.accuracy);
      return a <= 1.5 ? a * 100 : a;  // tolerate either scale
    }
    return null;
  }

  // Build one dataset per top signal.
  const labels = sorted.map((h) => h.date || h.ts || "");
  const datasets = top.map((name, i) => ({
    label: name,
    data: sorted.map((h) => _toPct(h?.signals?.[name])),
    borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length],
    borderWidth: 1.4,
    pointRadius: sorted.length < 6 ? 3 : 0,  // show dots when sparse
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
        legend: { display: true, position: "bottom",
                  labels: { color: c.dim, font: { size: 10 }, boxWidth: 8 } },
      },
      scales: {
        y: { min: 0, max: 100, ticks: { stepSize: 25 } },
        x: { ticks: { maxTicksLimit: 6 } },
      },
    },
    height,
  });
}
