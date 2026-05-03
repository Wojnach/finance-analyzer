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
 *   history: array of snapshots, each { date, signals: { signal: pct } }
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function accuracyChart({ history = [], topN = 8, height = 240 } = {}) {
  const c = getChartColors();
  const palette = [c.cyan, c.green, c.orange, c.yellow, c.blue, c.red, "#a855f7", "#ec4899"];

  // Aggregate signals by sample count to pick top-N.
  const counts = Object.create(null);
  for (const snap of history) {
    const sigs = snap?.signals || {};
    for (const [name, info] of Object.entries(sigs)) {
      const n = Number(info?.samples ?? info?.n ?? 0);
      if (!Number.isFinite(n)) continue;
      counts[name] = (counts[name] || 0) + n;
    }
  }
  const top = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([name]) => name);

  // Build one dataset per top signal.
  const labels = history.map((h) => h.date || h.ts || "");
  const datasets = top.map((name, i) => ({
    label: name,
    data: history.map((h) => {
      const v = h?.signals?.[name];
      return v && (v.pct ?? v.accuracy ?? v.accuracy_pct);
    }),
    borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length],
    borderWidth: 1.4,
    pointRadius: 0,
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
