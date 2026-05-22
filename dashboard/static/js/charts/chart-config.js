/*
 * chart-config.js — shared Chart.js mobile defaults.
 *
 * Track-6 decisions:
 *  - animation: 0 — skip the redraw dance on every refresh.
 *  - devicePixelRatio capped at 2 — high-DPR phones (3-4) don't pay back.
 *  - interaction.mode: 'index' — single tap = tooltip for that x-position.
 *  - maintainAspectRatio: false — let the container set the size.
 *  - responsive: true — resize on window/container change.
 */

import { getChartColors } from "../theme.js";

/* Terminal reskin 2026-05-22 — charts render in the monospace stack so
 * axis labels and tooltips match the rest of the terminal dashboard. */
const MONO =
  "'Cascadia Code','Fira Code','JetBrains Mono','SF Mono',Menlo,Consolas,monospace";

/**
 * Returns a mobile-friendly options object that callers can deep-merge with
 * chart-specific options.
 *
 * @param {object} [overrides] — caller-specific options
 * @returns {object}
 */
export function mobileDefaults(overrides = {}) {
  const colors = getChartColors();
  const base = {
    responsive: true,
    maintainAspectRatio: false,
    devicePixelRatio: Math.min(window.devicePixelRatio || 1, 2),
    animation: { duration: 0 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        display: false,
        position: "bottom",
        labels: { color: colors.dim, font: { size: 11, family: MONO } },
      },
      tooltip: {
        backgroundColor: colors.card,
        titleColor:  colors.text,
        bodyColor:   colors.text,
        borderColor: colors.grid,
        borderWidth: 1,
        padding: 8,
        cornerRadius: 0,
        titleFont: { family: MONO },
        bodyFont:  { family: MONO },
      },
    },
    scales: {
      x: {
        ticks: { color: colors.muted, font: { size: 10, family: MONO }, maxRotation: 0 },
        grid: { color: colors.grid, drawBorder: false },
      },
      y: {
        ticks: { color: colors.muted, font: { size: 10, family: MONO } },
        grid: { color: colors.grid, drawBorder: false },
      },
    },
  };
  return _deepMerge(base, overrides);
}

function _deepMerge(target, source) {
  if (!source || typeof source !== "object") return target;
  const out = { ...target };
  for (const k of Object.keys(source)) {
    const v = source[k];
    if (v && typeof v === "object" && !Array.isArray(v)
        && out[k] && typeof out[k] === "object" && !Array.isArray(out[k])) {
      out[k] = _deepMerge(out[k], v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

/**
 * Sparkline-only options — drops axis labels, gridlines, legend.
 */
export function sparklineOptions(overrides = {}) {
  return mobileDefaults({
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { display: false, grid: { display: false }, ticks: { display: false } },
      y: { display: false, grid: { display: false }, ticks: { display: false } },
    },
    elements: { point: { radius: 0 } },
    ...overrides,
  });
}
