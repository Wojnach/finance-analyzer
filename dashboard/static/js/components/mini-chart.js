/*
 * mini-chart.js — Chart.js lifecycle wrapper.
 *
 * Returns a wrapper Element + dispose() so views can clean up on unmount
 * (the legacy code's `equityChartInstance` global had no dispose path —
 * caused subtle memory leaks).
 *
 * Chart.js is loaded via UMD <script> in index.html; if missing or still
 * loading we fail gracefully (placeholder + console warning).
 */

import { mobileDefaults } from "../charts/chart-config.js";

/**
 * @param {{
 *   type: string,             // "line" | "bar" | "scatter" | ...
 *   data: object,             // Chart.js data object
 *   options?: object,         // overrides on top of mobileDefaults
 *   height?: number,          // pixel height of canvas wrapper
 * }} props
 * @returns {{ element: HTMLElement, chart: object|null, dispose: () => void }}
 */
export function miniChart({ type = "line", data, options = {}, height = 200 } = {}) {
  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.width = "100%";
  wrap.style.height = `${height}px`;

  if (typeof window === "undefined" || typeof window.Chart === "undefined") {
    const ph = document.createElement("div");
    ph.className = "empty";
    ph.textContent = "Chart library not yet loaded.";
    wrap.append(ph);
    console.warn("mini-chart: window.Chart not available; placeholder rendered.");
    return { element: wrap, chart: null, dispose: () => {} };
  }

  const canvas = document.createElement("canvas");
  wrap.append(canvas);

  let chart = null;
  try {
    chart = new window.Chart(canvas.getContext("2d"), {
      type,
      data,
      options: mobileDefaults(options),
    });
  } catch (e) {
    console.error("mini-chart: failed to create chart", e);
    const ph = document.createElement("div");
    ph.className = "banner banner--error";
    ph.textContent = "Chart failed to render.";
    wrap.replaceChildren(ph);
  }

  return {
    element: wrap,
    chart,
    dispose: () => {
      try { chart && chart.destroy && chart.destroy(); } catch (_) {}
      // Drop the canvas ref so GC can reclaim
      while (wrap.firstChild) wrap.removeChild(wrap.firstChild);
    },
  };
}
