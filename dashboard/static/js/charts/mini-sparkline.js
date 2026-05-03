/*
 * mini-sparkline.js — single-line sparkline for inline use (P&L card,
 * position cards). Strips all chart chrome.
 */

import { miniChart } from "../components/mini-chart.js";
import { sparklineOptions } from "./chart-config.js";

/**
 * @param {{
 *   values: number[],
 *   labels?: any[],
 *   color?: string,
 *   fillColor?: string,
 *   height?: number,
 * }} props
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function miniSparkline({ values = [], labels = [], color = null, fillColor = null, height = 40 } = {}) {
  const tone = _autoColor(values);
  const stroke = color || tone.line;
  const fill   = fillColor || tone.fill;

  const data = {
    labels: labels.length ? labels : values.map((_, i) => i),
    datasets: [{
      data: values,
      borderColor: stroke,
      backgroundColor: fill,
      borderWidth: 1.4,
      fill: true,
      tension: 0.25,
      pointRadius: 0,
    }],
  };
  const { element, dispose } = miniChart({
    type: "line",
    data,
    options: sparklineOptions(),
    height,
  });
  return { element, dispose };
}

function _autoColor(values) {
  // Up vs down: compare last to first.
  if (!values.length) return { line: "#6b7280", fill: "rgba(107,114,128,0.15)" };
  const a = Number(values[0]);
  const b = Number(values[values.length - 1]);
  if (Number.isFinite(a) && Number.isFinite(b)) {
    if (b >= a) return { line: "#00ff88", fill: "rgba(0,255,136,0.18)" };
    return                 { line: "#ff4444", fill: "rgba(255,68,68,0.18)" };
  }
  return { line: "#6b7280", fill: "rgba(107,114,128,0.15)" };
}
