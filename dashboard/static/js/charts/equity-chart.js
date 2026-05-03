/*
 * charts/equity-chart.js — full Patient + Bold equity curve with trade marks.
 */

import { miniChart } from "../components/mini-chart.js";
import { getChartColors } from "../theme.js";

/**
 * @param {{
 *   curve: object[],   // [{ts, patient_value_sek, bold_value_sek}] from portfolio_value_history.jsonl
 *   trades?: object[], // [{ts, action: "BUY"|"SELL", strategy: "patient"|"bold"}]
 *   height?: number,
 * }} props
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function equityChart({ curve = [], trades = [], height = 260 } = {}) {
  const c = getChartColors();
  const labels = curve.map((p) => p.ts || p.timestamp || "");
  // Field names match `portfolio_value_history.jsonl`: `patient_value_sek`
  // and `bold_value_sek`. The earlier `total_sek` / `total_sek_bold`
  // assumption in the redesign produced an empty chart on real data
  // (Codex P1 finding 2026-05-03).
  const datasets = [
    {
      label: "Patient",
      data: curve.map((p) => Number(p.patient_value_sek ?? p.total_sek)),
      borderColor: c.cyan,
      backgroundColor: "rgba(6,182,212,0.10)",
      borderWidth: 1.6,
      fill: true,
      pointRadius: 0,
      tension: 0.2,
      spanGaps: true,
    },
    {
      label: "Bold",
      data: curve.map((p) => Number(p.bold_value_sek ?? p.total_sek_bold)),
      borderColor: c.orange,
      backgroundColor: "rgba(249,115,22,0.10)",
      borderWidth: 1.6,
      fill: true,
      pointRadius: 0,
      tension: 0.2,
      spanGaps: true,
    },
  ];

  // BUY/SELL trade marks (small triangles)
  const trades_buy  = trades.filter((t) => (t.action || "").toUpperCase() === "BUY");
  const trades_sell = trades.filter((t) => (t.action || "").toUpperCase() === "SELL");
  if (trades_buy.length || trades_sell.length) {
    datasets.push(_tradeMarks(trades_buy,  c.green, "BUY"));
    datasets.push(_tradeMarks(trades_sell, c.red,   "SELL"));
  }

  return miniChart({
    type: "line",
    data: { labels, datasets },
    options: {
      plugins: {
        legend: { display: true, position: "bottom",
                  labels: { color: c.dim, font: { size: 11 }, boxWidth: 8 } },
      },
      scales: { x: { ticks: { maxTicksLimit: 6 } } },
    },
    height,
  });
}

function _tradeMarks(arr, color, label) {
  return {
    label,
    data: arr.map((t) => ({ x: t.ts, y: t.equity_sek ?? null })),
    borderColor: color,
    backgroundColor: color,
    showLine: false,
    pointStyle: label === "BUY" ? "triangle" : "rect",
    pointRadius: 5,
    pointHoverRadius: 7,
  };
}
