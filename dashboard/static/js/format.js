/*
 * format.js — value formatters.
 *
 * Ported from legacy index.html lines 853-879 (`fn`, `fs`, `fp`, `ft`, `eh`).
 * Names are kept short for terse render code. Each function tolerates
 * missing/undefined input and returns a placeholder ("--").
 */

/** Number with N decimal places. */
export function fn(v, decimals = 2) {
  if (v == null || !Number.isFinite(Number(v))) return "--";
  return Number(v).toFixed(decimals);
}

/** SEK formatter — abbreviates K / M for large amounts. */
export function fs(v) {
  if (v == null || !Number.isFinite(Number(v))) return "--";
  const n = Number(v);
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (abs >= 10_000)    return (n / 1_000).toFixed(1) + "K";
  if (abs >= 1_000)     return (n / 1_000).toFixed(2) + "K";
  return n.toFixed(0);
}

/** Price formatter — adapts decimal count to magnitude. */
export function fp(v) {
  if (v == null || !Number.isFinite(Number(v))) return "--";
  const n = Number(v);
  const abs = Math.abs(n);
  if (abs >= 10_000) return n.toFixed(0);
  if (abs >= 1_000)  return n.toFixed(1);
  if (abs >= 100)    return n.toFixed(2);
  if (abs >= 1)      return n.toFixed(3);
  return n.toFixed(5);
}

/** Percentage with sign. */
export function fpct(v, decimals = 2) {
  if (v == null || !Number.isFinite(Number(v))) return "--";
  const n = Number(v);
  return (n >= 0 ? "+" : "") + n.toFixed(decimals) + "%";
}

/** Compact ISO/epoch timestamp -> "HH:MM:SS" (local). */
export function ft(ts) {
  if (!ts) return "--";
  const d = ts instanceof Date ? ts : new Date(ts);
  if (Number.isNaN(d.getTime())) return "--";
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

/** Compact ISO/epoch timestamp -> "YYYY-MM-DD HH:MM:SS". */
export function ftFull(ts) {
  if (!ts) return "--";
  const d = ts instanceof Date ? ts : new Date(ts);
  if (Number.isNaN(d.getTime())) return "--";
  const pad = (x) => String(x).padStart(2, "0");
  return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate())
    + " " + pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
}

/** Relative-time formatter — "12s ago", "3m ago", "2h ago", "5d ago". */
export function fAgo(ts) {
  if (!ts) return "--";
  const d = ts instanceof Date ? ts : new Date(ts);
  if (Number.isNaN(d.getTime())) return "--";
  const seconds = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (seconds < 60)    return seconds + "s ago";
  if (seconds < 3600)  return Math.floor(seconds / 60) + "m ago";
  if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago";
  return Math.floor(seconds / 86400) + "d ago";
}

/** HTML-escape — match legacy `eh` semantics (XSS guard for innerHTML). */
export function eh(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Best-effort cast to number, or null. */
export function num(v) {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
