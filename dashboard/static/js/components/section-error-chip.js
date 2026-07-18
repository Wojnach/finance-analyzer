/*
 * section-error-chip.js — inline "this section is degraded" indicator.
 *
 * Used by home-page render functions when a `/api/system_status` (or
 * `/api/trading_status`) section carries an `error` string (parse/import/
 * logic failure) instead of silently rendering a fabricated "all clear" /
 * zero value. A degraded reader must never look like a healthy 0-count
 * (2026-07-18 Phase 2 home redesign — same principle as the backend's
 * `unresolved: None` guard in system_status.py).
 */

/** @returns {HTMLElement} */
export function sectionErrorChip(message) {
  const chip = document.createElement("div");
  chip.className = "chip";
  chip.style.cursor = "default";
  chip.style.whiteSpace = "normal";
  chip.style.textTransform = "none";
  chip.style.letterSpacing = "normal";
  chip.style.color = "var(--txm)";
  chip.style.marginBottom = "var(--sp-2)";
  chip.textContent = `section unavailable: ${message}`;
  return chip;
}
