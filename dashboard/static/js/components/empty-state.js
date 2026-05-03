/*
 * empty-state.js — placeholder for "no data yet" sections.
 * Returns a DOM node — never innerHTML to keep XSS-safe.
 */

/**
 * @param {string} message - the message to display
 * @returns {HTMLElement}
 */
export function emptyState(message) {
  const el = document.createElement("div");
  el.className = "empty";
  el.textContent = message || "No data yet.";
  return el;
}
