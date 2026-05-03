/*
 * error-banner.js — top-of-view error banner.
 */

/**
 * @param {string} message
 * @param {{onDismiss?: () => void}} [opts]
 * @returns {HTMLElement}
 */
export function errorBanner(message, opts = {}) {
  const el = document.createElement("div");
  el.className = "banner banner--error";
  el.textContent = message || "Something went wrong.";
  if (opts.onDismiss) {
    const close = document.createElement("button");
    close.type = "button";
    close.className = "icon-btn";
    close.setAttribute("aria-label", "Dismiss");
    close.textContent = "×";
    close.style.marginLeft = "auto";
    close.addEventListener("click", () => {
      el.remove();
      opts.onDismiss();
    });
    el.append(close);
  }
  return el;
}
