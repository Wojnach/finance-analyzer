/*
 * toast.js — transient bottom-of-screen notification.
 *
 * Singleton element appended to <body> lazily. Re-triggering while a
 * toast is visible resets its timer instead of stacking multiple toasts —
 * the dashboard only ever needs one message on screen at a time.
 */

let _el = null;
let _timer = null;

/**
 * @param {string} message
 * @param {{duration?: number}} [opts]
 */
export function showToast(message, { duration = 2200 } = {}) {
  if (!_el) {
    _el = document.createElement("div");
    _el.className = "toast";
    document.body.append(_el);
  }
  _el.textContent = message;
  _el.classList.add("show");
  if (_timer) clearTimeout(_timer);
  _timer = setTimeout(() => {
    _el.classList.remove("show");
    _timer = null;
  }, duration);
}
