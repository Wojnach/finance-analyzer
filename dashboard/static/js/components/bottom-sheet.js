/*
 * bottom-sheet.js — universal long-press / drill-down modal.
 *
 * Single instance bound to the existing #bottom-sheet element in index.html.
 * Open with a content Node + title; backdrop tap closes; Esc closes; the
 * panel can be swiped down (touchmove) to dismiss.
 */

const SHEET_ID = "bottom-sheet";
const CONTENT_ID = "bottom-sheet-content";

let _onClose = null;
let _bound = false;

function _ensureBindings() {
  if (_bound) return;
  const sheet = document.getElementById(SHEET_ID);
  if (!sheet) return;

  // Backdrop click
  sheet.querySelector("[data-bottom-sheet-close]")?.addEventListener("click", close);

  // Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && sheet.classList.contains("open")) close();
  });

  // Swipe-down gesture on the panel
  const panel = sheet.querySelector(".bottom-sheet__panel");
  if (panel) {
    let startY = null;
    panel.addEventListener("touchstart", (e) => {
      if (panel.scrollTop > 0) { startY = null; return; }
      startY = e.touches[0]?.clientY ?? null;
    }, { passive: true });
    panel.addEventListener("touchmove", (e) => {
      if (startY == null) return;
      const dy = (e.touches[0]?.clientY ?? startY) - startY;
      if (dy > 80) { startY = null; close(); }
    }, { passive: true });
  }

  _bound = true;
}

/**
 * Open the bottom sheet.
 * @param {{title?: string, content: Node, onClose?: () => void}} props
 */
export function open({ title = "", content, onClose = null } = {}) {
  _ensureBindings();
  const sheet = document.getElementById(SHEET_ID);
  const contentEl = document.getElementById(CONTENT_ID);
  if (!sheet || !contentEl) return;

  // Clear & populate
  while (contentEl.firstChild) contentEl.removeChild(contentEl.firstChild);
  if (title) {
    const t = document.createElement("div");
    t.className = "bottom-sheet__title";
    t.textContent = title;
    contentEl.append(t);
  }
  if (content instanceof Node) {
    contentEl.append(content);
  } else if (typeof content === "string") {
    const p = document.createElement("p");
    p.textContent = content;
    contentEl.append(p);
  }

  _onClose = onClose;
  sheet.classList.add("open");
  sheet.setAttribute("aria-hidden", "false");
}

export function close() {
  const sheet = document.getElementById(SHEET_ID);
  if (!sheet) return;
  sheet.classList.remove("open");
  sheet.setAttribute("aria-hidden", "true");
  if (_onClose) {
    try { _onClose(); } catch (e) { console.warn("bottom-sheet onClose threw", e); }
    _onClose = null;
  }
}

export function isOpen() {
  const sheet = document.getElementById(SHEET_ID);
  return !!sheet && sheet.classList.contains("open");
}

/**
 * Long-press helper. Attach to any element to open the sheet on long-press.
 * @param {HTMLElement} target
 * @param {(e: Event) => {title?: string, content: Node, onClose?: () => void} | null | undefined} sheetFn
 *        - returns the sheet props, or null/undefined to skip.
 * @param {number} [durationMs=420]
 * @returns {() => void} cleanup function
 */
export function bindLongPress(target, sheetFn, durationMs = 420) {
  let timer = null;
  let triggered = false;

  function start() {
    triggered = false;
    timer = window.setTimeout(() => {
      timer = null;
      const props = sheetFn?.();
      if (props) {
        triggered = true;
        open(props);
      }
    }, durationMs);
  }
  function cancel() {
    if (timer) { clearTimeout(timer); timer = null; }
  }

  target.addEventListener("touchstart", start,  { passive: true });
  target.addEventListener("touchend",   cancel, { passive: true });
  target.addEventListener("touchmove",  cancel, { passive: true });
  target.addEventListener("touchcancel", cancel, { passive: true });
  // Mouse equivalents for desktop
  target.addEventListener("mousedown", start);
  target.addEventListener("mouseup",   cancel);
  target.addEventListener("mouseleave", cancel);
  // Suppress click after a successful long-press
  target.addEventListener("click", (e) => {
    if (triggered) {
      e.stopPropagation();
      e.preventDefault();
      triggered = false;
    }
  }, true);

  return () => {
    cancel();
    target.removeEventListener("touchstart", start);
    target.removeEventListener("touchend",   cancel);
    target.removeEventListener("touchmove",  cancel);
    target.removeEventListener("touchcancel", cancel);
    target.removeEventListener("mousedown", start);
    target.removeEventListener("mouseup",   cancel);
    target.removeEventListener("mouseleave", cancel);
  };
}
