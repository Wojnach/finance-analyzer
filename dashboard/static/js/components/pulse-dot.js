/*
 * pulse-dot.js — single colored status dot (loop heartbeats, system pulse).
 */

const STATES = new Set(["ok", "warn", "fail", "idle"]);

/**
 * @param {{state: "ok"|"warn"|"fail"|"idle", label?: string, title?: string, onTap?: () => void}} props
 * @returns {HTMLElement}
 */
export function pulseDot({ state = "idle", label = "", title = "", onTap = null } = {}) {
  const wrap = document.createElement(onTap ? "button" : "span");
  wrap.className = "pulse-dot-wrap";
  wrap.style.display = "inline-flex";
  wrap.style.alignItems = "center";
  wrap.style.gap = "var(--sp-1)";
  if (onTap) {
    wrap.type = "button";
    wrap.style.cursor = "pointer";
    wrap.style.background = "transparent";
    wrap.style.border = "0";
    wrap.style.padding = "var(--sp-1)";
    wrap.style.minHeight = "var(--tap-min)";
    wrap.addEventListener("click", onTap);
  }
  if (title) wrap.title = title;

  const dot = document.createElement("span");
  dot.className = "pulse-dot " + (STATES.has(state) ? state : "idle");
  wrap.append(dot);

  if (label) {
    const txt = document.createElement("span");
    txt.style.fontSize = "var(--ty-sm)";
    txt.style.color = "var(--txd)";
    txt.textContent = label;
    wrap.append(txt);
  }

  return wrap;
}
