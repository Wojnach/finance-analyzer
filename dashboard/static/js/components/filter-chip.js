/*
 * filter-chip.js — toggleable chip with active/inactive state.
 */

/**
 * @param {{label: string, active?: boolean, onToggle?: (next: boolean) => void, value?: any}} props
 * @returns {HTMLElement}
 */
export function filterChip({ label = "", active = false, onToggle = null, value = null } = {}) {
  const el = document.createElement("button");
  el.type = "button";
  el.className = "chip chip--filter" + (active ? " active" : "");
  el.textContent = label;
  if (value != null) el.dataset.value = String(value);
  el.setAttribute("aria-pressed", String(!!active));

  if (onToggle) {
    el.addEventListener("click", () => {
      const next = !el.classList.contains("active");
      el.classList.toggle("active", next);
      el.setAttribute("aria-pressed", String(next));
      onToggle(next, value);
    });
  }
  return el;
}

/**
 * Convenience for a chip strip.
 * @param {Array<{label: string, value?: any, active?: boolean}>} items
 * @param {(value: any) => void} onSelect - receives the picked value
 * @returns {HTMLElement}
 */
export function chipStrip(items, onSelect) {
  const strip = document.createElement("div");
  strip.className = "chip-strip";
  items.forEach((it) => {
    const chip = filterChip({
      label: it.label,
      active: !!it.active,
      value: it.value ?? it.label,
      onToggle: () => {
        // Single-select strip: clear siblings, mark this one.
        strip.querySelectorAll(".chip").forEach((c) => {
          c.classList.remove("active");
          c.setAttribute("aria-pressed", "false");
        });
        chip.classList.add("active");
        chip.setAttribute("aria-pressed", "true");
        onSelect && onSelect(it.value ?? it.label);
      },
    });
    strip.append(chip);
  });
  return strip;
}
