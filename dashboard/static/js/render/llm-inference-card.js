/*
 * render/llm-inference-card.js — per-LLM inference success bars.
 *
 * Reads `/api/system_status` payload's `llm_inference` section and renders:
 *   - One row per LLM: name, success_pct, total calls
 *   - Inline progress bar with color by success threshold
 *   - Header with overall weighted average
 *
 * <80% bars go red, <95% yellow, ≥95% green — matches the system_status
 * severity thresholds.
 */

const RED_BELOW = 80.0;
const YELLOW_BELOW = 95.0;

/** @returns {HTMLElement} */
export function llmInferenceCard(llmPayload) {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.alignItems = "baseline";
  header.style.marginBottom = "var(--sp-2)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0";
  title.textContent = "LLM inference health";
  header.append(title);

  const overall = document.createElement("span");
  const overallPct = llmPayload?.overall_pct;
  overall.style.fontSize = "var(--ty-sm)";
  overall.style.color = _colorFor(overallPct);
  overall.style.fontWeight = "600";
  overall.textContent = overallPct == null ? "—" : `${overallPct.toFixed(1)}% avg`;
  header.append(overall);
  card.append(header);

  const models = Array.isArray(llmPayload?.models) ? llmPayload.models : [];
  if (!models.length) {
    const empty = document.createElement("div");
    empty.style.color = "var(--txm)";
    empty.style.fontSize = "var(--ty-sm)";
    empty.textContent = "no LLM telemetry yet";
    card.append(empty);
    return card;
  }

  for (const m of models) card.append(_modelRow(m));
  return card;
}

function _modelRow(m) {
  const row = document.createElement("div");
  row.style.padding = "var(--sp-1) 0";

  const top = document.createElement("div");
  top.style.display = "flex";
  top.style.justifyContent = "space-between";
  top.style.fontSize = "var(--ty-sm)";
  top.style.marginBottom = "2px";

  const name = document.createElement("span");
  name.textContent = m.name || m.key || "?";
  name.style.color = "var(--tx)";
  top.append(name);

  const stats = document.createElement("span");
  stats.style.color = "var(--txm)";
  stats.textContent = `${(m.success_pct ?? 0).toFixed(1)}% · ${_compactNumber(m.total ?? 0)}`;
  top.append(stats);
  row.append(top);

  const barWrap = document.createElement("div");
  barWrap.style.height = "6px";
  barWrap.style.background = "var(--bd)";
  barWrap.style.borderRadius = "3px";
  barWrap.style.overflow = "hidden";

  const bar = document.createElement("div");
  const pct = Math.max(0, Math.min(100, m.success_pct ?? 0));
  bar.style.height = "100%";
  bar.style.width = `${pct}%`;
  bar.style.background = _colorFor(pct);
  bar.style.transition = "width 200ms ease";
  barWrap.append(bar);
  row.append(barWrap);

  return row;
}

function _colorFor(pct) {
  if (pct == null) return "var(--txm)";
  if (pct < RED_BELOW) return "var(--red)";
  if (pct < YELLOW_BELOW) return "var(--yel)";
  return "var(--grn)";
}

function _compactNumber(n) {
  const num = Number(n);
  if (!Number.isFinite(num)) return "?";
  if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + "M";
  if (num >= 10_000) return (num / 1_000).toFixed(0) + "k";
  if (num >= 1_000) return (num / 1_000).toFixed(1) + "k";
  return String(num);
}
