/*
 * render/command-central-card.js — Layer 1 + Layer 2 + voters rollup.
 *
 * Renamed/broadened from layer2-activity-card.js (2026-07-18): this is
 * now the one card that answers "is the whole decision pipeline actually
 * running" — Layer 1 loop state, Layer 2 trigger/cost stats + Claude
 * gate, and a voter-state summary count. Keeps the previous card's
 * tap-to-/decisions behaviour on the main body; a separate "Manage ->"
 * row links to the future #control view (Phase 3 — the router falls
 * back gracefully until that view exists).
 *
 * Reads the full `/api/system_status` payload's `layer1`, `layer2`, and
 * `voters` sections.
 */

import * as router from "../router.js";
import { ft } from "../format.js";
import { sectionErrorChip } from "../components/section-error-chip.js";

/** @returns {HTMLElement} */
export function commandCentralCard(sys) {
  const layer2Payload = sys?.layer2;
  const layer1Payload = sys?.layer1;
  const votersPayload = sys?.voters;

  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const main = document.createElement("button");
  main.type = "button";
  main.className = "card--tap";
  main.style.display = "block";
  main.style.width = "100%";
  main.style.textAlign = "left";
  main.style.background = "transparent";
  main.style.border = "0";
  main.style.padding = "0";
  main.style.minHeight = "var(--tap-min)";
  main.addEventListener("click", () => router.navigate("decisions"));

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.alignItems = "baseline";
  header.style.marginBottom = "var(--sp-2)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0";
  title.textContent = "Command Central";
  header.append(title);

  const gate = layer2Payload?.gate;
  if (gate && gate.label && gate.label !== "UNKNOWN") {
    const frozen = gate.enabled === false;
    const pill = document.createElement("span");
    pill.textContent = frozen ? "⏸ FROZEN" : "● LIVE";
    pill.title =
      `config.layer2=${gate.config_layer2_enabled} · ` +
      `gate=${gate.claude_gate_enabled} · metals=${gate.metals_claude_enabled}`;
    pill.style.fontSize = "0.7rem";
    pill.style.fontWeight = "700";
    pill.style.padding = "1px 6px";
    pill.style.borderRadius = "999px";
    pill.style.marginLeft = "var(--sp-2)";
    pill.style.whiteSpace = "nowrap";
    pill.style.color = frozen ? "var(--yel)" : "var(--grn)";
    pill.style.border = `1px solid ${frozen ? "var(--yel)" : "var(--grn)"}`;
    header.append(pill);
  }

  const triggers = layer2Payload?.triggers_24h ?? 0;
  const pct = layer2Payload?.success_pct;
  const headline = document.createElement("span");
  headline.style.fontSize = "var(--ty-sm)";
  headline.style.fontWeight = "600";
  headline.style.color = _colorForPct(pct, triggers);
  headline.textContent = pct == null
    ? `${triggers} triggers`
    : `${triggers} triggers · ${pct.toFixed(0)}%`;
  header.append(headline);
  main.append(header);

  if (layer2Payload?.error) main.append(sectionErrorChip(layer2Payload.error));

  main.append(_layer1Row(layer1Payload));
  main.append(_sparkBars(layer2Payload?.spark_24h));

  const cost = layer2Payload?.cost_usd_24h;
  const inTok = layer2Payload?.input_tokens_24h;
  const outTok = layer2Payload?.output_tokens_24h;
  const cacheRead = layer2Payload?.cache_read_tokens_24h;
  if (Number.isFinite(cost) || Number.isFinite(inTok) || Number.isFinite(outTok)) {
    const costRow = document.createElement("div");
    costRow.style.marginTop = "var(--sp-2)";
    costRow.style.fontSize = "var(--ty-sm)";
    costRow.style.color = "var(--tx)";
    const totalTok = (Number(inTok) || 0) + (Number(outTok) || 0);
    const cachePart = Number(cacheRead) > 0 ? ` · cache ${_fmtTok(cacheRead)}` : "";
    costRow.textContent = `$${(cost || 0).toFixed(2)} · ${_fmtTok(totalTok)} tok${cachePart}`;
    main.append(costRow);
  }

  const latest = layer2Payload?.latest;
  if (latest) {
    const summary = document.createElement("div");
    summary.style.marginTop = "var(--sp-2)";
    summary.style.fontSize = "var(--ty-sm)";
    summary.style.color = "var(--txm)";
    summary.style.overflow = "hidden";
    summary.style.textOverflow = "ellipsis";
    summary.style.whiteSpace = "nowrap";
    const status = (latest.status || "?").toUpperCase();
    const dur = Number.isFinite(latest.duration_seconds)
      ? `${latest.duration_seconds.toFixed(0)}s`
      : "?";
    summary.textContent = `last: ${ft(latest.ts)} · ${latest.caller || "?"} · ${status} · ${dur}`;
    main.append(summary);
  }

  const votersLine = _votersSummaryLine(votersPayload);
  if (votersLine) main.append(votersLine);

  card.append(main);
  card.append(_manageLink());
  return card;
}

function _layer1Row(l1) {
  const row = document.createElement("div");
  row.style.fontSize = "var(--ty-sm)";
  row.style.marginBottom = "var(--sp-1)";
  if (!l1 || l1.error) {
    row.style.color = "var(--txm)";
    row.textContent = `Layer 1: unavailable${l1?.error ? ` (${l1.error})` : ""}`;
    return row;
  }
  const active = !!l1.active;
  const enabled = !!l1.enabled;
  const dot = document.createElement("span");
  dot.textContent = "● ";
  dot.style.color = active ? "var(--grn)" : "var(--red)";
  row.append(dot);
  const text = document.createElement("span");
  text.style.color = "var(--tx)";
  text.textContent = `Layer 1: ${active ? "active" : "stopped"} (${enabled ? "enabled" : "disabled"})`;
  row.append(text);
  return row;
}

function _votersSummaryLine(votersPayload) {
  if (!votersPayload || votersPayload.error) return null;
  const states = Object.values(votersPayload).map((v) => v?.state);
  if (!states.length) return null;
  const voting = states.filter((s) => s === "VOTING").length;
  const disabled = states.filter((s) => s === "DISABLED").length;
  const paused = states.length - voting - disabled;
  const row = document.createElement("div");
  row.style.marginTop = "var(--sp-2)";
  row.style.fontSize = "var(--ty-sm)";
  row.style.color = "var(--txm)";
  const parts = [`${voting} voting`, `${disabled} disabled`];
  if (paused > 0) parts.push(`${paused} paused`);
  row.textContent = parts.join(" · ");
  return row;
}

function _manageLink() {
  const row = document.createElement("button");
  row.type = "button";
  row.style.display = "block";
  row.style.width = "100%";
  row.style.textAlign = "left";
  row.style.background = "transparent";
  row.style.border = "0";
  row.style.borderTop = "1px solid var(--bdr)";
  row.style.marginTop = "var(--sp-2)";
  row.style.paddingTop = "var(--sp-2)";
  row.style.minHeight = "var(--tap-min)";
  row.style.color = "var(--cyn)";
  row.style.fontWeight = "600";
  row.style.fontSize = "var(--ty-sm)";
  row.style.cursor = "pointer";
  row.textContent = "Manage →";
  row.addEventListener("click", () => router.navigate("control"));
  return row;
}

function _sparkBars(values) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.alignItems = "flex-end";
  wrap.style.gap = "1px";
  wrap.style.height = "24px";
  wrap.style.padding = "2px 0";

  const arr = Array.isArray(values) && values.length === 24 ? values : new Array(24).fill(0);
  const max = Math.max(...arr, 1);
  for (let i = 0; i < arr.length; i++) {
    const v = Math.max(0, Number(arr[i]) || 0);
    const bar = document.createElement("span");
    bar.style.flex = "1";
    bar.style.height = `${(v / max) * 100}%`;
    bar.style.minHeight = "2px";
    bar.style.background = v ? "var(--cyn)" : "var(--bd)";
    bar.style.borderRadius = "1px";
    wrap.append(bar);
  }
  return wrap;
}

function _fmtTok(n) {
  const v = Number(n) || 0;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return String(v);
}

function _colorForPct(pct, triggers) {
  if (pct == null || triggers < 3) return "var(--txm)";
  if (pct < 60) return "var(--red)";
  if (pct < 85) return "var(--yel)";
  return "var(--grn)";
}
