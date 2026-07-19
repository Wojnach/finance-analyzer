/*
 * render/silver-pipeline.js — #silver page pipeline diagram (Phase 6).
 *
 * Two live-colored lanes: paper (Layer 1 signals → consensus → autonomous
 * rec) vs real money (metals_loop → grid_fisher/LLM → Avanza warrants).
 * Colors are derived from loop/heartbeat/gate state, NOT a literal
 * per-stage health check — no such instrumentation exists yet; each box
 * builder below documents exactly what its color is a proxy for.
 */

import { fAgo } from "../format.js";

const TICKER = "XAG-USD";

const LEVEL_COLOR = {
  green: "var(--grn)",
  amber: "var(--yel)",
  red: "var(--red)",
  grey: "var(--gry)",
  blue: "var(--blu)",
};

/**
 * @param {{sys: object|null, cs: object|null, registryApplicable: string[]|null, gridFisher: object|null}} props
 * @returns {HTMLElement}
 */
export function silverPipeline({
  sys,
  cs,
  registryApplicable,
  gridFisher,
} = {}) {
  const card = document.createElement("article");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  card.append(_lane("Paper — Layer 1", _lane1Boxes(sys, registryApplicable)));
  const div = document.createElement("div");
  div.style.height = "1px";
  div.style.background = "var(--bdr)";
  div.style.margin = "var(--sp-3) 0";
  card.append(div);
  card.append(
    _lane("Real money — metals_loop", _lane2Boxes(sys, cs, gridFisher)),
  );

  return card;
}

function _lane(label, boxes) {
  const wrap = document.createElement("div");
  const lbl = document.createElement("div");
  lbl.style.fontSize = "var(--ty-xs)";
  lbl.style.fontWeight = "700";
  lbl.style.color = "var(--txm)";
  lbl.style.marginBottom = "var(--sp-2)";
  lbl.textContent = label.toUpperCase();
  wrap.append(lbl);

  boxes.forEach((b, i) => {
    if (i > 0) wrap.append(_arrow());
    wrap.append(_box(b));
  });
  return wrap;
}

function _arrow() {
  const a = document.createElement("div");
  a.style.textAlign = "center";
  a.style.color = "var(--txm)";
  a.style.fontSize = "var(--ty-sm)";
  a.style.lineHeight = "1";
  a.style.margin = "2px 0";
  a.textContent = "↓";
  return a;
}

function _box({ label, subtitle, level, title }) {
  const el = document.createElement("div");
  const color = LEVEL_COLOR[level] || "var(--bdr)";
  el.style.border = `1px solid ${color}`;
  el.style.borderLeft = `4px solid ${color}`;
  el.style.borderRadius = "var(--rad-md)";
  el.style.padding = "var(--sp-2) var(--sp-3)";
  if (title) el.title = title;

  const l = document.createElement("div");
  l.style.fontSize = "var(--ty-sm)";
  l.style.fontWeight = "600";
  l.style.color = "var(--tx)";
  l.textContent = label;
  el.append(l);

  if (subtitle) {
    const s = document.createElement("div");
    s.style.fontSize = "var(--ty-xs)";
    s.style.color = "var(--txm)";
    s.style.marginTop = "2px";
    s.textContent = subtitle;
    el.append(s);
  }
  return el;
}

function _xagAggRow(sys) {
  const rows = sys?.signal_aggregate?.tickers;
  return Array.isArray(rows) ? rows.find((t) => t.ticker === TICKER) : null;
}

function _lane1Boxes(sys, registryApplicable) {
  const layer1 = sys?.layer1;
  const enabled = !!layer1?.enabled;
  const active = !!layer1?.active;
  const frozen = !!sys?.sources?.["signal_log.jsonl"]?.frozen;
  // 2026-07-18: there's no per-feed heartbeat — this is a proxy from the
  // Layer 1 systemd unit state + the signal-log freshness it produces, not a
  // literal per-stage health check on data_collector's Binance FAPI calls.
  const level = !enabled
    ? "grey"
    : !active
      ? "red"
      : frozen
        ? "amber"
        : "green";

  const row = _xagAggRow(sys);
  const nApplicable = Array.isArray(registryApplicable)
    ? registryApplicable.length
    : null;

  const l2gate = sys?.layer2?.gate;
  const gateLabel = l2gate?.label || "UNKNOWN";
  const recLevel =
    gateLabel === "ACTIVE" ? "green" : gateLabel === "FROZEN" ? "blue" : "grey";

  return [
    {
      label: "Price feed (Binance FAPI)",
      subtitle: layer1?.last_cycle_ts
        ? `last cycle ${fAgo(layer1.last_cycle_ts)}`
        : "no cycle recorded",
      level,
      title:
        "Proxy: pf-dataloop unit state + signal_log.jsonl freshness (no per-feed instrumentation exists).",
    },
    {
      label:
        nApplicable != null
          ? `Signal engine: ${nApplicable} applicable`
          : "Signal engine",
      subtitle: `${TICKER} registry snapshot`,
      level,
    },
    {
      label: "Consensus vote + horizon",
      subtitle: row
        ? `${row.consensus} · ${row.horizon || "1d (default)"}`
        : "no snapshot yet",
      level,
    },
    {
      label: "Autonomous recommendation",
      subtitle:
        gateLabel === "FROZEN"
          ? "Layer 2 off — Layer 3 autonomous fallback"
          : `Layer 2 ${gateLabel.toLowerCase()}`,
      level: recLevel,
    },
  ];
}

function _lane2Boxes(sys, cs, gridFisher) {
  const loop = cs?.loops?.["pf-metalsloop"];
  const loopEnabled = !!loop?.enabled;
  const loopActive = !!loop?.active;
  const hbFrozen = !!sys?.sources?.["metals_loop.heartbeat"]?.frozen;
  const hbAge = sys?.sources?.["metals_loop.heartbeat"]?.age_sec;
  const loopLevel = !loopEnabled
    ? "grey"
    : !loopActive
      ? "red"
      : hbFrozen
        ? "amber"
        : "green";

  const xagArmed = Object.values(gridFisher?.state?.by_instrument || {}).some(
    (i) => i.ticker === TICKER,
  );
  const llmEnabled = !!cs?.llm_enabled;

  const credsOk = !!sys?.avanza?.creds_configured;
  const unresolvedErrs = sys?.avanza?.unresolved_errors ?? 0;
  // Green requires BOTH creds configured AND no outstanding errors — creds
  // alone (the old check) said nothing about session/order-flow health.
  const avanzaLevel = !credsOk ? "red" : unresolvedErrs === 0 ? "green" : "amber";

  return [
    {
      label: "metals loop (60s + 10s fast-tick)",
      subtitle:
        hbAge != null
          ? `heartbeat ${fAgo(new Date(Date.now() - hbAge * 1000))}`
          : "no heartbeat yet",
      level: loopLevel,
    },
    {
      label: "LLM predictions / Grid Fisher",
      subtitle: `LLM: ${llmEnabled ? "active" : "paused"} · Grid Fisher: ${xagArmed ? "armed for XAG" : "not armed"}`,
      level: xagArmed ? "green" : "grey",
    },
    {
      label: "Avanza warrants",
      subtitle: !credsOk
        ? `not configured — ${unresolvedErrs} unresolved errors`
        : unresolvedErrs === 0
          ? "credentials configured"
          : `credentials configured — ${unresolvedErrs} unresolved errors`,
      level: avanzaLevel,
    },
  ];
}
