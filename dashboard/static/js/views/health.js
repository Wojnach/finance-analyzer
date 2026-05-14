/*
 * views/health.js — system health snapshot view.
 *
 * KPI grid (loop, agent, cycles, errors) + module failures + recent
 * errors digest. Uses /api/health.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fAgo, ftFull } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { pulseDot } from "../components/pulse-dot.js";

const POLL_KEY = "health";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.HEALTH, _renderBody));
    _unsubs.push(state.subscribe(state.Slots.LOOP_HEALTH, _renderBody));
    _unsubs.push(state.subscribe(state.Slots.SYSTEM_STATUS, _renderBody));

    polling.register(POLL_KEY, 60_000, async () => {
      const h = await fj("/api/health");
      if (h) state.set(state.Slots.HEALTH, h);
      const lh = await fj("/api/loop_health");
      if (lh) state.set(state.Slots.LOOP_HEALTH, lh);
      const ss = await fj("/api/system_status");
      if (ss) state.set(state.Slots.SYSTEM_STATUS, ss);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("health", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--health";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "Health";
  view.append(title);

  const body = document.createElement("div");
  body.dataset.slot = "body";
  view.append(body);
  return view;
}

function _renderBody() {
  if (!_root) return;
  const slot = _root.querySelector('[data-slot="body"]');
  if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const h = state.get(state.Slots.HEALTH);
  if (!h) {
    slot.append(emptyState("Loading health…"));
  } else {
    slot.append(_kpiGrid(h));
    if (Array.isArray(h.module_failures) && h.module_failures.length) {
      slot.append(_section("Module failures", _moduleFailures(h.module_failures)));
    }
    if (Array.isArray(h.recent_errors) && h.recent_errors.length) {
      slot.append(_section("Recent errors", _errorList(h.recent_errors)));
    }
  }

  // Unresolved critical_errors + contract_violations (mirrors home page widget tally).
  const ss = state.get(state.Slots.SYSTEM_STATUS);
  if (ss) {
    const errs = ss.errors || {};
    const cvs = ss.contract_violations || {};
    const errRecent = Array.isArray(errs.recent) ? errs.recent : [];
    const cvRecent = Array.isArray(cvs.recent) ? cvs.recent : [];
    if (errRecent.length || cvRecent.length) {
      slot.append(_section(
        `Unresolved (${errs.unresolved ?? 0} err · ${cvs.unresolved ?? 0} cv, 24h)`,
        _unresolvedList(errRecent, cvRecent),
      ));
    }
  }

  // Loop health rollup. Endpoint shape: {checked_at, loops:{name:{state, age_seconds, ...}}, any_unhealthy, unhealthy[]}.
  const rollup = state.get(state.Slots.LOOP_HEALTH);
  const loops = rollup?.loops;
  if (loops && typeof loops === "object" && Object.keys(loops).length) {
    slot.append(_section("Loop heartbeats", _loopList(loops, rollup)));
  }
}

function _kpiGrid(h) {
  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(140px, 1fr))";
  grid.style.gap = "var(--sp-2)";
  grid.style.marginBottom = "var(--sp-3)";

  const kpis = [
    { label: "Loop status", value: h.status || "—", color: _statusColor(h.status) },
    { label: "Heartbeat",   value: h.heartbeat_age_seconds != null ? `${h.heartbeat_age_seconds}s` : "—" },
    { label: "Cycles",      value: h.cycle_count ?? "—" },
    { label: "Signals OK",  value: `${h.signals_ok ?? "?"} / ${(h.signals_ok ?? 0) + (h.signals_failed ?? 0)}` },
    { label: "Errors",      value: h.error_count ?? 0, color: (h.error_count > 0) ? "var(--red)" : "var(--grn)" },
    { label: "Agent",       value: h.agent_silent ? `silent ${h.agent_silence_seconds ?? "?"}s` : "ok",
      color: h.agent_silent ? "var(--yel)" : "var(--grn)" },
  ];
  for (const k of kpis) {
    const card = document.createElement("article");
    card.className = "card";
    const lbl = document.createElement("div");
    lbl.className = "card__subtitle";
    lbl.textContent = k.label;
    const v = document.createElement("div");
    v.className = "num num--md";
    v.textContent = String(k.value);
    if (k.color) v.style.color = k.color;
    card.append(lbl, v);
    grid.append(card);
  }
  return grid;
}

function _section(title, body) {
  const wrap = document.createElement("section");
  wrap.style.marginTop = "var(--sp-3)";
  const t = document.createElement("div");
  t.className = "section-title";
  t.textContent = title;
  wrap.append(t, body);
  return wrap;
}

function _moduleFailures(items) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexWrap = "wrap";
  wrap.style.gap = "var(--sp-1)";
  for (const item of items) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.style.background = "rgba(255,68,68,0.15)";
    chip.style.color = "var(--red)";
    chip.style.borderColor = "rgba(255,68,68,0.3)";
    chip.textContent = typeof item === "string" ? item : (item.name || item.module || JSON.stringify(item));
    wrap.append(chip);
  }
  return wrap;
}

function _errorList(items) {
  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";
  for (const item of items.slice(0, 30)) {
    const row = document.createElement("div");
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";
    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.fontSize = "var(--ty-xs)";
    const left = document.createElement("span");
    left.style.color = "var(--txm)";
    left.textContent = ftFull(item?.ts || item?.timestamp || "");
    const right = document.createElement("span");
    right.style.color = "var(--red)";
    right.textContent = item?.category || item?.level || "error";
    top.append(left, right);
    const msg = document.createElement("div");
    msg.style.fontSize = "var(--ty-sm)";
    msg.style.marginTop = "2px";
    msg.style.color = "var(--tx)";
    msg.textContent = item?.message || item?.msg || JSON.stringify(item);
    row.append(top, msg);
    wrap.append(row);
  }
  return wrap;
}

function _unresolvedList(errs, cvs) {
  const items = [];
  for (const e of errs) {
    items.push({
      kind: "err",
      ts: e.ts || e.timestamp || "",
      label: e.category || e.caller || "error",
      msg: e.message || e.msg || "",
      raw: e,
    });
  }
  for (const v of cvs) {
    items.push({
      kind: "cv",
      ts: v.ts || v.timestamp || "",
      label: v.invariant || "violation",
      msg: v.message || v.msg || "",
      raw: v,
    });
  }
  items.sort((a, b) => (b.ts || "").localeCompare(a.ts || ""));

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";
  for (const it of items.slice(0, 30)) {
    const row = document.createElement("div");
    row.style.borderBottom = "1px solid var(--bdr)";

    const head = document.createElement("button");
    head.type = "button";
    head.style.display = "block";
    head.style.width = "100%";
    head.style.textAlign = "left";
    head.style.background = "transparent";
    head.style.border = "0";
    head.style.color = "inherit";
    head.style.padding = "var(--sp-2) var(--sp-3)";
    head.style.minHeight = "var(--tap-min)";
    head.style.cursor = "pointer";

    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.alignItems = "center";
    top.style.gap = "var(--sp-2)";
    top.style.fontSize = "var(--ty-xs)";
    const left = document.createElement("span");
    left.style.color = it.kind === "cv" ? "var(--yel)" : "var(--red)";
    left.style.fontWeight = "600";
    left.textContent = it.label;
    const right = document.createElement("span");
    right.style.color = "var(--txm)";
    right.style.display = "flex";
    right.style.alignItems = "center";
    right.style.gap = "var(--sp-1)";
    const tsSpan = document.createElement("span");
    tsSpan.textContent = ftFull(it.ts);
    const chev = document.createElement("span");
    chev.textContent = "›";
    chev.style.transition = "transform 120ms";
    right.append(tsSpan, chev);
    top.append(left, right);

    const msg = document.createElement("div");
    msg.style.fontSize = "var(--ty-sm)";
    msg.style.marginTop = "2px";
    msg.style.color = "var(--tx)";
    msg.textContent = it.msg;

    head.append(top, msg);

    const detail = document.createElement("pre");
    detail.style.display = "none";
    detail.style.margin = "0";
    detail.style.padding = "var(--sp-2) var(--sp-3)";
    detail.style.borderTop = "1px solid var(--bdr)";
    detail.style.background = "var(--bg)";
    detail.style.fontSize = "var(--ty-xs)";
    detail.style.color = "var(--txm)";
    detail.style.whiteSpace = "pre-wrap";
    detail.style.wordBreak = "break-word";
    detail.style.overflow = "auto";
    detail.style.maxHeight = "60vh";
    try {
      detail.textContent = JSON.stringify(it.raw, null, 2);
    } catch (_) {
      detail.textContent = String(it.msg || "");
    }

    head.addEventListener("click", () => {
      const open = detail.style.display !== "none";
      detail.style.display = open ? "none" : "block";
      chev.style.transform = open ? "rotate(0deg)" : "rotate(90deg)";
    });

    row.append(head, detail);
    wrap.append(row);
  }
  return wrap;
}

function _loopList(loops, rollup) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  wrap.style.gap = "var(--sp-1)";

  if (rollup?.any_unhealthy) {
    const banner = document.createElement("div");
    banner.className = "banner banner--error";
    banner.textContent = "Unhealthy loops: " + (rollup.unhealthy || []).join(", ");
    wrap.append(banner);
  }

  for (const [name, info] of Object.entries(loops)) {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.justifyContent = "space-between";
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.background = "var(--card)";
    row.style.border = "1px solid var(--bdr)";
    row.style.borderRadius = "var(--rad-md)";
    const left = document.createElement("div");
    left.style.display = "flex";
    left.style.alignItems = "center";
    left.style.gap = "var(--sp-2)";
    const stateName = _loopStateClass(info?.state);
    left.append(
      pulseDot({ state: stateName }),
      Object.assign(document.createElement("span"), { textContent: name }),
    );
    const right = document.createElement("div");
    right.style.fontSize = "var(--ty-xs)";
    right.style.color = "var(--txm)";
    const ageS = info?.age_seconds;
    right.textContent = `${info?.state ?? "?"} · age ${ageS ?? "?"}s`
      + (Number.isFinite(Number(ageS)) ? ` (${fAgo(Date.now() - 1000 * ageS)})` : "");
    row.append(left, right);
    wrap.append(row);
  }
  return wrap;
}

function _loopStateClass(state) {
  switch ((state || "").toLowerCase()) {
    case "fresh":   return "ok";
    case "stale":   return "warn";
    case "missing": return "fail";
    case "unparseable": return "fail";
    default:        return "idle";
  }
}

function _statusColor(s) {
  if (!s) return undefined;
  const v = String(s).toLowerCase();
  if (v.includes("ok") || v.includes("running") || v.includes("healthy")) return "var(--grn)";
  if (v.includes("warn") || v.includes("stale")) return "var(--yel)";
  if (v.includes("error") || v.includes("dead") || v.includes("fail")) return "var(--red)";
  return undefined;
}
