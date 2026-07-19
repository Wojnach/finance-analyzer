/*
 * views/cost.js — Claude CLI cost + token breakdown.
 *
 * Wraps /api/claude_cost (which wraps scripts/claude_cost_report.summarise).
 * Tables: totals, per-day, per-caller, per-model, layer2 by tier.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { emptyState } from "../components/empty-state.js";

const POLL_KEY = "cost";
const DEFAULT_DAYS = 7;

let _root = null;
let _unsubs = [];
let _days = DEFAULT_DAYS;

export const view = {
  mount(rootEl) {
    _root = rootEl;
    _days = DEFAULT_DAYS;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.CLAUDE_COST, _renderBody));

    polling.register(POLL_KEY, 120_000, async () => {
      const c = await fj(`/api/claude_cost?days=${_days}`);
      if (c) state.set(state.Slots.CLAUDE_COST, c);
      return c != null;
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("cost", view);

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--cost";

  const head = document.createElement("div");
  head.style.display = "flex";
  head.style.justifyContent = "space-between";
  head.style.alignItems = "baseline";
  head.style.gap = "var(--sp-2)";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "Claude CLI cost";
  head.append(title);

  const picker = document.createElement("div");
  picker.style.display = "flex";
  picker.style.gap = "var(--sp-1)";
  for (const d of [1, 7, 30]) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "card card--tap";
    b.style.padding = "var(--sp-1) var(--sp-2)";
    b.style.fontSize = "var(--ty-sm)";
    b.style.fontWeight = d === _days ? "700" : "400";
    b.textContent = `${d}d`;
    b.addEventListener("click", async () => {
      _days = d;
      const c = await fj(`/api/claude_cost?days=${_days}`);
      if (c) state.set(state.Slots.CLAUDE_COST, c);
      // re-render shell to update active state on picker
      while (_root.firstChild) _root.removeChild(_root.firstChild);
      _root.append(_renderShell());
      _renderBody();
    });
    picker.append(b);
  }
  head.append(picker);
  v.append(head);

  const body = document.createElement("div");
  body.dataset.slot = "body";
  v.append(body);
  return v;
}

function _renderBody() {
  if (!_root) return;
  const slot = _root.querySelector('[data-slot="body"]');
  if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const c = state.get(state.Slots.CLAUDE_COST);
  if (!c) {
    slot.append(emptyState("Loading cost…"));
    return;
  }
  if (c.error) {
    slot.append(emptyState(`Error: ${c.error}`));
    return;
  }

  slot.append(_totals(c.totals || {}));
  if (c.by_day) slot.append(_table("By day",
    ["date", "calls", "cost($)", "in_tok", "out_tok", "wall(s)"],
    Object.entries(c.by_day).map(([day, v]) => [
      day, v.calls, _money(v.cost_usd), _tok(v.input_tokens), _tok(v.output_tokens), _sec(v.duration_s),
    ])));
  if (c.by_caller) slot.append(_table("By caller",
    ["caller", "calls", "parsed", "cost($)", "in_tok", "out_tok", "wall(s)"],
    Object.entries(c.by_caller)
      .sort((a, b) => (b[1].cost_usd || 0) - (a[1].cost_usd || 0))
      .map(([k, v]) => [
        k, v.calls, v.parsed, _money(v.cost_usd), _tok(v.input_tokens), _tok(v.output_tokens), _sec(v.duration_s),
      ])));
  if (c.by_model) slot.append(_table("By model",
    ["model", "calls", "cost($)", "in_tok", "out_tok", "wall(s)"],
    Object.entries(c.by_model)
      .sort((a, b) => (b[1].cost_usd || 0) - (a[1].cost_usd || 0))
      .map(([k, v]) => [
        k, v.calls, _money(v.cost_usd), _tok(v.input_tokens), _tok(v.output_tokens), _sec(v.duration_s),
      ])));
  if (c.by_status) slot.append(_kv("Gate status", c.by_status));
  if (c.layer2_by_tier) slot.append(_table("Layer 2 by tier",
    ["tier", "calls", "ran", "wall(s)"],
    Object.entries(c.layer2_by_tier).map(([k, v]) => [
      `T${k}`, v.calls, v.ran, _sec(v.duration_s),
    ])));
  if (c.layer2_by_status) slot.append(_kv("Layer 2 status", c.layer2_by_status));
}

function _totals(t) {
  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(140px, 1fr))";
  grid.style.gap = "var(--sp-2)";
  grid.style.marginBottom = "var(--sp-3)";
  grid.style.marginTop = "var(--sp-2)";

  const kpis = [
    { label: "Cost", value: `$${(t.cost_usd || 0).toFixed(4)}`, color: "var(--cyn)" },
    { label: "Calls", value: `${t.parsed_rows ?? 0}/${t.gate_rows ?? 0}` },
    { label: "Input tok", value: _tok(t.input_tokens || 0) },
    { label: "Output tok", value: _tok(t.output_tokens || 0) },
    { label: "Cache reads", value: _tok(t.cache_read_tokens || 0), color: "var(--grn)" },
    { label: "Wall (min)", value: `${t.duration_minutes ?? 0}` },
  ];
  for (const k of kpis) {
    const c = document.createElement("article");
    c.className = "card";
    const l = document.createElement("div");
    l.className = "card__subtitle";
    l.textContent = k.label;
    const vEl = document.createElement("div");
    vEl.className = "num num--md";
    vEl.textContent = String(k.value);
    if (k.color) vEl.style.color = k.color;
    c.append(l, vEl);
    grid.append(c);
  }
  return grid;
}

function _table(title, headers, rows) {
  const wrap = document.createElement("section");
  wrap.style.marginTop = "var(--sp-3)";
  const t = document.createElement("div");
  t.className = "section-title";
  t.textContent = title;
  wrap.append(t);

  const scroll = document.createElement("div");
  scroll.style.overflowX = "auto";
  scroll.style.background = "var(--card)";
  scroll.style.border = "1px solid var(--bdr)";
  scroll.style.borderRadius = "var(--rad-md)";

  const tbl = document.createElement("table");
  tbl.style.width = "100%";
  tbl.style.borderCollapse = "collapse";
  tbl.style.fontSize = "var(--ty-sm)";

  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  for (const h of headers) {
    const th = document.createElement("th");
    th.textContent = h;
    th.style.textAlign = "left";
    th.style.padding = "var(--sp-1) var(--sp-2)";
    th.style.color = "var(--txm)";
    th.style.fontWeight = "600";
    th.style.borderBottom = "1px solid var(--bdr)";
    trh.append(th);
  }
  thead.append(trh);
  tbl.append(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = String(cell);
      td.style.padding = "var(--sp-1) var(--sp-2)";
      td.style.borderBottom = "1px solid var(--bdr)";
      tr.append(td);
    }
    tbody.append(tr);
  }
  tbl.append(tbody);
  scroll.append(tbl);
  wrap.append(scroll);
  return wrap;
}

function _kv(title, obj) {
  const wrap = document.createElement("section");
  wrap.style.marginTop = "var(--sp-3)";
  const t = document.createElement("div");
  t.className = "section-title";
  t.textContent = title;
  wrap.append(t);

  const chips = document.createElement("div");
  chips.style.display = "flex";
  chips.style.flexWrap = "wrap";
  chips.style.gap = "var(--sp-1)";
  for (const [k, v] of Object.entries(obj || {})) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = `${k}: ${v}`;
    chips.append(c);
  }
  wrap.append(chips);
  return wrap;
}

function _money(v) {
  return (Number(v) || 0).toFixed(4);
}
function _tok(n) {
  const v = Number(n) || 0;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return String(v);
}
function _sec(n) {
  return `${Math.round(Number(n) || 0)}`;
}
