/*
 * views/elongir.js — Elongir equity-bot view.
 *
 * KPI grid + halted badge. Mirrors the shape of views/golddigger.js.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fs, ftFull } from "../format.js";
import { emptyState } from "../components/empty-state.js";

const POLL_KEY = "elongir";
const SLOT = "elongir";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());
    _unsubs.push(state.subscribe(SLOT, _renderBody));

    polling.register(POLL_KEY, 60_000, async () => {
      const d = await fj("/api/elongir", { ttl: 5_000 });
      if (d) state.set(SLOT, d);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("elongir", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--elongir";
  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "Elongir";
  v.append(t);

  for (const slotName of ["summary", "halted", "log"]) {
    const slot = document.createElement("div");
    slot.dataset.slot = slotName;
    slot.style.marginBottom = "var(--sp-3)";
    v.append(slot);
  }
  return v;
}

function _renderBody() {
  if (!_root) return;
  const data = state.get(SLOT);
  const sumSlot = _root.querySelector('[data-slot="summary"]');
  if (!data) {
    while (sumSlot.firstChild) sumSlot.removeChild(sumSlot.firstChild);
    sumSlot.append(emptyState("Loading Elongir…"));
    return;
  }
  _renderSummary(data);
  _renderHalted(data);
  _renderLog(data);
}

function _renderSummary(data) {
  const slot = _root.querySelector('[data-slot="summary"]');
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const s = data.state || {};

  const card = document.createElement("article");
  card.className = "card";
  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(110px, 1fr))";
  grid.style.gap = "var(--sp-2)";

  grid.append(
    _kpi("Signal state",   s.signal_state ?? "—"),
    _kpi("Cash SEK",       fs(s.cash_sek)),
    _kpi("Position",       s.position ?? "—"),
    _kpi("Daily P&L",      fs(s.daily_pnl), _pnlColor(s.daily_pnl)),
    _kpi("Daily trades",   s.daily_trades ?? 0),
    _kpi("Total P&L",      fs(s.total_pnl), _pnlColor(s.total_pnl)),
    _kpi("W/L",            `${s.wins ?? 0} / ${s.losses ?? 0}`),
    _kpi("Max DD%",        _fmtNum(s.max_drawdown_pct, 2)),
    _kpi("Last trade",     s.last_trade_date ?? "—"),
  );
  card.append(grid);
  slot.append(card);
}

function _renderHalted(data) {
  const slot = _root.querySelector('[data-slot="halted"]');
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const s = data.state || {};
  if (s.halted == null) return;

  const card = document.createElement("article");
  card.className = "card";
  const row = document.createElement("div");
  row.style.display = "flex";
  row.style.alignItems = "center";
  row.style.gap = "var(--sp-2)";

  const badge = document.createElement("span");
  badge.style.fontSize = "var(--ty-xs)";
  badge.style.fontWeight = "600";
  badge.style.padding = "2px var(--sp-2)";
  badge.style.borderRadius = "4px";
  badge.style.color = "var(--bg)";
  if (s.halted) {
    badge.textContent = "HALTED";
    badge.style.background = "var(--red)";
  } else {
    badge.textContent = "RUNNING";
    badge.style.background = "var(--grn)";
  }
  row.append(badge);

  if (s.halted && s.halted_reason) {
    const reason = document.createElement("span");
    reason.style.fontSize = "var(--ty-sm)";
    reason.style.color = "var(--txd)";
    reason.textContent = s.halted_reason;
    row.append(reason);
  }
  card.append(row);
  slot.append(card);
}

function _renderLog(data) {
  const slot = _root.querySelector('[data-slot="log"]');
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const log = Array.isArray(data.log) ? data.log : [];
  if (!log.length) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Recent decisions";
  slot.append(title);

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const entry of log.slice(-10).reverse()) {
    const row = document.createElement("div");
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";
    row.style.fontSize = "var(--ty-sm)";
    row.style.display = "grid";
    row.style.gridTemplateColumns = "70px 1fr";
    row.style.gap = "var(--sp-2)";

    const ts = document.createElement("span");
    ts.style.color = "var(--txm)";
    ts.style.fontSize = "var(--ty-xs)";
    const tsRaw = entry.ts || entry.timestamp;
    ts.textContent = tsRaw ? ftFull(tsRaw).slice(11, 19) : "—";

    const summary = document.createElement("span");
    summary.style.overflow = "hidden";
    summary.style.textOverflow = "ellipsis";
    summary.style.whiteSpace = "nowrap";
    summary.textContent = _summarise(entry);
    summary.title = JSON.stringify(entry);

    row.append(ts, summary);
    wrap.append(row);
  }
  slot.append(wrap);
}

function _summarise(entry) {
  if (!entry || typeof entry !== "object") return String(entry ?? "");
  const parts = [];
  if (entry.action) parts.push(String(entry.action));
  if (entry.ticker) parts.push(String(entry.ticker));
  if (entry.reason) parts.push(String(entry.reason));
  if (entry.pnl != null) parts.push(`pnl=${entry.pnl}`);
  if (parts.length) return parts.join(" · ");
  return JSON.stringify(entry).slice(0, 120);
}

function _kpi(label, value, color) {
  const wrap = document.createElement("div");
  const l = document.createElement("div");
  l.className = "card__subtitle"; l.textContent = label;
  const v = document.createElement("div");
  v.className = "num num--md"; v.textContent = String(value ?? "—");
  if (color) v.style.color = color;
  wrap.append(l, v);
  return wrap;
}

function _fmtNum(v, d = 2) {
  if (v == null) return "—";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

function _pnlColor(p) {
  if (p == null) return undefined;
  const n = Number(p);
  if (!Number.isFinite(n)) return undefined;
  return n > 0 ? "var(--grn)" : n < 0 ? "var(--red)" : undefined;
}

