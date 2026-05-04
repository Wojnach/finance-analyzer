/*
 * views/fish.js — Fish-engine (metals straddle bot) view.
 *
 * KPI grid + ORB range row + recent log tail.
 * Mirrors the shape of views/golddigger.js.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fs, fAgo, ftFull } from "../format.js";
import { emptyState } from "../components/empty-state.js";

const POLL_KEY = "fish";
const SLOT = "fish";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());
    _unsubs.push(state.subscribe(SLOT, _renderBody));

    polling.register(POLL_KEY, 60_000, async () => {
      const d = await fj("/api/fish_engine", { ttl: 5_000 });
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
router.register("fish", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--fish";
  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "Fish engine";
  v.append(t);

  for (const slotName of ["summary", "orb", "log"]) {
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
    sumSlot.append(emptyState("Loading Fish engine…"));
    return;
  }
  _renderSummary(data);
  _renderOrb(data);
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

  // last_trade_ts is a Unix epoch float (seconds); fAgo expects ms.
  const lastTradeMs = Number.isFinite(Number(s.last_trade_ts))
    ? Number(s.last_trade_ts) * 1000
    : null;

  grid.append(
    _kpi("Mode",          s.mode ?? "—"),
    _kpi("Position",      s.position ?? "—"),
    _kpi("Session P&L",   fs(s.session_pnl), _pnlColor(s.session_pnl)),
    _kpi("Trades",        s.trade_count ?? 0),
    _kpi("W/L",           `${s.win_count ?? 0} / ${s.loss_count ?? 0}`),
    _kpi("Consec losses", s.consecutive_losses ?? 0),
    _kpi("Cooldown",      `${s.cooldown_seconds ?? 0}s`),
    _kpi("Last trade",    lastTradeMs != null ? fAgo(lastTradeMs) : "—"),
  );
  card.append(grid);
  slot.append(card);
}

function _renderOrb(data) {
  const slot = _root.querySelector('[data-slot="orb"]');
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const s = data.state || {};
  if (s.orb_range_low == null && s.orb_range_high == null && s.orb_range_formed == null) return;

  const card = document.createElement("article");
  card.className = "card";
  const t = document.createElement("div");
  t.className = "card__title";
  t.textContent = "ORB range";
  card.append(t);

  const row = document.createElement("div");
  row.style.fontSize = "var(--ty-sm)";
  row.style.color = "var(--txd)";
  row.style.marginTop = "var(--sp-1)";
  const lo = _fmtNum(s.orb_range_low, 2);
  const hi = _fmtNum(s.orb_range_high, 2);
  const formed = s.orb_range_formed ? "yes" : "no";
  row.textContent = `${lo} → ${hi}  (formed: ${formed})`;
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
  title.textContent = "Recent log";
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
  // Prefer a few well-known fields if present, else fall back to a stringified slice.
  const parts = [];
  if (entry.event)  parts.push(String(entry.event));
  if (entry.action) parts.push(String(entry.action));
  if (entry.side)   parts.push(String(entry.side));
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
