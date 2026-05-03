/*
 * views/golddigger.js — GoldDigger gold-cert bot view.
 *
 * Composite signal + position + score-history line + recent trades.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fpct, fp, ftFull, num } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { miniChart } from "../components/mini-chart.js";
import { getChartColors } from "../theme.js";

const POLL_KEY = "golddigger";

let _root = null;
let _disposeChart = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());
    _unsubs.push(state.subscribe(state.Slots.GOLDDIGGER, _renderBody));

    polling.register(POLL_KEY, 60_000, async () => {
      const d = await fj("/api/golddigger", { ttl: 5_000 });
      if (d) state.set(state.Slots.GOLDDIGGER, d);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    if (_disposeChart) { try { _disposeChart(); } catch (_) {} _disposeChart = null; }
    _root = null;
  },
};
router.register("golddigger", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--gd";
  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "GoldDigger";
  v.append(t);

  for (const slotName of ["summary", "signal", "position", "chart", "trades"]) {
    const slot = document.createElement("div");
    slot.dataset.slot = slotName;
    slot.style.marginBottom = "var(--sp-3)";
    v.append(slot);
  }
  return v;
}

function _renderBody() {
  if (!_root) return;
  const data = state.get(state.Slots.GOLDDIGGER);
  const sumSlot = _root.querySelector('[data-slot="summary"]');
  if (!data) {
    while (sumSlot.firstChild) sumSlot.removeChild(sumSlot.firstChild);
    sumSlot.append(emptyState("Loading GoldDigger…"));
    return;
  }
  _renderSummary(data);
  _renderSignal(data);
  _renderPosition(data);
  _renderChart(data);
  _renderTrades(data);
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
    _kpi("S(t)",       _fmtNum(s.composite_score ?? s.s_t, 3)),
    _kpi("Mode",       String(s.mode ?? "—")),
    _kpi("Session",    String(s.session_open ? "OPEN" : "CLOSED")),
    _kpi("XAU/USD",    fp(num(s.xau_usd ?? s.gold_usd))),
    _kpi("USD/SEK",    fp(num(s.usd_sek))),
    _kpi("Confirms",   String(s.confirms ?? 0)),
  );
  card.append(grid);
  slot.append(card);
}

function _renderSignal(data) {
  const slot = _root.querySelector('[data-slot="signal"]');
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const s = data.state?.signal || data.state?.composite_breakdown;
  if (!s || typeof s !== "object") return;
  const card = document.createElement("article");
  card.className = "card";
  const t = document.createElement("div");
  t.className = "card__title";
  t.textContent = "Signal breakdown";
  card.append(t);

  for (const [k, v] of Object.entries(s)) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.justifyContent = "space-between";
    row.style.padding = "var(--sp-1) 0";
    row.style.fontSize = "var(--ty-sm)";
    const a = document.createElement("span");
    a.textContent = k;
    a.style.color = "var(--txd)";
    const b = document.createElement("span");
    b.className = "num num--sm";
    b.textContent = v.toFixed(3);
    b.style.color = v > 0 ? "var(--grn)" : v < 0 ? "var(--red)" : "var(--txm)";
    row.append(a, b);
    card.append(row);
  }
  slot.append(card);
}

function _renderPosition(data) {
  const slot = _root.querySelector('[data-slot="position"]');
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const p = data.state?.position;
  if (!p || typeof p !== "object") return;
  const card = document.createElement("article");
  card.className = "card";
  const t = document.createElement("div");
  t.className = "card__title";
  t.textContent = "Position";
  card.append(t);

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(110px, 1fr))";
  grid.style.gap = "var(--sp-2)";
  grid.style.marginTop = "var(--sp-2)";
  grid.append(
    _kpi("Qty",     _fmtNum(p.quantity ?? p.qty, 2)),
    _kpi("Entry",   fp(num(p.entry_price))),
    _kpi("Current", fp(num(p.current_price))),
    _kpi("P&L %",   fpct(num(p.pnl_pct), 2), _pnlColor(p.pnl_pct)),
    _kpi("Stop",    fp(num(p.stop_price))),
  );
  card.append(grid);
  slot.append(card);
}

function _renderChart(data) {
  const slot = _root.querySelector('[data-slot="chart"]');
  if (_disposeChart) { try { _disposeChart(); } catch (_) {} _disposeChart = null; }
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const log = Array.isArray(data.log) ? data.log : [];
  if (log.length < 2) return;

  const c = getChartColors();
  const series = log.map((e) => num(e.composite_score ?? e.s_t)).filter((x) => x != null);
  if (series.length < 2) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Composite score history";
  slot.append(title);

  const built = miniChart({
    type: "line",
    data: {
      labels: log.map((e) => e.ts || e.timestamp),
      datasets: [{
        label: "S(t)",
        data: series,
        borderColor: c.yellow,
        backgroundColor: "rgba(234,179,8,0.10)",
        borderWidth: 1.6,
        fill: true,
        pointRadius: 0,
        tension: 0.2,
      }],
    },
    options: { scales: { x: { ticks: { maxTicksLimit: 6 } } } },
    height: 220,
  });
  slot.append(built.element);
  _disposeChart = built.dispose;
}

function _renderTrades(data) {
  const slot = _root.querySelector('[data-slot="trades"]');
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const trades = data.trades;
  if (!Array.isArray(trades) || !trades.length) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Trades";
  slot.append(title);

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";
  for (const t of trades.slice(-30).reverse()) {
    const row = document.createElement("div");
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";
    row.style.fontSize = "var(--ty-sm)";
    row.style.display = "grid";
    row.style.gridTemplateColumns = "70px 60px 1fr 80px";
    row.style.gap = "var(--sp-2)";
    const ts = document.createElement("span");
    ts.style.color = "var(--txm)"; ts.style.fontSize = "var(--ty-xs)";
    ts.textContent = ftFull(t.ts || t.timestamp).slice(11, 19);
    const action = document.createElement("span");
    const isBuy = (t.action || "").toUpperCase() === "BUY";
    action.className = "badge " + (isBuy ? "badge--BUY" : "badge--SELL");
    action.textContent = (t.action || "").toUpperCase();
    const reason = document.createElement("span");
    reason.style.overflow = "hidden";
    reason.style.textOverflow = "ellipsis";
    reason.style.whiteSpace = "nowrap";
    reason.title = t.reason || "";
    reason.textContent = t.reason || "";
    const sek = document.createElement("span");
    sek.style.textAlign = "right";
    sek.textContent = t.sek != null ? Number(t.sek).toFixed(0) + " kr" : "—";
    row.append(ts, action, reason, sek);
    wrap.append(row);
  }
  slot.append(wrap);
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
  const n = num(p); if (n == null) return undefined;
  return n > 0 ? "var(--grn)" : n < 0 ? "var(--red)" : undefined;
}
