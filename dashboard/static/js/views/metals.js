/*
 * views/metals.js — Metals subsystem view.
 *
 * Mobile-friendly re-layout from the legacy Metals tab. Cards instead of
 * dense flex rows. Single-axis intraday chart (legacy dual-axis is a known
 * mobile anti-pattern per Track-5).
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fpct, fp, fs, ftFull, num } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { positionCard } from "../components/position-card.js";
import { miniChart } from "../components/mini-chart.js";
import { getChartColors } from "../theme.js";

const POLL_KEY = "metals";

let _root = null;
let _disposeChart = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.METALS, _renderBody));

    polling.register(POLL_KEY, 60_000, async () => {
      const m = await fj("/api/metals", { ttl: 5_000 });
      if (m) state.set(state.Slots.METALS, m);
      return m != null;
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
router.register("metals", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--metals";

  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "Metals";
  v.append(t);

  for (const slotName of ["summary", "positions", "risk", "chart", "decisions"]) {
    const slot = document.createElement("div");
    slot.dataset.slot = slotName;
    slot.style.marginBottom = "var(--sp-3)";
    v.append(slot);
  }
  return v;
}

function _renderBody() {
  if (!_root) return;
  const data = state.get(state.Slots.METALS);
  if (!data) {
    const empty = _root.querySelector('[data-slot="summary"]');
    while (empty.firstChild) empty.removeChild(empty.firstChild);
    empty.append(emptyState("Loading metals…"));
    return;
  }

  _renderSummary(data);
  _renderPositions(data);
  _renderRisk(data);
  _renderChart(data);
  _renderDecisions(data);
}

function _slot(name) {
  return _root?.querySelector(`[data-slot="${name}"]`) || null;
}

function _renderSummary(data) {
  const slot = _slot("summary"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const ctx = data.context || {};
  const totals = ctx.totals || {};
  const under = ctx.underlying || {};

  const card = document.createElement("article");
  card.className = "card";

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(120px, 1fr))";
  grid.style.gap = "var(--sp-3)";

  grid.append(
    _kpi("P&L %",     fpct(num(totals.pnl_pct), 2), _pnlColor(totals.pnl_pct)),
    _kpi("Value",     fs(totals.value_sek)),
    _kpi("Invested",  fs(totals.invested_sek)),
    _kpi("Gold USD",  fp(under?.XAU?.price)),
    _kpi("Silver USD", fp(under?.XAG?.price)),
  );
  card.append(grid);
  slot.append(card);
}

function _kpi(label, value, color) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  const l = document.createElement("div");
  l.className = "card__subtitle";
  l.textContent = label;
  const v = document.createElement("div");
  v.className = "num num--md";
  v.textContent = value;
  if (color) v.style.color = color;
  wrap.append(l, v);
  return wrap;
}

function _pnlColor(p) {
  const n = num(p); if (n == null) return undefined;
  return n > 0 ? "var(--grn)" : n < 0 ? "var(--red)" : undefined;
}

function _renderPositions(data) {
  const slot = _slot("positions"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const positions = (data.context?.positions) || [];
  if (!Array.isArray(positions) || !positions.length) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Positions";
  slot.append(title);

  const strip = document.createElement("div");
  strip.className = "scroll-strip";
  for (const p of positions) {
    strip.append(positionCard({
      ticker: p.name || p.ticker || p.warrant || "warrant",
      side: (p.leverage_direction || "LONG").toUpperCase(),
      pnlPct: num(p.pnl_pct ?? p.pnl_percent),
      pricePerUnit: num(p.bid ?? p.last_price ?? p.value_per_unit),
      stopPrice: num(p.stop_price ?? p.barrier),
      stopDistancePct: num(p.stop_distance_pct ?? p.barrier_distance_pct),
    }));
  }
  slot.append(strip);
}

function _renderRisk(data) {
  const slot = _slot("risk"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const risk = data.context?.risk;
  if (!risk) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Risk";
  slot.append(title);

  const card = document.createElement("article");
  card.className = "card";
  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(120px, 1fr))";
  grid.style.gap = "var(--sp-3)";

  grid.append(
    _kpi("Drawdown",     fpct(num(risk.drawdown_pct ?? risk.dd_pct), 2)),
    _kpi("Stop prob 1d", fpct(num((risk.stop_prob_1d ?? 0) * 100), 1)),
    _kpi("Trade guard",  String(risk.trade_guard_status ?? "—")),
  );
  card.append(grid);
  slot.append(card);
}

function _renderChart(data) {
  const slot = _slot("chart"); if (!slot) return;
  if (_disposeChart) { try { _disposeChart(); } catch (_) {} _disposeChart = null; }
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const history = data.context?.price_history_recent || data.history;
  if (!Array.isArray(history) || history.length < 2) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Intraday — Silver USD";
  slot.append(title);

  const c = getChartColors();
  const labels = history.map((p) => p.ts || p.timestamp);
  // Single axis — silver underlying only. Warrant SEK is on a different
  // scale and dual-axis on phone is a known anti-pattern.
  const built = miniChart({
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "XAG-USD",
        data: history.map((p) => num(p.xag_usd ?? p.silver ?? p.price)),
        borderColor: c.cyan,
        backgroundColor: "rgba(6,182,212,0.10)",
        borderWidth: 1.6,
        fill: true,
        pointRadius: 0,
        tension: 0.2,
      }],
    },
    options: {
      plugins: {
        legend: { display: true, position: "bottom",
                  labels: { color: c.dim, font: { size: 11 }, boxWidth: 8 } },
      },
      scales: { x: { ticks: { maxTicksLimit: 6 } } },
    },
    height: 240,
  });
  slot.append(built.element);
  _disposeChart = built.dispose;
}

function _renderDecisions(data) {
  const slot = _slot("decisions"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const decisions = data.decisions;
  if (!Array.isArray(decisions) || !decisions.length) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Recent metals decisions";
  slot.append(title);

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const d of decisions.slice(0, 30)) {
    const row = document.createElement("div");
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";
    row.style.fontSize = "var(--ty-sm)";

    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.fontSize = "var(--ty-xs)";
    const ts = document.createElement("span");
    ts.style.color = "var(--txm)";
    ts.textContent = ftFull(d.ts || d.timestamp);
    const tier = document.createElement("span");
    tier.style.color = "var(--txd)";
    tier.textContent = "tier " + (d.tier ?? "?");
    top.append(ts, tier);

    const body = document.createElement("div");
    body.style.marginTop = "2px";
    // prediction became a dict in newer rows ({action, direction,
    // confidence, horizon}) — rendering it raw printed "[object Object]"
    // (2026-07-19). Compose a line from whichever shape the row has.
    let text;
    if (typeof d.prediction === "string" && d.prediction) {
      text = d.prediction;
    } else if (d.prediction && typeof d.prediction === "object") {
      const p = d.prediction;
      const parts = [p.action || d.action || "?"];
      if (p.direction) parts.push(p.direction);
      if (p.horizon) parts.push(p.horizon);
      if (p.confidence != null) parts.push(`conf ${p.confidence}`);
      text = parts.join(" · ");
      if (d.trigger) text += ` — ${d.trigger}`;
    } else {
      text = d.action || JSON.stringify(d);
    }
    body.textContent = text;

    row.append(top, body);
    wrap.append(row);
  }
  slot.append(wrap);
}
