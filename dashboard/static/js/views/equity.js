/*
 * views/equity.js — full equity curve view (P&L over time + trade marks).
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { ftFull } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { equityChart } from "../charts/equity-chart.js";

const POLL_KEY = "equity";

let _root = null;
let _disposeChart = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.EQUITY_CURVE, _renderBody));
    _unsubs.push(state.subscribe(state.Slots.TRADES,        _renderBody));

    polling.register(POLL_KEY, 5 * 60_000, async () => {
      const eq = await fj("/api/equity-curve", { ttl: 60_000 });
      if (eq) state.set(state.Slots.EQUITY_CURVE, Array.isArray(eq) ? eq : (eq?.curve || eq?.data || []));
      const tr = await fj("/api/trades", { ttl: 60_000 });
      if (tr) state.set(state.Slots.TRADES, Array.isArray(tr) ? tr : (tr?.trades || []));
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
router.register("equity", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--equity";

  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "Equity curve";
  v.append(t);

  const chart = document.createElement("div");
  chart.dataset.slot = "chart";
  chart.style.background = "var(--card)";
  chart.style.border = "1px solid var(--bdr)";
  chart.style.borderRadius = "var(--rad-md)";
  chart.style.padding = "var(--sp-2)";
  chart.style.marginBottom = "var(--sp-3)";
  v.append(chart);

  const tradesTitle = document.createElement("div");
  tradesTitle.className = "section-title";
  tradesTitle.textContent = "Recent trades";
  v.append(tradesTitle);

  const trades = document.createElement("div");
  trades.dataset.slot = "trades";
  v.append(trades);

  return v;
}

function _renderBody() {
  if (!_root) return;
  const chartSlot = _root.querySelector('[data-slot="chart"]');
  const tradesSlot = _root.querySelector('[data-slot="trades"]');
  if (!chartSlot || !tradesSlot) return;

  if (_disposeChart) { try { _disposeChart(); } catch (_) {} _disposeChart = null; }
  while (chartSlot.firstChild) chartSlot.removeChild(chartSlot.firstChild);
  while (tradesSlot.firstChild) tradesSlot.removeChild(tradesSlot.firstChild);

  const curve = state.get(state.Slots.EQUITY_CURVE);
  const trades = state.get(state.Slots.TRADES) || [];

  if (!Array.isArray(curve) || !curve.length) {
    chartSlot.append(emptyState("No equity-curve data yet."));
  } else {
    const built = equityChart({
      curve,
      trades: Array.isArray(trades) ? trades.slice(-50) : [],
      height: 260,
    });
    chartSlot.append(built.element);
    _disposeChart = built.dispose;
  }

  // Trades list
  if (!Array.isArray(trades) || !trades.length) {
    tradesSlot.append(emptyState("No trades yet."));
  } else {
    const wrap = document.createElement("div");
    wrap.style.background = "var(--card)";
    wrap.style.border = "1px solid var(--bdr)";
    wrap.style.borderRadius = "var(--rad-md)";
    trades.slice(-30).reverse().forEach((t) => {
      const row = document.createElement("div");
      row.style.display = "grid";
      row.style.gridTemplateColumns = "70px 70px 90px 1fr 80px";
      row.style.gap = "var(--sp-2)";
      row.style.padding = "var(--sp-2) var(--sp-3)";
      row.style.borderBottom = "1px solid var(--bdr)";
      row.style.fontSize = "var(--ty-sm)";

      const ts = document.createElement("span");
      ts.style.color = "var(--txm)";
      ts.style.fontSize = "var(--ty-xs)";
      ts.textContent = ftFull(t.ts || t.timestamp).slice(11, 19);

      const strat = document.createElement("span");
      strat.style.color = (t.strategy === "bold") ? "var(--org)" : "var(--cyn)";
      strat.style.fontSize = "var(--ty-xs)";
      strat.style.fontWeight = "600";
      strat.textContent = (t.strategy || "").toUpperCase();

      const action = document.createElement("span");
      const isBuy = (t.action || "").toUpperCase() === "BUY";
      action.className = "badge " + (isBuy ? "badge--BUY" : "badge--SELL");
      action.textContent = (t.action || "").toUpperCase();

      const tk = document.createElement("span");
      tk.textContent = t.ticker || "";
      tk.style.fontWeight = "600";

      const sek = document.createElement("span");
      sek.style.textAlign = "right";
      sek.style.color = "var(--txd)";
      // /api/trades emits total_sek (the field name on the underlying
      // transaction record). Accept ``sek`` as a legacy fallback so a
      // future API rename doesn't blank the column.
      const sekVal = t.total_sek ?? t.sek;
      sek.textContent = sekVal != null ? Number(sekVal).toFixed(0) + " kr" : "—";

      row.append(ts, strat, action, tk, sek);
      wrap.append(row);
    });
    tradesSlot.append(wrap);
  }
}
