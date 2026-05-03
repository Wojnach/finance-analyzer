/*
 * views/avanza.js — live Avanza account snapshot.
 *
 * Lets the user verify that the system's view of the brokerage is in
 * sync with what they see in the actual Avanza app. Cash + open
 * positions + open orders + active stop-losses, polled at the same
 * cadence the dashboard uses for other broker-state data.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fs, fp, fpct, fAgo, ftFull, num } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { errorBanner } from "../components/error-banner.js";

const POLL_KEY = "avanza";
const SLOT_AVANZA = "avanza";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(SLOT_AVANZA, _renderBody));

    polling.register(POLL_KEY, 60_000, async () => {
      const d = await fj("/api/avanza_account", { ttl: 5_000 });
      if (d) state.set(SLOT_AVANZA, d);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("avanza", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--avanza";

  const head = document.createElement("div");
  head.style.display = "flex";
  head.style.alignItems = "center";
  head.style.justifyContent = "space-between";
  head.style.marginBottom = "var(--sp-2)";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "Avanza account";
  head.append(title);

  const refresh = document.createElement("button");
  refresh.type = "button";
  refresh.className = "icon-btn";
  refresh.textContent = "Refresh";
  refresh.style.minWidth = "auto";
  refresh.style.padding = "var(--sp-1) var(--sp-2)";
  let resetTimer = null;
  refresh.addEventListener("click", () => {
    polling.fireNow(POLL_KEY);
    // Visual confirmation — flash green + checkmark for 1.4s. Mirrors the
    // pattern in views/settings.js so refresh-style buttons behave the same
    // everywhere.
    refresh.style.color = "var(--grn)";
    refresh.style.borderColor = "var(--grn)";
    refresh.textContent = "Refreshed ✓";
    if (resetTimer) clearTimeout(resetTimer);
    resetTimer = setTimeout(() => {
      refresh.style.color = "";
      refresh.style.borderColor = "";
      refresh.textContent = "Refresh";
      resetTimer = null;
    }, 1400);
  });
  head.append(refresh);
  view.append(head);

  const stamp = document.createElement("div");
  stamp.dataset.slot = "stamp";
  stamp.style.fontSize = "var(--ty-xs)";
  stamp.style.color = "var(--txm)";
  stamp.style.marginBottom = "var(--sp-2)";
  view.append(stamp);

  const errors = document.createElement("div");
  errors.dataset.slot = "errors";
  view.append(errors);

  for (const slot of ["cash", "positions", "orders", "stops"]) {
    const el = document.createElement("div");
    el.dataset.slot = slot;
    el.style.marginBottom = "var(--sp-3)";
    view.append(el);
  }
  return view;
}

function _slot(name) {
  return _root?.querySelector(`[data-slot="${name}"]`) || null;
}

function _renderBody() {
  if (!_root) return;
  const data = state.get(SLOT_AVANZA);
  if (!data) {
    const c = _slot("cash");
    if (c) { while (c.firstChild) c.removeChild(c.firstChild); c.append(emptyState("Loading Avanza state…")); }
    return;
  }

  // Stamp
  const stamp = _slot("stamp");
  if (stamp) {
    while (stamp.firstChild) stamp.removeChild(stamp.firstChild);
    if (data.ts) stamp.textContent = `Last fetched ${ftFull(data.ts)} (${fAgo(data.ts)})`;
  }

  // Errors
  const errSlot = _slot("errors");
  if (errSlot) {
    while (errSlot.firstChild) errSlot.removeChild(errSlot.firstChild);
    const errs = Array.isArray(data.errors) ? data.errors : [];
    if (errs.length) {
      errSlot.append(errorBanner(`${errs.length} subsection error${errs.length > 1 ? "s" : ""}: ${errs.join(" · ")}`));
    }
  }

  _renderCash(data.cash);
  _renderPositions(data.positions || []);
  _renderOrders(data.orders || []);
  _renderStops(data.stop_losses || []);
}

function _renderCash(cash) {
  const slot = _slot("cash"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Cash";
  slot.append(title);

  if (!cash) {
    slot.append(emptyState("Cash unavailable."));
    return;
  }
  const card = document.createElement("article");
  card.className = "card";
  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(120px, 1fr))";
  grid.style.gap = "var(--sp-3)";
  grid.append(
    _kpi("Buying power", fs(num(cash.buying_power))),
    _kpi("Total value",  fs(num(cash.total_value))),
    _kpi("Own capital",  fs(num(cash.own_capital))),
  );
  card.append(grid);
  slot.append(card);
}

function _renderPositions(positions) {
  const slot = _slot("positions"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = `Positions (${positions.length})`;
  slot.append(title);

  if (!positions.length) {
    slot.append(emptyState("No open positions."));
    return;
  }
  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const p of positions) {
    const row = document.createElement("article");
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";

    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.alignItems = "baseline";
    const name = document.createElement("div");
    name.style.fontWeight = "600";
    name.style.fontSize = "var(--ty-md)";
    name.style.overflow = "hidden";
    name.style.textOverflow = "ellipsis";
    name.style.whiteSpace = "nowrap";
    name.textContent = p.name || p.orderbook_id || "—";
    name.title = `${p.name} · obid ${p.orderbook_id} · acct ${p.account_id}`;
    const pnl = document.createElement("div");
    const pct = num(p.profit_percent);
    pnl.className = "num num--md";
    pnl.classList.add(pct > 0 ? "pos" : pct < 0 ? "neg" : "flat");
    pnl.textContent = fpct(pct, 2);
    top.append(name, pnl);

    const meta = document.createElement("div");
    meta.style.display = "flex";
    meta.style.justifyContent = "space-between";
    meta.style.fontSize = "var(--ty-xs)";
    meta.style.color = "var(--txm)";
    meta.style.marginTop = "var(--sp-1)";
    meta.append(
      _span(`${p.volume ?? "?"} sh @ ${fp(num(p.last_price))}`),
      _span(`${fs(num(p.value))} ${p.currency || ""}`),
    );

    const sub = document.createElement("div");
    sub.style.display = "flex";
    sub.style.justifyContent = "space-between";
    sub.style.fontSize = "var(--ty-xs)";
    sub.style.color = "var(--txm)";
    sub.append(
      _span(`acq ${fs(num(p.acquired_value))}`),
      _span(`day ${fpct(num(p.change_percent), 2)}`),
    );

    row.append(top, meta, sub);
    wrap.append(row);
  }
  slot.append(wrap);
}

function _renderOrders(orders) {
  const slot = _slot("orders"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = `Open orders (${orders.length})`;
  slot.append(title);

  if (!orders.length) {
    slot.append(emptyState("No open orders."));
    return;
  }
  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const o of orders) {
    const row = document.createElement("article");
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";
    row.style.display = "grid";
    row.style.gridTemplateColumns = "60px 1fr 80px 70px";
    row.style.gap = "var(--sp-2)";
    row.style.alignItems = "center";
    row.style.fontSize = "var(--ty-sm)";

    const sideEl = document.createElement("span");
    const side = (o.side || "").toUpperCase();
    sideEl.className = "badge " + (side === "BUY" ? "badge--BUY" : side === "SELL" ? "badge--SELL" : "badge--HOLD");
    sideEl.textContent = side || "?";

    const obid = document.createElement("span");
    obid.style.overflow = "hidden";
    obid.style.textOverflow = "ellipsis";
    obid.style.whiteSpace = "nowrap";
    obid.textContent = o.orderbook_id || "—";
    obid.title = `order ${o.order_id}`;

    const px = document.createElement("span");
    px.className = "num";
    px.style.textAlign = "right";
    px.textContent = `${o.volume ?? "?"} @ ${fp(num(o.price))}`;

    const status = document.createElement("span");
    status.style.fontSize = "var(--ty-xs)";
    status.style.color = "var(--txm)";
    status.style.textAlign = "right";
    status.textContent = o.status || "—";

    row.append(sideEl, obid, px, status);
    wrap.append(row);
  }
  slot.append(wrap);
}

function _renderStops(stops) {
  const slot = _slot("stops"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = `Stop-losses (${stops.length})`;
  slot.append(title);

  if (!stops.length) {
    slot.append(emptyState("No active stop-losses."));
    return;
  }
  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const s of stops) {
    const row = document.createElement("article");
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";
    row.style.fontSize = "var(--ty-sm)";

    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    const obid = document.createElement("span");
    obid.style.fontWeight = "600";
    obid.textContent = s.orderbook_id || "—";
    obid.title = `stop_id ${s.stop_id} · acct ${s.account_id}`;
    const status = document.createElement("span");
    status.style.fontSize = "var(--ty-xs)";
    status.style.color = "var(--txm)";
    status.textContent = s.status || "—";
    top.append(obid, status);

    const sub = document.createElement("div");
    sub.style.display = "flex";
    sub.style.justifyContent = "space-between";
    sub.style.fontSize = "var(--ty-xs)";
    sub.style.color = "var(--txd)";
    sub.style.marginTop = "var(--sp-1)";
    sub.append(
      _span(`trig ${fp(num(s.trigger_price))} (${s.trigger_type || "—"})`),
      _span(`sell ${fp(num(s.sell_price))} × ${s.volume ?? "?"}`),
    );

    row.append(top, sub);
    wrap.append(row);
  }
  slot.append(wrap);
}

function _kpi(label, value, color) {
  const wrap = document.createElement("div");
  const l = document.createElement("div");
  l.className = "card__subtitle";
  l.textContent = label;
  const v = document.createElement("div");
  v.className = "num num--md";
  v.textContent = String(value ?? "—");
  if (color) v.style.color = color;
  wrap.append(l, v);
  return wrap;
}

function _span(text) {
  const s = document.createElement("span");
  s.textContent = text;
  return s;
}
