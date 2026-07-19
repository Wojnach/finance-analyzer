/*
 * views/prices.js — live ticker prices the system is reading.
 *
 * Lets the user verify "is the system seeing the same number I see in
 * the Avanza app?". Pulls /api/summary for Tier-1 underlying USD prices
 * + FX rate, and /api/avanza_account for held warrant last-prices in
 * SEK. Side-by-side with timestamps so freshness is obvious.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fp, fs, fAgo, ftFull, num } from "../format.js";
import { emptyState } from "../components/empty-state.js";

const POLL_KEY = "prices";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.SUMMARY, _renderBody));
    _unsubs.push(state.subscribe("avanza", _renderBody));

    polling.register(POLL_KEY, 60_000, async () => {
      const s = await fj("/api/summary", { ttl: 5_000 });
      if (s) state.set(state.Slots.SUMMARY, s);
      const a = await fj("/api/avanza_account", { ttl: 5_000 });
      if (a) state.set("avanza", a);
      return s != null || a != null;
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("prices", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--prices";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "Live prices";
  view.append(title);

  const sub = document.createElement("div");
  sub.style.fontSize = "var(--ty-sm)";
  sub.style.color = "var(--txd)";
  sub.style.marginBottom = "var(--sp-3)";
  sub.textContent = (
    "What the system is reading right now. Compare with the Avanza app: "
    + "USD numbers come from Binance / Alpaca / yfinance; warrant SEK "
    + "numbers come from Avanza itself."
  );
  view.append(sub);

  for (const slot of ["underlyings", "fx", "warrants"]) {
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
  const sum = state.get(state.Slots.SUMMARY);
  const avanza = state.get("avanza");

  _renderUnderlyings(sum);
  _renderFx(sum);
  _renderHeldWarrants(avanza);
}

function _renderUnderlyings(sum) {
  const slot = _slot("underlyings"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Underlying tickers (USD)";
  slot.append(title);

  const sigs = sum?.signals?.signals;
  if (!sigs || typeof sigs !== "object") {
    slot.append(emptyState("Loading underlyings…"));
    return;
  }
  const ts = sum?.signals?.timestamp || sum?.signals?.ts;
  const fxRate = num(sum?.signals?.fx_rate);
  const tickerOrder = ["BTC-USD", "ETH-USD", "MSTR", "XAU-USD", "XAG-USD"];

  const list = document.createElement("div");
  list.style.background = "var(--card)";
  list.style.border = "1px solid var(--bdr)";
  list.style.borderRadius = "var(--rad-md)";

  const seen = new Set();
  const ordered = [
    ...tickerOrder.filter((t) => sigs[t]).map((t) => [t, sigs[t]]),
    ...Object.entries(sigs).filter(([t]) => !tickerOrder.includes(t)),
  ];

  for (const [ticker, sig] of ordered) {
    if (seen.has(ticker)) continue;
    seen.add(ticker);
    const usd = num(sig?.price_usd);
    const sek = (usd != null && fxRate != null) ? usd * fxRate : null;

    const row = document.createElement("article");
    row.style.display = "grid";
    row.style.gridTemplateColumns = "90px 1fr auto";
    row.style.gap = "var(--sp-2)";
    row.style.alignItems = "center";
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";

    const tk = document.createElement("div");
    tk.style.fontWeight = "600";
    tk.textContent = ticker.replace(/-USD$/, "");
    const usdEl = document.createElement("div");
    usdEl.className = "num num--md";
    usdEl.textContent = usd != null ? `$${fp(usd)}` : "—";
    const sekEl = document.createElement("div");
    sekEl.style.fontSize = "var(--ty-xs)";
    sekEl.style.color = "var(--txm)";
    sekEl.style.textAlign = "right";
    sekEl.textContent = sek != null ? `${fs(sek)} SEK` : "";
    row.append(tk, usdEl, sekEl);
    list.append(row);
  }
  slot.append(list);

  if (ts) {
    const stamp = document.createElement("div");
    stamp.style.fontSize = "var(--ty-xs)";
    stamp.style.color = "var(--txm)";
    stamp.style.marginTop = "var(--sp-1)";
    stamp.textContent = `Source freshness: ${ftFull(ts)} (${fAgo(ts)})`;
    slot.append(stamp);
  }
}

function _renderFx(sum) {
  const slot = _slot("fx"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const fxRate = num(sum?.signals?.fx_rate);
  if (fxRate == null) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "FX";
  slot.append(title);

  const card = document.createElement("article");
  card.className = "card";
  card.style.display = "flex";
  card.style.justifyContent = "space-between";
  card.style.alignItems = "center";
  const left = document.createElement("div");
  const lbl = document.createElement("div");
  lbl.className = "card__title";
  lbl.textContent = "USD/SEK";
  const hint = document.createElement("div");
  hint.className = "card__subtitle";
  hint.textContent = "Used for SEK ⇄ USD conversions across the system";
  left.append(lbl, hint);
  const v = document.createElement("div");
  v.className = "num num--lg";
  v.textContent = fp(fxRate);
  card.append(left, v);
  slot.append(card);
}

function _renderHeldWarrants(avanza) {
  const slot = _slot("warrants"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const positions = avanza?.positions;
  if (!Array.isArray(positions) || !positions.length) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = `Held warrants — Avanza last prices (${positions.length})`;
  slot.append(title);

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const p of positions) {
    const row = document.createElement("article");
    row.style.display = "grid";
    row.style.gridTemplateColumns = "1fr auto auto";
    row.style.gap = "var(--sp-2)";
    row.style.alignItems = "center";
    row.style.padding = "var(--sp-2) var(--sp-3)";
    row.style.borderBottom = "1px solid var(--bdr)";

    const name = document.createElement("div");
    name.style.fontWeight = "600";
    name.style.fontSize = "var(--ty-md)";
    name.style.overflow = "hidden";
    name.style.textOverflow = "ellipsis";
    name.style.whiteSpace = "nowrap";
    name.textContent = p.name || p.orderbook_id || "—";
    name.title = p.orderbook_id || "";

    const price = document.createElement("div");
    price.className = "num num--md";
    price.textContent = `${fp(num(p.last_price))} ${p.currency || ""}`.trim();

    const obid = document.createElement("div");
    obid.style.fontSize = "var(--ty-xs)";
    obid.style.color = "var(--txm)";
    obid.textContent = `obid ${p.orderbook_id ?? "—"}`;

    row.append(name, price, obid);
    wrap.append(row);
  }
  slot.append(wrap);

  if (avanza?.ts) {
    const stamp = document.createElement("div");
    stamp.style.fontSize = "var(--ty-xs)";
    stamp.style.color = "var(--txm)";
    stamp.style.marginTop = "var(--sp-1)";
    stamp.textContent = `Avanza freshness: ${ftFull(avanza.ts)} (${fAgo(avanza.ts)})`;
    slot.append(stamp);
  }
}
