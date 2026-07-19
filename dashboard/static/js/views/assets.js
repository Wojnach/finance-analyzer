/*
 * views/assets.js — tradeable-asset universe view.
 *
 * What the loops will buy/sell on Avanza: metals warrants, crypto
 * warrants, oil warrants. Read-only. Pulls /api/tradeable_assets.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fAgo, ftFull } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { errorBanner } from "../components/error-banner.js";

const POLL_KEY = "assets";
const SLOT = "assets";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());
    _unsubs.push(state.subscribe(SLOT, _renderBody));

    polling.register(POLL_KEY, 5 * 60_000, async () => {
      const d = await fj("/api/tradeable_assets", { ttl: 60_000 });
      if (d) state.set(SLOT, d);
      return d != null;
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("assets", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--assets";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "Tradeable assets";
  view.append(title);

  const sub = document.createElement("div");
  sub.style.fontSize = "var(--ty-sm)";
  sub.style.color = "var(--txd)";
  sub.style.marginBottom = "var(--sp-3)";
  sub.textContent = (
    "What the loops will buy or sell on Avanza. Side-check the orderbook "
    + "ids and leverage match what you see in the Avanza app."
  );
  view.append(sub);

  const stamp = document.createElement("div");
  stamp.dataset.slot = "stamp";
  stamp.style.fontSize = "var(--ty-xs)";
  stamp.style.color = "var(--txm)";
  stamp.style.marginBottom = "var(--sp-2)";
  view.append(stamp);

  const errors = document.createElement("div");
  errors.dataset.slot = "errors";
  view.append(errors);

  for (const slot of ["metals", "crypto", "oil"]) {
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
  const data = state.get(SLOT);
  if (!data) {
    const m = _slot("metals");
    if (m) { while (m.firstChild) m.removeChild(m.firstChild); m.append(emptyState("Loading catalog…")); }
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
      errSlot.append(errorBanner(`${errs.length} catalog error${errs.length > 1 ? "s" : ""}: ${errs.join(" · ")}`));
    }
  }

  _renderCategory("metals", "Metals (XAG / XAU)", data.metals_warrants);
  _renderCategory("crypto", "Crypto (BTC / ETH)", data.crypto_warrants);
  _renderCategory("oil",    "Oil (WTI / Brent)",  data.oil_warrants);
}

function _renderCategory(slotName, title, warrants) {
  const slot = _slot(slotName); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const t = document.createElement("div");
  t.className = "section-title";
  const count = warrants && typeof warrants === "object" ? Object.keys(warrants).length : 0;
  t.textContent = `${title} (${count})`;
  slot.append(t);

  if (!count) {
    slot.append(emptyState(`No ${title.toLowerCase()} loaded. Catalog refresh may not have run yet.`));
    return;
  }

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const [key, w] of Object.entries(warrants)) {
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
    name.textContent = w.name || key;
    name.title = key;

    const dir = document.createElement("span");
    const direction = String(w.direction || "").toUpperCase();
    dir.className = "badge " + (direction === "SHORT" ? "badge--SELL" : "badge--BUY");
    dir.textContent = direction || "—";

    top.append(name, dir);
    row.append(top);

    const meta = document.createElement("div");
    meta.style.display = "flex";
    meta.style.justifyContent = "space-between";
    meta.style.fontSize = "var(--ty-xs)";
    meta.style.color = "var(--txm)";
    meta.style.marginTop = "var(--sp-1)";

    const left = document.createElement("span");
    const obId = w.ob_id || w.orderbook_id || w.id || "—";
    left.textContent = `obid ${obId}`;

    const right = document.createElement("span");
    const lev = w.leverage != null ? `${w.leverage}x` : "—";
    const under = w.underlying || w.underlying_ticker || "—";
    right.textContent = `${under} · ${lev}${w.barrier ? ` · barrier ${w.barrier}` : ""}`;

    meta.append(left, right);
    row.append(meta);

    if (w.issuer || w.api_type) {
      const sub = document.createElement("div");
      sub.style.fontSize = "var(--ty-xs)";
      sub.style.color = "var(--txm)";
      sub.textContent = [w.issuer, w.api_type].filter(Boolean).join(" · ");
      row.append(sub);
    }

    wrap.append(row);
  }
  slot.append(wrap);
}
