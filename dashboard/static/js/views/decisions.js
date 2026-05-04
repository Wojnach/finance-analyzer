/*
 * views/decisions.js — Layer 2 decision history (M5 user moment).
 *
 * Filter chip strips for action / ticker / strategy. Card list (last 100).
 * Tap a card → #decisions/<id> detail view.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { decisionCard } from "../components/decision-card.js";
import { emptyState } from "../components/empty-state.js";
import { filterChip } from "../components/filter-chip.js";
import { fAgo } from "../format.js";
import { view as detailView } from "./decision-detail.js";

const POLL_KEY = "decisions.list";
const POLL_KEY_SYS = "decisions.system_status";

const ACTIONS  = ["ALL", "BUY", "SELL", "HOLD"];
const TICKERS  = ["ALL", "BTC-USD", "ETH-USD", "MSTR", "XAG-USD", "XAU-USD"];
const STRATS   = ["ALL", "patient", "bold"];

let _root = null;
let _filter = { action: "ALL", ticker: "ALL", strategy: "ALL" };
let _unsubs = [];
let _activeChild = null;        // detail-view delegation tracker

export const view = {
  mount(rootEl, params) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);

    // /#decisions/<id> → detail view
    if (params && (typeof params === "string" || typeof params === "object")) {
      _activeChild = detailView;
      detailView.mount(rootEl, params);
      return;
    }
    _activeChild = null;

    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.DECISIONS, _renderList));
    _unsubs.push(state.subscribe(state.Slots.SYSTEM_STATUS, _renderList));
    polling.register(POLL_KEY, 60_000, _refresh);
    polling.register(POLL_KEY_SYS, 60_000, _refreshSystemStatus);
  },
  unmount() {
    if (_activeChild) {
      try { _activeChild.unmount && _activeChild.unmount(); } catch (_) {}
      _activeChild = null;
    } else {
      for (const off of _unsubs) try { off(); } catch (_) {}
      _unsubs = [];
      polling.unregister(POLL_KEY);
      polling.unregister(POLL_KEY_SYS);
    }
    _root = null;
  },
};

router.register("decisions", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--decisions";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "Decisions";
  view.append(title);

  const bannerSlot = document.createElement("div");
  bannerSlot.dataset.slot = "banner";
  view.append(bannerSlot);

  // Action filter strip
  view.append(_renderChipStrip("Action", ACTIONS, _filter.action, (v) => {
    _filter.action = v; _refresh();
  }));
  view.append(_renderChipStrip("Ticker", TICKERS, _filter.ticker, (v) => {
    _filter.ticker = v; _refresh();
  }, _shortenTicker));
  view.append(_renderChipStrip("Strategy", STRATS, _filter.strategy, (v) => {
    _filter.strategy = v; _refresh();
  }, _capitalize));

  const list = document.createElement("div");
  list.dataset.slot = "list";
  view.append(list);

  return view;
}

function _renderChipStrip(label, options, current, onSelect, transform = null) {
  const wrap = document.createElement("div");
  wrap.style.marginBottom = "var(--sp-2)";

  const lbl = document.createElement("div");
  lbl.style.fontSize = "var(--ty-xs)";
  lbl.style.color = "var(--txm)";
  lbl.style.textTransform = "uppercase";
  lbl.style.letterSpacing = "0.5px";
  lbl.style.marginBottom = "var(--sp-1)";
  lbl.textContent = label;
  wrap.append(lbl);

  const strip = document.createElement("div");
  strip.className = "chip-strip";
  options.forEach((opt) => {
    const chip = filterChip({
      label: transform ? transform(opt) : opt,
      active: opt === current,
      value: opt,
      onToggle: () => {
        strip.querySelectorAll(".chip").forEach((c) => {
          c.classList.remove("active");
          c.setAttribute("aria-pressed", "false");
        });
        chip.classList.add("active");
        chip.setAttribute("aria-pressed", "true");
        onSelect(opt);
      },
    });
    strip.append(chip);
  });
  wrap.append(strip);
  return wrap;
}

function _shortenTicker(t) {
  if (t === "ALL") return "ALL";
  return t.replace(/-USD$/, "");
}
function _capitalize(s) {
  if (!s || s === "ALL") return s;
  return s[0].toUpperCase() + s.slice(1);
}

async function _refresh() {
  const url = _buildUrl();
  const data = await fj(url, { ttl: 5_000 });
  if (data) state.set(state.Slots.DECISIONS, data);
}

async function _refreshSystemStatus() {
  const d = await fj("/api/system_status", { ttl: 5_000 });
  if (d) state.set(state.Slots.SYSTEM_STATUS, d);
}

function _buildUrl() {
  const qs = new URLSearchParams({ limit: "100" });
  if (_filter.action !== "ALL")   qs.set("action", _filter.action);
  if (_filter.ticker !== "ALL")   qs.set("ticker", _filter.ticker);
  if (_filter.strategy !== "ALL") qs.set("strategy", _filter.strategy);
  return "/api/decisions?" + qs.toString();
}

function _renderList() {
  if (!_root) return;
  const slot = _root.querySelector('[data-slot="list"]');
  const bannerSlot = _root.querySelector('[data-slot="banner"]');
  if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const data = state.get(state.Slots.DECISIONS);
  const arr = Array.isArray(data) ? data : (data?.decisions || []);

  if (bannerSlot) {
    while (bannerSlot.firstChild) bannerSlot.removeChild(bannerSlot.firstChild);
    const banner = _buildContextBanner(arr);
    if (banner) bannerSlot.append(banner);
  }

  if (!arr.length) {
    slot.append(emptyState("No decisions match the current filters."));
    return;
  }
  arr.forEach((d, idx) => {
    const card = decisionCard({
      ts: d.ts || d.timestamp,
      ticker: d.ticker || d.trigger_ticker || "",
      trigger: d.trigger,
      regime: d.regime,
      patient: d.decisions?.patient || d.patient,
      bold:    d.decisions?.bold    || d.bold,
      onTap: () => router.navigate("decisions", String(d.ts || d.timestamp || idx)),
    });
    card.style.marginBottom = "var(--sp-2)";
    slot.append(card);
  });
}

function _decisionAction(d, side) {
  const slot = d?.decisions?.[side] || d?.[side];
  return (slot?.action || "").toUpperCase();
}

function _buildContextBanner(arr) {
  if (!Array.isArray(arr) || !arr.length) return null;
  const head = arr[0];
  const headHold = _decisionAction(head, "patient") === "HOLD" && _decisionAction(head, "bold") === "HOLD";
  if (!headHold) return null;

  const sys = state.get(state.Slots.SYSTEM_STATUS);
  const tickers = sys?.signal_aggregate?.tickers;
  if (!Array.isArray(tickers) || !tickers.length) return null;
  // Match RANGING / RANGING_LOW_VOL / etc. — engine emits prefixed regime tags.
  const ranging = tickers.some((t) => /^RANGING/i.test(String(t?.regime || "")));
  if (!ranging) return null;

  let lastNonHold = null;
  for (const d of arr) {
    if (_decisionAction(d, "patient") !== "HOLD" || _decisionAction(d, "bold") !== "HOLD") {
      lastNonHold = d.ts || d.timestamp;
      break;
    }
  }

  const banner = document.createElement("div");
  banner.style.background = "var(--card)";
  banner.style.borderLeft = "3px solid var(--cyn)";
  banner.style.padding = "var(--sp-2) var(--sp-3)";
  banner.style.marginBottom = "var(--sp-2)";
  banner.style.borderRadius = "4px";
  banner.style.fontSize = "var(--ty-sm)";
  banner.style.color = "var(--txm)";
  banner.textContent = lastNonHold
    ? `Range-bound regime · last non-HOLD ${fAgo(lastNonHold)}`
    : "Range-bound regime · no non-HOLD decisions in this window";
  return banner;
}
