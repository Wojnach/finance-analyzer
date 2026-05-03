/*
 * views/signals.js — Signal heatmap + per-signal accuracy + history chart.
 *
 * Three sub-tabs (segmented control at top):
 *  1. Heatmap (default) — Track-5 transposed, sticky cells, color-only.
 *     Ticker chip-bar at top to switch BTC/ETH/MSTR/XAG/XAU.
 *  2. Accuracy — signal-row list sorted by accuracy desc, horizon toggle.
 *  3. History — top-N accuracy lines over time.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { emptyState } from "../components/empty-state.js";
import { filterChip } from "../components/filter-chip.js";
import { renderHeatmap } from "../render/signals-heatmap.js";
import { renderAccuracyPanel } from "../render/accuracy.js";
import { accuracyChart } from "../charts/accuracy-chart.js";

const POLL_KEY_HEATMAP = "signals.heatmap";
const POLL_KEY_ACC     = "signals.accuracy";

const DEFAULT_TICKERS = ["BTC-USD", "ETH-USD", "MSTR", "XAG-USD", "XAU-USD"];
const HORIZONS = ["1d", "3d", "5d", "10d"];

let _root = null;
let _activeTicker = null;
let _activeHorizon = "1d";
let _activeTab = "heatmap";          // heatmap | accuracy | history
let _disposeChart = null;
let _unsubs = [];

export const view = {
  mount(rootEl, params) {
    _root = rootEl;

    // params can be a single ticker (e.g. /#signals/BTC-USD) — drill from Home
    if (typeof params === "string" && params) _activeTicker = params;

    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.SIGNAL_HEATMAP, _renderActiveTab));
    _unsubs.push(state.subscribe(state.Slots.ACCURACY,        _renderActiveTab));
    _unsubs.push(state.subscribe(state.Slots.ACCURACY_HISTORY, _renderActiveTab));

    polling.register(POLL_KEY_HEATMAP, 60_000, async () => {
      const d = await fj("/api/signal-heatmap");
      if (d) state.set(state.Slots.SIGNAL_HEATMAP, d);
    });
    polling.register(POLL_KEY_ACC, 5 * 60_000, async () => {
      const d = await fj("/api/accuracy");
      if (d) state.set(state.Slots.ACCURACY, d);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY_HEATMAP);
    polling.unregister(POLL_KEY_ACC);
    if (_disposeChart) { try { _disposeChart(); } catch (_) {} _disposeChart = null; }
    _root = null;
  },
};
router.register("signals", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--signals";

  // Sub-tab segmented control
  const tabs = document.createElement("div");
  tabs.className = "chip-strip";
  ["heatmap", "accuracy", "history"].forEach((name) => {
    tabs.append(filterChip({
      label: name[0].toUpperCase() + name.slice(1),
      active: name === _activeTab,
      value: name,
      onToggle: () => {
        _activeTab = name;
        tabs.querySelectorAll(".chip").forEach((c) => {
          c.classList.toggle("active", c.dataset.value === name);
          c.setAttribute("aria-pressed", c.dataset.value === name ? "true" : "false");
        });
        _renderActiveTab();
      },
    }));
  });
  view.append(tabs);

  // Body slot
  const body = document.createElement("div");
  body.dataset.slot = "body";
  body.style.marginTop = "var(--sp-3)";
  view.append(body);

  return view;
}

function _renderActiveTab() {
  if (!_root) return;
  const slot = _root.querySelector('[data-slot="body"]');
  if (!slot) return;
  // Dispose any previous chart instance before rebuilding
  if (_disposeChart) { try { _disposeChart(); } catch (_) {} _disposeChart = null; }
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  if (_activeTab === "heatmap")  return _renderHeatmapTab(slot);
  if (_activeTab === "accuracy") return _renderAccuracyTab(slot);
  if (_activeTab === "history")  return _renderHistoryTab(slot);
}

function _renderHeatmapTab(slot) {
  const data = state.get(state.Slots.SIGNAL_HEATMAP);
  const tickers = (Array.isArray(data?.tickers) && data.tickers.length)
    ? data.tickers : DEFAULT_TICKERS;
  if (!_activeTicker) _activeTicker = tickers[0];

  // Ticker chip strip
  const tickerStrip = document.createElement("div");
  tickerStrip.className = "chip-strip";
  tickers.forEach((t) => {
    tickerStrip.append(filterChip({
      label: t.replace(/-USD$/, ""),
      active: t === _activeTicker,
      value: t,
      onToggle: () => {
        _activeTicker = t;
        _renderActiveTab();
      },
    }));
  });
  slot.append(tickerStrip);

  if (!data) {
    slot.append(emptyState("Loading heatmap…"));
    return;
  }

  // Pull per-ticker accuracy for the bottom-sheet drill content.
  const accAll = state.get(state.Slots.ACCURACY) || {};
  const horizon = _activeHorizon;
  const accSlice = accAll[horizon]?.signals || {};
  const accuracyMap = Object.create(null);
  for (const [name, info] of Object.entries(accSlice)) {
    const v = info?.pct ?? info?.accuracy ?? info?.accuracy_pct;
    if (Number.isFinite(Number(v))) accuracyMap[name] = Number(v);
  }
  const disabled = new Set(data?.disabled_signals || []);

  slot.append(renderHeatmap({
    ticker: _activeTicker,
    data,
    accuracy: accuracyMap,
    disabled,
  }));
}

function _renderAccuracyTab(slot) {
  // Horizon chip-bar
  const strip = document.createElement("div");
  strip.className = "chip-strip";
  HORIZONS.forEach((h) => {
    strip.append(filterChip({
      label: h,
      active: h === _activeHorizon,
      value: h,
      onToggle: () => {
        _activeHorizon = h;
        _renderActiveTab();
      },
    }));
  });
  slot.append(strip);

  const data = state.get(state.Slots.ACCURACY);
  if (!data) {
    slot.append(emptyState("Loading accuracy…"));
    return;
  }
  slot.append(renderAccuracyPanel({ horizon: _activeHorizon, data }));
}

async function _renderHistoryTab(slot) {
  let history = state.get(state.Slots.ACCURACY_HISTORY);
  if (!history) {
    slot.append(emptyState("Loading accuracy history…"));
    const fetched = await fj("/api/accuracy-history?limit=120", { ttl: 5 * 60_000 });
    if (fetched) state.set(state.Slots.ACCURACY_HISTORY, fetched);
    return; // rerender will run via state subscription
  }
  const arr = Array.isArray(history) ? history : (history?.snapshots || history);
  if (!Array.isArray(arr) || !arr.length) {
    slot.append(emptyState("No accuracy history data yet."));
    return;
  }
  const built = accuracyChart({ history: arr, topN: 8, height: 260 });
  slot.append(built.element);
  _disposeChart = built.dispose;
}
