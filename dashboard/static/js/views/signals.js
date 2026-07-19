/*
 * views/signals.js — Signal heatmap + per-signal accuracy + history chart.
 *
 * Three sub-tabs (segmented control at top):
 *  1. Heatmap (default) — Track-5 transposed, sticky cells, color-only.
 *     Ticker chip-bar at top to switch BTC/ETH/MSTR/XAG/XAU. Inactive
 *     (all-HOLD) rows sort to the bottom with a "hide inactive" toggle.
 *  2. Accuracy — signal-row list sorted by accuracy desc, horizon toggle;
 *     also a per-ticker matrix sub-view (rows=tickers, cols=horizons).
 *  3. History — consensus line + a per-signal small-multiple (selector
 *     chips, capped at 6 lines).
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { emptyState } from "../components/empty-state.js";
import { filterChip } from "../components/filter-chip.js";
import { renderHeatmap } from "../render/signals-heatmap.js";
import {
  renderAccuracyPanel,
  renderPerTickerTable,
} from "../render/accuracy.js";
import {
  accuracyChart,
  consensusHistoryChart,
  rankSignalsByVolume,
} from "../charts/accuracy-chart.js";

const POLL_KEY_HEATMAP = "signals.heatmap";
const POLL_KEY_ACC = "signals.accuracy";

const DEFAULT_TICKERS = ["BTC-USD", "ETH-USD", "MSTR", "XAG-USD", "XAU-USD"];
const HORIZONS = ["1d", "3d", "5d", "10d"];
const ACCURACY_SUBVIEWS = [
  ["signals", "Per-signal"],
  ["tickers", "Per-ticker"],
];
const HISTORY_CANDIDATE_N = 12; // chip pool size
const HISTORY_MAX_SELECTED = 6; // max lines on the per-signal chart

let _root = null;
let _activeHorizon = "1d";
let _activeTab = "heatmap"; // heatmap | accuracy | history
let _accuracySubView = "signals"; // signals | tickers
let _historySelected = null; // Set<string> | null (null = not yet defaulted)
let _disposeCharts = [];
let _unsubs = [];

export const view = {
  mount(rootEl, _params) {
    // Note: home cards may navigate with a ticker param (e.g. /#signals/BTC-USD)
    // for future drill-by-ticker. The heatmap currently shows all tickers
    // as columns, so the param is informational only — not yet used.
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.SIGNAL_HEATMAP, _renderActiveTab));
    _unsubs.push(state.subscribe(state.Slots.ACCURACY, _renderActiveTab));
    _unsubs.push(
      state.subscribe(state.Slots.ACCURACY_HISTORY, _renderActiveTab),
    );

    polling.register(POLL_KEY_HEATMAP, 60_000, async () => {
      const d = await fj("/api/signal-heatmap");
      if (d) state.set(state.Slots.SIGNAL_HEATMAP, d);
      return d != null;
    });
    polling.register(POLL_KEY_ACC, 5 * 60_000, async () => {
      const d = await fj("/api/accuracy");
      if (d) state.set(state.Slots.ACCURACY, d);
      return d != null;
    });
  },
  unmount() {
    for (const off of _unsubs)
      try {
        off();
      } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY_HEATMAP);
    polling.unregister(POLL_KEY_ACC);
    for (const dispose of _disposeCharts)
      try {
        dispose();
      } catch (_) {}
    _disposeCharts = [];
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
    tabs.append(
      filterChip({
        label: name[0].toUpperCase() + name.slice(1),
        active: name === _activeTab,
        value: name,
        onToggle: () => {
          _activeTab = name;
          tabs.querySelectorAll(".chip").forEach((c) => {
            c.classList.toggle("active", c.dataset.value === name);
            c.setAttribute(
              "aria-pressed",
              c.dataset.value === name ? "true" : "false",
            );
          });
          _renderActiveTab();
        },
      }),
    );
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
  // Dispose any previous chart instance(s) before rebuilding
  for (const dispose of _disposeCharts)
    try {
      dispose();
    } catch (_) {}
  _disposeCharts = [];
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  if (_activeTab === "heatmap") return _renderHeatmapTab(slot);
  if (_activeTab === "accuracy") return _renderAccuracyTab(slot);
  if (_activeTab === "history") return _renderHistoryTab(slot);
}

function _renderHeatmapTab(slot) {
  const data = state.get(state.Slots.SIGNAL_HEATMAP);
  if (!data) {
    slot.append(emptyState("Loading heatmap…"));
    return;
  }

  // /api/signal-heatmap shape is {heatmap: {ticker: {signal: "BUY"|"SELL"|"HOLD"}}}
  // — no per-timeframe nesting. We render rows=signals × cols=tickers and
  // use the chip-bar to filter accuracy-data scope only (drill content),
  // not the heatmap shape.
  const tickers =
    Array.isArray(data?.tickers) && data.tickers.length
      ? data.tickers
      : DEFAULT_TICKERS;

  // Pull per-ticker accuracy for the bottom-sheet drill content.
  const accAll = state.get(state.Slots.ACCURACY) || {};
  const horizon = _activeHorizon;
  const accSlice = accAll[horizon]?.signals || {};
  const accuracyMap = Object.create(null);
  // 2026-05-05: also surface the disable reason in the bottom-sheet so
  // tapping a "DISABLED" cell explains *why* (e.g. "pending live validation").
  const disabledReasons = Object.create(null);
  for (const [name, info] of Object.entries(accSlice)) {
    const v = info?.pct ?? info?.accuracy ?? info?.accuracy_pct;
    if (Number.isFinite(Number(v))) accuracyMap[name] = Number(v);
    if (info?.disabled_reason) disabledReasons[name] = info.disabled_reason;
  }
  const disabled = new Set(data?.disabled_signals || []);

  slot.append(
    renderHeatmap({
      data,
      tickers,
      accuracy: accuracyMap,
      disabled,
      disabledReasons,
    }),
  );
}

function _renderAccuracyTab(slot) {
  // Sub-view toggle: per-signal list vs per-ticker matrix. The per-ticker
  // matrix shows all horizons at once (cols=horizons), so the horizon
  // chip-bar below only applies to the per-signal sub-view.
  const subStrip = document.createElement("div");
  subStrip.className = "chip-strip";
  ACCURACY_SUBVIEWS.forEach(([val, label]) => {
    subStrip.append(
      filterChip({
        label,
        active: val === _accuracySubView,
        value: val,
        onToggle: () => {
          _accuracySubView = val;
          _renderActiveTab();
        },
      }),
    );
  });
  slot.append(subStrip);

  const data = state.get(state.Slots.ACCURACY);
  if (!data) {
    slot.append(emptyState("Loading accuracy…"));
    return;
  }

  if (_accuracySubView === "tickers") {
    slot.append(renderPerTickerTable({ data, horizons: HORIZONS }));
    return;
  }

  // Horizon chip-bar
  const strip = document.createElement("div");
  strip.className = "chip-strip";
  HORIZONS.forEach((h) => {
    strip.append(
      filterChip({
        label: h,
        active: h === _activeHorizon,
        value: h,
        onToggle: () => {
          _activeHorizon = h;
          _renderActiveTab();
        },
      }),
    );
  });
  slot.append(strip);
  slot.append(renderAccuracyPanel({ horizon: _activeHorizon, data }));
}

async function _renderHistoryTab(slot) {
  let history = state.get(state.Slots.ACCURACY_HISTORY);
  if (!history) {
    slot.append(emptyState("Loading accuracy history…"));
    const fetched = await fj("/api/accuracy-history?limit=120", {
      ttl: 5 * 60_000,
    });
    if (fetched) state.set(state.Slots.ACCURACY_HISTORY, fetched);
    return; // rerender will run via state subscription
  }
  const arr = Array.isArray(history) ? history : history?.snapshots || history;
  if (!Array.isArray(arr) || !arr.length) {
    slot.append(emptyState("No accuracy history data yet."));
    return;
  }

  // Surface a hint when there are too few snapshots to be a useful line
  // chart. Without this an empty-looking chart is confusing — the
  // accuracy_snapshots writer needs ~7+ daily entries before trend lines
  // tell you anything beyond the per-signal accuracy panel.
  if (arr.length < 5) {
    const hint = document.createElement("div");
    hint.className = "banner banner--info";
    hint.textContent =
      `Only ${arr.length} accuracy snapshot${arr.length === 1 ? "" : "s"} so far. ` +
      "Trend lines need ~7+ daily snapshots; check back in a few days.";
    slot.append(hint);
  }

  // Consensus gets its own always-shown chart — mashing it into the
  // per-signal chart made both unreadable (2026-07-18 small-multiples split).
  const consensusTitle = document.createElement("div");
  consensusTitle.className = "section-title";
  consensusTitle.textContent = "Consensus accuracy";
  slot.append(consensusTitle);
  const consensusBuilt = consensusHistoryChart({ history: arr, height: 200 });
  slot.append(consensusBuilt.element);
  _disposeCharts.push(consensusBuilt.dispose);

  // Per-signal chart: default selection = top-5 by sample volume, chosen
  // once per session; user can swap via chips (capped at 6 total lines).
  const ranked = rankSignalsByVolume(arr);
  const candidates = ranked.slice(0, HISTORY_CANDIDATE_N).map((r) => r.name);
  if (_historySelected == null) {
    _historySelected = new Set(ranked.slice(0, 5).map((r) => r.name));
  }

  const perSigTitle = document.createElement("div");
  perSigTitle.className = "section-title";
  perSigTitle.style.marginTop = "var(--sp-3)";
  perSigTitle.textContent = "Per-signal accuracy";
  slot.append(perSigTitle);

  const chipStrip = document.createElement("div");
  chipStrip.className = "chip-strip";
  candidates.forEach((name) => {
    chipStrip.append(
      filterChip({
        label: name,
        active: _historySelected.has(name),
        value: name,
        onToggle: (next) => {
          if (next && _historySelected.size >= HISTORY_MAX_SELECTED) {
            // Cap hit — reject the add. _renderActiveTab() rebuilds the chip
            // strip from _historySelected, which snaps this chip back off.
            _renderActiveTab();
            return;
          }
          if (next) _historySelected.add(name);
          else _historySelected.delete(name);
          _renderActiveTab();
        },
      }),
    );
  });
  slot.append(chipStrip);

  const cap = document.createElement("div");
  cap.style.fontSize = "var(--ty-xs)";
  cap.style.color = "var(--txm)";
  cap.style.margin = "var(--sp-1) 0 var(--sp-2)";
  cap.textContent = `${_historySelected.size}/${HISTORY_MAX_SELECTED} signals shown`;
  slot.append(cap);

  const selected = candidates.filter((n) => _historySelected.has(n));
  if (!selected.length) {
    slot.append(emptyState("Select at least one signal above."));
    return;
  }
  const built = accuracyChart({ history: arr, signals: selected, height: 260 });
  slot.append(built.element);
  _disposeCharts.push(built.dispose);
}
