/*
 * views/silver.js — #silver command page (Phase 6, 2026-07-18).
 *
 * XAG-USD operational picture — the user's stated main instrument focus.
 * Composes six existing/new endpoints client-side (system_status,
 * control/state, control/registry, silver/accuracy, accuracy, metals,
 * warrants, grid-fisher) into one page instead of making the user hop
 * between Home / Signals / Control / Metals to answer "what is the system
 * actually doing with silver right now". Rendering for each section lives
 * in its own render/silver-*.js module (mirrors render/accuracy.js,
 * render/voters-card.js etc.) — this file is just fetch/poll/slot wiring.
 *
 * Sections top → bottom: pipeline diagram, component health, accuracy
 * matrix, live votes, trade panel. Every section renders its own
 * sectionErrorChip when ITS fetch fails, without blanking whatever data is
 * already on screen.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { sectionErrorChip } from "../components/section-error-chip.js";
import { votersCard } from "../render/voters-card.js";
import { silverPipeline } from "../render/silver-pipeline.js";
import { silverComponentGrid } from "../render/silver-components.js";
import { silverAccuracyMatrix } from "../render/silver-accuracy.js";
import { silverLiveVotes } from "../render/silver-votes.js";
import { silverTradePanel } from "../render/silver-trade.js";

const TICKER = "XAG-USD";

const POLL_SYS = "silver.system_status";
const POLL_CONTROL = "silver.control_state";
const POLL_REGISTRY = "silver.registry";
const POLL_ACC = "silver.accuracy";
const POLL_ACC_GLOBAL = "silver.accuracy_global";
const POLL_METALS = "silver.metals";
const POLL_WARRANTS = "silver.warrants";
const POLL_GRID = "silver.grid_fisher";
const ALL_POLL_KEYS = [
  POLL_SYS,
  POLL_CONTROL,
  POLL_REGISTRY,
  POLL_ACC,
  POLL_ACC_GLOBAL,
  POLL_METALS,
  POLL_WARRANTS,
  POLL_GRID,
];

// Local (non-shared) state slots — system_status/metals/warrants/accuracy
// reuse the shared state.Slots so other views' data stays warm on nav.
const SLOT_CONTROL = "silver.controlState";
const SLOT_REGISTRY = "silver.registry";
const SLOT_ACC = "silver.accuracy";
const SLOT_GRID = "silver.gridFisher";

let _root = null;
let _unsubs = [];
// Per-section fetch failure flags — independent of state so a failed
// fetch never clears previously-good data, only annotates it as stale.
const _fail = {
  system: false,
  control: false,
  registry: false,
  accuracy: false,
  accGlobal: false,
  metals: false,
  warrants: false,
  grid: false,
};

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(
      state.subscribe(state.Slots.SYSTEM_STATUS, () => {
        _renderPipeline();
        _renderVotes();
        _renderTrade();
        // voters card inside Component health reads sys.voters — without
        // this, a registry fetch landing first freezes "no voter data yet"
        // until the next registry poll (race seen live 2026-07-18).
        _renderComponents();
      }),
    );
    _unsubs.push(
      state.subscribe(SLOT_CONTROL, () => {
        _renderPipeline();
        _renderTrade();
      }),
    );
    _unsubs.push(state.subscribe(SLOT_REGISTRY, _renderComponents));
    _unsubs.push(state.subscribe(SLOT_ACC, _renderAccuracy));
    _unsubs.push(state.subscribe(state.Slots.ACCURACY, _renderAccuracy));
    _unsubs.push(state.subscribe(state.Slots.METALS, _renderTrade));
    _unsubs.push(state.subscribe(state.Slots.WARRANTS, _renderTrade));
    _unsubs.push(state.subscribe(SLOT_GRID, _renderTrade));

    polling.register(POLL_SYS, 30_000, async () => {
      const d = await fj("/api/system_status");
      _fail.system = d == null;
      if (d) state.set(state.Slots.SYSTEM_STATUS, d);
      else {
        _renderPipeline();
        _renderVotes();
        _renderTrade();
      }
    });
    polling.register(POLL_CONTROL, 30_000, async () => {
      const d = await fj("/api/control/state");
      _fail.control = d == null;
      if (d) state.set(SLOT_CONTROL, d);
      else {
        _renderPipeline();
        _renderTrade();
      }
    });
    polling.register(POLL_REGISTRY, 60_000, async () => {
      const d = await fj(`/api/control/registry?ticker=${TICKER}`, {
        ttl: 30_000,
      });
      _fail.registry = d == null;
      if (d) state.set(SLOT_REGISTRY, d.registry);
      else _renderComponents();
    });
    polling.register(POLL_ACC, 5 * 60_000, async () => {
      const d = await fj(`/api/silver/accuracy?ticker=${TICKER}`, {
        ttl: 60_000,
      });
      _fail.accuracy = d == null;
      if (d) state.set(SLOT_ACC, d);
      else _renderAccuracy();
    });
    polling.register(POLL_ACC_GLOBAL, 5 * 60_000, async () => {
      const d = await fj("/api/accuracy");
      _fail.accGlobal = d == null;
      if (d) state.set(state.Slots.ACCURACY, d);
      else _renderAccuracy();
    });
    polling.register(POLL_METALS, 60_000, async () => {
      const d = await fj("/api/metals", { ttl: 5_000 });
      _fail.metals = d == null;
      if (d) state.set(state.Slots.METALS, d);
      else _renderTrade();
    });
    polling.register(POLL_WARRANTS, 60_000, async () => {
      const d = await fj("/api/warrants", { ttl: 5_000 });
      _fail.warrants = d == null;
      if (d) state.set(state.Slots.WARRANTS, d);
      else _renderTrade();
    });
    polling.register(POLL_GRID, 60_000, async () => {
      const d = await fj("/api/grid-fisher", { ttl: 5_000 });
      _fail.grid = d == null;
      if (d) state.set(SLOT_GRID, d);
      else _renderTrade();
    });

    _renderPipeline();
    _renderComponents();
    _renderAccuracy();
    _renderVotes();
    _renderTrade();
  },
  unmount() {
    for (const off of _unsubs)
      try {
        off();
      } catch (_) {}
    _unsubs = [];
    for (const key of ALL_POLL_KEYS) polling.unregister(key);
    _root = null;
  },
};
router.register("silver", view);

// ---------------------------------------------------------------------------
// Shell
// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--silver";

  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "Silver — XAG-USD";
  v.append(t);

  for (const slotName of [
    "pipeline",
    "components",
    "accuracy",
    "votes",
    "trade",
  ]) {
    const slot = document.createElement("div");
    slot.dataset.slot = slotName;
    slot.style.marginBottom = "var(--sp-3)";
    v.append(slot);
  }
  return v;
}

function _slot(name) {
  return _root ? _root.querySelector(`[data-slot="${name}"]`) : null;
}

function _replaceSlot(name, title, node, errChip) {
  const slot = _slot(name);
  if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  const h = document.createElement("div");
  h.className = "section-title";
  h.textContent = title;
  slot.append(h);
  if (errChip) slot.append(errChip);
  slot.append(node);
}

// ---------------------------------------------------------------------------
// Section renderers — each pulls its own slice of state + failure flags
// ---------------------------------------------------------------------------

function _renderPipeline() {
  const sys = state.get(state.Slots.SYSTEM_STATUS);
  const cs = state.get(SLOT_CONTROL);
  const registry = state.get(SLOT_REGISTRY);
  const gridFisher = state.get(SLOT_GRID);
  const node = silverPipeline({
    sys,
    cs,
    registryApplicable: registry?.applicable,
    gridFisher,
  });
  const err =
    _fail.system || _fail.control
      ? sectionErrorChip(
          "system_status/control fetch failed — showing last known state",
        )
      : null;
  _replaceSlot("pipeline", "Pipeline", node, err);
}

function _renderComponents() {
  const registry = state.get(SLOT_REGISTRY);
  const node = document.createElement("div");
  node.append(silverComponentGrid({ registryTicker: registry }));
  const sys = state.get(state.Slots.SYSTEM_STATUS);
  node.append(votersCard(sys?.voters));
  const err = _fail.registry
    ? sectionErrorChip("registry fetch failed — showing last known state")
    : null;
  _replaceSlot("components", "Component health", node, err);
}

function _renderAccuracy() {
  const silverAcc = state.get(SLOT_ACC);
  const accGlobal = state.get(state.Slots.ACCURACY);
  const node = silverAccuracyMatrix({ silverAcc, accGlobal });
  const err =
    _fail.accuracy || _fail.accGlobal
      ? sectionErrorChip("accuracy fetch failed — showing last known state")
      : null;
  _replaceSlot("accuracy", "Accuracy matrix", node, err);
}

function _renderVotes() {
  const sys = state.get(state.Slots.SYSTEM_STATUS);
  const node = silverLiveVotes({ sys });
  const err = _fail.system
    ? sectionErrorChip("system_status fetch failed — showing last known state")
    : null;
  _replaceSlot("votes", "Live votes", node, err);
}

function _renderTrade() {
  const cs = state.get(SLOT_CONTROL);
  const sys = state.get(state.Slots.SYSTEM_STATUS);
  const metals = state.get(state.Slots.METALS);
  const warrants = state.get(state.Slots.WARRANTS);
  const gridFisher = state.get(SLOT_GRID);
  const node = silverTradePanel({ cs, sys, metals, warrants, gridFisher });
  const err =
    _fail.control || _fail.metals || _fail.warrants || _fail.grid
      ? sectionErrorChip(
          "one or more trade-panel fetches failed — showing last known state",
        )
      : null;
  _replaceSlot("trade", "Trade panel", node, err);
}
