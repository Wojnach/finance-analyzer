/*
 * state.js — module-scoped singleton state store.
 *
 * Replaces the ~30 bare globals (T, paused, cdv, lastRefreshTime, equityChartInstance,
 * _accuracySort, _decFilters, _msgFilters, ...) that lived at the top of the
 * legacy index.html script section.
 *
 * Pattern: module-private object + small public API. Views call `set()` /
 * `get()`; nothing reaches in directly. View-side subscriptions via
 * `subscribe(key, fn)` for reactive re-render.
 */

const _data = Object.create(null);
const _subs = Object.create(null); // key -> Set<fn>

/**
 * Read a value. Returns undefined if never set.
 */
export function get(key) {
  return _data[key];
}

/**
 * Read or supply a default if undefined.
 */
export function getOr(key, defaultValue) {
  return key in _data ? _data[key] : defaultValue;
}

/**
 * Write a value and notify subscribers if it changed (shallow ref check).
 */
export function set(key, value) {
  const prev = _data[key];
  if (prev === value) return value;
  _data[key] = value;
  const subs = _subs[key];
  if (subs) {
    for (const fn of subs) {
      try { fn(value, prev); } catch (e) { console.error("state subscriber error", key, e); }
    }
  }
  return value;
}

/**
 * Update an object slot by merging keys. Notifies subscribers.
 * If the slot is not currently an object, it is replaced.
 */
export function patch(key, partial) {
  const prev = _data[key];
  const next = (prev && typeof prev === "object" && !Array.isArray(prev))
    ? { ...prev, ...partial }
    : { ...partial };
  return set(key, next);
}

/**
 * Subscribe to changes for `key`. Returns an unsubscribe function.
 */
export function subscribe(key, fn) {
  if (typeof fn !== "function") return () => {};
  if (!_subs[key]) _subs[key] = new Set();
  _subs[key].add(fn);
  return () => _subs[key]?.delete(fn);
}

/**
 * Snapshot for debugging only — never mutate the returned object.
 */
export function snapshot() {
  return { ..._data };
}

// Default UI state slots — listed for discoverability; consumers don't need to
// pre-populate, but having them documented in one place avoids string-typo bugs.
export const Slots = Object.freeze({
  // Server data
  SUMMARY:          "summary",
  DECISIONS:        "decisions",
  TRIGGERS:         "triggers",
  ACCURACY:         "accuracy",
  ACCURACY_HISTORY: "accuracyHistory",
  SIGNAL_HEATMAP:   "signalHeatmap",
  WARRANTS:         "warrants",
  RISK:             "risk",
  HEALTH:           "health",
  LOOP_HEALTH:      "loopHealth",
  METALS:           "metals",
  GOLDDIGGER:       "golddigger",
  EQUITY_CURVE:     "equityCurve",
  TRADES:           "trades",
  MESSAGES:         "messages",
  LORA:             "lora",
  MARKET_HEALTH:    "marketHealth",
  SYSTEM_STATUS:    "systemStatus",
  TRADING_STATUS:   "tradingStatus",
  CLAUDE_COST:      "claudeCost",

  // UI state
  ROUTE:            "ui.route",
  PAUSED:           "ui.paused",
  THEME:            "ui.theme",
  LAST_REFRESH:     "ui.lastRefresh",
  ERROR:            "ui.error",
});
