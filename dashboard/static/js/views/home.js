/*
 * views/home.js — the default route. The "first screen" on a phone.
 *
 * Cards (top → bottom, ordered by user-moments M1 → M5):
 *   1. P&L glance + 24h sparkline
 *   2. Open positions strip
 *   3. Active consensus row
 *   4. Latest decision card
 *   5. System pulse strip
 *
 * Polls /api/summary (60s), /api/risk (5min), /api/warrants (60s),
 * /api/loop_health (60s). Re-renders on state change.
 */

import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fpct, fp, num } from "../format.js";
import { pnlCard }       from "../components/pnl-card.js";
import { positionCard }  from "../components/position-card.js";
import { consensusChip } from "../components/consensus-chip.js";
import { decisionCard }  from "../components/decision-card.js";
import { pulseDot }      from "../components/pulse-dot.js";
import { emptyState }    from "../components/empty-state.js";
import { miniSparkline } from "../charts/mini-sparkline.js";
import * as router from "../router.js";

const POLL_KEY_SUMMARY = "home.summary";
const POLL_KEY_RISK    = "home.risk";
const POLL_KEY_WARR    = "home.warrants";
const POLL_KEY_LOOPS   = "home.loops";

let _root = null;
let _disposeSparkline = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    // Subscriptions
    _unsubs.push(
      state.subscribe(state.Slots.SUMMARY,     _renderAll),
      state.subscribe(state.Slots.RISK,        _renderAll),
      state.subscribe(state.Slots.WARRANTS,    _renderAll),
      state.subscribe(state.Slots.LOOP_HEALTH, _renderAll),
      state.subscribe(state.Slots.LAST_REFRESH, _renderRefreshDot),
    );

    // Polling — Track-6 cadence
    polling.register(POLL_KEY_SUMMARY, 60_000, async () => {
      const d = await fj("/api/summary");
      if (d) state.set(state.Slots.SUMMARY, d);
    });
    polling.register(POLL_KEY_LOOPS, 60_000, async () => {
      const d = await fj("/api/loop_health");
      if (d) state.set(state.Slots.LOOP_HEALTH, d);
    });
    polling.register(POLL_KEY_WARR, 60_000, async () => {
      const d = await fj("/api/warrants");
      if (d) state.set(state.Slots.WARRANTS, d);
    });
    polling.register(POLL_KEY_RISK, 5 * 60_000, async () => {
      const d = await fj("/api/risk");
      if (d) state.set(state.Slots.RISK, d);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY_SUMMARY);
    polling.unregister(POLL_KEY_RISK);
    polling.unregister(POLL_KEY_WARR);
    polling.unregister(POLL_KEY_LOOPS);
    if (_disposeSparkline) { try { _disposeSparkline(); } catch (_) {} _disposeSparkline = null; }
    _root = null;
  },
};

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--home";

  const slots = ["pnl", "positions", "consensus", "decision", "pulse", "below"];
  for (const id of slots) {
    const slot = document.createElement("div");
    slot.dataset.slot = id;
    slot.style.marginBottom = "var(--sp-3)";
    view.append(slot);
  }
  return view;
}

function _renderAll() {
  if (!_root) return;
  _renderPnL();
  _renderPositions();
  _renderConsensus();
  _renderLatestDecision();
  _renderPulse();
}

function _slot(name) {
  return _root?.querySelector(`[data-slot="${name}"]`) || null;
}

// 1. P&L card -----------------------------------------------------------------
function _renderPnL() {
  const slot = _slot("pnl"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const summary = state.get(state.Slots.SUMMARY);
  const warrants = state.get(state.Slots.WARRANTS);
  if (!summary && !warrants) {
    slot.append(emptyState("Loading P&L…"));
    return;
  }

  const sig = summary?.signals || {};
  const port = summary?.portfolio || {};
  const portBold = summary?.portfolio_bold || {};
  const sigPort = sig.portfolio || {};

  // Patient/Bold come from /api/summary; warrants come from /api/warrants
  const patient = {
    label: "Patient",
    value: num(sigPort.total_sek ?? port.total_sek),
    deltaPct: num(sigPort.pnl_pct ?? port.pnl_pct),
  };
  const bold = {
    label: "Bold",
    value: num(portBold?.total_sek),
    deltaPct: _pnlPctFor(portBold),
  };
  const wPct = _warrantPnLPct(warrants);
  const wVal = _warrantTotal(warrants);
  const warrantsCard = {
    label: "Warrants",
    value: wVal,
    deltaPct: wPct,
  };

  // Sparkline from equity-curve fetch (deferred — best-effort fallback to none).
  // Field names match portfolio_value_history.jsonl: patient_value_sek + bold_value_sek.
  let sparkEl = null;
  if (_disposeSparkline) { try { _disposeSparkline(); } catch (_) {} _disposeSparkline = null; }
  const eq = state.get(state.Slots.EQUITY_CURVE);
  if (Array.isArray(eq) && eq.length > 1) {
    const recent = eq.slice(-96); // last ~24h at 15min cadence
    const values = recent
      .map((r) => {
        const p = num(r?.patient_value_sek);
        const b = num(r?.bold_value_sek);
        if (p == null && b == null) return null;
        return (p || 0) + (b || 0);
      })
      .filter((x) => x != null);
    if (values.length > 1) {
      const sp = miniSparkline({ values, height: 36 });
      sparkEl = sp.element;
      _disposeSparkline = sp.dispose;
    }
  }

  slot.append(pnlCard({ patient, bold, warrants: warrantsCard, sparkline: sparkEl, title: "Portfolio" }));

  _updateHeaderPnL(patient.deltaPct);
}

function _updateHeaderPnL(pct) {
  const el = document.getElementById("header-pnl");
  if (!el) return;
  el.classList.remove("pos", "neg", "flat");
  if (pct == null) { el.textContent = "--"; el.classList.add("flat"); return; }
  el.classList.add(pct > 0 ? "pos" : pct < 0 ? "neg" : "flat");
  el.textContent = fpct(pct, 2);
}

function _pnlPctFor(p) {
  if (!p) return null;
  const total = num(p.total_sek);
  const cash  = num(p.cash_sek);
  const init  = num(p.starting_cash) ?? 500_000; // default per CLAUDE.md
  if (total == null) return null;
  return ((total - init) / init) * 100;
}

function _warrantTotal(w) {
  if (!w?.holdings) return null;
  // Sum (qty * last_known_price) across holdings; fall back to entry price.
  let total = 0;
  for (const h of Object.values(w.holdings)) {
    if (!h) continue;
    const qty = num(h.shares ?? h.units ?? h.qty);
    if (qty == null) continue;
    const px = num(h.current_price ?? h.last_price ?? h.entry_price ?? h.price);
    if (px == null) continue;
    total += qty * px;
  }
  return total > 0 ? total : null;
}

function _warrantPnLPct(w) {
  if (!w?.holdings) return null;
  let invested = 0, current = 0;
  for (const h of Object.values(w.holdings)) {
    if (!h) continue;
    const qty   = num(h.shares ?? h.units ?? h.qty);
    const entry = num(h.entry_price ?? h.cost_per_unit);
    const px    = num(h.current_price ?? h.last_price ?? h.entry_price);
    if (qty == null || entry == null || px == null) continue;
    invested += qty * entry;
    current  += qty * px;
  }
  if (invested <= 0) return null;
  return ((current - invested) / invested) * 100;
}

// 2. Open positions strip -----------------------------------------------------
function _renderPositions() {
  const slot = _slot("positions"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const summary = state.get(state.Slots.SUMMARY);
  const warrants = state.get(state.Slots.WARRANTS);
  const positions = _gatherPositions(summary, warrants);
  if (!positions.length) return; // nothing to show, leave slot empty

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Open positions";
  slot.append(title);

  const strip = document.createElement("div");
  strip.className = "scroll-strip";
  positions.forEach((p) => {
    strip.append(positionCard({
      ticker: p.ticker,
      side: p.side,
      pnlPct: p.pnlPct,
      pricePerUnit: p.price,
      stopPrice: p.stopPrice,
      stopDistancePct: p.stopDistancePct,
      onTap: () => router.navigate("signals", p.ticker),
    }));
  });
  slot.append(strip);
}

function _gatherPositions(summary, warrants) {
  const positions = [];
  // Patient
  const patient = summary?.portfolio?.holdings || {};
  for (const [ticker, h] of Object.entries(patient)) {
    if (!h?.shares || h.shares <= 0) continue;
    positions.push(_positionFromHolding(ticker, h, "Patient", summary));
  }
  // Bold
  const bold = summary?.portfolio_bold?.holdings || {};
  for (const [ticker, h] of Object.entries(bold)) {
    if (!h?.shares || h.shares <= 0) continue;
    positions.push(_positionFromHolding(ticker, h, "Bold", summary));
  }
  // Warrants
  for (const [name, h] of Object.entries(warrants?.holdings || {})) {
    if (!h) continue;
    const qty = num(h.shares ?? h.units ?? h.qty);
    if (qty == null || qty <= 0) continue;
    positions.push({
      ticker: name,
      side: "LONG",
      pnlPct: _holdingPnLPct(h),
      price: num(h.current_price ?? h.last_price),
      stopPrice: num(h.stop_price),
      stopDistancePct: _stopDistancePct(h),
    });
  }
  return positions;
}

function _positionFromHolding(ticker, h, _strategy, summary) {
  const sigForTicker = summary?.signals?.signals?.[ticker] || summary?.signals?.[ticker] || {};
  const livePrice = num(sigForTicker.price ?? sigForTicker.last_price);
  const entry = num(h.cost_per_unit ?? h.entry_price ?? h.cost_basis);
  return {
    ticker,
    side: "LONG",
    pnlPct: (livePrice != null && entry != null && entry > 0)
      ? ((livePrice - entry) / entry) * 100 : null,
    price: livePrice,
    stopPrice: num(h.stop_price ?? h.stop),
    stopDistancePct: _stopDistanceFrom(livePrice, h.stop_price ?? h.stop),
  };
}

function _holdingPnLPct(h) {
  const px = num(h.current_price ?? h.last_price);
  const entry = num(h.entry_price ?? h.cost_per_unit);
  if (px == null || entry == null || entry <= 0) return null;
  return ((px - entry) / entry) * 100;
}
function _stopDistancePct(h) {
  const px = num(h.current_price ?? h.last_price);
  return _stopDistanceFrom(px, h.stop_price ?? h.stop);
}
function _stopDistanceFrom(px, stop) {
  const p = num(px), s = num(stop);
  if (p == null || s == null || p <= 0) return null;
  return ((p - s) / p) * 100;
}

// 3. Active consensus row -----------------------------------------------------
function _renderConsensus() {
  const slot = _slot("consensus"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const summary = state.get(state.Slots.SUMMARY);
  const sigs = summary?.signals?.signals;
  const tfs  = summary?.signals?.timeframes;
  if (!sigs || typeof sigs !== "object") return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "Consensus";
  slot.append(title);

  const strip = document.createElement("div");
  strip.className = "scroll-strip";

  for (const [ticker, sig] of Object.entries(sigs)) {
    const action = (sig?.consensus || sig?.action || "HOLD").toUpperCase();
    const votes = {
      buy:  num(sig?.buy_votes  ?? sig?.votes?.buy),
      sell: num(sig?.sell_votes ?? sig?.votes?.sell),
      hold: num(sig?.hold_votes ?? sig?.votes?.hold),
    };
    // Map timeframes to a tiny strip
    const tfStrip = {};
    if (tfs?.[ticker]) {
      for (const [tf, val] of Object.entries(tfs[ticker])) {
        const a = val?.action || val?.consensus;
        if (a) tfStrip[tf] = a;
      }
    }
    strip.append(consensusChip({
      ticker, action, votes, timeframes: tfStrip,
      onTap: () => router.navigate("signals", ticker),
    }));
  }
  slot.append(strip);
}

// 4. Latest Layer-2 decision --------------------------------------------------
function _renderLatestDecision() {
  const slot = _slot("decision"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  // Pull from /api/decisions (small one-shot — reuse cache).
  // Fired here on first render and on summary updates so it stays fresh.
  fj("/api/decisions?limit=1", { ttl: 30_000 }).then((data) => {
    while (slot.firstChild) slot.removeChild(slot.firstChild);
    const arr = Array.isArray(data) ? data : (data?.decisions || []);
    if (!arr.length) {
      slot.append(emptyState("No decisions yet."));
      return;
    }
    const d = arr[0];
    const card = decisionCard({
      ts: d.ts || d.timestamp,
      ticker: d.ticker || d.trigger_ticker || "",
      trigger: d.trigger,
      regime: d.regime,
      patient: d.decisions?.patient || d.patient,
      bold:    d.decisions?.bold    || d.bold,
      onTap: () => router.navigate("decisions"),
    });
    const title = document.createElement("div");
    title.className = "section-title";
    title.textContent = "Latest decision";
    slot.append(title, card);
  }).catch((e) => {
    console.warn("home: decisions fetch failed", e);
  });
}

// 5. System pulse strip -------------------------------------------------------
function _renderPulse() {
  const slot = _slot("pulse"); if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const rollup = state.get(state.Slots.LOOP_HEALTH);
  // /api/loop_health returns {checked_at, loops:{name:{state, age_seconds, payload, error, path}}, any_unhealthy, unhealthy[]}.
  // Earlier versions iterated the rollup directly; that broke pulse-dot
  // colors and the home-screen badge.  Codex P1 finding 2026-05-03.
  const loops = rollup?.loops;
  if (!loops || typeof loops !== "object" || !Object.keys(loops).length) return;

  const title = document.createElement("div");
  title.className = "section-title";
  title.textContent = "System pulse";
  slot.append(title);

  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexWrap = "wrap";
  wrap.style.gap = "var(--sp-2)";

  for (const [name, info] of Object.entries(loops)) {
    const stateName = _loopStateClass(info?.state);
    wrap.append(pulseDot({
      state: stateName,
      label: name.replace(/^PF-?/i, ""),
      title: `state: ${info?.state ?? "?"}, age ${info?.age_seconds ?? "?"}s`
        + (info?.error ? ` · ${info.error}` : ""),
      onTap: () => router.navigate("health"),
    }));
  }
  slot.append(wrap);
}

function _loopStateClass(state) {
  switch ((state || "").toLowerCase()) {
    case "fresh":   return "ok";
    case "stale":   return "warn";
    case "missing": return "fail";
    case "unparseable": return "fail";
    default:        return "idle";
  }
}

function _renderRefreshDot() {
  const dot = document.getElementById("refresh-dot");
  if (!dot) return;
  // Brief flash on refresh
  dot.classList.remove("paused");
  dot.style.animation = "none";
  // Force reflow then re-enable
  void dot.offsetWidth;
  dot.style.animation = "";
}

// Self-register with the router
router.register("home", view);
