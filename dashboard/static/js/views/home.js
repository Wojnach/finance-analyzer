/*
 * views/home.js — system-health-first home (2026-05-04 redesign).
 *
 * The previous home led with simulated-portfolio P&L. The user
 * deprioritised that — the Patient/Bold portfolios almost never trade
 * and the dashboard's job is to answer "is the system actually
 * working?" at a glance. The old P&L hero now lives at #portfolio
 * under More.
 *
 * Card stack (top → bottom):
 *   1. System status hero      (GREEN/YELLOW/RED + reasons)
 *   2. Trading status          (per-Avanza-bot state with reason)
 *   3. LLM inference health    (per-model success bars)
 *   4. Layer 2 activity        (24h spark + latest invocation)
 *   5. Signal pulse            (abstain rate per ticker)
 *   6. Errors & violations     (unresolved tail)
 *   7. P&L footer              (tiny line linking to /portfolio)
 *
 * Polls /api/system_status (30s) and /api/trading_status (30s). Both
 * endpoints have 30s server-side TTL caches so even with multiple
 * dashboard tabs open the load on disk is bounded.
 */

import * as state from "../state.js";
import * as polling from "../polling.js";
import * as router from "../router.js";
import { fj } from "../fetch.js";
import { fpct, fs } from "../format.js";

import { systemStatusHero }    from "../render/system-status-hero.js";
import { tradingStatusCard }   from "../render/trading-status-card.js";
import { llmInferenceCard }    from "../render/llm-inference-card.js";
import { layer2ActivityCard }  from "../render/layer2-activity-card.js";
import { signalPulseCard }     from "../render/signal-pulse-card.js";
import { errorsPanel }         from "../render/errors-panel.js";

const POLL_KEY_SYS = "home.system_status";
const POLL_KEY_TRD = "home.trading_status";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(
      state.subscribe(state.Slots.SYSTEM_STATUS, _renderAll),
      state.subscribe(state.Slots.TRADING_STATUS, _renderAll),
      state.subscribe(state.Slots.LAST_REFRESH, _renderRefreshDot),
    );

    polling.register(POLL_KEY_SYS, 30_000, async () => {
      const d = await fj("/api/system_status");
      if (d) state.set(state.Slots.SYSTEM_STATUS, d);
    });
    polling.register(POLL_KEY_TRD, 30_000, async () => {
      const d = await fj("/api/trading_status");
      if (d) state.set(state.Slots.TRADING_STATUS, d);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY_SYS);
    polling.unregister(POLL_KEY_TRD);
    _root = null;
  },
};

// ---------------------------------------------------------------------------

function _renderShell() {
  const view = document.createElement("section");
  view.className = "view view--home";

  const slots = ["hero", "trading", "llm", "layer2", "signals", "errors", "footer"];
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
  const sys = state.get(state.Slots.SYSTEM_STATUS);
  const trd = state.get(state.Slots.TRADING_STATUS);

  _replaceSlot("hero",    sys ? systemStatusHero(sys)             : _placeholder("Loading system status…"));
  _replaceSlot("trading", trd ? tradingStatusCard(trd)            : _placeholder("Loading bot states…"));
  _replaceSlot("llm",     sys ? llmInferenceCard(sys.llm_inference)        : _placeholder("Loading LLM telemetry…"));
  _replaceSlot("layer2",  sys ? layer2ActivityCard(sys.layer2)             : _placeholder("Loading Layer 2 activity…"));
  _replaceSlot("signals", sys ? signalPulseCard(sys.signal_aggregate)      : _placeholder("Loading signals…"));
  _replaceSlot("errors",  sys ? errorsPanel(sys)                            : _placeholder("Loading error log…"));
  _replaceSlot("footer",  sys ? _pnlFooter(sys.pnl_footer)                  : _placeholder(""));

  _updateHeaderBadge(sys);
}

function _replaceSlot(name, child) {
  const slot = _root?.querySelector(`[data-slot="${name}"]`);
  if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  if (child) slot.append(child);
}

function _placeholder(text) {
  const div = document.createElement("div");
  div.className = "card";
  div.style.padding = "var(--sp-3)";
  div.style.color = "var(--txm)";
  div.style.fontSize = "var(--ty-sm)";
  div.textContent = text;
  return div;
}

// 7. P&L footer (deprioritised — links to /portfolio under More) -------------
function _pnlFooter(pnl) {
  if (!pnl || (pnl.patient_value_sek == null && pnl.bold_value_sek == null)) {
    return document.createElement("div");
  }
  const row = document.createElement("button");
  row.type = "button";
  row.className = "card card--tap";
  row.style.display = "flex";
  row.style.justifyContent = "space-between";
  row.style.alignItems = "center";
  row.style.padding = "var(--sp-2) var(--sp-3)";
  row.style.minHeight = "var(--tap-min)";
  row.style.background = "transparent";
  row.style.textAlign = "left";
  row.style.cursor = "pointer";
  row.addEventListener("click", () => router.navigate("portfolio"));

  const label = document.createElement("span");
  label.style.color = "var(--txm)";
  label.style.fontSize = "var(--ty-sm)";
  label.textContent = "Simulated portfolios →";
  row.append(label);

  const stats = document.createElement("span");
  stats.style.fontSize = "var(--ty-sm)";
  stats.style.color = "var(--txd)";
  const ppct = _pnlPct(pnl.patient_value_sek, pnl.patient_starting_sek);
  const bpct = _pnlPct(pnl.bold_value_sek, pnl.bold_starting_sek);
  stats.textContent = `P ${fs(pnl.patient_value_sek)} (${fpct(ppct)}) · B ${fs(pnl.bold_value_sek)} (${fpct(bpct)})`;
  row.append(stats);

  return row;
}

function _pnlPct(value, starting) {
  if (!Number.isFinite(value) || !Number.isFinite(starting) || starting <= 0) return null;
  return ((value - starting) / starting) * 100;
}

// Header status badge — small colored dot in the page header so the
// status is visible even when scrolled past the hero.
function _updateHeaderBadge(sys) {
  const el = document.getElementById("header-pnl");
  if (!el) return;
  el.classList.remove("pos", "neg", "flat");
  if (!sys) {
    el.textContent = "—";
    el.classList.add("flat");
    return;
  }
  const overall = (sys.overall || "").toUpperCase();
  if (overall === "GREEN")       { el.textContent = "OK"; el.classList.add("pos"); }
  else if (overall === "YELLOW") { el.textContent = "!";  el.classList.add("flat"); }
  else if (overall === "RED")    { el.textContent = "✕";  el.classList.add("neg"); }
  else                            { el.textContent = "—"; el.classList.add("flat"); }
}

function _renderRefreshDot() {
  const dot = document.getElementById("refresh-dot");
  if (!dot) return;
  dot.classList.remove("paused");
  dot.style.animation = "none";
  void dot.offsetWidth;
  dot.style.animation = "";
}

router.register("home", view);
