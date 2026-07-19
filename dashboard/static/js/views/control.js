/*
 * views/control.js — Command Central (Phase 3, 2026-07-18).
 *
 * Frontend for dashboard/control.py's write API — the only place in the
 * dashboard that changes anything. Three sections:
 *
 *   - LLM votes: master pause toggle (data/local_llm.disabled), plus the
 *     existing Voters card (read-only) so remote/herc2 gate state is
 *     visible right next to the switch that affects it.
 *   - Instruments: per-ticker tracked toggle (data/control/instruments.json —
 *     not yet consumed by any loop; Phase 4.3 wiring).
 *   - Loops: per allowlisted pf-* unit, live state + start/stop/restart.
 *     Every loop button needs a double-tap to fire (arm on first tap,
 *     3s window, fire on second) since these are real systemd actions
 *     on a public-facing route.
 *
 * All writes go through the small local `_post()` helper (not fetch.js's
 * `fpost`, which discards the response body on non-2xx — this view wants
 * the actual error/rate-limit detail for its toasts).
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { showToast } from "../components/toast.js";
import { emptyState } from "../components/empty-state.js";
import { pulseDot } from "../components/pulse-dot.js";
import { votersCard } from "../render/voters-card.js";

const POLL_KEY = "control";
const SYS_POLL_KEY = "control.system_status";

let _root = null;
let _unsubs = [];
let _controlState = null; // last /api/control/state payload

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.SYSTEM_STATUS, _renderVoters));

    polling.register(SYS_POLL_KEY, 30_000, async () => {
      const d = await fj("/api/system_status");
      if (d) state.set(state.Slots.SYSTEM_STATUS, d);
      return d != null;
    });

    polling.register(POLL_KEY, 10_000, async () => {
      const d = await fj("/api/control/state");
      if (d) {
        _controlState = d;
        _renderLlm();
        _renderInstruments();
        _renderLoops();
      }
      return d != null;
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    polling.unregister(SYS_POLL_KEY);
    _root = null;
    _controlState = null;
  },
};
router.register("control", view);

// ---------------------------------------------------------------------------
// Shell
// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--control";

  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "Control";
  v.append(t);

  const hint = document.createElement("div");
  hint.className = "card__subtitle";
  hint.style.marginBottom = "var(--sp-3)";
  hint.textContent = "Every action here is logged to data/control/audit.jsonl.";
  v.append(hint);

  for (const slotName of ["llm", "voters", "instruments", "loops"]) {
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

function _replaceSlot(name, node) {
  const el = _slot(name);
  if (!el) return;
  while (el.firstChild) el.removeChild(el.firstChild);
  if (node) el.append(node);
}

// ---------------------------------------------------------------------------
// LLM votes
// ---------------------------------------------------------------------------

function _renderLlm() {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0 0 var(--sp-2) 0";
  title.textContent = "LLM votes";
  card.append(title);

  if (!_controlState) {
    card.append(emptyState("Loading…"));
    _replaceSlot("llm", card);
    return;
  }

  const enabled = !!_controlState.llm_enabled;
  const row = document.createElement("div");
  row.style.display = "flex";
  row.style.alignItems = "center";
  row.style.justifyContent = "space-between";
  row.style.gap = "var(--sp-3)";

  const label = document.createElement("div");
  label.style.fontSize = "var(--ty-sm)";
  label.style.color = "var(--tx)";
  label.textContent = enabled
    ? "Local model inference: enabled"
    : "Local model inference: paused";
  row.append(label);

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "icon-btn";
  btn.style.minWidth = "auto";
  btn.style.padding = "var(--sp-2) var(--sp-3)";
  if (enabled) {
    btn.style.color = "var(--red)";
    btn.style.borderColor = "var(--red)";
  }
  btn.textContent = enabled ? "Pause" : "Resume";
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    const r = await _post("/api/control/llm", { enabled: !enabled });
    btn.disabled = false;
    _toastResult(r, r.ok ? `LLM votes ${r.data.llm_enabled ? "enabled" : "paused"}` : null);
    await _refetchState();
  });
  row.append(btn);
  card.append(row);

  _replaceSlot("llm", card);
}

// ---------------------------------------------------------------------------
// Voters (read-only — remote/herc2 state lives here until Phase 4 registry)
// ---------------------------------------------------------------------------

function _renderVoters() {
  const sys = state.get(state.Slots.SYSTEM_STATUS);
  _replaceSlot("voters", sys ? votersCard(sys.voters) : emptyState("Loading voter states…"));
}

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

function _renderInstruments() {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0 0 var(--sp-2) 0";
  title.textContent = "Instruments";
  card.append(title);

  if (!_controlState) {
    card.append(emptyState("Loading…"));
    _replaceSlot("instruments", card);
    return;
  }

  const note = document.createElement("div");
  note.className = "card__subtitle";
  note.style.marginBottom = "var(--sp-2)";
  note.textContent =
    "Stored in data/control/instruments.json — the component registry (live since 2026-07-18) is the enablement authority; per-signal overrides via registry_overrides.json.";
  card.append(note);

  const tickers = Object.keys(_controlState.instruments || {}).sort();
  for (const ticker of tickers) {
    card.append(_instrumentRow(ticker, !!_controlState.instruments[ticker]?.tracked));
  }

  _replaceSlot("instruments", card);
}

function _instrumentRow(ticker, tracked) {
  const row = document.createElement("div");
  row.style.display = "flex";
  row.style.alignItems = "center";
  row.style.justifyContent = "space-between";
  row.style.padding = "var(--sp-2) 0";
  row.style.borderTop = "1px solid var(--bdr)";
  row.style.minHeight = "var(--tap-min)";

  const label = document.createElement("span");
  label.style.fontSize = "var(--ty-sm)";
  label.style.fontWeight = "600";
  label.style.color = "var(--tx)";
  label.textContent = ticker;
  row.append(label);

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "icon-btn";
  btn.style.minWidth = "auto";
  btn.style.padding = "var(--sp-1) var(--sp-2)";
  btn.textContent = tracked ? "Tracked" : "Untracked";
  if (!tracked) btn.style.color = "var(--txm)";
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    const r = await _post("/api/control/instrument", { ticker, tracked: !tracked });
    btn.disabled = false;
    _toastResult(r, r.ok ? `${ticker} ${r.data.tracked ? "tracked" : "untracked"}` : null);
    await _refetchState();
  });
  row.append(btn);

  return row;
}

// ---------------------------------------------------------------------------
// Loops
// ---------------------------------------------------------------------------

function _renderLoops() {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0 0 var(--sp-2) 0";
  title.textContent = "Loops";
  card.append(title);

  if (!_controlState) {
    card.append(emptyState("Loading…"));
    _replaceSlot("loops", card);
    return;
  }

  const units = Object.keys(_controlState.loops || {}).sort();
  if (!units.length) {
    card.append(emptyState("No loops in the allowlist."));
  } else {
    for (const unit of units) card.append(_loopRow(unit, _controlState.loops[unit]));
  }

  _replaceSlot("loops", card);
}

function _loopRow(unit, info) {
  const row = document.createElement("div");
  row.style.padding = "var(--sp-2) 0";
  row.style.borderTop = "1px solid var(--bdr)";

  const top = document.createElement("div");
  top.style.display = "flex";
  top.style.alignItems = "center";
  top.style.justifyContent = "space-between";
  top.style.gap = "var(--sp-2)";
  top.style.marginBottom = "var(--sp-2)";

  const left = pulseDot({
    state: info?.active ? "ok" : "fail",
    label: `${unit} · ${info?.enabled ? "enabled" : "disabled"}`,
  });
  top.append(left);
  row.append(top);

  const actions = document.createElement("div");
  actions.style.display = "flex";
  actions.style.gap = "var(--sp-2)";
  actions.append(_armableButton("Start", false, () => _runLoopAction(unit, "start")));
  actions.append(_armableButton("Restart", false, () => _runLoopAction(unit, "restart")));
  actions.append(_armableButton("Stop", true, () => _runLoopAction(unit, "stop")));
  row.append(actions);

  return row;
}

async function _runLoopAction(unit, action) {
  const r = await _post("/api/control/loop", { unit, action });
  _toastResult(r, r.ok ? `${unit}: ${action} ok` : null, r.data?.stderr);
  await _refetchState();
}

/**
 * A button that requires two taps to fire: first tap arms it ("tap again
 * to confirm", 3s window), second tap runs `onConfirm`. Used for every
 * loop start/stop/restart button — these are real systemd actions on a
 * public-facing route.
 */
function _armableButton(label, danger, onConfirm) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "icon-btn";
  btn.style.minWidth = "auto";
  btn.style.padding = "var(--sp-1) var(--sp-2)";
  btn.style.fontSize = "var(--ty-sm)";
  if (danger) {
    btn.style.color = "var(--red)";
    btn.style.borderColor = "var(--red)";
  }
  btn.textContent = label;

  let armed = false;
  let timer = null;
  const reset = () => {
    armed = false;
    if (timer) clearTimeout(timer);
    timer = null;
    btn.textContent = label;
  };

  btn.addEventListener("click", async () => {
    if (!armed) {
      armed = true;
      btn.textContent = "Tap again to confirm";
      timer = setTimeout(reset, 3000);
      return;
    }
    reset();
    btn.disabled = true;
    await onConfirm();
    btn.disabled = false;
  });

  return btn;
}

// ---------------------------------------------------------------------------
// POST helper + shared refresh/toast plumbing
// ---------------------------------------------------------------------------

/**
 * POST JSON and return { ok, status, data }. Unlike fetch.js's `fpost`,
 * this keeps the response body on non-2xx (400/429/502) so callers can
 * show a specific reason instead of a generic failure toast.
 */
async function _post(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => null);
    return { ok: r.ok, status: r.status, data };
  } catch (err) {
    console.warn("control post failed:", url, err);
    return { ok: false, status: 0, data: null };
  }
}

function _toastResult(r, successMessage, detail = "") {
  if (r.ok) {
    showToast(successMessage || "Done");
    return;
  }
  if (r.status === 429) {
    showToast("Rate limited — wait a moment and try again");
    return;
  }
  const reason = detail || r.data?.error || `HTTP ${r.status || "?"}`;
  showToast(`Failed: ${reason}`);
}

async function _refetchState() {
  // On failure, fj() has already surfaced the global error banner
  // (state.Slots.ERROR) — leave the last-known-good render in place
  // rather than blanking a working section over a transient fetch error.
  const d = await fj("/api/control/state");
  if (d) {
    _controlState = d;
    _renderLlm();
    _renderInstruments();
    _renderLoops();
  }
}
