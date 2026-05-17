/*
 * views/loop_processes.js — running-loop duplicate-detection view.
 *
 * Replaces the visual cue the user lost when scheduled-task popup
 * windows were hidden via run-hidden.vbs. Shows one row per known
 * loop with PID count + uptime + a colour-coded badge:
 *   green  = exactly 1 PID running (healthy)
 *   grey   = 0 PIDs (not running — may be intentional, e.g. DRY_RUN
 *            loops not yet started)
 *   red    = 2+ PIDs (duplicate — race risk on shared state)
 *
 * Backed by /api/loop-processes. See portfolio/loop_processes.py.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { fAgo } from "../format.js";
import { emptyState } from "../components/empty-state.js";

const POLL_KEY = "loop_processes";
const SLOT = "loopProcesses";   // ad-hoc slot — state.js Slots not extended for this view
const POLL_MS = 30_000;

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(SLOT, _renderBody));

    polling.register(POLL_KEY, POLL_MS, async () => {
      const data = await fj("/api/loop-processes");
      if (data) state.set(SLOT, data);
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("loop-processes", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--loop-processes";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "Running loops";
  v.append(title);

  const hint = document.createElement("div");
  hint.className = "card__subtitle";
  hint.style.marginBottom = "var(--sp-2)";
  hint.textContent = "Replaces the popup-window visual cue. Red = duplicate (race risk).";
  v.append(hint);

  const body = document.createElement("div");
  body.dataset.slot = "body";
  v.append(body);
  return v;
}

function _renderBody() {
  if (!_root) return;
  const slot = _root.querySelector('[data-slot="body"]');
  if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const data = state.get(SLOT);
  if (!data) {
    slot.append(emptyState("Loading running-loop data…"));
    return;
  }
  if (data.error) {
    slot.append(emptyState(`Endpoint error: ${data.error}`));
    return;
  }

  const loops = Array.isArray(data.loops) ? data.loops : [];
  if (!loops.length) {
    slot.append(emptyState("No loops registered."));
    return;
  }

  // Rollup banner on top so glance-detection works without scrolling.
  const banner = document.createElement("article");
  banner.className = "card";
  banner.style.marginBottom = "var(--sp-3)";
  banner.style.background = data.any_duplicate ? "rgba(255,68,68,0.12)" : "rgba(0,200,0,0.08)";
  banner.style.borderColor = data.any_duplicate ? "rgba(255,68,68,0.35)" : "rgba(0,200,0,0.25)";
  const bigLabel = document.createElement("div");
  bigLabel.className = "card__title";
  bigLabel.textContent = data.any_duplicate ? "DUPLICATE LOOPS DETECTED" : "No duplicates";
  bigLabel.style.color = data.any_duplicate ? "var(--red)" : "var(--grn)";
  const sub = document.createElement("div");
  sub.className = "card__subtitle";
  const dupNames = loops.filter(L => L.duplicate).map(L => L.name);
  sub.textContent = data.any_duplicate
    ? `Duplicates: ${dupNames.join(", ")} — run scripts/win/verify-tasks.ps1 + kill orphan PIDs (see docs/HIDDEN_TASKS.md)`
    : `${loops.filter(L => L.count > 0).length}/${loops.length} loops running.`;
  banner.append(bigLabel, sub);
  slot.append(banner);

  // Per-loop rows.
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  wrap.style.gap = "var(--sp-1)";
  for (const L of loops) {
    wrap.append(_loopRow(L));
  }
  slot.append(wrap);
}

function _loopRow(L) {
  const row = document.createElement("article");
  row.className = "card";
  row.style.display = "grid";
  row.style.gridTemplateColumns = "1fr auto auto auto";
  row.style.alignItems = "center";
  row.style.gap = "var(--sp-2)";

  const name = document.createElement("div");
  const n = document.createElement("div");
  n.className = "card__title";
  n.textContent = L.name;
  const p = document.createElement("div");
  p.className = "card__subtitle";
  p.style.fontFamily = "monospace";
  p.textContent = L.pattern;
  name.append(n, p);

  const count = document.createElement("div");
  count.className = "num num--md";
  count.textContent = String(L.count);
  if (L.duplicate) {
    count.style.color = "var(--red)";
  } else if (L.count === 0) {
    count.style.color = "var(--txm)";
  } else {
    count.style.color = "var(--grn)";
  }

  const uptime = document.createElement("div");
  uptime.className = "card__subtitle";
  if (L.oldest_uptime_seconds != null) {
    // fAgo expects an ISO string, not seconds — feed it the start time.
    const startIso = (L.process_started_at && L.process_started_at[0]) || null;
    uptime.textContent = startIso ? `up ${fAgo(startIso)}` : `up ${L.oldest_uptime_seconds}s`;
  } else {
    uptime.textContent = "—";
  }

  const pids = document.createElement("div");
  pids.className = "card__subtitle";
  pids.style.fontFamily = "monospace";
  pids.textContent = L.pids.length ? `pid ${L.pids.join(",")}` : "";

  row.append(name, count, uptime, pids);
  return row;
}
