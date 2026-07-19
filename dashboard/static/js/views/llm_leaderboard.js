/*
 * views/llm_leaderboard.js — per-LLM scorecard (accuracy + Brier + status).
 *
 * Reads /api/llm-leaderboard. Sorts by accuracy desc, dimming rows with
 * <30 matched outcomes so the eye lands on signals with enough samples to
 * be meaningful. Status chip on the right (shadow/promoted/retired) plus
 * promotion-criteria hint when the signal is within reach of promotion.
 *
 * Security: every field is rendered via textContent / dataset; no
 * innerHTML interpolation. The endpoint already JSON-encodes everything
 * but defense-in-depth is cheap.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { emptyState } from "../components/empty-state.js";

const POLL_KEY = "llm_leaderboard";
const SLOT = "llm_leaderboard";

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    // No Slots registration needed: state.set/get/subscribe key on plain
    // strings (see views/silver.js SLOT_* pattern). The old attempt to
    // write into state.Slots threw "object is not extensible" in strict
    // mode — Slots is Object.freeze'd — killing the whole view mount
    // (2026-07-19).
    _unsubs.push(state.subscribe(SLOT, _renderBody));

    // 5min cadence — endpoint is 5min-cached server-side, no point
    // polling faster than the cache TTL.
    polling.register(POLL_KEY, 300_000, async () => {
      const data = await fj("/api/llm-leaderboard");
      if (data) state.set(SLOT, data);
      return data != null;
    });
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    _root = null;
  },
};
router.register("llm-leaderboard", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--llm-leaderboard";

  const title = document.createElement("h1");
  title.className = "section-title";
  title.textContent = "LLM leaderboard";
  v.append(title);

  const sub = document.createElement("p");
  sub.className = "section-subtitle";
  sub.textContent =
    "Per-LLM accuracy + Brier joined with shadow-registry status. " +
    "Rows with <30 matched outcomes are dimmed — too few samples to trust.";
  sub.style.color = "var(--txm)";
  sub.style.fontSize = "13px";
  sub.style.marginTop = "calc(-1 * var(--sp-2))";
  sub.style.marginBottom = "var(--sp-3)";
  v.append(sub);

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
    slot.append(emptyState("Loading leaderboard…"));
    return;
  }

  const signals = Array.isArray(data.signals) ? data.signals.slice() : [];
  if (!signals.length) {
    slot.append(emptyState("No signals registered."));
    return;
  }

  // Sort: highest accuracy first, then by sample count for ties. Signals
  // with null accuracy (no matched outcomes yet) sink to the bottom.
  signals.sort((a, b) => {
    const aAcc = a.accuracy ?? -Infinity;
    const bAcc = b.accuracy ?? -Infinity;
    if (bAcc !== aAcc) return bAcc - aAcc;
    return (b.n_with_outcome ?? 0) - (a.n_with_outcome ?? 0);
  });

  slot.append(_buildTable(signals));

  if (data.updated_ts) {
    const stamp = document.createElement("div");
    stamp.style.color = "var(--txm)";
    stamp.style.fontSize = "11px";
    stamp.style.marginTop = "var(--sp-2)";
    stamp.textContent = `Updated: ${data.updated_ts}`;
    slot.append(stamp);
  }
}

function _buildTable(signals) {
  const wrap = document.createElement("div");
  wrap.className = "card";
  wrap.style.padding = "var(--sp-2)";
  wrap.style.overflowX = "auto";

  const table = document.createElement("table");
  table.style.width = "100%";
  table.style.borderCollapse = "collapse";
  table.style.fontSize = "13px";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Signal", "Status", "Acc", "Brier", "Matched", "Days"]) {
    const th = document.createElement("th");
    th.textContent = label;
    th.style.textAlign = label === "Signal" ? "left" : "right";
    th.style.padding = "6px 8px";
    th.style.borderBottom = "1px solid var(--brd)";
    th.style.color = "var(--txm)";
    th.style.fontWeight = "600";
    headRow.append(th);
  }
  thead.append(headRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  for (const s of signals) {
    tbody.append(_buildRow(s));
  }
  table.append(tbody);
  wrap.append(table);
  return wrap;
}

function _buildRow(s) {
  const tr = document.createElement("tr");
  const dim = (s.n_with_outcome ?? 0) < 30;
  if (dim) tr.style.opacity = "0.55";

  const tdName = document.createElement("td");
  tdName.textContent = s.name ?? "—";
  tdName.style.padding = "6px 8px";
  tdName.style.fontWeight = "600";
  tr.append(tdName);

  const tdStatus = document.createElement("td");
  tdStatus.append(_statusChip(s.status));
  tdStatus.style.padding = "6px 8px";
  tdStatus.style.textAlign = "right";
  tr.append(tdStatus);

  const tdAcc = document.createElement("td");
  tdAcc.textContent = s.accuracy == null ? "—" : `${(s.accuracy * 100).toFixed(1)}%`;
  tdAcc.style.padding = "6px 8px";
  tdAcc.style.textAlign = "right";
  if (s.accuracy != null) tdAcc.style.color = _accColor(s.accuracy);
  tr.append(tdAcc);

  const tdBrier = document.createElement("td");
  tdBrier.textContent = s.brier == null ? "—" : s.brier.toFixed(3);
  tdBrier.style.padding = "6px 8px";
  tdBrier.style.textAlign = "right";
  tdBrier.style.color = "var(--txm)";
  tr.append(tdBrier);

  const tdMatched = document.createElement("td");
  const n = s.n_with_outcome ?? 0;
  const nTot = s.n_samples ?? 0;
  tdMatched.textContent = `${n}/${nTot}`;
  tdMatched.style.padding = "6px 8px";
  tdMatched.style.textAlign = "right";
  tdMatched.style.color = "var(--txm)";
  tr.append(tdMatched);

  const tdDays = document.createElement("td");
  tdDays.textContent = s.days_in_shadow == null ? "—" : `${s.days_in_shadow}d`;
  tdDays.style.padding = "6px 8px";
  tdDays.style.textAlign = "right";
  tdDays.style.color = "var(--txm)";
  tr.append(tdDays);

  return tr;
}

function _statusChip(status) {
  const chip = document.createElement("span");
  chip.textContent = status || "—";
  chip.style.padding = "2px 6px";
  chip.style.borderRadius = "4px";
  chip.style.fontSize = "11px";
  chip.style.fontWeight = "600";
  chip.style.textTransform = "uppercase";
  if (status === "promoted") {
    chip.style.background = "var(--grn-bg, rgba(40, 180, 99, 0.18))";
    chip.style.color = "var(--grn, #28b463)";
  } else if (status === "retired") {
    chip.style.background = "var(--red-bg, rgba(231, 76, 60, 0.18))";
    chip.style.color = "var(--red, #e74c3c)";
  } else if (status === "shadow") {
    chip.style.background = "var(--ylw-bg, rgba(241, 196, 15, 0.18))";
    chip.style.color = "var(--ylw, #f1c40f)";
  } else {
    chip.style.background = "var(--bg2)";
    chip.style.color = "var(--txm)";
  }
  return chip;
}

function _accColor(acc) {
  if (acc >= 0.55) return "var(--grn, #28b463)";
  if (acc >= 0.47) return "var(--ylw, #f1c40f)";
  return "var(--red, #e74c3c)";
}
