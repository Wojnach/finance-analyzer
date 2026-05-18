/*
 * views/pickups.js — scheduled-pickup status view.
 *
 * Shows every entry from /api/pickups (data/pending_pickups.json).
 * Each pickup is a one-shot verification job auto-run by
 * scripts/process_pending_pickups.py via the PF-PendingPickups
 * Windows scheduled task (daily 08:00 CET).
 *
 * Rows are sorted server-side by days_until_due ascending: overdue
 * pickups first (negative delta), upcoming last. Status is colour-
 * coded:
 *   red    = overdue + pending      (cron hasn't picked it up)
 *   yellow = due today              (pending, days_until_due <= 1)
 *   blue   = future                 (pending, days_until_due > 1)
 *   green  = completed
 *   grey   = error                  (handler returned verdict=error)
 *
 * All DOM nodes are built via createElement+textContent. innerHTML is
 * avoided so that a corrupted data/pending_pickups.json containing
 * HTML entities in `title`, `id`, or `last_verdict` cannot reach the
 * page renderer (XSS hardening, even though the JSON is trusted).
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";

const POLL_KEY = "pickups";
const SLOT = "pickups";
const POLL_MS = 60_000;

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());
    _unsubs.push(state.subscribe(SLOT, _renderBody));
    polling.register(POLL_KEY, POLL_MS, async () => {
      const data = await fj("/api/pickups");
      if (data) state.set(SLOT, data);
    });
  },
  unmount() {
    polling.unregister(POLL_KEY);
    _unsubs.forEach((u) => u());
    _unsubs = [];
    _root = null;
  },
};

function _el(tag, opts = {}, children = []) {
  const node = document.createElement(tag);
  if (opts.className) node.className = opts.className;
  if (opts.text !== undefined) node.textContent = opts.text;
  for (const c of children) {
    if (c) node.appendChild(c);
  }
  return node;
}

function _renderShell() {
  const wrap = _el("div", { className: "view" });
  const header = _el("header");
  header.appendChild(_el("h1", { text: "Scheduled pickups" }));
  header.appendChild(
    _el("p", {
      className: "sub",
      text: "Auto-run verification jobs (PF-PendingPickups daily 08:00 CET).",
    }),
  );
  wrap.appendChild(header);
  const body = _el("div");
  body.id = "pickups-body";
  body.appendChild(_el("em", { text: "Loading…" }));
  wrap.appendChild(body);
  return wrap;
}

function _renderBody(data) {
  const body = _root?.querySelector("#pickups-body");
  if (!body) return;
  while (body.firstChild) body.removeChild(body.firstChild);
  if (!data || !Array.isArray(data.pickups) || data.pickups.length === 0) {
    body.appendChild(_el("em", { text: "No pickups scheduled." }));
    return;
  }

  const table = _el("table", { className: "data-table" });
  const thead = _el("thead");
  const headRow = _el("tr");
  for (const t of ["ID", "Title", "Due", "Δ", "Status", "Last run"]) {
    headRow.appendChild(_el("th", { text: t }));
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = _el("tbody");
  for (const p of data.pickups) {
    const cls = _classify(p);
    const row = _el("tr", { className: `row-${cls}` });

    const idCell = _el("td");
    idCell.appendChild(_el("code", { text: p.id ?? "?" }));
    row.appendChild(idCell);

    row.appendChild(_el("td", { text: p.title ?? "" }));
    row.appendChild(_el("td", {
      text: p.due_ts ? new Date(p.due_ts).toLocaleString() : "—",
    }));

    const delta = p.days_until_due == null
      ? "—"
      : (p.days_until_due >= 0
          ? `in ${p.days_until_due.toFixed(1)}d`
          : `${(-p.days_until_due).toFixed(1)}d overdue`);
    row.appendChild(_el("td", {
      className: cls === "overdue" ? "warn" : "",
      text: delta,
    }));

    const statusCell = _el("td", { text: (p.status ?? "?") + " " });
    if (p.last_verdict) {
      statusCell.appendChild(_el("span", {
        className: `badge badge-${p.last_verdict}`,
        text: p.last_verdict,
      }));
    }
    row.appendChild(statusCell);

    row.appendChild(_el("td", {
      text: p.last_run_ts ? new Date(p.last_run_ts).toLocaleString() : "—",
    }));

    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  body.appendChild(table);

  body.appendChild(_el("p", {
    className: "sub",
    text:
      "Source: data/pending_pickups.json. Add new pickups by editing the JSON + " +
      "whitelisting the handler in scripts/process_pending_pickups.py:_HANDLERS.",
  }));
}

function _classify(p) {
  if (p.status === "error") return "error";
  if (p.status === "completed") return "completed";
  if (p.days_until_due == null) return "future";
  if (p.days_until_due < 0) return "overdue";
  if (p.days_until_due <= 1) return "today";
  return "future";
}

router.register("pickups", view);
