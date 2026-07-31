/*
 * views/swedbank.js — externally-custodied share accounts, monitoring only.
 *
 * Reads /api/swedbank, which serves the snapshot written by
 * data/swedbank_loop.py. The endpoint never prices on demand: the Avanza
 * session is shared with the real-money metals loop, so a page refresh must
 * not be able to contend for it. The consequence is that what you see can be
 * stale, so snapshot age is rendered prominently rather than implied away.
 *
 * Honesty rules this view enforces:
 *   - Snapshot age is always shown, and turns amber/red as it grows.
 *   - A position priced from a degraded source (Alpaca fallback, or a cached
 *     last-good price) is labelled as such; it is never drawn like a live one.
 *   - A position marked at bid/ask mid because `last` was stale shows "mid".
 *   - Spread is a first-class column. The operator executes these by hand, so
 *     spread is a real personal cost, not a footnote.
 *   - Positions with no price are listed separately and excluded from totals,
 *     because a total that quietly omits a holding is worse than a visible gap.
 *
 * Orderbook IDs are shown as plain text, NOT as links. Certificates, warrants
 * and equities live under different Avanza URL paths, and a deep link that
 * resolves to the wrong product would send a manual order to the wrong
 * security. Showing the ID is unambiguous; guessing a URL is not.
 *
 * All nodes are built with createElement + textContent; no innerHTML.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";

const POLL_KEY = "swedbank";
const SLOT = "swedbank";
const POLL_MS = 30_000;

const STALE_WARN_S = 180;
const STALE_BAD_S = 900;
const WIDE_SPREAD_PCT = 0.5;

let _root = null;
let _unsubs = [];

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());
    _unsubs.push(state.subscribe(SLOT, _renderBody));
    polling.register(POLL_KEY, POLL_MS, async () => {
      const data = await fj("/api/swedbank");
      if (data) state.set(SLOT, data);
      return data != null;
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
  if (opts.title) node.title = opts.title;
  for (const c of children) if (c) node.appendChild(c);
  return node;
}

function _num(x, dp = 2) {
  if (x == null || Number.isNaN(x)) return "—";
  return x.toLocaleString("sv-SE", {
    minimumFractionDigits: dp,
    maximumFractionDigits: dp,
  });
}

function _signed(x, dp = 2) {
  if (x == null) return "—";
  return (x >= 0 ? "+" : "") + _num(x, dp);
}

function _age(s) {
  if (s == null) return "—";
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function _renderShell() {
  const wrap = _el("div", { className: "view" });
  const header = _el("header");
  header.appendChild(_el("h1", { text: "Swedbank book" }));
  header.appendChild(
    _el("p", {
      className: "sub",
      text:
        "Externally-custodied share accounts. Monitoring only — this system " +
        "never places orders; you execute manually on Avanza.",
    }),
  );
  wrap.appendChild(header);
  const body = _el("div");
  body.id = "swedbank-body";
  body.appendChild(_el("em", { text: "Loading…" }));
  wrap.appendChild(body);
  return wrap;
}

function _freshnessBanner(data) {
  const age = data.snapshot_age_s;
  const loopUp = data.loop && data.loop.running;
  let cls = "ok";
  let msg = `Snapshot ${_age(age)} old.`;
  if (age == null || age > STALE_BAD_S) {
    cls = "bad";
    msg = `Snapshot ${_age(age)} old — prices are NOT live.`;
  } else if (age > STALE_WARN_S) {
    cls = "warn";
    msg = `Snapshot ${_age(age)} old — refreshing may be stalled.`;
  }
  if (!loopUp) {
    cls = cls === "ok" ? "warn" : cls;
    msg +=
      " Monitoring loop is not running (systemctl --user start pf-swedbank).";
  }
  return _el("p", { className: `banner ${cls}`, text: msg });
}

function _flagsFor(r) {
  const f = [];
  if (r.stale_last) f.push("mid");
  if (r.degraded) f.push("degraded");
  if (r.spread_pct != null && r.spread_pct > WIDE_SPREAD_PCT) {
    f.push(`spread ${r.spread_pct.toFixed(2)}%`);
  }
  return f.join(" · ");
}

function _holdingsTable(rows, currency) {
  const table = _el("table", { className: "data-table" });
  const thead = _el("thead");
  const hr = _el("tr");
  for (const h of [
    "Instrument",
    "Qty",
    "Mark",
    "Spread",
    "Age",
    `Value ${currency}`,
    "P&L",
    "P&L %",
    "Flags",
    "OB",
  ]) {
    hr.appendChild(_el("th", { text: h }));
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = _el("tbody");
  for (const r of rows) {
    const tr = _el("tr");
    tr.appendChild(_el("td", { text: r.name, title: r.key }));
    tr.appendChild(_el("td", { text: String(r.qty) }));
    tr.appendChild(
      _el("td", {
        text: _num(r.mark),
        title:
          r.mark_basis === "mid"
            ? "marked at bid/ask mid: last was stale"
            : "last",
      }),
    );
    tr.appendChild(
      _el("td", {
        text: r.spread_pct == null ? "—" : `${r.spread_pct.toFixed(2)}%`,
      }),
    );
    tr.appendChild(_el("td", { text: _age(r.age_s), title: r.source || "" }));
    tr.appendChild(_el("td", { text: _num(r.value) }));
    const pnl = _el("td", { text: _signed(r.pnl) });
    if (r.pnl != null) pnl.className = r.pnl >= 0 ? "pos" : "neg";
    tr.appendChild(pnl);
    const pct = _el("td", {
      text: r.pnl_pct == null ? "—" : `${_signed(r.pnl_pct, 1)}%`,
    });
    if (r.pnl_pct != null) pct.className = r.pnl_pct >= 0 ? "pos" : "neg";
    tr.appendChild(pct);
    tr.appendChild(_el("td", { className: "sub", text: _flagsFor(r) }));
    tr.appendChild(_el("td", { className: "sub", text: r.avanza_ob || "—" }));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  return table;
}

function _totalsBlock(t, currency) {
  const dl = _el("div", { className: "stat-row" });
  const add = (label, value, cls) => {
    const box = _el("div", { className: "stat" });
    box.appendChild(_el("div", { className: "stat-label", text: label }));
    const v = _el("div", { className: `stat-value ${cls || ""}`, text: value });
    box.appendChild(v);
    dl.appendChild(box);
  };
  add("Value", `${_num(t.total_value)} ${currency}`);
  add("Cost", _num(t.cost_basis));
  add(
    "Unrealized",
    t.pnl == null ? "—" : _signed(t.pnl),
    t.pnl == null ? "" : t.pnl >= 0 ? "pos" : "neg",
  );
  add(
    "Unrealized %",
    t.pnl_pct == null ? "—" : `${_signed(t.pnl_pct, 2)}%`,
    t.pnl_pct == null ? "" : t.pnl_pct >= 0 ? "pos" : "neg",
  );
  add("Cash", _num(t.cash));
  return dl;
}

function _renderBody(data) {
  const body = _root?.querySelector("#swedbank-body");
  if (!body) return;
  while (body.firstChild) body.removeChild(body.firstChild);

  if (!data) {
    body.appendChild(_el("em", { text: "Loading…" }));
    return;
  }
  if (data.error) {
    body.appendChild(_el("p", { className: "banner bad", text: data.error }));
    return;
  }
  if (data.available === false) {
    body.appendChild(
      _el("p", {
        className: "banner warn",
        text: data.reason || "No snapshot.",
      }),
    );
    return;
  }

  const currency = data.base_currency || "SEK";
  body.appendChild(_freshnessBanner(data));
  body.appendChild(_el("h2", { text: "All accounts" }));
  body.appendChild(_totalsBlock(data.total || {}, currency));

  if (data.fx && Object.keys(data.fx).length) {
    const pairs = Object.entries(data.fx)
      .map(([k, v]) => `${k} ${_num(v, 4)}`)
      .join("  ·  ");
    body.appendChild(_el("p", { className: "sub", text: `FX: ${pairs}` }));
  }

  const unpriced = data.unpriced || [];
  if (unpriced.length) {
    body.appendChild(
      _el("p", {
        className: "banner bad",
        text:
          `${unpriced.length} position(s) have no price and are EXCLUDED from the ` +
          `totals above: ${unpriced.map((u) => `${u[0]}/${u[1]}`).join(", ")}`,
      }),
    );
  }

  body.appendChild(_el("h2", { text: "Consolidated" }));
  body.appendChild(_holdingsTable(data.consolidated || [], currency));

  for (const [label, acc] of Object.entries(data.accounts || {})) {
    body.appendChild(_el("h2", { text: label }));
    body.appendChild(
      _totalsBlock({ ...acc, total_value: acc.total_value }, currency),
    );
    body.appendChild(_holdingsTable(acc.holdings || [], currency));
  }

  body.appendChild(
    _el("p", {
      className: "sub",
      text:
        "Prices via Avanza (real-time), Alpaca fallback for US names when the " +
        "Avanza session is down. Rows flagged 'mid' are marked at bid/ask mid " +
        "because the last print was stale. Orderbook IDs are shown as text, not " +
        "links, so a wrong URL can never route a manual order to the wrong product.",
    }),
  );
}

router.register("swedbank", view);
