/*
 * views/decision-detail.js — full single-decision drill view.
 *
 * Reached via /#decisions/<id> where <id> is the timestamp string.
 * Loads from state.DECISIONS first (fast back-from-list path), falls
 * back to a fresh /api/decisions fetch.
 */

import * as router from "../router.js";  // used for back-navigation
import * as state from "../state.js";
import { fj } from "../fetch.js";
import { fAgo, ftFull } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { errorBanner } from "../components/error-banner.js";

let _root = null;

export const view = {
  async mount(rootEl, params) {
    _root = rootEl;
    const id = typeof params === "string" ? params : (params?.id || "");
    while (_root.firstChild) _root.removeChild(_root.firstChild);

    _root.append(_renderHeader(id));
    _root.append(emptyState("Loading…"));

    const decision = await _findDecision(id);
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderHeader(id));

    if (!decision) {
      _root.append(errorBanner(`Decision "${id}" not found.`));
      return;
    }
    _root.append(_renderBody(decision));
  },
  unmount() {
    _root = null;
  },
};

// Note: decision-detail does NOT self-register with the router. It is
// invoked by views/decisions.js when /#decisions/<id> resolves with params.
// The list view delegates to this view's mount/unmount. Keeping a single
// registered name ("decisions") avoids hash ambiguity.

// ---------------------------------------------------------------------------

function _renderHeader(id) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.alignItems = "center";
  wrap.style.justifyContent = "space-between";
  wrap.style.marginBottom = "var(--sp-3)";

  const back = document.createElement("button");
  back.type = "button";
  back.className = "icon-btn";
  back.style.minWidth = "auto";
  back.style.padding = "var(--sp-1) var(--sp-2)";
  back.textContent = "← Decisions";
  back.addEventListener("click", () => router.navigate("decisions"));
  wrap.append(back);

  const t = document.createElement("div");
  t.className = "card__meta";
  t.textContent = id ? "id: " + id.slice(0, 19) : "";
  wrap.append(t);

  return wrap;
}

function _renderBody(d) {
  const body = document.createElement("section");

  // Top metadata card
  const meta = document.createElement("article");
  meta.className = "card";
  const tsLine = document.createElement("div");
  tsLine.className = "card__meta";
  tsLine.textContent = ftFull(d.ts || d.timestamp) + "  ·  " + fAgo(d.ts || d.timestamp);
  meta.append(tsLine);

  if (d.trigger) {
    const tr = document.createElement("div");
    tr.style.fontWeight = "600";
    tr.style.marginTop = "var(--sp-2)";
    tr.textContent = d.trigger;
    meta.append(tr);
  }
  if (d.regime) {
    const rg = document.createElement("div");
    rg.style.fontSize = "var(--ty-sm)";
    rg.style.color = "var(--txd)";
    rg.textContent = "Regime: " + d.regime;
    meta.append(rg);
  }
  if (d.reflection) {
    const rf = document.createElement("div");
    rf.style.marginTop = "var(--sp-2)";
    rf.style.fontSize = "var(--ty-sm)";
    rf.style.color = "var(--tx)";
    rf.style.fontStyle = "italic";
    rf.textContent = "“" + d.reflection + "”";
    meta.append(rf);
  }
  body.append(meta);

  // Two strategy decisions
  body.append(_strategyBlock("Patient", "var(--cyn)", d.decisions?.patient || d.patient));
  body.append(_strategyBlock("Bold",    "var(--org)", d.decisions?.bold    || d.bold));

  // Ticker outlooks
  if (d.tickers && typeof d.tickers === "object") {
    const card = document.createElement("article");
    card.className = "card";
    card.style.marginTop = "var(--sp-3)";
    const t = document.createElement("div");
    t.className = "card__title";
    t.textContent = "Ticker outlooks";
    card.append(t);
    for (const [tk, info] of Object.entries(d.tickers)) {
      const row = document.createElement("div");
      row.style.padding = "var(--sp-2) 0";
      row.style.borderBottom = "1px solid var(--bdr)";
      const top = document.createElement("div");
      top.style.display = "flex";
      top.style.justifyContent = "space-between";
      const tname = document.createElement("strong");
      tname.textContent = tk;
      const out = document.createElement("span");
      out.style.fontSize = "var(--ty-xs)";
      out.style.color = "var(--txm)";
      out.textContent = (info?.outlook || "—") + " · conv " + (info?.conviction ?? "?");
      top.append(tname, out);
      row.append(top);
      if (info?.thesis) {
        const th = document.createElement("div");
        th.style.fontSize = "var(--ty-sm)";
        th.style.color = "var(--txd)";
        th.style.marginTop = "var(--sp-1)";
        th.textContent = info.thesis;
        row.append(th);
      }
      if (info?.levels && Array.isArray(info.levels)) {
        const lv = document.createElement("div");
        lv.style.fontSize = "var(--ty-xs)";
        lv.style.color = "var(--txm)";
        lv.style.marginTop = "var(--sp-1)";
        lv.textContent = "Levels: " + info.levels.join(" / ");
        row.append(lv);
      }
      card.append(row);
    }
    body.append(card);
  }

  // Watchlist
  if (Array.isArray(d.watchlist) && d.watchlist.length) {
    const card = document.createElement("article");
    card.className = "card";
    card.style.marginTop = "var(--sp-3)";
    const t = document.createElement("div");
    t.className = "card__title";
    t.textContent = "Watchlist";
    card.append(t);
    const ul = document.createElement("ul");
    ul.style.paddingLeft = "var(--sp-4)";
    ul.style.marginTop = "var(--sp-2)";
    d.watchlist.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = String(item);
      li.style.fontSize = "var(--ty-sm)";
      li.style.color = "var(--txd)";
      li.style.padding = "var(--sp-1) 0";
      ul.append(li);
    });
    card.append(ul);
    body.append(card);
  }

  return body;
}

function _strategyBlock(label, color, info) {
  const card = document.createElement("article");
  card.className = "card";
  card.style.marginTop = "var(--sp-3)";
  card.style.borderLeft = `3px solid ${color}`;

  const head = document.createElement("div");
  head.className = "card__header";
  const t = document.createElement("div");
  t.className = "card__title";
  t.style.color = color;
  t.textContent = label;
  head.append(t);
  if (info?.action) {
    const badge = document.createElement("span");
    badge.className = "badge " + _badgeClass(info.action);
    badge.textContent = info.action;
    head.append(badge);
  }
  card.append(head);

  if (info?.reasoning) {
    const r = document.createElement("p");
    r.style.fontSize = "var(--ty-sm)";
    r.style.color = "var(--tx)";
    r.style.lineHeight = "1.6";
    r.style.marginTop = "var(--sp-2)";
    r.textContent = info.reasoning;
    card.append(r);
  }
  return card;
}

function _badgeClass(action) {
  switch ((action || "").toUpperCase()) {
    case "BUY": case "STRONG_BUY":   return "badge--BUY";
    case "SELL": case "STRONG_SELL": return "badge--SELL";
    default:                          return "badge--HOLD";
  }
}

async function _findDecision(id) {
  const cached = state.get(state.Slots.DECISIONS);
  const arr = Array.isArray(cached) ? cached : (cached?.decisions || []);
  const hit = arr.find((d) => String(d.ts || d.timestamp) === id);
  if (hit) return hit;
  // Fallback fetch
  const data = await fj(`/api/decisions?limit=500`, { ttl: 10_000 });
  const all = Array.isArray(data) ? data : (data?.decisions || []);
  return all.find((d) => String(d.ts || d.timestamp) === id) || null;
}
