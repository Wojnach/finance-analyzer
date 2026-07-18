/*
 * render/silver-trade.js — #silver page "Trade panel" (Phase 6).
 * metals_loop heartbeat, open XAG positions (metals context + warrants
 * holdings), Grid Fisher's XAG instrument ladders, last 3 metals decisions.
 */

import { fAgo, fs, ftFull, num } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { pulseDot } from "../components/pulse-dot.js";
import { positionCard } from "../components/position-card.js";

const TICKER = "XAG-USD";

/**
 * @param {{cs: object|null, sys: object|null, metals: object|null,
 *   warrants: object|null, gridFisher: object|null}} props
 * @returns {HTMLElement}
 */
export function silverTradePanel({
  cs,
  sys,
  metals,
  warrants,
  gridFisher,
} = {}) {
  const card = document.createElement("article");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const loop = cs?.loops?.["pf-metalsloop"];
  const hbAge = sys?.sources?.["metals_loop.heartbeat"]?.age_sec;
  card.append(
    pulseDot({
      state: loop?.active ? "ok" : "fail",
      label:
        `metals loop · ${loop?.enabled ? "enabled" : "disabled"}` +
        (hbAge != null
          ? ` · heartbeat ${fAgo(new Date(Date.now() - hbAge * 1000))}`
          : ""),
    }),
  );

  card.append(_subtitle("Open XAG positions"));
  card.append(_positionsSection(metals, warrants));

  card.append(_subtitle("Grid Fisher — XAG instruments"));
  card.append(_gridFisherSection(gridFisher));

  card.append(_subtitle("Last metals decisions"));
  card.append(_decisionsSection(metals));

  return card;
}

function _subtitle(text) {
  const el = document.createElement("div");
  el.className = "card__subtitle";
  el.style.marginTop = "var(--sp-3)";
  el.textContent = text;
  return el;
}

function _positionsSection(metals, warrants) {
  const positions = Object.values(metals?.context?.positions || {}).filter(
    (p) => /silver|xag/i.test(p.name || p.ticker || ""),
  );
  const warrantHoldings = Object.values(warrants?.holdings || {}).filter((p) =>
    /silver|xag/i.test(p.name || p.ticker || p.warrant || ""),
  );
  const all = [...positions, ...warrantHoldings];

  if (!all.length) return emptyState("No open XAG positions.");

  const strip = document.createElement("div");
  strip.className = "scroll-strip";
  for (const p of all) {
    strip.append(
      positionCard({
        ticker: p.name || p.ticker || p.warrant || "warrant",
        side: (p.leverage_direction || "LONG").toUpperCase(),
        pnlPct: num(p.pnl_pct ?? p.pnl_percent),
        pricePerUnit: num(p.bid ?? p.last_price ?? p.value_per_unit),
      }),
    );
  }
  return strip;
}

function _gridFisherSection(gridFisher) {
  const xagInstruments = Object.values(
    gridFisher?.state?.by_instrument || {},
  ).filter((i) => i.ticker === TICKER);
  if (!xagInstruments.length) return emptyState("No XAG instruments armed.");

  const wrap = document.createElement("div");
  for (const inst of xagInstruments) {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.justifyContent = "space-between";
    row.style.fontSize = "var(--ty-sm)";
    row.style.padding = "var(--sp-1) 0";
    row.style.borderTop = "1px solid var(--bdr)";
    const left = document.createElement("span");
    left.textContent = inst.cert_name || inst.ob_id;
    const right = document.createElement("span");
    right.style.color = "var(--txm)";
    right.textContent = `${inst.active_direction || "?"} · ${inst.inventory_units ?? 0}u · ${fs(inst.session_pnl_sek ?? 0)}`;
    row.append(left, right);
    wrap.append(row);
  }
  return wrap;
}

function _decisionsSection(metals) {
  const decisions = Array.isArray(metals?.decisions)
    ? metals.decisions.slice(-3).reverse()
    : [];
  if (!decisions.length) return emptyState("No recent metals decisions.");

  const wrap = document.createElement("div");
  for (const d of decisions) {
    const row = document.createElement("div");
    row.style.padding = "var(--sp-1) 0";
    row.style.borderTop = "1px solid var(--bdr)";
    row.style.fontSize = "var(--ty-sm)";

    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.fontSize = "var(--ty-xs)";
    top.style.color = "var(--txm)";
    const xagVote = d?.llm?.[TICKER]?.consensus || d?.signals?.[TICKER]?.action;
    top.textContent = `${ftFull(d.ts || d.timestamp)}${xagVote ? " · XAG: " + xagVote : ""}`;
    row.append(top);

    const body = document.createElement("div");
    body.textContent = d.action || d.reasoning || "—";
    row.append(body);

    wrap.append(row);
  }
  return wrap;
}
