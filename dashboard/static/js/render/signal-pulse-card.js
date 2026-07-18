/*
 * render/signal-pulse-card.js — at-a-glance signal abstain/vote rate.
 *
 * Reads `/api/system_status` payload's `signal_aggregate` section. The
 * goal is the user's question: "if most of the signals are abstaining
 * and holding, surface that". For each Tier-1 ticker we show:
 *   - Consensus action badge (BUY/SELL/HOLD)
 *   - Vote breakdown: e.g. "5B / 2S / 43H · 75% abstain"
 *
 * Tap a row → /signals/<ticker> for full per-signal drill-down.
 */

import * as router from "../router.js";
import { sectionErrorChip } from "../components/section-error-chip.js";

const ACTION_COLORS = {
  BUY: "var(--grn)",
  SELL: "var(--red)",
  HOLD: "var(--txm)",
};

/** @returns {HTMLElement} */
export function signalPulseCard(signalAgg) {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0 0 var(--sp-2) 0";
  title.textContent = "Signal pulse";
  card.append(title);

  if (signalAgg?.error) card.append(sectionErrorChip(signalAgg.error));

  const tickers = Array.isArray(signalAgg?.tickers) ? signalAgg.tickers : [];
  if (!tickers.length) {
    if (!signalAgg?.error) {
      const empty = document.createElement("div");
      empty.style.color = "var(--txm)";
      empty.style.fontSize = "var(--ty-sm)";
      empty.textContent = "no signal snapshot yet";
      card.append(empty);
    }
    return card;
  }

  for (const t of tickers) card.append(_tickerRow(t));
  return card;
}

function _tickerRow(t) {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "card--tap";
  row.style.display = "flex";
  row.style.justifyContent = "space-between";
  row.style.alignItems = "center";
  row.style.gap = "var(--sp-2)";
  row.style.width = "100%";
  row.style.padding = "var(--sp-2) 0";
  row.style.background = "transparent";
  row.style.border = "0";
  row.style.borderTop = "1px solid var(--bd)";
  row.style.minHeight = "var(--tap-min)";
  row.style.textAlign = "left";
  row.style.cursor = "pointer";
  row.addEventListener("click", () => router.navigate("signals", t.ticker));

  const left = document.createElement("div");
  left.style.flex = "1";
  left.style.minWidth = "0";

  const tickerRow = document.createElement("div");
  tickerRow.style.display = "flex";
  tickerRow.style.alignItems = "center";
  tickerRow.style.gap = "6px";
  const ticker = document.createElement("span");
  ticker.style.fontWeight = "600";
  ticker.style.color = "var(--tx)";
  ticker.textContent = t.ticker;
  tickerRow.append(ticker);

  // 2026-07-18: XAG is the user's stated main instrument focus — a direct
  // link to its dedicated command page (#silver), separate from the
  // generic #signals drill-down the rest of the row triggers on tap.
  if (t.ticker === "XAG-USD") {
    const link = document.createElement("span");
    link.style.fontSize = "var(--ty-xs)";
    link.style.fontWeight = "700";
    link.style.color = "var(--blu)";
    link.textContent = "Silver →";
    link.addEventListener("click", (e) => {
      e.stopPropagation();
      router.navigate("silver");
    });
    tickerRow.append(link);
  }
  left.append(tickerRow);

  const detail = document.createElement("div");
  detail.style.display = "flex";
  detail.style.alignItems = "center";
  detail.style.gap = "6px";
  detail.style.fontSize = "var(--ty-sm)";
  detail.style.color = "var(--txm)";

  // horizon = vote target timeframe (2026-07-18); older snapshots lack it.
  // Broken out into its own pill (rather than inline text) so it reads as
  // a distinct dimension, not just another number in the B/S/H tally.
  const horizon = t.horizon || "1d (default)";
  const hPill = document.createElement("span");
  hPill.textContent = horizon;
  hPill.style.fontSize = "var(--ty-xs)";
  hPill.style.fontWeight = "600";
  hPill.style.padding = "0 4px";
  hPill.style.border = "1px solid var(--bdr2)";
  hPill.style.borderRadius = "3px";
  hPill.style.color = "var(--txd)";
  hPill.style.whiteSpace = "nowrap";
  detail.append(hPill);

  const total = t.total ?? 0;
  const buy = t.buy ?? 0;
  const sell = t.sell ?? 0;
  const hold = t.hold ?? 0;
  const abstainPct = total > 0 ? Math.round((hold / total) * 100) : 0;
  const rest = document.createElement("span");
  rest.textContent = `${buy}B · ${sell}S · ${hold}H — ${abstainPct}% abstaining`;
  detail.append(rest);

  left.append(detail);
  row.append(left);

  const action = (t.consensus || "HOLD").toUpperCase();
  const badge = document.createElement("span");
  badge.textContent = action;
  badge.style.fontSize = "var(--ty-xs)";
  badge.style.fontWeight = "700";
  badge.style.padding = "2px var(--sp-2)";
  badge.style.borderRadius = "4px";
  badge.style.background = ACTION_COLORS[action] || "var(--txm)";
  badge.style.color = "var(--bg)";
  badge.style.minWidth = "44px";
  badge.style.textAlign = "center";
  row.append(badge);

  return row;
}
