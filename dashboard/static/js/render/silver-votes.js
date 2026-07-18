/*
 * render/silver-votes.js — #silver page "Live votes" card (Phase 6).
 * Current XAG-USD signal_aggregate row (consensus/buy/sell/hold/horizon)
 * plus phi4_mini's dynamic voter state — the one voter whose availability
 * flips live on the herc2 remote-LLM gate (see system_status.py's _voters).
 */

import { emptyState } from "../components/empty-state.js";

const TICKER = "XAG-USD";

/** @param {{sys: object|null}} props */
export function silverLiveVotes({ sys } = {}) {
  const card = document.createElement("article");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const rows = sys?.signal_aggregate?.tickers;
  const row = Array.isArray(rows)
    ? rows.find((t) => t.ticker === TICKER)
    : null;

  if (!row) {
    card.append(emptyState("No signal snapshot yet."));
  } else {
    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.alignItems = "center";
    const action = (row.consensus || "HOLD").toUpperCase();
    const badge = document.createElement("span");
    badge.className =
      "badge " +
      (action === "BUY"
        ? "badge--BUY"
        : action === "SELL"
          ? "badge--SELL"
          : "badge--HOLD");
    badge.textContent = action;
    const detail = document.createElement("span");
    detail.style.fontSize = "var(--ty-sm)";
    detail.style.color = "var(--txm)";
    const abstainPct =
      row.total > 0 ? Math.round((row.hold / row.total) * 100) : 0;
    detail.textContent = `${row.buy}B · ${row.sell}S · ${row.hold}H — ${abstainPct}% abstaining · ${row.horizon || "1d (default)"}`;
    top.append(detail, badge);
    card.append(top);
  }

  const phi4 = sys?.voters?.phi4_mini;
  if (phi4) card.append(_phi4Row(phi4));

  return card;
}

function _phi4Row(phi4) {
  const div = document.createElement("div");
  div.style.marginTop = "var(--sp-2)";
  div.style.paddingTop = "var(--sp-2)";
  div.style.borderTop = "1px solid var(--bdr)";

  const lbl = document.createElement("span");
  lbl.style.fontSize = "var(--ty-sm)";
  lbl.style.fontWeight = "600";
  lbl.textContent = "phi4_mini: ";

  const st = (phi4.state || "?").toUpperCase();
  const color =
    st === "VOTING"
      ? "var(--grn)"
      : st.startsWith("GATED") || st.startsWith("PAUSED")
        ? "var(--yel)"
        : "var(--gry)";
  const pill = document.createElement("span");
  pill.textContent = st;
  pill.style.fontSize = "var(--ty-xs)";
  pill.style.fontWeight = "700";
  pill.style.padding = "1px 6px";
  pill.style.borderRadius = "999px";
  pill.style.color = color;
  pill.style.border = `1px solid ${color}`;
  div.append(lbl, pill);

  if (phi4.reason) {
    const reason = document.createElement("div");
    reason.style.fontSize = "var(--ty-xs)";
    reason.style.color = "var(--txm)";
    reason.style.marginTop = "2px";
    reason.textContent = phi4.reason;
    div.append(reason);
  }
  return div;
}
