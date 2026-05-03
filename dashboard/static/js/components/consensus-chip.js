/*
 * consensus-chip.js — per-ticker consensus pill: action + vote count + 7-tf strip.
 *
 * Used on the Home screen "Active consensus" row and on the Signals heatmap
 * header.
 */

const TF_ORDER = ["now", "12h", "2d", "7d", "1mo", "3mo", "6mo"];

/**
 * @param {{
 *   ticker: string,
 *   action: "BUY"|"SELL"|"HOLD"|"STRONG_BUY"|"STRONG_SELL",
 *   votes?: { buy?: number, sell?: number, hold?: number },
 *   timeframes?: Record<string, "BUY"|"SELL"|"HOLD">,
 *   onTap?: () => void,
 * }} props
 * @returns {HTMLElement}
 */
export function consensusChip({ ticker = "", action = "HOLD", votes, timeframes, onTap = null } = {}) {
  const card = document.createElement("article");
  card.className = "card" + (onTap ? " card--tap" : "");
  card.style.minWidth = "140px";
  card.style.padding = "var(--sp-2)";
  if (onTap) card.addEventListener("click", onTap);

  const top = document.createElement("div");
  top.style.display = "flex";
  top.style.justifyContent = "space-between";
  top.style.alignItems = "center";

  const t = document.createElement("div");
  t.className = "card__title";
  t.style.fontSize = "var(--ty-md)";
  t.textContent = ticker;
  top.append(t);

  const badge = document.createElement("span");
  badge.className = "badge " + _badgeClass(action);
  badge.textContent = action.replace("_", " ");
  top.append(badge);
  card.append(top);

  if (votes) {
    const vt = document.createElement("div");
    vt.style.fontSize = "var(--ty-xs)";
    vt.style.color = "var(--txm)";
    vt.style.marginTop = "var(--sp-1)";
    const buy = votes.buy ?? 0, sell = votes.sell ?? 0, hold = votes.hold ?? 0;
    vt.textContent = `${buy}B / ${sell}S / ${hold}H`;
    card.append(vt);
  }

  if (timeframes) {
    const strip = document.createElement("div");
    strip.style.display = "flex";
    strip.style.gap = "2px";
    strip.style.marginTop = "var(--sp-1)";
    TF_ORDER.forEach((tf) => {
      const cell = document.createElement("span");
      const a = (timeframes[tf] || "").toUpperCase();
      cell.style.flex = "1";
      cell.style.height = "10px";
      cell.style.borderRadius = "2px";
      cell.style.background =
        a === "BUY"  ? "var(--hm-buy)"  :
        a === "SELL" ? "var(--hm-sell)" :
                       "var(--hm-hold)";
      cell.title = `${tf}: ${a || "—"}`;
      strip.append(cell);
    });
    card.append(strip);
  }

  return card;
}

function _badgeClass(action) {
  switch ((action || "").toUpperCase()) {
    case "BUY":         return "badge--BUY";
    case "STRONG_BUY":  return "badge--BUY";
    case "SELL":        return "badge--SELL";
    case "STRONG_SELL": return "badge--SELL";
    default:            return "badge--HOLD";
  }
}
