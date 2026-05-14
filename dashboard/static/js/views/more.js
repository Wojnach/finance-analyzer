/*
 * views/more.js — list of secondary destinations under the "More" tab.
 *
 * Each item navigates to a top-level route (#health, #messages, etc.).
 */

import * as router from "../router.js";

const ITEMS = [
  { route: "avanza",     label: "Avanza",      hint: "Live broker sync — cash, positions, orders, stops" },
  { route: "prices",     label: "Live prices", hint: "What the system reads — verify against Avanza app" },
  { route: "assets",     label: "Assets",      hint: "Tradeable warrant catalog — what the loops can buy/sell" },
  { route: "portfolio",  label: "Portfolio",   hint: "Patient + Bold simulated P&L (legacy home)" },
  { route: "health",     label: "Health",      hint: "Loops, signals, errors" },
  { route: "cost",       label: "Claude cost", hint: "CLI tokens + $ rollup (7d default)" },
  { route: "messages",   label: "Messages",    hint: "Telegram log + saved-only" },
  { route: "metals",     label: "Metals",      hint: "XAG/XAU warrants" },
  { route: "golddigger", label: "GoldDigger",  hint: "Gold-cert bot" },
  { route: "equity",     label: "Equity",      hint: "P&L curve + trade marks" },
  { route: "settings",   label: "Settings",    hint: "Theme, polling, /legacy" },
];

export const view = {
  mount(rootEl) {
    while (rootEl.firstChild) rootEl.removeChild(rootEl.firstChild);

    const view = document.createElement("section");
    view.className = "view view--more";

    const title = document.createElement("h1");
    title.className = "section-title";
    title.textContent = "More";
    view.append(title);

    const list = document.createElement("nav");
    list.style.display = "flex";
    list.style.flexDirection = "column";
    list.style.gap = "var(--sp-2)";

    for (const it of ITEMS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "card card--tap";
      btn.style.display = "flex";
      btn.style.justifyContent = "space-between";
      btn.style.alignItems = "center";
      btn.style.minHeight = "var(--tap-min)";
      btn.style.textAlign = "left";

      const meta = document.createElement("div");
      const t = document.createElement("div");
      t.className = "card__title";
      t.textContent = it.label;
      const h = document.createElement("div");
      h.className = "card__subtitle";
      h.textContent = it.hint;
      meta.append(t, h);
      btn.append(meta);

      const chev = document.createElement("span");
      chev.style.color = "var(--txm)";
      chev.style.fontSize = "20px";
      chev.textContent = "›";
      btn.append(chev);

      btn.addEventListener("click", () => router.navigate(it.route));
      list.append(btn);
    }
    view.append(list);
    rootEl.append(view);
  },
  unmount() {},
};
router.register("more", view);
