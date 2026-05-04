/*
 * render/errors-panel.js — unresolved errors + contract violations.
 *
 * Reads `/api/system_status` payload's `errors` and `contract_violations`
 * sections. Tap → /health for full triage.
 */

import * as router from "../router.js";
import { ft } from "../format.js";

/** @returns {HTMLElement} */
export function errorsPanel(payload) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "card card--tap";
  card.style.display = "block";
  card.style.width = "100%";
  card.style.textAlign = "left";
  card.style.padding = "var(--sp-3)";
  card.style.minHeight = "var(--tap-min)";
  card.addEventListener("click", () => router.navigate("health"));

  const errs = payload?.errors || {};
  const cvs = payload?.contract_violations || {};
  const errCount = errs.unresolved ?? 0;
  const cvCount = cvs.unresolved ?? 0;
  const total = errCount + cvCount;

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.alignItems = "baseline";
  header.style.marginBottom = "var(--sp-2)";

  const title = document.createElement("div");
  title.className = "section-title";
  title.style.margin = "0";
  title.textContent = "Errors & violations";
  header.append(title);

  const tally = document.createElement("span");
  tally.style.fontSize = "var(--ty-sm)";
  tally.style.fontWeight = "600";
  tally.style.color = total > 0 ? "var(--red)" : "var(--grn)";
  tally.textContent = total > 0
    ? `${errCount} err · ${cvCount} cv (24h)`
    : "0 unresolved";
  header.append(tally);
  card.append(header);

  if (!total) {
    const ok = document.createElement("div");
    ok.style.color = "var(--txm)";
    ok.style.fontSize = "var(--ty-sm)";
    ok.textContent = "all clear";
    card.append(ok);
    return card;
  }

  // Combine and sort by ts (newest first), cap at 5.
  const items = [];
  for (const e of (errs.recent || [])) {
    items.push({ kind: "err", ts: e.ts, label: e.category || e.caller || "error", msg: e.message });
  }
  for (const v of (cvs.recent || [])) {
    items.push({ kind: "cv", ts: v.ts, label: v.invariant || "violation", msg: v.message });
  }
  items.sort((a, b) => (b.ts || "").localeCompare(a.ts || ""));

  const list = document.createElement("div");
  list.style.display = "flex";
  list.style.flexDirection = "column";
  list.style.gap = "var(--sp-1)";
  for (const it of items.slice(0, 5)) list.append(_errorRow(it));
  card.append(list);

  return card;
}

function _errorRow(it) {
  const row = document.createElement("div");
  row.style.fontSize = "var(--ty-sm)";
  row.style.color = "var(--tx)";
  row.style.borderTop = "1px solid var(--bd)";
  row.style.paddingTop = "var(--sp-1)";

  const top = document.createElement("div");
  top.style.display = "flex";
  top.style.justifyContent = "space-between";
  top.style.gap = "var(--sp-2)";

  const tag = document.createElement("span");
  tag.style.fontWeight = "600";
  tag.style.color = it.kind === "cv" ? "var(--yel)" : "var(--red)";
  tag.textContent = it.label;
  top.append(tag);

  const ts = document.createElement("span");
  ts.style.color = "var(--txm)";
  ts.style.fontSize = "var(--ty-xs)";
  ts.textContent = ft(it.ts);
  top.append(ts);
  row.append(top);

  const msg = document.createElement("div");
  msg.style.color = "var(--txm)";
  msg.style.fontSize = "var(--ty-xs)";
  msg.style.overflow = "hidden";
  msg.style.textOverflow = "ellipsis";
  msg.style.whiteSpace = "nowrap";
  msg.textContent = it.msg || "";
  row.append(msg);

  return row;
}
