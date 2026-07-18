/*
 * render/errors-panel.js — unresolved system errors + contract violations,
 * plus a compact Avanza status chip.
 *
 * Reads `/api/system_status` payload's `errors`, `contract_violations`,
 * and `avanza` sections. Tap the main panel -> /health for full triage;
 * tap the Avanza chip -> /avanza.
 *
 * 2026-07-18: split system errors from Avanza-session noise. Avanza
 * session/account-mismatch failures (BankID re-auth pending) used to
 * flood this panel — 267+ avanza_session_consecutive_failures entries
 * alone. `avanza_*`-category errors now get their own chip; everything
 * else (including `auth_failure`, which is a Claude CLI OAuth failure,
 * NOT Avanza — see system_status.py::_is_avanza_category) stays here.
 * Also: a degraded errors/violations reader (backend `error` field) now
 * shows a chip instead of silently rendering as "0 unresolved" — that
 * used to look identical to a genuinely clean state.
 */

import * as router from "../router.js";
import { ft } from "../format.js";
import { sectionErrorChip } from "../components/section-error-chip.js";

/** @returns {HTMLElement} */
export function errorsPanel(payload) {
  const card = document.createElement("section");
  card.className = "card";
  card.style.padding = "var(--sp-3)";

  const errs = payload?.errors || {};
  const cvs = payload?.contract_violations || {};

  const main = document.createElement("button");
  main.type = "button";
  main.className = "card--tap";
  main.style.display = "block";
  main.style.width = "100%";
  main.style.textAlign = "left";
  main.style.background = "transparent";
  main.style.border = "0";
  main.style.padding = "0";
  main.style.minHeight = "var(--tap-min)";
  main.addEventListener("click", () => router.navigate("health"));

  // recent_system: non-Avanza rows, computed backend-side from the FULL
  // unresolved list (not the pre-capped `recent`) so a system error can't
  // get pushed out just because Avanza noise is more recent. Fall back to
  // client-side filtering of `recent` for resilience if an older backend
  // build hasn't shipped the field yet.
  const recentSystem = Array.isArray(errs.recent_system)
    ? errs.recent_system
    : (Array.isArray(errs.recent) ? errs.recent : [])
        .filter((e) => !String(e?.category || "").startsWith("avanza_"));

  const errCount = errs.system_unresolved ?? (
    errs.unresolved != null && errs.avanza_unresolved != null
      ? errs.unresolved - errs.avanza_unresolved
      : errs.unresolved ?? 0
  );
  const cvCount = cvs.unresolved ?? 0;
  const total = (errCount || 0) + cvCount;

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
  main.append(header);

  // A degraded reader (errs.error / cvs.error) must never look like the
  // "0 unresolved" clean state above — surface it explicitly.
  if (errs.error) main.append(sectionErrorChip(`errors check degraded — ${errs.error}`));
  if (cvs.error) main.append(sectionErrorChip(`violations check degraded — ${cvs.error}`));

  if (total > 0) {
    const items = [];
    for (const e of recentSystem) {
      items.push({ kind: "err", ts: e.ts, label: e.category || e.caller || "error", msg: e.message });
    }
    for (const v of (cvs.recent || [])) {
      items.push({ kind: "cv", ts: v.ts, label: v.invariant || "violation", msg: v.message });
    }
    items.sort((a, b) => (b.ts || "").localeCompare(a.ts || ""));

    if (items.length) {
      const list = document.createElement("div");
      list.style.display = "flex";
      list.style.flexDirection = "column";
      list.style.gap = "var(--sp-1)";
      for (const it of items.slice(0, 5)) list.append(_errorRow(it));
      main.append(list);
    }
  } else if (!errs.error && !cvs.error) {
    // total (system errs + cv) is 0 here, but that must NOT be conflated
    // with "nothing unresolved anywhere" — Avanza-category errors are
    // counted separately and can still be nonzero (jsdom smoke test
    // caught this: this branch used to say "all clear" while the Avanza
    // chip below showed unresolved errors in the same render).
    const ok = document.createElement("div");
    ok.style.color = "var(--txm)";
    ok.style.fontSize = "var(--ty-sm)";
    ok.textContent = (errs.avanza_unresolved ?? 0) > 0
      ? "no system errors — see Avanza chip below"
      : "all clear";
    main.append(ok);
  }

  card.append(main);
  card.append(_avanzaChip(payload?.avanza));
  return card;
}

function _errorRow(it) {
  const row = document.createElement("div");
  row.style.fontSize = "var(--ty-sm)";
  row.style.color = "var(--tx)";
  row.style.borderTop = "1px solid var(--bdr)";
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

function _avanzaChip(av) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.style.marginTop = "var(--sp-2)";
  chip.style.minHeight = "var(--tap-min)";
  chip.addEventListener("click", (e) => {
    e.stopPropagation();
    router.navigate("avanza");
  });

  const info = av || {};
  const n = info.unresolved_errors ?? 0;
  let color = "var(--grn)";
  let label = "Avanza: OK";
  if (info.creds_configured === false) {
    color = "var(--red)";
    label = "Avanza: not configured";
  } else if (info.creds_configured == null) {
    color = "var(--txm)";
    label = "Avanza: unknown";
  } else if (n > 5) {
    color = "var(--red)";
    label = `Avanza: ${n} unresolved`;
  } else if (n > 0) {
    color = "var(--yel)";
    label = `Avanza: ${n} unresolved`;
  }
  chip.style.color = color;
  chip.style.borderColor = color;
  chip.textContent = label;
  return chip;
}
