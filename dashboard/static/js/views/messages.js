/*
 * views/messages.js — Telegram messages log.
 *
 * Inherited from legacy: card-list (already mobile-friendly per inventory).
 * Track-4 found 42% of telegram_messages.jsonl is saved-only — added a
 * "Saved-only" filter chip to surface those entries.
 */

import * as router from "../router.js";
import * as state from "../state.js";
import * as polling from "../polling.js";
import { fj } from "../fetch.js";
import { ftFull } from "../format.js";
import { emptyState } from "../components/empty-state.js";
import { filterChip } from "../components/filter-chip.js";

const POLL_KEY = "messages";

const CATEGORIES = [
  "ALL", "trade", "analysis", "iskbets", "bigbet", "digest",
  "invocation", "regime", "error", "fx_alert",
];

let _root = null;
let _filter = { category: "ALL", search: "", savedOnly: false };
let _unsubs = [];
let _searchTimer = null;

export const view = {
  mount(rootEl) {
    _root = rootEl;
    while (_root.firstChild) _root.removeChild(_root.firstChild);
    _root.append(_renderShell());

    _unsubs.push(state.subscribe(state.Slots.MESSAGES, _renderList));

    polling.register(POLL_KEY, 60_000, _refresh);
  },
  unmount() {
    for (const off of _unsubs) try { off(); } catch (_) {}
    _unsubs = [];
    polling.unregister(POLL_KEY);
    if (_searchTimer) { clearTimeout(_searchTimer); _searchTimer = null; }
    _root = null;
  },
};
router.register("messages", view);

// ---------------------------------------------------------------------------

function _renderShell() {
  const v = document.createElement("section");
  v.className = "view view--messages";

  const t = document.createElement("h1");
  t.className = "section-title";
  t.textContent = "Messages";
  v.append(t);

  // Category strip
  const catStrip = document.createElement("div");
  catStrip.className = "chip-strip";
  for (const cat of CATEGORIES) {
    catStrip.append(filterChip({
      label: cat === "ALL" ? "All" : cat,
      active: cat === _filter.category,
      value: cat,
      onToggle: () => {
        _filter.category = cat;
        catStrip.querySelectorAll(".chip").forEach((c) => {
          c.classList.toggle("active", c.dataset.value === cat);
          c.setAttribute("aria-pressed", c.dataset.value === cat ? "true" : "false");
        });
        _refresh();
      },
    }));
  }
  v.append(catStrip);

  // Saved-only toggle
  const toggleWrap = document.createElement("div");
  toggleWrap.style.margin = "var(--sp-2) 0";
  const savedChip = filterChip({
    label: "Saved-only",
    active: _filter.savedOnly,
    onToggle: (next) => {
      _filter.savedOnly = next;
      _renderList();
    },
  });
  toggleWrap.append(savedChip);
  v.append(toggleWrap);

  // Search
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search messages…";
  search.style.width = "100%";
  search.style.padding = "var(--sp-2) var(--sp-3)";
  search.style.background = "var(--card)";
  search.style.border = "1px solid var(--bdr)";
  search.style.borderRadius = "var(--rad-md)";
  search.style.color = "var(--tx)";
  search.style.fontSize = "var(--ty-md)";
  search.style.minHeight = "var(--tap-min)";
  search.value = _filter.search;
  search.addEventListener("input", () => {
    if (_searchTimer) clearTimeout(_searchTimer);
    _searchTimer = setTimeout(() => {
      _filter.search = search.value || "";
      _refresh();
    }, 300);
  });
  v.append(search);

  const list = document.createElement("div");
  list.dataset.slot = "list";
  list.style.marginTop = "var(--sp-3)";
  v.append(list);
  return v;
}

async function _refresh() {
  const url = _buildUrl();
  const data = await fj(url, { ttl: 5_000 });
  if (data) state.set(state.Slots.MESSAGES, data);
  return data != null;
}

function _buildUrl() {
  const qs = new URLSearchParams({ limit: "200" });
  if (_filter.category !== "ALL") qs.set("category", _filter.category);
  if (_filter.search)             qs.set("search", _filter.search);
  return "/api/telegrams?" + qs.toString();
}

function _renderList() {
  if (!_root) return;
  const slot = _root.querySelector('[data-slot="list"]');
  if (!slot) return;
  while (slot.firstChild) slot.removeChild(slot.firstChild);

  const data = state.get(state.Slots.MESSAGES);
  let arr = Array.isArray(data) ? data : (data?.messages || []);
  if (_filter.savedOnly) arr = arr.filter((m) => m && m.sent === false);

  if (!arr.length) {
    slot.append(emptyState("No messages match the current filters."));
    return;
  }

  const wrap = document.createElement("div");
  wrap.style.background = "var(--card)";
  wrap.style.border = "1px solid var(--bdr)";
  wrap.style.borderRadius = "var(--rad-md)";

  for (const msg of arr) {
    wrap.append(_renderMessage(msg));
  }
  slot.append(wrap);
}

function _renderMessage(msg) {
  const row = document.createElement("article");
  row.style.padding = "var(--sp-2) var(--sp-3)";
  row.style.borderBottom = "1px solid var(--bdr)";

  const head = document.createElement("div");
  head.style.display = "flex";
  head.style.alignItems = "center";
  head.style.gap = "var(--sp-2)";
  head.style.fontSize = "var(--ty-xs)";

  // Category chip
  if (msg?.category) {
    const cat = document.createElement("span");
    cat.className = "chip";
    cat.style.fontSize = "var(--ty-xs)";
    cat.style.padding = "1px var(--sp-1)";
    cat.style.minHeight = "auto";
    cat.textContent = msg.category;
    head.append(cat);
  }
  const ts = document.createElement("span");
  ts.style.color = "var(--txm)";
  ts.textContent = ftFull(msg?.ts || msg?.timestamp);
  head.append(ts);

  if (msg?.sent === false) {
    const tag = document.createElement("span");
    tag.style.marginLeft = "auto";
    tag.style.color = "var(--yel)";
    tag.style.fontWeight = "600";
    tag.textContent = "saved-only";
    head.append(tag);
  }
  row.append(head);

  // Body — preserve newlines via textContent + white-space: pre-wrap
  const body = document.createElement("div");
  body.style.fontFamily = "var(--ty-mono)";
  body.style.fontSize = "var(--ty-sm)";
  body.style.whiteSpace = "pre-wrap";
  body.style.wordBreak = "break-word";
  body.style.color = "var(--tx)";
  body.style.marginTop = "var(--sp-1)";
  body.textContent = String(msg?.text ?? msg?.message ?? "");
  row.append(body);
  return row;
}
