# PLAN — Mobile-First Dashboard Redesign

**Date:** 2026-05-03
**Branch:** `feat/mobile-dashboard-redesign-2026-05-03`
**Worktree:** `/mnt/q/finance-analyzer/.worktrees/mobile-dashboard-2026-05-03`
**Goal:** Replace the desktop-first 3,211-line single-file dashboard with a
mobile-first, ES-module, PWA-capable dashboard while preserving all existing
functionality and zero-changing the Flask `/api/*` surface.

---

## Context

Current state:
- `dashboard/app.py` — 1,445 lines, 38 routes, healthy. **No changes here**
  except adding a `/legacy` route in Batch 1.
- `dashboard/static/index.html` — 3,211 lines, single file (CSS+JS+HTML),
  Chart.js CDN. Top-nav with 10 tabs that horizontally scrolls on phone.
  Three `@media` blocks (1200/800/480) that mostly shrink fonts.
- The user has stated they want the dashboard primarily phone-displayed.

Research deliverables (Tracks 1-6) under `docs/research/2026-05-03-mobile-dashboard/`:
- `00-synthesis.md` — research summary + final design decisions.
- `01-current-inventory.md` — 39 widgets, 33 endpoints catalogued.
- `02-usage-signal.md` — no usage data; recommend opt-in access log (deferred).
- `03-user-moments.md` — 7 user moments (M1-M7) drive home priorities.
- `04-telegram-overlap.md` — top 5 mobile widgets; 42% of TG is saved-only.
- `05-comparable-products.md` — 9 apps surveyed; heatmap pattern, anti-patterns.
- `06-tech-constraints.md` — stack/PWA/charts/auth/polling decisions.

Design spec at `docs/superpowers/specs/2026-05-03-mobile-dashboard-redesign-design.md`.

Old plan (`midfinance follow-ups`) was completed and merged 2026-05-02 (commit
`40197785`); this plan supersedes it.

---

## What this PR does

The plan ships in 9 batches. Each batch is committed independently; tests run
between batches per `/fgl` protocol. After all batches: Codex adversarial
review, fix findings, merge to main, push.

### Batch 1 — Skeleton + foundation modules (~9 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/index.html` | rewrite | ~120-line skeleton: `<head>` (manifest, viewport-fit=cover, apple-touch-icon, Chart.js UMD), header shell, `<main id="root">`, bottom-nav placeholder, `<script type="module" src="/static/js/main.js">`. |
| `dashboard/static/index_legacy.html` | new from current | Preserve the existing experience under `/legacy`. |
| `dashboard/static/css/tokens.css` | new | CSS variables (palette extracted from current `:root` + `html.light`, plus new spacing/typography tokens). |
| `dashboard/static/css/base.css` | new | reset, body, scrollbar, `font-variant-numeric: tabular-nums` defaults. |
| `dashboard/static/css/layout.css` | new | header, bottom-nav, view container, `safe-area-inset-*`, `overscroll-behavior-y: contain`. |
| `dashboard/static/css/components.css` | new | cards, chips, badges, accordion, bottom sheet, signal heatmap cells, color-flash animation. |
| `dashboard/static/css/responsive.css` | new | mobile-first media queries → tablet → desktop. |
| `dashboard/app.py` | edit | Add `@app.route("/legacy")` returning `index_legacy.html`. ≤8 LOC. |
| `tests/test_dashboard_legacy_route.py` | new | Verify `/legacy` returns the old file when authed. |

Risk: `index.html` rewrite is destructive but `/legacy` route preserves
fallback. Tests cover it.

### Batch 2 — Core JS modules (~7 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/js/main.js` | new | Entry: register SW, init state/theme, init router, mount initial view. |
| `dashboard/static/js/state.js` | new | Singleton state object (replaces 30 bare globals from current `index.html:834-848`). |
| `dashboard/static/js/fetch.js` | new | `fj()` fetch wrapper with retry, ttl cache, error surface. |
| `dashboard/static/js/format.js` | new | `fn`/`fs`/`fp`/`ft` formatters (extracted from current 853-879). |
| `dashboard/static/js/theme.js` | new | `initTheme`, `toggleTheme`, `getChartColors` (extracted from current 927-944). |
| `dashboard/static/js/router.js` | new | Hash router; mount/unmount lifecycle for views. |
| `dashboard/static/js/polling.js` | new | Interval registry, Page Visibility API hook, per-section cadence. |

These modules are zero-render — no DOM mutations yet. Tests after this batch
just confirm the JS parses (one Python test that fetches each module path
through Flask and checks 200 + JS content type).

### Batch 3 — Reusable components (~10 files)

All under `dashboard/static/js/components/`:

- `pnl-card.js` — three-number card + delta + sparkline option.
- `position-card.js` — ticker/side/P&L%/sparkline/distance-to-stop bar.
- `consensus-chip.js` — action + vote count + 7-tf strip.
- `decision-card.js` — chip-action/ticker/ts/reason → tap drill.
- `signal-row.js` — name/B-S-H badge/accuracy bar/sample count.
- `pulse-dot.js` — single colored dot with hover label.
- `mini-chart.js` — Chart.js wrapper applying mobile defaults (DPR cap,
  animation: 0, interaction.mode: 'index', maintainAspectRatio: false).
- `accordion.js` — collapsible card.
- `filter-chip.js` — toggleable chip.
- `bottom-sheet.js` — universal long-press drill; backdrop + swipe-down dismiss.
- `empty-state.js`, `error-banner.js` — small.

Each component exports a render function returning a DOM node. No event
handlers attached automatically — caller wires them.

### Batch 4 — Home view + bottom-nav (~3 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/js/views/home.js` | new | Mounts the 5 home cards; subscribes to polling for `/api/summary`, `/api/risk`, `/api/warrants`, `/api/loop_health`. |
| `dashboard/static/js/charts/mini-sparkline.js` | new | 24h equity sparkline used on the Home P&L card. |
| `dashboard/static/js/charts/chart-config.js` | new | Shared Chart.js mobile defaults. |

After this batch the bare-minimum mobile dashboard works for the most
frequent moments (M1, M3, M4). Subsequent batches add depth.

### Batch 5 — Decisions + decision-detail (~3 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/js/views/decisions.js` | new | Filter chips at top + decision card list; pull `/api/decisions`. |
| `dashboard/static/js/views/decision-detail.js` | new | Full Layer 2 decision detail (Patient + Bold blocks, ticker outlooks, watchlist). |
| `dashboard/static/js/render/decisions.js` | new | Render helpers extracted from current `loadDecisions`/`renderDecisions` (lines 2730-3034). |

### Batch 6 — Signals heatmap + accuracy (~4 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/js/views/signals.js` | new | Heatmap (default) + per-signal accuracy + history sub-sections. |
| `dashboard/static/js/render/signal-cards.js` | new | rCards/rHeat/rMkt extracted from current 1041-1213. |
| `dashboard/static/js/render/accuracy.js` | new | Accuracy + accuracy-history rendering extracted from current 1595-2625. |
| `dashboard/static/js/charts/accuracy-chart.js` | new | Chart.js config for accuracy history. |

### Batch 7 — More + Health + Messages + Settings (~5 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/js/views/more.js` | new | List view of sub-sections; routes to /#more/<sub>. |
| `dashboard/static/js/views/health.js` | new | KPI grid + module failures + recent errors digest badge. |
| `dashboard/static/js/views/messages.js` | new | Existing card-list pattern + Saved-only filter chip (Track 4). |
| `dashboard/static/js/views/settings.js` | new | Theme, refresh interval, pause, logout, /legacy link. |
| `dashboard/static/js/render/telegrams.js` | new | rTel extracted from 1298. |

### Batch 8 — Metals + GoldDigger + Equity + LoRA (~6 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/js/views/metals.js` | new | Re-laid-out metals view (cards from existing data). |
| `dashboard/static/js/views/golddigger.js` | new | GoldDigger view. |
| `dashboard/static/js/views/equity.js` | new | Full equity curve view (Chart.js single-axis, large). |
| `dashboard/static/js/render/portfolio.js` | new | rTrades/rBoldSummary/rHoldings/loadWarrants/loadRisk. |
| `dashboard/static/js/render/lora.js` | new | rLora (desktop-only — hidden under 1024px). |
| `dashboard/static/js/charts/{equity,metals,gd}-chart.js` | new | Chart configs (3 small files). |

### Batch 9 — PWA + tests + docs (~10 files)

| File | Action | Purpose |
|------|--------|---------|
| `dashboard/static/manifest.webmanifest` | new | name/short_name/display=standalone/theme/icons. |
| `dashboard/static/sw.js` | new | Service worker: precache shell + Chart.js, network-first for `/api/*`, offline fallback page. |
| `dashboard/static/icons/icon-192.png` | new | PWA icon (placeholder generated in this PR). |
| `dashboard/static/icons/icon-512.png` | new | PWA icon. |
| `dashboard/static/icons/icon-512-maskable.png` | new | maskable variant. |
| `dashboard/static/icons/apple-touch-icon-180.png` | new | iOS icon. |
| `tests/test_dashboard_static_assets.py` | new | Verify manifest, sw.js, icons all served. |
| `tests/test_dashboard_skeleton.py` | new | Verify index.html parses, has manifest link, viewport-fit=cover, links Chart.js + main.js. |
| `docs/CHANGELOG.md` | edit | Entry. |
| `docs/SESSION_PROGRESS.md` | edit | Session note. |
| `docs/TESTING.md` | edit | Manual phone smoke-test checklist. |

### Codex adversarial review

After Batch 9, run:
```
/codex:adversarial-review --wait --scope branch --effort xhigh
```
Address P1/P2 findings. Document P3 decisions in a follow-up commit.

### Test, merge, push

- `.venv/Scripts/python.exe -m pytest tests/ -n auto`
- All current tests must remain green (backend untouched).
- New static-asset and skeleton tests pass.
- Merge into main.
- Push via Windows git: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`
- `git worktree remove /mnt/q/finance-analyzer/.worktrees/mobile-dashboard-2026-05-03 && git branch -d feat/mobile-dashboard-redesign-2026-05-03`

---

## Risks (from spec §11) and mitigations

| Risk | Mitigation |
|------|------------|
| User opens dashboard during cut-over and it breaks | `/legacy` route preserves the old file; mention in TG. |
| Service worker caches an old shell | Versioned cache name; `skipWaiting` + `clients.claim` on install/activate; `?nosw=1` bypass. |
| Chart.js perf on phone | Mobile defaults (animation:0, DPR cap); evaluate after live test. Swap to uPlot/Lightweight in v2 only if needed. |
| Polling drains battery | Page Visibility API; per-section cadence; pause when hidden. |
| CF Access redirect breaks on cellular | Existing 1-year cookie + `?token=` fallback; auth.py unchanged. |
| Tests don't catch UI regressions | Add manual phone smoke-test checklist to TESTING.md. |

## What could break (concrete)

- **`index.html` rewrite** is the highest-blast-radius change. Mitigation:
  preserve the current file as `index_legacy.html`, add `/legacy` route,
  test before merging.
- **Chart instance lifecycle** when a view unmounts must dispose the Chart.js
  instance to avoid memory leaks (current code's `equityChartInstance` global
  has no dispose path). `mini-chart.js` will own this.
- **Polling registration** must unregister on view unmount or the closed-view
  endpoints keep firing. `polling.js` controller enforces this.
- **`pf_dashboard_token` cookie** must continue to be sent on all `/api/*`
  fetches. Default `credentials: 'same-origin'` covers it; do not write
  `credentials: 'omit'` anywhere.

## Execution order rationale

Batches 1-2 are pure foundation (no UI yet); they don't break anything and
can ship independently. Batches 3-4 build the home screen, the highest-
value moment. Batches 5-8 fill out the remaining views. Batch 9 ships PWA
+ tests + docs.

If we run out of session before completing all 9, stop at the latest
clean-state batch (home works, /legacy preserves the rest) and ship as a
partial release. The user will already see the new home view; everything
else falls back to the legacy file via the More tab's "Legacy view" link.

## Outcomes (filled in during execution)

- Batch 1 — *pending*
- Batch 2 — *pending*
- Batch 3 — *pending*
- Batch 4 — *pending*
- Batch 5 — *pending*
- Batch 6 — *pending*
- Batch 7 — *pending*
- Batch 8 — *pending*
- Batch 9 — *pending*
- Codex review — *pending*
- Tests — *pending*
- Merge + push — *pending*
