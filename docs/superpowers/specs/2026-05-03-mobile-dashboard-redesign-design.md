# Mobile-First Dashboard Redesign — Design Spec

**Date:** 2026-05-03
**Branch:** `feat/mobile-dashboard-redesign-2026-05-03`
**Author:** main (Claude)
**Status:** decision-spec — synthesized from research Tracks 1-6.

---

## 1. Overview

The finance-analyzer dashboard is currently a single 3,211-line `index.html`
with embedded CSS+JS, served by Flask on port 5055 behind Cloudflare Access.
Top-nav scrollable tabs over a desktop-grid layout; partial responsive CSS
that mostly shrinks fonts at 480px.

This spec replaces it with a **mobile-first** dashboard that:

- Renders cleanly on a 390×844 iPhone viewport (and similarly on Android).
- Progressively enhances to wider screens — single codebase, mobile-first CSS.
- Splits the monolithic file into modular static assets (no build step).
- Adds PWA bones (manifest + service worker) so the dashboard installs on the
  home screen and survives the lock-screen.
- Reorganises information so the *first screen* answers the user's most
  frequent on-phone question without scrolling.
- Doesn't duplicate Telegram (already pushes alerts and digests).

## 2. Goals & non-goals

### Goals
1. **Phone-first experience** — first paint usable on iOS Safari + Android
   Chrome over 4G.
2. **No feature loss** — every existing tab continues to be reachable and
   functional. Nothing is silently removed.
3. **Battery-friendly polling** — pause when tab hidden; longer cadences for
   non-active sections.
4. **Add-to-home-screen** — installable, standalone, custom icon.
5. **Clean code** — split single file into focused modules, each ≤300 lines.
6. **Backwards-compatible API** — no `/api/*` endpoint changes, no auth
   changes.
7. **Codex-reviewed before merge** per `/fgl` protocol.
8. **Tested** — pytest suite stays green; new modules ship with view-render
   smoke tests where feasible.

### Non-goals (this iteration)
- Framework adoption (Preact/React/Vue/etc.). Vanilla JS + ES modules suffices.
- Switch away from Chart.js. Keep the library; address mobile pain via config.
- Real-time push (websockets / SSE). Polling is fine.
- Backend refactor. We may add a request-log hook (Track 2) but otherwise
  the Flask app is unchanged.
- Bilingual / i18n. English only.
- Accessibility audit beyond touch targets + colour contrast. WCAG-AA is a
  goal but not a gating concern; future iteration.
- Multi-user features. Single-user dashboard.

## 3. User moments

From `docs/research/2026-05-03-mobile-dashboard/03-user-moments.md`. These
drive home-screen content and navigation depth:

| ID | Moment | Frequency | Primary need |
|----|--------|-----------|--------------|
| M1 | Morning anchor (07-09 CET) | daily ~5 min | overnight P&L delta + market summary |
| M2 | Trigger drill-down (TG alert → drill) | 5-15×/day on vol days | signal vote breakdown + chart |
| M3 | Position check during volatility | 1-5×/week, multi-min | live P&L, distance to stop |
| M4 | System pulse (background curiosity) | several/day, 10-20s | loop heartbeats, errors |
| M5 | Decision backlog review (evening) | daily ~10min | filterable Layer 2 history |
| M6 | Research mode (weekend) | weekly ~30-90min | signal heatmap, accuracy |
| M7 | Trade-fill confirmation (post-execute) | 0-10/day | latest tx + position state |

## 4. Information architecture

### 4.1 Top-level navigation: bottom-nav, 4 items

| Tab | Symbol | Covers moments | Endpoints |
|-----|--------|----------------|-----------|
| Home | ⌂ | M1, M3, M4 (lite), M7 | `/api/summary`, `/api/risk`, `/api/warrants`, `/api/loop_health` |
| Decisions | ◉ | M2 (entry), M5 | `/api/decisions`, `/api/triggers` |
| Signals | ▦ | M2 (drill), M6 | `/api/signal-heatmap`, `/api/accuracy`, `/api/accuracy-history` |
| More | ⋯ | M4 (full), M5 (msgs) | other tabs nested |

Bottom-nav is fixed-position with `safe-area-inset-bottom`. Active tab is
underlined and coloured (cyan on dark, blue on light). 56 px tall icons + label.

The "More" tab opens a list view with sub-sections:
- Health (full system pulse)
- Messages (Telegram log)
- Metals (warrant subsystem)
- GoldDigger (gold cert bot)
- Equity Curve
- Settings (theme, refresh, pause, logout)
- Legacy view link (`?legacy=1`)

### 4.2 Routing
URL hash routing: `/#home`, `/#decisions`, `/#signals`, `/#more/health`, etc.
Browser back-button works. Initial load with no hash → home. The previous
default of "Accuracy" is preserved as a deep link the user can pin.

### 4.3 Above-the-fold home screen (first render)

Track 4 confirmed the high-value-on-mobile widgets are exactly the ones
Telegram cannot fit: equity curve, position cards, signal heatmap (compressed),
health, and the trigger/decision feed.

Content stack (top → bottom, fits in ~720 px viewport with 60 px header and
56 px bottom nav):

1. **P&L glance + sparkline** — three numbers (Patient / Bold / Warrants)
   with delta vs yesterday's close, color-coded. A 24-hour mini equity-curve
   sparkline below the numbers. Tap → Equity Curve view. Height: ~140 px.
   *Justification (Track 4):* Telegram only carries point values; the visual
   curve is dashboard-unique.
2. **Open positions strip** — horizontal scroll-snap of position cards: ticker,
   side, P&L %, mini sparkline, distance-to-stop bar. Tap card →
   ticker drill-down (M2). Long-press → bottom sheet with full position
   detail + recent decisions on this ticker. Height: ~150 px.
3. **Active consensus row** — for each ticker BTC/ETH/MSTR/XAG/XAU: chip
   with action + vote count `5B/3S/22H` + tiny color-coded heatmap-strip
   showing the 7-timeframe alignment. Tap → Signals heatmap for that ticker.
   Height: ~120 px.
4. **Latest decision card** — most recent Layer 2 row. Action chip + ticker +
   1-line reason + timestamp. Tap → full decision detail. Height: ~80 px.
5. **System pulse strip** — colored dots for each loop (PF-DataLoop,
   PF-MetalsLoop, PF-CryptoLoop, PF-OilLoop, PF-MstrLoop, PF-GoldDigger).
   Errors-since-last-visit badge (digest, not push — Track 4: ~50/day errors
   would otherwise spam). Tap → Health view. Height: ~60 px.

Below the fold (scroll):

6. **Today's market summary** — collapsed by default. Expands to show
   `morning_briefing.json` highlights (key levels, watchlist, market_outlook).
   *Track 4:* Market context (F&G/DXY/VIX) is heavily duplicated in TG
   analysis-message footers — collapse it.
7. **Recent trades** — last 5 transactions from both strategies.
8. **Risk snapshot** — VaR + Monte Carlo p5/p50/p95 for held tickers. Same
   data as current `/api/risk`, condensed to 3-row card list.

The previous "Overview" tab content is decomposed into these home cards;
not a single tab anymore. The "Recent Telegram Messages" collapsible on the
old Overview is **dropped** (Track 4: redundant with Messages tab).

## 5. Detailed view specs

### 5.1 Home (`/#home`)
See §4.3.

Polling: every 60 s while document.visibilityState is "visible". Pause on hide.
Endpoints: `/api/summary`, `/api/risk`, `/api/warrants`, `/api/loop_health`.

`/api/lora-status` is **dropped on mobile** (LoRA training is desktop concern).
Behind a `?desktop=1` flag or breakpoint ≥1024 px, the LoRA card reappears.

### 5.2 Decisions (`/#decisions`)

- Filter strip at top: chips for action {ALL, BUY, SELL, HOLD}, ticker
  {ALL, BTC, ETH, MSTR, XAG, XAU}, strategy {ALL, Patient, Bold}. Selected
  chips coloured.
- List of decision cards (last 100). Each card: ts (relative), ticker badge,
  action chip × 2 (Patient/Bold), 2-line reasoning preview. Tap → full detail.
- Full detail view (`/#decisions/<id>`): same content as current `decDetail`
  panel, single column, scrollable.
- Pull-to-refresh re-queries `/api/decisions?limit=100&...` with current filters.

### 5.3 Signals (`/#signals`)

Three sub-sections, segmented control at top:

**Sub: Heatmap** (default — Track 5 cleanest answer)
- Sticky chip-bar at top: BTC | ETH | MSTR | XAG | XAU (one ticker at a time)
- Transposed grid: rows = signals (≤33), columns = timeframes (`5m 15m 1h 4h 1d 3d 1w` mapped to our 7-tf set)
- Sticky leftmost column (signal name truncated to 12 chars)
- Sticky top row (timeframe header)
- Cells are color-only — 5-class scale: STRONG_BUY → BUY → HOLD → SELL → STRONG_SELL.
  No numbers in the cell. Disabled signals are diagonally-striped grey.
- ~38×24 px cells → ~7 cols × 14 rows visible per screen
- **Long-press a cell** → bottom sheet with: signal name (full), timeframe, vote,
  confidence %, recent accuracy %, sample size, 3-line rationale.
- **Tap a row label** → filter heatmap to that signal across all tickers
- Single consensus chip above the grid showing the current ticker's verdict
  (`STRONG_BUY 0.78`, etc.).

**Sub: Per-signal accuracy** (the M6 use case)
- 1d/3d/5d/10d horizon toggle (sticky chip-bar)
- Vertical list, sorted by accuracy desc. Each row: signal name + horizontal
  bar (color-coded over/under 47%) + percentage + sample size.
- Tap row → bottom sheet with calibration plot + recent activations.
- Filter chip "show only force-HOLD" / "show only active" toggles greying.

**Sub: History chart**
- Re-implementation of current `accChart`. Top-8 signals by samples,
  selectable via chip toggles below the chart. Single series at a time
  by default to avoid line-spaghetti on phone.

### 5.4 More (`/#more`)

Root: list of links. Each link goes to `/#more/<sub>`.

Sub-sections preserved with mobile-friendly redesign each:

- **Health** (`/#more/health`) — KPI grid (4 cards: Loop Status, Agent Status,
  Cycles, Errors) + Module Failures chip list + Recent Errors timeline. Re-uses
  current `/api/health` shape.
- **Messages** (`/#more/messages`) — **higher priority than original draft**:
  Track 4 found 42% of `telegram_messages.jsonl` (invocation, health,
  crypto_report, golddigger categories) is *saved-only and never delivered* —
  the Messages tab is the sole interface for ~2000 entries. Keep card list
  pattern (already mobile-friendly per inventory). Search + chip filter.
  Add a "Saved-only" filter chip to surface the unrouted entries.
- **Metals** (`/#more/metals`) — re-laid-out from current Metals tab. Each
  panel becomes a vertical card: Summary banner → Position cards (already
  card-shaped) → Risk & Signals (collapse stop-prob table to chip list) →
  Technicals (single column per timeframe, accordion) → Decisions card list →
  Intraday chart (single y-axis, switch warrant/underlying via tab).
- **GoldDigger** (`/#more/golddigger`) — Composite score + position + score
  history chart + trades. Re-laid-out from current.
- **Equity Curve** (`/#more/equity`) — Single chart, large touch targets for
  zoom/pan. BUY/SELL annotations as small markers; tap shows tooltip.
- **Settings** (`/#more/settings`) — theme toggle, refresh interval (15s /
  60s / 5m / paused), logout (clear cookie), legacy-view link.

### 5.5 Trigger Timeline
Subsumed into Decisions tab via a "Show triggers" toggle (or kept as
`/#decisions?view=triggers`). Already vertical-list-friendly.

## 6. Component inventory

These reusable JS components (each in its own file) drive the views:

- **PnLCard** — three-number summary + delta + sparkline option.
- **PositionCard** — ticker, side, P&L%, distance-to-stop bar, sparkline.
- **DecisionCard** — chip-action, ticker, ts, reason. Tap → drill.
- **SignalRow** — name, B/S/H badge, accuracy bar, tap → drill.
- **ConsensusChip** — action + vote count `5B/3S/22H`.
- **PulseDot** — single colored dot with hover/tap label.
- **MiniChart** — Chart.js wrapper with mobile defaults (no axis labels,
  responsive, capped height, touch tooltip).
- **AccordionSection** — collapsible card with title + chevron.
- **FilterChip** — toggleable chip with active state.
- **EmptyState** — "no data yet" placeholder.
- **ErrorBanner** — top-of-page error with dismiss.

## 7. Behaviour

### 7.1 Polling

Single polling controller in `js/polling.js`:
- Interval registry: `{home: 60_000, lora: 300_000, ...}`
- Pauses when `document.visibilityState === 'hidden'` (Page Visibility API).
- Resumes immediately when visible; refreshes once on resume.
- Pull-to-refresh on each view triggers immediate refresh.
- Each view registers/unregisters its endpoints on activate/deactivate.

Inactive tabs do **not** poll. Active tab polls only its declared endpoints.
Total polling on the active "Home" tab: ~3 endpoints / 60 s. About 6 KB / minute.

### 7.2 Live updates UX

- Refresh dot pulses while polling.
- Pause button toggles polling globally; persists in localStorage.
- "Last updated 12s ago" caption per card; updates every second using a
  single ticker (existing pattern).

### 7.3 Theme
Existing dark/light toggle preserved. Persists in `localStorage("pi-theme")`.
Both themes follow `prefers-color-scheme` on first visit.

### 7.4 PWA

- `static/manifest.webmanifest`: name="Portfolio Intelligence",
  short_name="PI", display="standalone", theme_color="#1a1a2e",
  background_color="#1a1a2e", start_url="/", icons (192, 512, maskable).
- `static/sw.js`: install precaches the static shell (HTML, CSS, JS,
  manifest, fallback offline page). Fetch handler:
  - `/api/*` → network-first with 3s timeout, no caching (always-fresh).
  - Static assets → cache-first.
  - Navigation requests → network-first with offline fallback to a cached
    "you're offline" page.
- iOS-specific `<meta name="apple-mobile-web-app-capable" content="yes">`
  and `apple-touch-icon` link.
- Service worker registered after `load` to avoid blocking first paint.
- Service worker scope is `/`; CF Access cookie traverses normally because
  the SW is on the same origin and credentials-include is the fetch default.

### 7.5 Auth
Unchanged. Existing `/dashboard/auth.py` handles CF Access header → cookie.
Cookie has 1-year rolling refresh. Logout in Settings sets cookie expiry to 0.

### 7.6 Touch ergonomics

- Tap targets ≥48×48 CSS px (Material spec; clears WCAG 2.5.5 + Apple HIG).
  Inline icons in dense rows may drop to 36 px iff ≥8 px from neighbours.
- Bottom nav: `calc(56px + env(safe-area-inset-bottom))` for iPhone home
  indicator. Add `viewport-fit=cover` to viewport meta.
- **No pull-to-refresh in v1** (Track 6 decision). Disable Chrome's native
  page-reload PTR via `overscroll-behavior-y: contain` on `<body>`.
  Visibility-resume + auto-poll cover the refresh need.
- **No swipe-tab navigation.** Conflicts with horizontal table scroll.
- **Long-press** as the "more data" gesture (Track 5 pattern). Heatmap cells,
  decision cards, position cards all open a bottom sheet on long-press.
- **Bottom-sheet drill-down** is the universal detail pattern — avoids losing
  context the way a full page navigation would.
- Touch tooltip mode for charts: `interaction.mode: 'index'` so a single tap
  shows the indicator-line tooltip for that x-position.
- `font-variant-numeric: tabular-nums` on all numeric cells for
  decimal-alignment.
- Color-flash on number changes: 250 ms green/red flash on price-tick cells,
  then revert. Encodes momentum without consuming space.

### 7.7 Performance budget (Track 6)

| Bucket | Budget (gzipped) | Notes |
|--------|------------------|-------|
| Shell HTML + critical CSS | ≤ 20 KB | Inline above-the-fold styles |
| Non-critical CSS | ≤ 10 KB | Loaded async via `<link>` |
| Chart.js (CDN) | ~65 KB | Cached after first load |
| App JS modules total | ≤ 80 KB | Down from ~85 KB single-file today |
| **Cold first paint over wire** | **≤ 100 KB** | HTML+CSS+app-JS, before Chart.js |
| **Total cold load** | **≤ 175 KB** | Including Chart.js |
| FCP target | < 1.5 s on 4G | 1.5 Mbps, 100 ms RTT |
| TTI target | < 3.0 s | Parse+execute the shell |
| Subsequent (SW) | < 500 ms | Shell from cache |

Chart.js stays UMD via `<script src="...chart.umd.min.js">` *before* the
module entry script. Modules use the global `Chart` — no ESM import dance
([Chart.js #10163](https://github.com/chartjs/Chart.js/issues/10163)).

### 7.8 Optional: minimal access logging

Add a flag-gated `@app.after_request` that appends `{ts, path, method,
status, ua_class, ms}` to `data/dashboard_access.jsonl` for `/api/*` only.
Disabled by default; enabled via `config.dashboard.access_log = true`.
This is from Track 2's recommendation.

## 8. Tech stack & code organisation

### 8.1 No build step

Vanilla JS, ES modules, served as static files by Flask.
Browser support target: Safari 14+, Chrome 90+, Firefox 90+ (covers all
current iOS/Android phones in active use). ES modules and dynamic imports
universally supported.

### 8.2 File layout (under `dashboard/static/`)

Aligned with Track 6's recommendation, with `views/` and `components/` dirs
added for the new mobile structure:

```
static/
├── index.html                  # ~120-line skeleton: <head>, layout shell, <script type="module">
├── index_legacy.html           # the previous index.html, served at /legacy
├── manifest.webmanifest
├── sw.js                       # service worker
├── css/
│   ├── tokens.css              # CSS variables (existing palette + new spacing/typography tokens)
│   ├── base.css                # reset, typography, body, scrollbars
│   ├── layout.css              # header, bottom-nav, safe-area, view container
│   ├── components.css          # cards, chips, badges, charts, accordion, bottom sheet
│   └── responsive.css          # breakpoints (mobile-first → tablet → desktop)
├── js/
│   ├── main.js                 # entry: register SW, init router, mount initial view
│   ├── state.js                # module-scoped state store (replaces 30 bare globals)
│   ├── fetch.js                # fj(), retry, ttl cache, error display, auth-aware
│   ├── format.js               # fn, fs, fp, ft formatters (extracted from current 853-879)
│   ├── theme.js                # initTheme, updateChartColors
│   ├── router.js               # hash routing + view lifecycle (mount/unmount)
│   ├── polling.js              # interval registry, visibility-aware, per-section cadence
│   ├── components/
│   │   ├── pnl-card.js
│   │   ├── position-card.js
│   │   ├── consensus-chip.js
│   │   ├── decision-card.js
│   │   ├── signal-row.js
│   │   ├── pulse-dot.js
│   │   ├── mini-chart.js
│   │   ├── accordion.js
│   │   ├── filter-chip.js
│   │   ├── bottom-sheet.js     # universal drill pattern
│   │   ├── empty-state.js
│   │   └── error-banner.js
│   ├── views/                  # mobile-first views (mount/unmount lifecycle)
│   │   ├── home.js
│   │   ├── decisions.js
│   │   ├── decision-detail.js
│   │   ├── signals.js
│   │   ├── more.js
│   │   ├── health.js
│   │   ├── messages.js
│   │   ├── metals.js
│   │   ├── golddigger.js
│   │   ├── equity.js
│   │   └── settings.js
│   ├── render/                 # render helpers extracted from current code
│   │   ├── signal-cards.js     # rCards/rHeat/rMkt
│   │   ├── portfolio.js        # rTrades/rBoldSummary/rHoldings/loadWarrants/loadRisk
│   │   ├── accuracy.js         # renderAccuracy/loadAccuracy/loadAccuracyHistory
│   │   ├── lora.js             # rLora (desktop-only)
│   │   └── telegrams.js        # rTel
│   └── charts/
│       ├── chart-config.js     # shared mobile defaults (DPR cap, animation off, index tooltip)
│       ├── equity-chart.js
│       ├── accuracy-chart.js
│       ├── metals-chart.js
│       ├── gd-chart.js
│       └── mini-sparkline.js
└── icons/
    ├── icon-192.png
    ├── icon-512.png
    ├── icon-512-maskable.png
    └── apple-touch-icon-180.png
```

Three rules during the migration (Track 6):
1. **One module ↔ one render group.** No cross-imports between render
   modules — they all import from `state.js`, `fetch.js`, `format.js`,
   `theme.js`. Flat dependency graph.
2. **`state.js` exports a singleton object** (not bare globals).
3. **Chart.js stays UMD** (`<script src="...chart.umd.min.js">` *before* the
   module entry). Modules use the global `Chart`.

`index.html` becomes a minimal skeleton (~120 lines) loading `js/main.js` as a
module. Each `views/*.js` exports `{mount, unmount}` lifecycle. Each
`components/*.js` exports a render function returning a DOM node.

### 8.3 Legacy fallback

The existing `index.html` is preserved as `static/index_legacy.html` and
served at `/legacy`. The new home page at `/` is the redesigned dashboard.
A "Legacy view" link in Settings points to `/legacy` for emergency access
during the transition (or for desktop users who prefer the old layout).

After 14 days of stable operation, `index_legacy.html` and the `/legacy`
route can be removed.

## 9. Migration / rollout

1. Implement modules in the worktree (this PR).
2. Run pytest + spawn dashboard server locally, smoke-test every view on
   a phone-sized window (Chrome dev tools mobile emulator).
3. Codex adversarial review on the branch.
4. Address findings.
5. Merge to main. Push.
6. The user reloads `https://<dashboard-host>/` on phone; new UI lands.
7. If issues, the `/legacy` route preserves the old experience while we fix.

## 10. Testing strategy

### 10.1 Existing tests
- All current pytest suites must remain green (`pytest -n auto`).
- The dashboard backend is unchanged (no API edits) so backend tests are
  unaffected.

### 10.2 New tests
- `tests/test_dashboard_static_assets.py` — every static asset listed in
  the manifest exists and is non-empty.
- `tests/test_dashboard_skeleton.py` — `index.html` parses, has manifest
  link, viewport meta, and references the expected JS entry.
- `tests/test_dashboard_legacy_route.py` — `/legacy` returns
  `index_legacy.html` while authed.
- (Optional, if tooling supports) Lighthouse-ish smoke: not required for
  MVP, future iteration.

### 10.3 Manual validation
- Author-driven phone smoke test via Chrome dev tools (390×844 viewport).
- Lighthouse audit run locally (Performance, PWA scores).
- Visual review of every view on iOS Safari + Android Chrome via the live
  Cloudflare-tunnelled URL.

## 11. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Single user opens dashboard during the cut-over and it breaks | high | medium | Keep `/legacy` route with old dashboard; user can fall back. Communicate via TG. |
| Service worker caches an old shell and user sees stale UI | medium | medium | Versioned cache name, `skipWaiting` + `clients.claim` on install/activate; hard-reload bypass via `?nosw=1`. |
| Chart.js mobile perf is worse than expected | medium | low | Defer charts on home; use sparklines only above-fold. If problematic, swap to uPlot/Lightweight Charts in a follow-up. |
| Polling on mobile drains battery | low | low | Page Visibility API pauses polling when hidden. Optional longer interval in Settings. |
| CF Access redirect breaks on phone reload | low | high | The 1-year cookie should keep it logged in. If broken, fall back to /legacy URL with `?token=XXX` query token (still supported by `auth.py`). |
| Tests we don't have to catch UI regressions | high | medium | Document the manual phone smoke-test checklist in `docs/TESTING.md`. |

## 11.5 Known issues to flag but not fix in this PR

- **`data/metals_loop.py:959`** has a `send_telegram()` that bypasses
  `send_or_store`, so 30+ FISH/L3/TRADE/AVANZA-SESSION templates have no
  `category` field and can't be filtered in the Messages tab. Track 4 found
  this; out of scope for the dashboard redesign but worth a follow-up issue.
- **LOOP CONTRACT spam**: ~50 alerts/day (Track 4 sample). The mobile
  dashboard handles this with the digest badge, but the underlying
  generator should be calmed. Out of scope; track separately.

## 12. Out of scope (this iteration; followups)

- Real-time push updates (websockets/SSE).
- Deep-link from Telegram messages to specific dashboard views (would need
  message-side changes — track separately).
- Notifications API (browser push). Telegram is push.
- Offline mode for /api/* responses (currently network-first). A future
  iteration could add a 5-min stale-while-revalidate cache.
- Per-user customization (drag-to-reorder home cards, hide cards). Single
  user; postpone.
- Replacement of Chart.js. Possible future swap to Lightweight Charts for
  smaller payload + better candlestick support, but not now.
- New `/api/*` endpoints (e.g., a dedicated `/api/home_summary` for one-shot
  home fetch). The first iteration uses existing endpoints; we may
  consolidate in a follow-up if home polling is meaningfully slow.

## 13. Open questions to validate during implementation

- Does iOS Safari preserve the CF Access + Flask cookie across full app
  closures + cold launch? Test on real device.
- Does adding `<meta name="apple-mobile-web-app-capable" content="yes">`
  break the Cloudflare Access redirect (it sometimes interferes with
  redirects on older iOS)? Test before locking in.
- Does the existing 60s `/api/summary` payload size stay reasonable on
  cellular? If >50 KB we may want a thinner `/api/home` projection later.

## 14. Acceptance criteria

- [ ] Every existing tab is reachable from the new layout.
- [ ] No `/api/*` endpoint is removed or modified in shape.
- [ ] All current pytest tests remain green.
- [ ] New dashboard renders cleanly at 390×844, 768×1024, 1440×900 viewports.
- [ ] PWA install prompt appears in Chrome on Android; "Add to Home Screen"
  works on iOS Safari.
- [ ] Service worker controls navigation; offline fallback page appears
  when network is off.
- [ ] Codex adversarial-review (`xhigh`) findings P1/P2 addressed.
- [ ] Documentation updated: `docs/CHANGELOG.md`, `docs/SESSION_PROGRESS.md`,
  CLAUDE.md if architecture entry needs revising.

## 15. Decision log

(To be appended during implementation.)

- 2026-05-03 — Stack: vanilla JS + ES modules, no build step. Reasoning:
  single-user codebase, low maintenance ceiling, ESM is universally supported
  on target browsers.
- 2026-05-03 — Charts: keep Chart.js for now. Reasoning: scope discipline,
  swap optional in follow-up.
- 2026-05-03 — Migration: parallel `/legacy` route for 14 days. Reasoning:
  reversibility, user has a fallback.
- 2026-05-03 — Tracks 4/5/6 may add or override decisions; this section
  will be updated when those research deliverables land.
