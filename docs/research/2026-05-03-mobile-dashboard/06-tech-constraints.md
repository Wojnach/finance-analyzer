# 06 — Technical Constraints (Mobile Dashboard)

**Date:** 2026-05-03
**Author:** mobile-dashboard agent
**Scope:** Tech-stack constraints, library choices, auth/PWA gotchas, performance budget for the mobile-first redesign of `dashboard/static/index.html`.
**Status:** Decisions only. Implementation belongs to a later memo.

## Baseline (what we have today)

Numbers measured against current artifacts:

- `dashboard/app.py` — 1,445 lines, 38 routes (`grep "@app.route" dashboard/app.py`).
- `dashboard/static/index.html` — 3,211 lines, **151.7 KB** uncompressed (single file, embedded CSS, embedded JS, Chart.js v4.4.1 from jsDelivr CDN).
- `dashboard/auth.py:103-163` — auth chain: CF Access header → cookie (`pf_dashboard_token`, 365-day max-age, `httponly+secure+samesite=Lax`) → `?token=` query → bearer.
- Refresh cadence (`dashboard/static/index.html:3188-3201`): 60-second `setInterval`, single `/api/summary` fetch, then sequential `loadWarrants()` + `loadRisk()` + `/api/lora-status`.
- Lazy-loaded tabs (`index.html:990-993`): decisions, messages, health load only when their tab opens. Good pattern, keep it.
- Static-fallback mode (`index.html:2988-3030`): the page already detects "no Flask backend" and serves cached JSON from a static origin. This is essentially a baked-in offline strategy that pre-dates any PWA work.
- Existing responsive CSS (`index.html:450-483`): three breakpoints (1200/800/480), `min-height: 44px` already on `.nav-tab`. So mobile is partly thought through but desktop-first.
- **No** `<link rel="manifest">`, **no** service worker, **no** `theme-color`, **no** `apple-touch-icon` — confirmed by grep returning zero hits.

## Topic 1 — Stack: vanilla, modules, build, or framework?

### Options

| | Maintenance load | Readability of large file | IDE support | Build step | Future-proof |
|--|--|--|--|--|--|
| (a) Pure vanilla, single file | Lowest | Bad at 3.2K lines | Poor | None | Forever |
| (b) Vanilla + ES modules, no build | Low | Good | Good (with JSDoc + types via `// @ts-check`) | None | Forever |
| (c) esbuild/Vite tiny build | Medium | Good | Excellent | One npm command + watch | Until npm/node breaks |
| (d) Tiny framework (Preact/Lit/Alpine) | Higher (framework upgrades) | Best | Excellent | Optional but usually yes | Subject to library churn |

### Reasoning

The user is a single trader-developer; the dashboard runs on one machine and is deployed by `git pull`. The cost we're optimizing against is *future me debugging this at 02:00 after silver crashes*. That argues hard against:

- A build step we have to remember to run before deploying. (We already have a Flask `static/` + Cloudflare Tunnel pipeline that works without one. Adding `npm run build` adds a class of failure where the deployed bundle drifts from source.)
- A framework that ships its own reactivity model. Even Preact at ~3 KB gzipped ([Preact docs](https://preactjs.com/), [Medium 2025 comparison](https://medium.com/@marketing_96787/preact-vs-react-in-2025-which-javascript-framework-delivers-the-best-performance-f2ded55808a4)) implies a hooks model and component lifecycle. Alpine.js (~15 KB) is closer to "vanilla with directives" but hides logic in HTML attributes — bad for the kind of debugging this dashboard needs. Lit is ergonomic but couples us to web components, where IDE/devtools support is still patchier than plain HTML.
- The 3,200-line file. It is genuinely hard to read. We've felt this — see line 836-848 of `index.html`, where global `var cdv = 60; var cdi = null; var paused = false; var lastRefreshTime = null;` are declared bare.

But (b) — ES modules served as static `.js` files via `<script type="module" src="...">` — buys us most of (c)'s readability with none of the build cost:

- Browsers natively parse modules. No bundler.
- Flask serves them via `static/` already.
- We get `import`/`export`, real scoping, and IDE go-to-definition.
- HTTP/2 multiplexing via Cloudflare means parallel module fetches don't stall on first paint the way they would on HTTP/1.1.
- We can incrementally migrate: extract one section at a time without breaking the rest.

The "no IDE support" argument against (a) is real. With (b) we get it cheaply.

### Decision

**(b) Vanilla + ES modules, no build step.** Continue using Chart.js via CDN script tag (UMD). Put our own code into `static/js/*.js` modules, loaded via one `<script type="module" src="/static/js/main.js">`. Chart.js as UMD must come *before* the module script.

Reasoning: lowest maintenance, eliminates the 3.2K-line readability problem, no toolchain, no framework churn.

## Topic 2 — PWA feasibility

### What "minimum viable PWA" means here

For a single-user dashboard behind Cloudflare Access, the value of PWA is narrow:

1. **Add to home screen** with a proper icon (looks better than a Safari favicon, hides the URL bar).
2. **Standalone display mode** — full-bleed, no browser chrome.
3. **Some offline shell** — show a useful "last known state" when on the metro / no signal, instead of a Safari error page.

We do **not** need: push notifications (Telegram already does this — `portfolio/telegram_notifications.py`), background sync, install prompts (the user is the only user), web share targets.

### Manifest — straightforward

A `dashboard/static/manifest.webmanifest` with:
- `name`, `short_name`
- `display: standalone`
- `theme_color: #1a1a2e` (matches `--bg` at `index.html:15`)
- `background_color: #1a1a2e`
- `icons[]` with at least 192x192 and 512x512 PNGs
- `start_url: "/"` (or `/?source=pwa` if we want to track install→launch)

Plus `<meta name="apple-mobile-web-app-capable" content="yes">` and `<link rel="apple-touch-icon">` for iOS Add-to-Home-Screen.

### Service worker — conditional yes

The classic SW value is offline cache. But we already have a static-fallback mode (`index.html:2988-3030`) that serves cached JSON from a separate static origin. That's a hand-rolled pseudo-PWA. A real SW would unify this and let us cache the HTML/CSS/JS shell so the app loads instantly even on cold start.

**The Cloudflare Access interaction is the real question.** A service worker intercepts every `fetch` for its scope. CF Access guards every request with the `CF_Authorization` cookie. Three known failure modes:

1. **SW fetch from cache served as 200 OK while CF would have returned 302 to login.** If our SW serves a cached `/api/summary` after the CF_Authorization cookie expired, the user sees stale data and never gets prompted to re-auth. Acceptable for read-only data, but we have to be deliberate about it.
2. **Subrequest cookie collision.** Per [Cloudflare docs](https://community.cloudflare.com/t/cloudflare-access-and-worker-subrequests/184512), when a Cloudflare *Worker* (not browser SW) makes a subrequest to an Access-protected origin, the response's `Set-Cookie` for `CF_Authorization` can clobber the original session. This does **not** apply to browser service workers — they live on the client, not in CF's edge — but it's worth understanding so we don't confuse the two.
3. **`fetch` credentials.** CF Access requires the cookie. Browser `fetch()` defaults to `same-origin` credentials, which is what we want. No code change needed *unless* we accidentally write `credentials: 'omit'` somewhere — guard against that.

**Recommended cache strategy:**
- **HTML/CSS/JS shell:** `cache-first, fallback to network` — version-pinned via filename hash or `?v=2026-05-03` query string.
- **`/api/*` JSON:** `network-first, fallback to cache, max-age 60s`. Never serve stale auth-failed responses.
- **Chart.js CDN:** `cache-first, max-age 30 days` — it's CDN-stable.

If the SW gets a 401/403 from `/api/*`, **do not cache it**, fall through to network so the user sees the auth prompt.

### iOS "Add to home screen" caveats

From [MagicBell PWA iOS guide](https://www.magicbell.com/blog/pwa-ios-limitations-safari-support-complete-guide) and [Brainhub 2025](https://brainhub.eu/library/pwa-on-ios):

- **Storage isolation:** "Cookies, Web Storage, and IndexedDB are isolated and separate from Safari and other icons of the same PWAs". This is the killer caveat. **The CF Access cookie set in Safari is *not* visible to the home-screen PWA.** First launch from home screen will redirect to CF login again. After that, the PWA has its own cookie jar — once authenticated there, it persists separately.
- **Cache API ~50 MB limit** on iOS Safari. Plenty for our shell (HTML+CSS+JS+Chart.js < 300 KB) and JSON cache.
- **7-day script-writable storage cap** if the PWA isn't used. The user opens it daily, so this is moot in practice.
- **Service workers run on iOS** since iOS 11.3, but with tighter quotas than Chrome.
- **No SW background updates**, no background sync, no periodic sync. Battery-conservative — fine, we don't need them.

### Decision

**Yes to manifest + apple-touch-icon, optional minimal service worker for shell+chart cache only.** No background sync, no push, no fancy strategies. Auth-bearing `/api/*` calls go straight to network and are *not* cached. Treat the PWA as "Safari with a home-screen icon and standalone display" — that captures 90% of the value at 5% of the complexity.

The user must expect to re-authenticate once after first install on home screen (separate cookie jar). Document this.

## Topic 3 — Chart.js on mobile

### Current state

`<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js">` at `index.html:7`. Used for: `equityChart`, `accChart`, `metalsPriceChart`, `gdScoreChart`. UMD bundle is ~190 KB minified, ~65 KB gzipped (per [Chart.js v4 docs](https://www.chartjs.org/docs/latest/getting-started/integration.html)).

### Mobile pain points (researched)

1. **Touch event suppression** when `chartjs-plugin-zoom` is added globally — non-zoomed charts also start swallowing scroll touches ([chartjs-plugin-zoom #311](https://github.com/chartjs/chartjs-plugin-zoom/issues/311)). If we add zoom, scope it per-chart only.
2. **Hammer.js dependency** — chartjs-plugin-zoom v2+ requires `hammerjs` for pinch ([chartjs-plugin-zoom guide](https://www.chartjs.org/chartjs-plugin-zoom/latest/guide/)). Hammer is ~7 KB gzipped but unmaintained since 2020. Acceptable, but worth knowing.
3. **Canvas DPR perf** at high pixel ratios on iPhones — Chart.js redraws on every animation frame; we should set `responsive: true, maintainAspectRatio: false, animation: { duration: 0 }` or short for trade-relevant charts to stop the GPU/CPU dance.
4. **Bundle size** is the biggest sin for mobile-first.

### Alternatives surveyed

| Library | Bundle size (min+gzip) | Mobile/touch | Strengths | Weaknesses |
|--|--|--|--|--|
| Chart.js v4 (UMD) | ~65 KB | OK; needs zoom plugin | Familiar, all chart types, what we already have | Heaviest of the four |
| Lightweight Charts v5 | ~35 KB | Excellent — financial use case is the design center | OHLC/area/line, snappy on mobile, 50K+ candles smooth | Time-series only — bad fit for our equity-curve-style line charts where x-axis is sometimes labels not timestamps; not drop-in API-compatible |
| ApexCharts | ~120 KB | Decent | Pretty, lots of types | Heaviest, weakest perf |
| uPlot | ~50 KB | Touch is meh, perf is best-in-class | 100K pts/ms, 10% CPU vs Chart.js's 40% on 3,600 pts at 60fps ([cprimozic notes](https://cprimozic.net/notes/posts/my-thoughts-on-the-uplot-charting-library/)) | Time-series only, no built-in touch zoom, plain visuals |

Sources: [Index.dev 2026 comparison](https://www.index.dev/skill-vs-skill/tradingview-vs-lightweight-charts-vs-chartjs), [Lightweight Charts v5 announcement](https://www.tradingview.com/blog/en/tradingview-lightweight-charts-version-5-50837/), [uPlot vs others](https://cprimozic.net/notes/posts/my-thoughts-on-the-uplot-charting-library/).

### Reasoning

We have four chart types in use today: equity curve (line), accuracy bar/line, metals intraday price (multi-axis), GoldDigger score (bar/line). All are time-series-friendly except `accChart`, which sometimes plots categorical accuracy buckets.

A switch to Lightweight Charts would force us to keep Chart.js for `accChart` *anyway*, which means shipping both — net loss. Switching to uPlot drops us back to plain visuals and we'd have to re-implement zoom for touch; the perf gain is moot because we're plotting <300 points per chart.

The mobile pain isn't Chart.js perf — it's touch ergonomics (no pinch-zoom on intraday) and bundle weight. Pinch-zoom is *nice to have*, not load-bearing. The user looks at the chart and reads the numbers from the table.

### Decision

**Keep Chart.js v4. Skip the zoom plugin.** If pinch-zoom for the intraday metals chart becomes a real ask, evaluate `chartjs-plugin-zoom` *scoped to that one chart only* (to avoid the touch-event suppression bug). Don't add a second chart library.

Mobile config tweaks to apply: `animation.duration: 0` for refreshing charts, `devicePixelRatio: Math.min(window.devicePixelRatio, 2)` to cap canvas resolution, `interaction.mode: 'index'` for finger-friendly tooltips.

## Topic 4 — Auth on mobile

### Failure modes to anticipate

Combining what's in `dashboard/auth.py` with the iOS/CF research:

1. **First visit from a new mobile browser** — neither cookie nor CF Access header present.
   - Path A: User opens `https://dashboard.example.com` in Safari → CF Access intercepts → forces SSO redirect → on success, CF sets `CF_Authorization` cookie + injects `Cf-Access-Authenticated-User-Email` header for the proxied request → `dashboard/auth.py:131` accepts → `_refresh_cookie()` sets the `pf_dashboard_token` cookie. Both cookies now present.
   - Path B: User shares a link with `?token=...` → `auth.py:140-142` matches → cookie set in one round-trip. Better for "scan QR code on phone, one-tap login" UX but currently requires the user to manually paste.

2. **Cellular vs Wi-Fi.** No fundamental difference for the redirect flow. Latency is higher on cellular (Cloudflare Tunnel adds ~80-150ms vs ~30-50ms on home Wi-Fi), but Cloudflare's Anycast means the CF Access SSO flow stays close to the user. **Practical impact:** the SSO redirect chain (3-4 hops: app → CF login → IdP → CF callback → app) takes ~1-2s on Wi-Fi, ~3-5s on cellular. Acceptable.

3. **Safari ITP.** Per [JenTis 2024 ITP guide](https://www.jentis.com/blog/how-to-work-with-safari-itp-limitations) and [Snowplow ITP](https://snowplow.io/blog/tracking-cookies-length):
   - **JS-set cookies:** capped at 7 days. Not us — `pf_dashboard_token` is set server-side via `Set-Cookie` (`auth.py:90-100`).
   - **Server-set HttpOnly+Secure cookies:** persist up to 400 days *if* the server's IP matches the website's IP. Behind Cloudflare Tunnel, the origin IP differs from the public IP — Safari 16.4+ may treat the dashboard origin as a "CNAME-cloaked" tracker. **Risk:** even our 365-day cookie could get capped to 7 days under aggressive ITP heuristics.
   - **30-day inactivity purge:** if the user doesn't visit for 30 days, all storage (cookies, IndexedDB) is wiped. The user opens the dashboard daily — moot.
   - **Mitigation:** `_refresh_cookie()` (`auth.py:90`) resets max-age on every request, so as long as the user visits within whatever window ITP enforces (worst case 7 days), the cookie keeps sliding forward. This is **already correctly designed** and should survive ITP.

4. **CF Access cookie SameSite.** Per [Cloudflare WAF docs on SameSite](https://developers.cloudflare.com/waf/troubleshooting/samesite-cookie-interaction/), `Strict` causes redirect loops in the CF Access SSO flow. CF defaults to `Lax`, which works for top-level navigation. Our `pf_dashboard_token` is also `samesite=Lax` (`auth.py:98`) — correct.

5. **Standalone PWA cookie isolation.** As covered in Topic 2, the home-screen PWA has its own cookie jar separate from Safari. **First open from home screen ≈ first visit from a new browser.** The user goes through the SSO redirect once, then the PWA's CF_Authorization persists.

6. **iOS 26 Safari WebSocket bug.** Per [Jack Pearce's writeup](https://www.jackpearce.co.uk/posts/debugging-websocket-upgrade-failures-safari-ios26/), Safari 26 with iCloud Private Relay sends malformed CONNECT requests through Cloudflare Tunnel for WebSocket upgrades. **If we add SSE/WebSocket** (Topic 7), we may hit this. SSE over plain HTTP/1.1 is fine — the bug is WebSocket-specific. Plan around this.

### Decision

**No code change to auth.** The existing chain (`dashboard/auth.py:103-163`) handles mobile correctly. Document for the user:
- First visit from a new browser/PWA = expect SSO redirect.
- After that, 1-year rolling cookie + per-request refresh = effectively never re-auth.
- ITP worst-case forces re-auth weekly; daily use means we never see it.
- Avoid WebSockets (use SSE or polling) to dodge the iOS 26 Private Relay bug.

## Topic 5 — Code organization for the new mobile UI

### What we have and the readability problem

`index.html` 3,211 lines = `<head>` (8) + `<style>` (476) + `<body>` (markup, ~340 lines) + `<script>` (~2,380 lines). The script section has 35 distinct top-level functions (`grep "function r\|function load\|function init\|function render"` returned 35). Several render-cluster groups: signal cards, heatmap, market summary, equity curve, accuracy, decisions, messages, health, metals, golddigger, warrants, risk.

This is the single biggest day-to-day cost. Anyone (the user, future Claude sessions) reading this file scrolls forever to find anything.

### Options

| | Steps to deploy | IDE support | Migration cost | Order-dependency risk |
|--|--|--|--|--|
| Multiple `<script src="...">` | Same as today | Decent | Low — physical copy-paste of code blocks | High — globals, must keep load order |
| `<script type="module" src="...">` ESM | Same as today | Excellent | Medium — must add explicit `import`/`export` | Low — modules manage their own deps |
| esbuild bundle | New: `npm run build` step before deploy | Excellent | High — sets up package.json, watcher, CI | None — bundler resolves order |

### Reasoning

Plain `<script src="">` files preserve the implicit-globals model and require us to remember every global's load order. We already have ~30 globals in `index.html:834-848` and across functions. Splitting into 8 files via `<script>` tags would just turn one 3.2K-line file into eight 400-line files that can only be understood as a whole. Bad trade.

ESM with `<script type="module">` gives us:
- Explicit imports — IDE shows you exactly what each module needs.
- Per-module top-level scope — globals stop leaking accidentally.
- No build step — Flask serves `static/js/*.js`, browser fetches and parses natively.
- HTTP/2 multiplexing means 8-12 modules in parallel doesn't slow first paint.
- Migration is incremental: pull out one render group at a time.

esbuild/Vite would help if we adopted TypeScript or wanted minification. We probably want neither for this codebase. If we ever want minification later, we can add esbuild as a one-line CI step without changing the source layout.

### Recommended directory structure

```
dashboard/static/
  index.html          # shell only: <head>, layout, <script type="module" src="js/main.js">
  manifest.webmanifest
  sw.js               # service worker, scope = /, registered from main.js
  icons/              # PWA icons (192, 512, apple-touch)
  css/
    base.css          # CSS variables, reset, typography (currently lines 9-100 of index.html)
    layout.css        # grid, header, nav (lines 100-450)
    components.css    # cards, tables, charts (lines 200-450)
    responsive.css    # the three @media blocks (lines 450-483)
  js/
    main.js           # entry: imports + init + setInterval + visibility hook
    state.js          # global state object (replaces bare `var` at lines 834-848)
    fetch.js          # fj(), _fjStatic(), _withApiToken(), error display
    format.js         # fn, fs, fp, ft, ftFull, eh — the formatters at lines 853-879
    theme.js          # initTheme, updateChartColors, getChartColors
    nav.js            # switchTab, mobile bottom-nav
    render/
      signal-cards.js   # rCards, rHeat, rMkt
      portfolio.js      # rTrades, rBoldSummary, rHoldings, loadWarrants, loadRisk
      equity-curve.js   # loadEquityCurve, renderTradeTimeline
      accuracy.js       # renderAccuracy, loadAccuracy, loadAccuracyHistory
      decisions.js      # loadDecisions, renderDecisions
      messages.js       # loadMessages
      health.js         # loadHealth
      metals.js         # loadMetals + 5 metals renderers
      golddigger.js     # loadGoldDigger + 5 GD renderers
      lora.js           # rLora
      telegrams.js      # rTel
    charts/
      chart-config.js   # shared Chart.js options (mobile DPR cap, animation off)
      equity-chart.js   # equity curve specific
      metals-chart.js
      gd-chart.js
```

Three rules to follow during the migration:
1. **One module ↔ one render group.** No cross-imports between render modules — they all import from `state.js`, `fetch.js`, `format.js`, `theme.js`. Flat dependency graph.
2. **`state.js` exports a singleton object**, not bare globals. Hooks into `theme.js` and the visibility-change handler in `main.js` go through the state object.
3. **Chart.js stays a UMD `<script>` tag in `index.html`.** It registers a `Chart` global. Modules import nothing for it — just use `Chart`. Avoids the chart.js ESM tree-shake registration ceremony ([Chart.js issue #10163](https://github.com/chartjs/Chart.js/issues/10163)).

### Decision

**ES modules served as static files. No build step. Directory structure above.** `<script type="module" src="/static/js/main.js">` in `index.html`, with Chart.js loaded via UMD `<script>` *before* the module script. Migrate one render group per session — the file will shrink incrementally without ever being broken.

## Topic 6 — Touch ergonomics

### Tap targets

[WCAG 2.5.5 (AAA)](https://www.w3.org/WAI/WCAG21/Understanding/target-size.html) requires 44×44 CSS px. [WCAG 2.5.8 (AA, 2.2)](https://www.allaccessible.org/blog/wcag-258-target-size-minimum-implementation-guide) requires 24×24 minimum. Apple HIG: 44pt. Material Design: 48dp. Safe number: **48 CSS px** — clears both platforms with a small buffer.

Current state (`index.html:462,467,469`): `.nav-tab { min-height: 44px }`, `.icon-btn { min-width: 36px; min-height: 36px }` (sub-spec — fix), `.atab { min-height: 36px }` (sub-spec — fix).

**Apply globally to interactive elements:** `min-height: 48px; min-width: 48px` for buttons/links/tabs. Allow inline icon-buttons inside dense table rows to use 36px *only if* row spacing puts them ≥8px apart from neighbors (per [Smashing Mag tap targets](https://www.smashingmagazine.com/2023/04/accessible-tap-target-sizes-rage-taps-clicks/)).

### Bottom nav + safe-area

Per [WebDong iPhone home indicator](https://www.webdong.dev/en/shortpost/iphone-home-indicator-safe-area-css/) and [Bram.us PWA safe-area](https://www.bram.us/2017/12/10/customizing-pull-to-refresh-and-overflow-effects-with-css-overscroll-behavior/):

- Add `viewport-fit=cover` to the existing `<meta name="viewport">` at `index.html:5`.
- Bottom-nav container: `padding-bottom: env(safe-area-inset-bottom)`. Use `calc(56px + env(safe-area-inset-bottom))` for full-bar height on iPhones with home indicator (typically +34px).
- Top-nav (current `.hdr` / `.nav-tabs`): `padding-top: env(safe-area-inset-top)` if we ever go full-bleed in standalone mode.

Bottom-nav placement: reasonable on phones <428px wide. The current `.nav-tabs` (10 tabs) is too wide for a bottom bar — that's a UI redesign concern (other memos), but the constraint is: a bottom bar fits ~5 items max at 48-56px each.

### Pull-to-refresh

Two options:
1. **Native browser PTR.** Chrome on Android refreshes the page. Safari on iOS doesn't have built-in PTR for arbitrary pages (only Safari's own UI). Inconsistent ([Bram.us](https://www.bram.us/2017/12/10/customizing-pull-to-refresh-and-overflow-effects-with-css-overscroll-behavior/)).
2. **Custom PTR** that calls `refresh()`. ~50 lines of JS — listen to `touchstart`/`touchmove` at the top of the scroll container, animate a spinner, fire the same `refresh()` we already have at `index.html:3147`.

For our use case, native PTR (Chrome's "reload the entire page") is wrong — it nukes our in-memory state including chart instances. **Recommended:** disable native PTR with `overscroll-behavior-y: contain` on `<body>`, ship a small custom PTR that triggers `refresh()` only. This keeps the page mounted.

Alternative: skip PTR entirely, rely on the existing 60s auto-refresh + a visible "Refresh now" button in the header (already exists implicitly via `cdv` countdown). This is the lower-risk move — PTR feels native but it's not load-bearing for this app.

### Swipe gestures

The dashboard has dense tabular data (signal heatmap = 13+ tickers × 7 timeframes; accuracy table; decisions table). Swipe gestures over these are **risky**:

- Horizontal swipe to scroll wide tables — already in use via `overflow-x: auto` (`index.html:463-464`). Keep it.
- Horizontal swipe to navigate between tabs — **don't**. Conflicts with table scrolling, false-positives are infuriating.
- Swipe-to-dismiss for cards — only if the card has a visible "x" button as the primary affordance. Swipe is the secondary path.
- Long-press for context menu — useful for "show this trade's full reasoning" on the decisions table. Cheap to add, low risk.

### Content-specific guidance

- **Signal heatmap rows** (`rHeat`, `index.html:1132`): keep horizontal scroll, add sticky first column (ticker name), make each cell ≥48px wide for taps, show a bottom-sheet detail on tap.
- **Decision cards** (`renderDecisions`, `index.html:2771`): full-width on mobile, vertical stack of action/ticker/confidence/reason. Tap-to-expand reason; current rendering inlines them.
- **Equity curve & metals charts:** chart-as-hero on mobile, table-of-trades collapsible below. Set `Chart.js` `interaction.mode: 'index'` so a single tap shows tooltip for that x-position (no fiddly point hits).
- **Holdings header strip** (`index.html:497-518`): currently 4-5 stat cards in a row. On mobile, becomes 2 rows of 2-3 cards or a horizontal-scroll strip.

### Decision

- **48×48 px global tap-target floor**, table-row icons can drop to 36 with adequate spacing.
- **`viewport-fit=cover` + safe-area-inset-bottom** on any bottom-fixed UI.
- **No swipe-tab-navigation.** Keep horizontal scroll for tables. Long-press for "show full detail" is acceptable.
- **No PTR in v1.** `overscroll-behavior-y: contain` on `<body>` to disable Chrome's native page-reload PTR. Add custom PTR later if the user explicitly asks.

## Topic 7 — Performance budget

### Targets

Working assumption: 4G at ~1.5 Mbps (~190 KB/s) effective downlink, 100ms RTT, single connection (HTTP/2 multiplex). Targets:

- **First Contentful Paint (FCP):** < 1.5s on cold load. Means first byte of HTML must arrive ~150ms after request, and HTML+critical CSS must be < 30 KB to fit in the first ~200ms of TCP.
- **Time to Interactive (TTI):** < 3.0s. Means JS payload (parse+execute) for the shell < ~150 KB.
- **Subsequent loads:** < 500ms (SW shell cache).

### Current bytes

- `index.html` (single file): 151.7 KB uncompressed. With gzip on the Flask response (Cloudflare auto-applies brotli at the edge), wire size is ~28 KB. Acceptable.
- `chart.umd.min.js` from jsDelivr: 195 KB minified, ~65 KB gzipped/brotli. Cached after first load.
- No other assets.

So the cold first-load is **~93 KB over the wire** (HTML + Chart.js gzipped). On 4G, ~500ms transfer. Should hit FCP under 1.5s.

After the ESM split (Topic 5), we'll have ~8-12 small `.js` files. HTTP/2 multiplexes them, but each one adds parse cost. Aim to keep total app-JS under **80 KB gzipped** (we're under that today, the split shouldn't grow it).

### Polling strategy

Current state (`index.html:3186-3201`): `setInterval` every 1s decrements a counter, calls `refresh()` when it hits zero (60s cycle). `refresh()` does:
- 1× `/api/summary` (the big one, includes signals/portfolio/portfolio_bold/telegrams)
- 1× `/api/lora-status`
- 1× `/api/warrants` (via `loadWarrants`)
- 1× `/api/risk` (via `loadRisk`)

That's **4 HTTP requests every 60s** = 5,760/day per open tab. On cellular, each request has ~100ms RTT minimum, so ~400ms of network spending per minute. Battery cost is real on iPhones — every wake of the cell radio costs ~1-3 seconds of high-power state.

Improvements available, in order of impact:

1. **Pause when hidden.** [MDN Page Visibility API](https://developer.mozilla.org/en-US/docs/Web/API/Page_Visibility_API). When `document.hidden === true`, stop the interval. On `visibilitychange` to visible, fire one immediate refresh and resume. This is the single biggest battery win — backgrounded tabs do *zero* network work.

2. **Backoff when on cellular.** `navigator.connection?.effectiveType` returns `"slow-2g" | "2g" | "3g" | "4g"`. On `"slow-2g"`/`"2g"`, lengthen interval to 5min. Optional but cheap.

3. **Server-sent events (SSE) for triggers.** Layer 2 trigger detection (`portfolio/trigger.py`) is event-based — when a trigger fires, we want the UI to know within seconds, not up to 60s later. SSE (one persistent HTTP connection, server pushes JSON lines) is a good fit:
   - Native browser support, no library.
   - Works fine through Cloudflare Tunnel (unlike WebSocket, no iOS 26 Private Relay bug).
   - Auto-reconnects on drop.
   - Backend cost: one Flask thread per connected client. We're a single user. Acceptable.

   **But this is a v2 enhancement.** v1 should ship with polling-only and reduced cadence.

4. **Per-section cadence.** Not everything needs to refresh every 60s.

### Recommended polling cadence per-section

| Endpoint | Current | Recommended | Rationale |
|--|--|--|--|
| `/api/summary` | 60s | 60s on Wi-Fi, 120s on cellular, **paused when hidden** | Hot path; signal/portfolio change fast |
| `/api/lora-status` | 60s | 5min | LoRA state changes hourly at most |
| `/api/warrants` | 60s | 60s when warrants tab is open, 5min otherwise | Active position monitoring |
| `/api/risk` | 60s | 5min | Drawdown circuit-breaker state, slow-changing |
| `/api/decisions` | tab-open only | tab-open + 60s while open | Already good |
| `/api/messages` | tab-open only | tab-open + 30s while open | Telegram inbound ≤ 30s old is fine |
| `/api/health` | tab-open only | tab-open + 60s while open | Already good |
| `/api/metals` | tab-open only | tab-open + 30s while open + paused when hidden | Active trading data |
| `/api/golddigger` | tab-open only | tab-open + 60s while open | Slower-changing |
| `/api/equity-curve` | tab-open only | tab-open only (no interval) | Daily-resolution data, refresh on visibility return is enough |
| `/api/accuracy*` | tab-open only | tab-open only | Slow-moving |
| `/api/triggers` | tab-open only | tab-open only | Historical |

### JS payload budget

| Bucket | Budget (gzipped) | Comment |
|--|--|--|
| Shell HTML + critical CSS | ≤ 20 KB | Inline above-the-fold styles |
| Non-critical CSS | ≤ 10 KB | Loaded async via `<link>` |
| Chart.js (CDN) | ~65 KB | Cached after first load |
| App JS modules total | ≤ 80 KB | Down from ~85 KB in current single-file (after dedup) |
| **Cold first paint over wire** | **≤ 100 KB** | Including HTML+CSS+app-JS, before Chart.js arrives |
| **Total cold load** | **≤ 175 KB** | Including Chart.js |

For comparison: a single `nytimes.com` mobile page is 5-8 MB. We're aiming for one fiftieth of that. Achievable.

### Decision

- **Add Page Visibility API hook.** Pause the 60s interval when `document.hidden`, fire one refresh on return.
- **Per-section cadence above** — hot endpoints stay 30-60s, slow endpoints move to 5min.
- **Keep polling, don't add SSE in v1.** Plan SSE as a v2 enhancement once mobile UI is stable.
- **Performance budget: ≤ 100 KB cold first paint**, ≤ 175 KB total cold load. Measure with Chrome DevTools Network tab on first deploy.

---

## Decisions made

1. **Stack:** vanilla JS + ES modules served as static files, no build step, Chart.js continues via CDN UMD tag.
2. **PWA:** ship manifest + apple-touch-icon + minimal service worker for shell+chart cache only; `/api/*` always go to network; document one-time PWA re-auth.
3. **Charts:** keep Chart.js v4, no second library, no zoom plugin in v1; tweak mobile config (animation off, DPR cap, index tooltip mode).
4. **Auth:** no code change to `dashboard/auth.py`; existing 1-year rolling cookie + per-request refresh handles iOS ITP and CF Access correctly; avoid WebSockets to dodge iOS 26 Private Relay bug.
5. **Code organization:** split `index.html` into ESM modules under `dashboard/static/{css,js,js/render,js/charts}/`, one render group per module, flat dependency graph through `state.js`/`fetch.js`/`format.js`/`theme.js`.
6. **Touch ergonomics:** 48×48 px global tap-target floor, `viewport-fit=cover` + safe-area-inset, no swipe-tab-nav, disable native PTR via `overscroll-behavior-y: contain`, no custom PTR in v1.
7. **Performance:** ≤100 KB cold first paint budget, Page Visibility API to pause polling when hidden, per-section cadence (hot endpoints 30-60s, slow endpoints 5min), SSE deferred to v2.

---

## References

External:
- [Chart.js v4 docs](https://www.chartjs.org/docs/latest/getting-started/integration.html)
- [Chart.js bundle size discussion #10163](https://github.com/chartjs/Chart.js/issues/10163)
- [chartjs-plugin-zoom mobile issue #311](https://github.com/chartjs/chartjs-plugin-zoom/issues/311)
- [TradingView Lightweight Charts v5](https://www.tradingview.com/blog/en/tradingview-lightweight-charts-version-5-50837/)
- [uPlot perf notes (cprimozic)](https://cprimozic.net/notes/posts/my-thoughts-on-the-uplot-charting-library/)
- [Index.dev Charts comparison 2026](https://www.index.dev/skill-vs-skill/tradingview-vs-lightweight-charts-vs-chartjs)
- [Cloudflare Access authorization cookie docs](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/)
- [Cloudflare WAF SameSite docs](https://developers.cloudflare.com/waf/troubleshooting/samesite-cookie-interaction/)
- [Cloudflare Access + Worker subrequest cookie collision](https://community.cloudflare.com/t/cloudflare-access-and-worker-subrequests/184512)
- [iOS 26 Safari WebSocket + CF Tunnel bug](https://www.jackpearce.co.uk/posts/debugging-websocket-upgrade-failures-safari-ios26/)
- [PWA iOS limitations 2025-2026 (MagicBell)](https://www.magicbell.com/blog/pwa-ios-limitations-safari-support-complete-guide)
- [PWA iOS storage isolation (Brainhub 2025)](https://brainhub.eu/library/pwa-on-ios)
- [Safari ITP cookie rules (JenTis)](https://www.jentis.com/blog/how-to-work-with-safari-itp-limitations)
- [Safari ITP first-party cookie 7-day cap (Snowplow)](https://snowplow.io/blog/tracking-cookies-length)
- [Preact size + comparison (Medium 2025)](https://medium.com/@marketing_96787/preact-vs-react-in-2025-which-javascript-framework-delivers-the-best-performance-f2ded55808a4)
- [WCAG 2.5.5 Target Size](https://www.w3.org/WAI/WCAG21/Understanding/target-size.html)
- [WCAG 2.5.8 Target Size Minimum](https://www.allaccessible.org/blog/wcag-258-target-size-minimum-implementation-guide)
- [Smashing Mag accessible tap targets](https://www.smashingmagazine.com/2023/04/accessible-tap-target-sizes-rage-taps-clicks/)
- [iPhone safe-area-inset (WebDong)](https://www.webdong.dev/en/shortpost/iphone-home-indicator-safe-area-css/)
- [overscroll-behavior (Bram.us)](https://www.bram.us/2017/12/10/customizing-pull-to-refresh-and-overflow-effects-with-css-overscroll-behavior/)
- [Page Visibility API (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Page_Visibility_API)

Internal:
- `dashboard/app.py` — Flask app, 38 routes
- `dashboard/auth.py:103-163` — auth chain (CF Access header → cookie → query → bearer)
- `dashboard/auth.py:90-100` — `_refresh_cookie` rolling 365-day refresh
- `dashboard/static/index.html:5` — viewport meta tag (needs `viewport-fit=cover` added)
- `dashboard/static/index.html:7` — Chart.js v4.4.1 CDN script
- `dashboard/static/index.html:450-483` — current responsive @media blocks
- `dashboard/static/index.html:834-848` — global state vars (target for `state.js` module)
- `dashboard/static/index.html:990-993` — lazy-loaded tab pattern (already correct)
- `dashboard/static/index.html:2988-3030` — existing static-fallback fetch logic (informs SW design)
- `dashboard/static/index.html:3147-3183` — `refresh()` flow (target for visibility-aware polling)
- `dashboard/static/index.html:3186-3201` — countdown + setInterval (target for visibility hook)
