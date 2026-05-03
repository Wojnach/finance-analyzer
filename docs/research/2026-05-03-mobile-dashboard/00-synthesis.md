# Mobile Dashboard Redesign — Research Synthesis

**Date:** 2026-05-03
**Branch:** `feat/mobile-dashboard-redesign-2026-05-03`
**Inputs:**
- Track 1 — `01-current-inventory.md` (39 widgets, 33 endpoints)
- Track 2 — `02-usage-signal.md` (gap → propose minimal access log)
- Track 3 — `03-user-moments.md` (7 user moments, M1-M7)
- Track 4 — `04-telegram-overlap.md` (top 5 mobile-home widgets, 42% of TG is saved-only)
- Track 5 — `05-comparable-products.md` (9 apps, heatmap pattern, anti-patterns)
- Track 6 — `06-tech-constraints.md` (7 decisions, ESM, Chart.js, PWA, polling)

## Where the tracks converged

1. **Bottom-nav, 4–5 destinations is universal** (Track 5: every one of 9 apps).
   Track 4 ranked: Equity / Positions / Heatmap / Health / Triggers+Decisions
   as the highest-value mobile views. Track 3's M1/M3/M4 align.
2. **Don't duplicate Telegram** — Track 4 explicit. Track 3 user-moments
   M2/M5 already framed dashboard as drill-down, not push.
3. **Vanilla JS + ES modules, no build step** — Track 6 decision; aligns with
   user's single-dev maintenance constraint (Track 3 indirectly).
4. **Long-press + bottom sheet for detail** — Track 5 strongest pattern;
   Track 6 confirms touch-OK.
5. **Pull-to-refresh is anti-pattern in v1** — both Track 5 (real-time data
   shouldn't need pull) and Track 6 (PTR nukes in-memory state).
6. **Per-section polling cadence + Page Visibility API** — Track 6
   battery + bytes; Track 4 implies 60s on Home is the right hot-path
   cadence.

## Where the tracks diverged or refined each other

1. **Heatmap representation.** My initial Track 3 instinct was per-ticker
   accordion. Track 5 cleanly proposed: rows = signals, cols = timeframes,
   one ticker at a time, sticky-leftmost, color-cell-only. **This wins** —
   it gives 14 visible signals × 7 timeframes per screen vs ~5 visible per
   ticker per screen.
2. **Tap-target floor.** Track 3 + first draft used 44 px. Track 6
   recommended **48 px** (Material spec; clears WCAG + Apple HIG with
   buffer). Adopted.
3. **Pull-to-refresh.** Initial spec had it as a primary refresh path.
   Track 5 + Track 6 both reject for v1. Adopted: rely on visibility-resume
   + manual refresh button.
4. **Messages tab priority.** Track 3 deprioritized as "evening / weekend
   thing". Track 4 surfaced that 42% of `telegram_messages.jsonl` is
   *saved-only* — the Messages tab is the SOLE interface for those entries.
   Promoted: keep Messages well-built and add a "Saved-only" filter.
5. **Equity Curve to Home.** Wasn't in initial Track 3 home priorities;
   Track 4 ranked it #1 because TG only carries point values, no curve.
   Adopted: P&L card on home includes a 24h sparkline.
6. **iOS PWA cookie isolation.** Track 6 raised: home-screen PWA has its
   own cookie jar separate from Safari → first launch needs SSO redirect.
   Documented as a known caveat, no code change.

## Final design decisions (recap from spec)

1. **Stack:** vanilla JS + ES modules served as static files, no build
   step. Chart.js v4 stays via CDN UMD `<script>`.
2. **Layout:** bottom-nav with 4 items (Home / Decisions / Signals / More).
   Mobile-first CSS, progressive enhance to wider screens.
3. **Home:** 5 cards (P&L+sparkline / Positions strip / Active consensus /
   Latest decision / System pulse), then collapsibles below the fold.
4. **Routing:** hash-based (`/#home`, `/#decisions/<id>`, etc.).
5. **PWA:** manifest + apple-touch-icon + minimal SW (cache shell + Chart.js
   only; `/api/*` always to network). Document one-time PWA re-auth.
6. **Charts:** Chart.js v4, animation off, DPR cap, index tooltip mode.
   No second library; no zoom plugin in v1.
7. **Polling:** Page Visibility API + per-section cadence (Home `/api/summary`
   60s, slow endpoints 5min). SSE deferred to v2.
8. **Touch:** 48×48 px tap target floor; long-press + bottom sheet drill;
   no swipe-tab-nav; PTR disabled (`overscroll-behavior-y: contain`).
9. **Performance budget:** ≤100 KB first paint, ≤175 KB total cold load.
10. **Migration:** parallel `/legacy` route preserves the old dashboard for
    14 days. Tests cover routing + asset presence; manual phone smoke-test
    documented in `docs/TESTING.md`.

## Open items deferred from the design

- Custom pull-to-refresh, or replace with SSE — v2.
- Per-user customization (drag-reorder home cards, hide cards) — single
  user, postpone.
- Replacement of Chart.js with Lightweight Charts — possible, but TBD on
  whether candlestick is worth the second library.
- Deep-link from Telegram to dashboard views — needs message-side change,
  track separately.
- Optional access log (`config.dashboard.access_log = true`) — Track 2's
  Track-2 recommendation to capture usage data going forward.
- Address `data/metals_loop.py:959` Telegram-bypass + LOOP-CONTRACT spam —
  separate issues, not part of this dashboard PR.
