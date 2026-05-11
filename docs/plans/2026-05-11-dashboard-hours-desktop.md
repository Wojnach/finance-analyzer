# Plan: Unified trading hours + Desktop-mode toggle + Two investigations

Date: 2026-05-11
Branch: `feat/dashboard-hours-desktop-2026-05-11`
Worktree: `.worktrees/feat-dashboard-hours-desktop`

## Background

User reported three issues against the dashboard:

1. **Dashboard says loops are "OUTSIDE_HOURS"** at 14:23 CEST. The constant in
   `dashboard/trading_status.py` is 15:30–21:55 Europe/Stockholm — GoldDigger's
   US-focused window — incorrectly applied to all four Avanza bots.
   Correct intent: combined EU + US trading window 08:30–21:30 CET.

2. **`/api/avanza_account` shows positions Beammwave B / NextEra / Vertiv** on
   account 1625505. User reports these are NOT in his trading account.
   Background agent investigating root cause.

3. **Dashboard mobile-first layout is uncomfortable on desktop.** User wants a
   header toggle so PC browsers can opt into a wider desktop layout, while
   mobile remains the default.

A fourth concern surfaced from the contract violations dashboard chip:
`signal_log_reconciliation` is escalating 22x consecutive — JSONL 1411 entries
vs SQLite 7934. Background agent investigating.

## Goals

| # | What | Files (estimate) |
|---|---|---|
| 1 | Unify session window to 08:30–21:30 Europe/Stockholm across all four Avanza bots | `dashboard/trading_status.py`, `portfolio/golddigger/config.py` |
| 2 | Add desktop-mode toggle button in dashboard header; persist in localStorage; CSS forces ≥1024px layout when active | `dashboard/static/index.html`, `dashboard/static/css/layout.css`, `dashboard/static/css/responsive.css`, `dashboard/static/js/main.js` (or new `desktop-mode.js`) |
| 3 | (Pending agent) Fix Avanza positions account filter if root cause confirms a bug | TBD by agent finding |
| 4 | (Pending agent) Fix signal_log JSONL/SQLite divergence if real | TBD by agent finding |

## Why

- **Hours**: dashboard chip is the user's primary at-a-glance health view. A
  false "OUTSIDE_HOURS" status during the EU session reads as "system idle"
  when bots are in fact eligible to trade — actively misleading.
- **Desktop toggle**: dashboard is mobile-first per the redesign, but the user
  uses it from a desktop browser daily. The bottom-nav and narrow column waste
  horizontal screen real estate without delivering mobile ergonomics. A toggle
  preserves the mobile default for phone use while letting desktop reclaim
  the layout.

## What could break

- **Hours change**: GoldDigger currently only trades during US-overlap hours
  (15:30–21:55). Widening to 08:30–21:30 lets it try entries during EU-only
  hours when gold typically moves less. Mitigation: this is a *config*
  default; the existing entry signal threshold and spread cap still gate it.
  If it makes bad trades, revert in one commit.
- **Desktop toggle**: incorrectly written CSS could break the mobile-default
  layout. Mitigation: gate desktop styles behind `:root.desktop-mode` so the
  default code path is unchanged. Test on phone-width viewport first.
- **Worktree symlinks**: config.json is not replicated → full pytest will
  spew failures unrelated to this change. Run targeted tests only and
  document any flags.

## Execution order

1. Plan doc committed to worktree
2. Apply hours edits in worktree (2 files, 1 commit)
3. Implement desktop-mode toggle (3-4 files, 1 commit)
4. Run targeted pytest
5. Wait for background agents on positions + signal_log; integrate fixes only if root cause is small + safe; otherwise spin off into follow-up tickets
6. Codex review on worktree branch HEAD
7. Address P1/P2; document P3 deferral
8. Merge to main fast-forward
9. Push via cmd.exe
10. Worktree cleanup

## Out of scope (deferred)

- Per-bot session windows (each bot reading its own config) — the current
  unified 08:30–21:30 covers all four. Per-bot windows can come later if
  GoldDigger needs back to US-only.
- A "force refresh all caches" UI — the per-view Refresh buttons already
  exist on Avanza tab. Other views auto-poll. Out of scope.
- Avanza account scoping if root-cause turns out to be DEFAULT_ACCOUNT_ID
  mapping to a different account than user thinks — that needs user
  confirmation, not a unilateral fix.

## Notes for future-self

- The dashboard chip displays "OUTSIDE_HOURS" per-bot. If GoldDigger
  legitimately re-narrows back to US-only later, push the per-bot window
  back into `_golddigger` instead of the module constant.
- The desktop toggle uses `:root.desktop-mode`. Any future CSS gated only by
  `@media (min-width: 1024px)` will need a mirror rule under
  `:root.desktop-mode { ... }` to honor the toggle.
