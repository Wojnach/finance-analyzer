# Swedbank book — handoff / resume point

**Written 2026-07-31 so a future session can pick this up cold.** Read this
first, then `docs/PLAN.md` (design + premortem) and `docs/SESSION_PROGRESS.md`.

## Where things are

Worktree `.worktrees/swedbank-book`, branch `feature/swedbank-book`, merged
fast-forward into local `main`. **Nothing is pushed to GitHub and nothing should
be** — see Privacy below. herc2 was synced to `0663f5d7` via
`scripts/deck/sync-repo-to-herc.sh`.

## Privacy — read before committing anything

`github.com/Wojnach/finance-analyzer` is **PUBLIC** (verified: anonymous API
returns `"private": false`). The operator decided 2026-07-31 to make it private
and to stop pushing entirely; only the Deck and herc2 need the code.

- `data/swedbank_*` is gitignored **and** in `.git/info/exclude` (the latter
  covers every worktree regardless of which branch is checked out — a
  branch-only rule left the real book exposed as `??` in the main checkout).
- No real quantities, cost basis, account totals, P&L or account labels in git.
  Tests are synthetic. Docs use A/B/C.
- Never route this data through `dashboard/export_static.py` or
  `dashboard/static/api-data/` — served with **no auth**.
- **OPEN: 11 credentials sit in public git history at commit `338b6000`**
  (added 2026-03-15 15:27, untracked 83 min later in `d9dbe707`; untracking does
  not remove history). Verified publicly fetchable. Tracked as pending-pickup
  `ROTATE-LEAKED-CREDS`, which prints at every session start. Binance/Alpaca are
  read-only in this codebase (verified: zero order/withdraw calls), but the
  Telegram token and api_server jwt/password are not.

## What is DONE and verified live

| Piece                              | File                                                    | Status                                        |
| ---------------------------------- | ------------------------------------------------------- | --------------------------------------------- |
| Pinned instrument table (26)       | `portfolio/swedbank/instruments.py`                     | 26/26 verified against upstream names         |
| Derivation (FX solve, qty, cost)   | `portfolio/swedbank/snapshot.py`                        | reconciles to <2 SEK on the real book         |
| Pricing (Avanza + Alpaca fallback) | `portfolio/swedbank/pricing.py`                         | 26/26 live in ~1.5s                           |
| Book model + valuation             | `portfolio/swedbank/book.py`                            | live against the real book                    |
| CLI `show` / `quotes`              | `portfolio/swedbank/cli.py`                             | working                                       |
| Monitoring loop                    | `data/swedbank_loop.py`                                 | 2 cycles verified, lock + heartbeat + SIGTERM |
| systemd unit                       | `scripts/deck/install-swedbank-loop.sh`                 | installed, **not enabled**                    |
| Dashboard route + tab              | `dashboard/app.py` `/api/swedbank`, `views/swedbank.js` | 200 with auth, 401 without                    |
| Deck→herc sync                     | `scripts/deck/sync-repo-to-herc.sh`                     | synced successfully                           |

**137 offline tests pass.** Pre-change baseline was 38 failures; the suite is
flaky (37–54 across runs) so ALWAYS capture a baseline before judging. Every
apparent new failure was confirmed pre-existing on `main`.

## SIGNALS LAYER — now WORKING (updated 2026-07-31)

`portfolio/swedbank/ohlcv.py` + `portfolio/swedbank/signals.py` are built and
verified against live Avanza data. 143 offline tests pass.

Live proof: NVDA -> SELL conf 0.41 off 252 Avanza bars, 8 buy / 7 sell / 74 hold
across 9 applicable signals, 4 price targets with fill probabilities.
MINI-TSMC -> SELL conf 0.48 computed on its **TSM underlying**, confirming the
leveraged-product mapping.

Three shape bugs were found only by running it, never by reading:
- `generate_signal` returns a **3-tuple** `(action, confidence, extra)`, not a
  dict (canonical caller: `portfolio/main.py:512`).
- Votes live at `extra["_votes"]` / `extra["_raw_votes"]`. Guessing
  `"votes"`/`"signals"` yielded all-zero counts — a plausible but empty
  consensus, the exact silent-wrongness class this subsystem exists to avoid.
- `compute_targets` returns `extremes`/`targets`/`recommended`. There is no
  `expected_move_pct`.

Confirmed safe: `flush_sentiment_state()` is called from exactly one place,
`main.py:757` (Layer 1 only). This module never calls it, so the cross-process
clobber risk does not apply. The sentiment signal does still *compute*
(read-only, in-memory), which costs latency and hits Reddit — Reddit currently
403s, harmlessly.

### Bugs found and fixed in the signals layer (all silent, none raised)

Four string/enum contracts were assumed from the caller's vocabulary instead of
read from the callee. Every one produced confident, well-formed, WRONG output:

1. `generate_signal` returns a 3-tuple, not a dict — crashed every call.
2. `extra["regime"]` does not exist; it is `extra["_regime"]`. Regime was always
   empty, so `compute_targets(regime=...)` ran as regime-unknown everywhere.
3. `side` was `"LONG"`/`"SHORT"`; `price_targets` branches on lowercase
   `"buy"`/`"sell"` (price_targets.py:93,137,281-287). Every comparison fell to
   the else branch — trajectories ran with inverted directional logic while
   still returning plausible targets.
4. `_hours_remaining` hardcoded UTC constants (15.5 STO / 20.0 US) that are only
   correct during summer time — off by an hour for ~5 months a year — and could
   return 70h+ over a weekend, scaling the projection by sqrt(70). Rewritten
   with `zoneinfo`, returns `(hours, market_open)`, and caps to one session
   (6.5h US / 8.5h STO) when the market is shut.

**Lesson worth keeping: verify every cross-module string against its consumer.**
Reading this module's own code would not have caught any of the four.

FIXED: `projected_range_pct` was always None because `extremes` is a PERCENTILE
dict (`p10/p25/p50/p75/p90`), not high/low. Now reports the p10-p90 band.

### Signals are wired END-TO-END (2026-07-31, final state)

Loop computes signals every `SIGNAL_EVERY_N_CYCLES = 15` — a full pass is ~42s
vs ~1.5s for prices, so per-cycle would be wasteful and would hammer the shared
Avanza session. Between passes the prior result carries forward with its
ORIGINAL `signals_computed_at`. Signal failure cannot break pricing.

Route exposes `signals_age_s` separately from `snapshot_age_s` (different
clocks). Both derive from the payload's own timestamps, never a `stat()` —
`_read_json` is TTL-cached and pairing cached content with a fresh stat reports
stale data as fresh.

**Verified live: 26/26 signals, 0 errors, 41.8s; prices 26/26 in 1.47s;
endpoint returns snapshot_age_s 56.9 alongside signals_age_s 15.0.**

**172 offline tests pass** (32 of them covering ohlcv + signals specifically).

Remaining known gaps: the Fable 5 review never delivered (agents hit their
session limit at 16:34); P0-1 (api_get performs browser recovery internally) is
bounded to 3 failures but not properly fixed; `reconcile()` still not wired into
`cmd_sync`; `require_auth` fails OPEN if `dashboard_token` is ever absent.

## PREVIOUSLY IN FLIGHT (superseded)

New, **untested and uncommitted at time of writing**:

- `portfolio/swedbank/ohlcv.py` — Avanza price-chart → DataFrame, Alpaca
  fallback for US names only. Stockholm has no fallback and RAISES.
- `portfolio/swedbank/signals.py` — `evaluate()` / `evaluate_universe()` /
  `applicable_for()` / `log_snapshot()`, plus `_trajectory()` via
  `price_targets.compute_targets`.

Subagents were dispatched for: `tests/test_swedbank_signals.py` (agent
`sig-tests`), and loop+route+view wiring (agent `wire-ui`). If those did not
finish, their briefs are reproducible from the sections below.

### Remaining work, in priority order

1. Verify `signals.evaluate()` end-to-end against live Avanza data.
2. Wire signals into `data/swedbank_loop.py::cycle()` — every
   `SIGNAL_EVERY_N_CYCLES=15` cycles, NOT every cycle (each signal is an OHLCV
   fetch + indicator computation). Must never break pricing: try/except, store
   `{"error": ...}`, keep the valuation.
3. Expose `signals` + `signals_age_s` in `/api/swedbank` (heredoc edits only).
4. Render in `views/swedbank.js`; an errored signal must NOT render as HOLD.
5. Re-run the full suite against the 38-failure baseline; adversarial review.

## Hard-won constraints (do not relearn these)

- **`dashboard/app.py`, `portfolio/signal_engine.py`, `dashboard/system_status.py`,
  `portfolio/accuracy_stats.py`, `dashboard/trading_status.py`,
  `portfolio/loop_processes.py` are NOT black-clean.** Editing them with
  Edit/Write triggers a format-on-save hook that rewrites the whole file. Use
  `python3 - <<'PYEOF'` heredoc patches with `assert src.count(old)==1`.
- **A fresh git worktree has no gitignored runtime files.** `config.json`,
  `data/avanza_session.json`, `data/avanza_storage_state.json` must be
  symlinked in or every live-data call fails while unit tests pass.
- **`.gitignore`'s `_*.py` scratch rule also matches `__init__.py`**, silently
  dropping new packages' init files. Fixed with `!**/__init__.py`.
- **Never add these tickers to `tickers.STOCK_SYMBOLS`.** `alpha_vantage.py:238`
  iterates it against a 25/day budget; 19 extra names starve Tier-1's
  fundamentals. Also drags in `earnings_calendar.py` and NYSE-hours GPU gating
  (wrong session for the Stockholm half). Carry `asset_class` on the instrument
  instead.
- **Never call the sentiment vote path.** `signal_engine.flush_sentiment_state()`
  is a whole-dict overwrite from a per-process copy — a second process clobbers
  Layer 1's tickers.
- **Never write to `data/signal_log.jsonl`.** `accuracy_stats.signal_accuracy()`
  blends all tickers into one global per-signal figure Tier-1 falls back on, and
  our rows would evict real history from its 50k-row tail. Use
  `data/swedbank_signal_log.jsonl`.
- **Avanza calls must stay sequential.** The real-money metals loop shares the
  session. A full 26-instrument sweep is 1.5s — 2.5% duty cycle at 60s — so
  concurrency buys nothing and risks the Playwright context.
- **`ssh herc2 true` always fails** — herc2 is Windows, there is no `true`.
- **Windows OpenSSH scp rejects `/c/Users/...`** — use home-relative paths.

## Known-open findings (from Codex + Claude adversarial reviews)

Fixed: FX hard-coded 10.50 placeholder, partial-cost-basis P&L inflation, FX
ambiguity, stale-quote validity, non-positive marks, heartbeat leaking the book
total, prefix-match wrong-instrument, the pinned-ID test that verified nothing,
`--once` lock bypass, `sync` exit code, `keys=[]` sweeping everything,
integration tests running live by default, snapshot age from a cached read.

**Still open:**

- **P0-1** `api_get` internally performs browser recovery, so this loop can
  disturb the metals session. Bounded to 3 consecutive failures instead of 26;
  the real fix is a read-only Avanza client sharing no browser context.
  `TODO: MANUAL REVIEW` in `pricing.py`.
- **P1** `reconcile()` is written and tested but not wired into `cmd_sync`.
- **P1** `parse_markdown_table` silently skips unparseable rows.
- **P2** `require_auth` fails OPEN when `dashboard_token` is absent from config.
- **P2** cached quotes have no maximum age — a 3-week-old mark stays in totals.
- **P2** `/api/swedbank` `loop.running` never checks heartbeat freshness.
- **P2** `acquire_singleton_lock` TOCTOU (inherited from `oil_loop.py`), plus a
  local bug: `except (OSError, ProcessLookupError)` precedes
  `except PermissionError`, and PermissionError subclasses OSError, so the
  assume-alive branch is unreachable.

## Commands

```bash
cd /home/deck/projects/finance-analyzer
.venv/bin/python -m portfolio.swedbank show      # live valuation
.venv/bin/python -m portfolio.swedbank quotes    # raw sweep
.venv/bin/python -u data/swedbank_loop.py --once
.venv/bin/python -m pytest tests/test_swedbank_*.py -q
scripts/deck/with-herc.sh scripts/deck/sync-repo-to-herc.sh   # sync to herc2
systemctl --user enable --now pf-swedbank        # NOT enabled by default
```

The loop is deliberately not enabled: every other `pf-*` unit is disabled and
auto-starting one would override a pause the operator chose.
