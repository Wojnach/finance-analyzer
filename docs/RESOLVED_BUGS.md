# Resolved Bugs Archive

Historical bug tracking from `docs/SYSTEM_OVERVIEW.md`, archived 2026-05-04.
All entries below have been fixed. See git history for implementation details.

## 2026-03 (BUG-15 through BUG-159)

- BUG-15 through BUG-22: Fixed 2026-03-08 session
- BUG-23 through BUG-27: Fixed 2026-03-09 (signal validation, None ticker, OSError, heartbeat)
- BUG-28: Enhanced signal failures silently count as HOLD (addressed by ARCH-12)
- BUG-29: `_vote_correct()` treats 0% change as incorrect (fixed with `_MIN_CHANGE_PCT`)
- BUG-30: `load_json()` TOCTOU race — fixed 2026-03-10
- BUG-31/84: `_compute_adx()` not cached (fixed 2026-03-19)
- BUG-34 through BUG-38: Fixed 2026-03-11 (portfolio_mgr TOCTOU, health TOCTOU, inversion cap)
- BUG-39 through BUG-46: Fixed 2026-03-12 (agent completion, digest isolation, TOCTOU-safe I/O)
- BUG-47 through BUG-50: Fixed 2026-03-14 (IO safety sweep)
- BUG-51: Signal failure tracking ephemeral (fixed 2026-03-16)
- BUG-52: `total_applicable` hardcoded (fixed 2026-03-16)
- BUG-71 through BUG-79: Fixed 2026-03-18 (golddigger/elongir config, dead variables)
- BUG-80 through BUG-84: Fixed 2026-03-19 (sentiment stopwords, ADX cache, ruff)
- BUG-85 through BUG-100: Fixed 2026-03-20 (thread safety, NaN, confidence penalties, circuit breaker)
- BUG-101 through BUG-106: Fixed 2026-03-21 (crash safety, thread safety, alert routing)
- BUG-107 through BUG-114: Fixed 2026-03-22/23 (zero-division, digest, outcome tracker, forecast)
- BUG-115 through BUG-127: Fixed 2026-03-24/25 (structure logging, trigger prune, fx_rates, FOMC, health)
- BUG-128 through BUG-131: Fixed 2026-03-26 (avanza offset, Playwright locks, dashboard cache, Telegram truncation)
- BUG-133 through BUG-139: Fixed 2026-03-27 (accuracy cache, regime horizon, utility horizon, SQLite leak)
- BUG-143 through BUG-148: Fixed 2026-03-29 (unanimity penalty, forecast regime, meta_learner)
- BUG-157 through BUG-159: Fixed 2026-03-31 (loop var capture, undefined datetime, raise chaining)

## 2026-04 (BUG-160 through BUG-242)

- BUG-160: 3 signals missing from SIGNAL_NAMES — fixed 2026-04-01
- BUG-161: metals_loop.py raw JSONL appends — fixed 2026-04-01
- BUG-165: llama_server.py model swap race condition — fixed 2026-04-02
- BUG-166: shared_state thundering herd — fixed 2026-04-02
- BUG-168 through BUG-170: Fixed 2026-04-04 (llama_server, indicators lock, fear_greed atomic)
- BUG-176 through BUG-182: Fixed 2026-04-08 (concentration, pool timeout, collector timeout, ADX, GPU lock)
- BUG-183/184: Fixed 2026-04-09 (dead code, shadowed test)
- BUG-196 through BUG-199: Fixed 2026-04-15 (relative paths, dead code, import retry, trigger dedup)
- BUG-200 through BUG-205: Fixed 2026-04-16 (auth detection, journal grace, monotonic timing)
- BUG-206 through BUG-213: Fixed 2026-04-20/21 (regime mismatch, signal exception, indicators poison, config destroy, order limit, rate limiter, loading timestamps, drawdown wiring)
- BUG-214 through BUG-218: Fixed 2026-04-21 (drawdown, fx lock, MC seed, swing exit, econ bias)
- BUG-219 through BUG-221: Fixed 2026-04-23/24 (trade guards pnl_pct, outcome tracker, daily_digest tz)
- BUG-230 through BUG-235: Fixed 2026-04-28 (CORS, heartbeat, portfolio NaN, fish monitor, signal engine, dashboard errors)
- BUG-236 through BUG-242: Fixed 2026-04-30 (crypto datetime, accuracy imports, singleton lock, ruff)
- BUG-243/244: Fixed 2026-05-01 (encoding on open() calls)

## Architecture (ARCH-10 through ARCH-30)

All completed. Key items: signal validation (ARCH-10), confidence caps (ARCH-11),
failure tracking (ARCH-12), dynamic total_applicable (ARCH-14), JSONL utilities (ARCH-15),
accuracy blending (ARCH-23/24), regime context (ARCH-25/26), trade guards helper (ARCH-29).

## Refactoring (REF-3 through REF-55)

All completed. 2000+ ruff auto-fix violations, 100+ manual fixes across 70+ sessions.

## Deferred (not fixed)

- ARCH-17: main.py re-exports 100+ symbols (obscures boundaries)
- ARCH-18/BUG-162: metals_loop.py monolith (7,699 lines)
- ARCH-19: No CI/CD pipeline
- ARCH-20: No type checking (mypy)
- ARCH-21: autonomous.py 500+ line functions
- ARCH-22: agent_invocation.py module-level globals
- BUG-132: orb_predictor.py fetches 5000+ candles uncached
- BUG-149: meta_learner orphaned (predict() never called)
- TEST-1: GPU gate zero test coverage
- TEST-3: 26+ pre-existing test failures
