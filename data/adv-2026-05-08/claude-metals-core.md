# Adversarial Review: metals-core subsystem (2026-05-08)

[P0] data/metals_loop.py:1491
**`SILVER_VELOCITY_WINDOW` deque accessed without locking.**
Problem: Fast-tick appends to `_silver_fast_prices` while main cycle iterates POSITIONS
concurrently. A tick during resize/read can hit a partial deque state.
Fix: `threading.Lock()` around all deque appends and reads.

[P0] data/metals_loop.py:4910-4915
**Stop-loss placement does not verify trigger > barrier.**
Problem: 3% distance check is from current bid only, not from MINI barrier. A stop
placed at or below the barrier triggers instantly at knock-out.
Fix: `if trigger_price <= barrier: log("SKIP barrier crash"); continue` before placement.

[P0] portfolio/exit_optimizer.py:184
**Hardcoded `usdsek: float = 10.85` baked into `MarketSnapshot` default.**
Problem: SEK/USD historical range 9.5–11.5; 3–8% mis-pricing on every ExitPlan that
relies on the default. Live warrant trading hits this.
Fix: Inject live FX from `fx_rates.py` at call site; remove the static default or set
it to NaN to force callers to provide it.

[P0] portfolio/fin_snipe.py:38
**Stop-loss list response shape assumed dict.**
Problem: `payload.get("orders", [])` fails if `api_get("/_api/trading/stoploss")`
returns a list directly — silent crash on mismatched response shape.
Fix: `if isinstance(payload, list): orders = payload` before `.get()`.

[P0] portfolio/metals_cross_assets.py:95-96
**`get_copper_data()` advertises Binance freshness but silently falls back to yfinance
EOD.**
Problem: When `price_source` fails the caller believes it has 7.7s-fresh data; actually
gets 15–30 min delayed data. No metric/log distinguishes the source.
Fix: Set `fetched_from` field on returned dict and emit a warning log when fallback
fires.

[P1] portfolio/microstructure_state.py:209
**`persist_state()` re-acquires `_buffer_lock` recursively.**
Problem: Locked section calls `get_microstructure_state()` which calls `record_ofi()`,
which itself reacquires the same lock. If thread is preempted between release and
reacquire, fast-tick interleavings corrupt OFI history ordering.
Fix: Either use RLock + audit re-entrance, or call `get_microstructure_state()`
outside the persist lock.

[P1] data/metals_loop.py:1498
**Velocity-alert dedupe key uses wall-clock time.**
Problem: `int((time.time() - 2) // 300)` not monotonic. NTP step backward can fire same
key twice in succession.
Fix: Use `time.monotonic()` or maintain `_silver_alerted_levels` set with explicit
membership check.

[P1] data/metals_loop.py:729 + 7348-7350
**POSITIONS dict mutated during fast-tick iteration.**
Problem: `_silver_fast_tick` reads `POSITIONS.items()` (line 1099) without holding any
lock; main cycle deletes/flips entries simultaneously. KeyError or stale-flag reads.
Fix: Threading.RLock around all POSITIONS access, or snapshot via `dict(POSITIONS)`
before iteration.

[P1] portfolio/metals_ladder.py:52-56
**`translate_underlying_target()` divides by `current_underlying_price` with no
zero-guard.**
Problem: Stale fetch_klines returning 0 (rare but possible) divides-by-zero, ladder
breaks silently inside try/except higher up.
Fix: `if current_underlying_price <= 0: return 0.0` early-return.

[P1] data/metals_loop.py:1467-1472
**`_silver_consecutive_down` counter triggered by `<` not `!=`.**
Problem: Two ticks reading the same cached price increment the counter (0.0 < 0.0 is
false, but with a tiny float jitter `0.0 < 0.000001` is true). Phantom "consecutive
down" events on stale data.
Fix: `if abs(price - prev_price) < eps: skip; elif price < prev_price - 0.001: increment`.

[P1] data/metals_loop.py:1043-1058
**Entry-tick loops run when no entry candidate exists.**
Problem: If `*_ENTRY_FAST_TICK_ENABLED` is true but no eligible position, `entry_tick_active`
stays True; a 0.1s/tick loop adds 6s+ of latency every 60s cycle. Performance cliff.
Fix: Gate on actual eligibility (open candidate) or market-hours window; short-circuit
when neither.

[P1] portfolio/orb_predictor.py:257-258
**`max(full_day, ...)` fails on empty list.**
Problem: Holiday or partial-day data leaves `full_day=[]`; `max()` raises ValueError;
wrapped or unwrapped, the whole prediction returns nothing.
Fix: `if not full_day: return None` before the max calls.

[P1] data/metals_loop.py:4743
**Iterating `POSITIONS` to count active silver positions without snapshot.**
Problem: Concurrent mutation makes the count nondeterministic; size-of-dict-changed
during iteration error in CPython is also possible.
Fix: `positions_copy = dict(POSITIONS); count = sum(...)`.

[P1] data/metals_loop.py:1423-1426
**Cached underlying prices used without staleness check.**
Problem: When live fetch fails, `_underlying_prices` cache is read with no max-age
guard; a 10-min-old price triggers false velocity alarms.
Fix: Timestamp cache entries; reject if older than 5 minutes.

[P2] portfolio/metals_orderbook.py
**Concurrent persistence of `microstructure_state.json`.**
Problem: Fast-tick + main cycle may both call atomic_write_json on the same target.
Verify atomic_write_json semantics in file_utils (rename-replace on Windows is not
atomic if dest open).
Fix: Single-writer rule (only main cycle persists); fast-tick only updates in-memory.

[P2] data/metals_loop.py:1502-1504
**Misleading constant name `SILVER_VELOCITY_ALERT_PCT = -0.8`.**
Problem: Threshold of -0.8% over 3 minutes is intraday noise, not a "rapid drop". The
log message "RAPID DROP" creates alert fatigue.
Fix: Rename to `_NOISE_FLOOR_PCT` and only log "rapid drop" when threshold passes
~-1.5%.

## Summary

5 P0 (concurrency on shared state, barrier-blind stop placement, hardcoded FX, response-
shape mismatch, silent fallback masquerading as fresh) + 11 P1 + 2 P2.
The metals loop's combined fast-tick + main-cycle architecture has multiple unguarded
shared-state reads — biggest correctness risk in the subsystem.
