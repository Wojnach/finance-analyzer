# Metals State Hardening Plan

## Addendum: Metals Loop Restart Wiring

Updated: 2026-03-11
Worktree: `Q:\finance-analyzer`

### Goal

Restore an actual auto-start / auto-restart path for `data/metals_loop.py` so the
metals execution loop survives crashes and machine logons without depending on a
manual launch.

### Findings

- The repo already has the intended wrapper for the metals loop:
  `scripts/win/metals-loop.bat`
- That wrapper auto-restarts `data/metals_loop.py` after crashes, but no active
  scheduled task currently owns it.
- The live task that looked like the metals owner, `PF-SilverORB`, is still wired
  to the legacy script `scripts/pf-silver-orb.bat`, and that script launches
  `data/silver_monitor.py`, not `data/metals_loop.py`.
- This explains the gap around the reported `silver301` event:
  the metals loop restart chain failed because the metals loop was never attached
  to Task Scheduler in the first place.
- The current metals wrapper also restarts forever even if a second launcher hits
  the singleton lock, because `data/metals_loop.py` exits cleanly on duplicate
  detection and the wrapper cannot distinguish that from a crash.

### Risks

- Registering a new live task can create duplicate loop trees if the wrapper does
  not stop on duplicate-instance exits.
- Repointing the existing `PF-SilverORB` task would blur responsibility between
  the silver sidecar and the brokered metals loop.
- Any task registration change should preserve current Telegram behavior and avoid
  disabling the existing silver monitor path unexpectedly.

### Execution Order

1. Add a distinct duplicate-instance exit code to `data/metals_loop.py`.
2. Teach `scripts/win/metals-loop.bat` to stop instead of restart-spinning on that
   duplicate exit code.
3. Add a dedicated Task Scheduler installer for `PF-MetalsLoop` that launches the
   canonical metals wrapper.
4. Update ops docs so the canonical owner of `data/metals_loop.py` is explicit.
5. Register the task, start it, and verify process/task state.

## Addendum: Silver Task Ownership Cleanup

Updated: 2026-03-11
Worktree: `Q:\finance-analyzer`

### Goal

Remove the ambiguity between `PF-SilverORB` and `PF-SilverMonitor` so there is a
single canonical scheduled-task owner for `data/silver_monitor.py`.

### Findings

- `PF-SilverORB` and `PF-SilverMonitor` both ultimately target
  `data/silver_monitor.py`.
- `PF-SilverORB` is the legacy path:
  - task action: `scripts/pf-silver-orb.bat`
  - that batch file launches `data/silver_monitor.py` directly
- `PF-SilverMonitor` is the maintained path:
  - task action: `scripts/win/silver-monitor.bat`
  - that wrapper already has duplicate-instance handling and cleaner restart logic
- The only unique operational advantage left on `PF-SilverORB` was the extra
  logon trigger fallback; that should live on `PF-SilverMonitor` instead.

### Risks

- Switching task ownership while a live monitor process is running can briefly
  leave no active silver monitor if the handoff is done in the wrong order.
- Leaving the legacy helper scripts untouched would allow future operators to
  recreate the old duplicated task topology by accident.

### Execution Order

1. Repoint legacy launcher/helpers to the canonical `scripts/win/silver-monitor.bat`.
2. Update `PF-SilverMonitor` install flow to include the logon fallback.
3. Stop the legacy `PF-SilverORB` process tree, start `PF-SilverMonitor`, and
   verify exactly one silver monitor tree is active.
4. Disable the `PF-SilverORB` task so ownership is unambiguous but rollback stays easy.
5. Update docs to mark `PF-SilverORB` as legacy / disabled.

## Addendum: Canonical Avanza Control Facade

Updated: 2026-03-11
Worktree: `Q:\finance-analyzer`

### Goal

Expose one canonical repo-level Avanza module for both account reads and
browser-session trading actions, then move the live metals / GoldDigger code to
that module instead of importing mixed helpers directly.

### Findings

- There is already a real Avanza library in `portfolio/`:
  - `portfolio/avanza_session.py` handles the BankID-backed Playwright session
  - `portfolio/avanza_client.py` exposes account reads plus TOTP-backed order APIs
- There is also a parallel metals-specific execution helper:
  - `data/metals_avanza_helpers.py`
  - it performs the working Playwright-page quote/order/stop-loss calls used by
    the metals loop and GoldDigger
- The current split is confusing for operators and maintainers:
  - strategy code imports `metals_avanza_helpers` directly for writes
  - other code imports `portfolio.avanza_client`
  - docs still point at `portfolio/avanza_client.py` even when the live runtime
    is actually using the metals helper path
- `portfolio/avanza_client.py` also contains a stale session-auth path that does
  not match the current `portfolio/avanza_session.py` API and should be repaired
  before we bless it as part of the shared backbone.

### Risks

- Replacing the metals helper implementation outright would be risky because it
  already works for the live brokered flows.
- Adding a facade without migrating live call sites would create a third Avanza
  path instead of reducing fragmentation.
- Trading writes must keep using the Playwright-authenticated page path until a
  fully verified alternative exists.

### Execution Order

1. Repair the stale session-auth flow inside `portfolio/avanza_client.py`.
2. Add `portfolio/avanza_control.py` as the canonical facade over the existing
   working account/session/helper implementations.
3. Point live strategy code (`metals_loop`, `metals_swing_trader`,
   GoldDigger runner/data provider) at the facade.
4. Add targeted tests for the repaired session path and facade behavior.
5. Update Avanza docs so operators know the canonical import path.

## Addendum: Layer 2 Message Readability Hardening

Updated: 2026-03-11
Worktree: `Q:\finance-analyzer`

### Goal

Make Layer 2 Telegram messages consistently human-readable and traceable without
changing trading behavior, alert categories, or Telegram enablement.

### Findings

- The malformed `*T1 CHECK*` message at `2026-03-10T19:40:04Z` was produced by
  the Layer 2 quick-check path, not by GoldDigger or the metals loop.
- `CLAUDE.md` still instructs the Layer 2 agent to write directly to
  `data/telegram_messages.jsonl` and call Telegram via raw `requests.post`.
- That bypasses the shared routing/sanitization path in
  `portfolio/message_store.py`, so bad control characters or mojibake can land
  in both the log and Telegram unchanged.
- The bad payload shape matches this failure mode:
  - embedded control bytes (`\u0005`, `\u0013`, `\u0000`)
  - mojibake / broken symbol sequences
  - readable structure otherwise, which points to model-output text quality
    rather than a broken scheduler or wrong sender

### Risks

- Over-aggressive sanitization could remove legitimate Markdown formatting.
- Updating the Layer 2 instructions must preserve the current rule that HOLD
  analysis still reaches Telegram normally.
- Historical malformed log lines should not be rewritten blindly because the
  JSONL file is also an audit trail.

### Execution Order

1. Add small shared text-sanitization helpers in `portfolio/message_store.py`.
2. Route `log_message()` and `send_or_store()` through the same cleaned text.
3. Update `CLAUDE.md` so Layer 2 uses `send_or_store(...)` instead of direct
   JSONL writes / raw Telegram HTTP calls.
4. Add targeted regression tests for control-char stripping and common mojibake
   normalization.
5. Run targeted tests only for this batch.

## Addendum: Notification Abbreviation Cleanup

Updated: 2026-03-11
Worktree: `Q:\finance-analyzer`

### Goal

Make the highest-volume Telegram notifications understandable to an operator who
does not already know the house abbreviations.

### Findings

- The user example came from `data/metals_loop.py` autonomous alerts, not from
  the main loop:
  - `bid`, `e`, `b`, `pk`, `DD`, `LLM`, `min`, `chr`, `T3`, `u`
- The main portfolio autonomous alerts in `portfolio/autonomous.py` use a
  parallel shorthand style:
  - `F&G`, `B/S/H` vote triplets, `P:` / `B:`, `acc`, `p46%`, `sh`
- Both paths are code-generated, which means we can improve readability
  consistently without depending on model prompting.
- Layer 2 / Claude-written messages still need prompt guidance, but the most
  repetitive readability pain is in these autonomous builders.

### Decision

Introduce a small shared notification-text helper and update the autonomous
builders to prefer plain labels:

- `Fear & Greed` instead of `F&G`
- `buy / sell / hold votes` instead of `4B/2S/14H`
- `Patient portfolio` / `Bold portfolio` instead of `P:` / `B:`
- `Consensus accuracy` / `confidence` instead of `acc` / `p46%`
- `Entry`, `Bid`, `Profit/loss`, `Off peak`, `Stop-loss`, `Drawdown`,
  `Tier 3`, `units` instead of terse local abbreviations

### Risks

- More readable text is longer, so the Apple Watch first line still needs to
  stay compact.
- The monospace ticker rows must remain short enough that Telegram messages do
  not balloon past the 4096-character limit on busy updates.
- Existing tests assert some old shorthand fragments and will need to be
  updated alongside the implementation.

### Execution Order

1. Add shared wording helpers under `portfolio/`.
2. Update `data/metals_loop.py` autonomous Telegram formatting.
3. Update `portfolio/autonomous.py` Mode A / Mode B Telegram formatting.
4. Add prompt guidance in `CLAUDE.md` to avoid unexplained abbreviations in
   model-written Telegrams.
5. Add/update targeted tests for both builders.
6. Run targeted tests for autonomous + metals formatting.

## Addendum: Runtime Singleton Hardening

Updated: 2026-03-10
Worktree: `Q:\finance-analyzer`

### Goal

Stop duplicate long-running monitor processes from stacking up in the live repo
without changing trading enablement or muting Telegram notifications.

### Scope

- `data/silver_monitor.py`
- `portfolio/golddigger/runner.py`
- `scripts/win/silver-monitor.bat`
- small shared helper for non-blocking singleton file locks

### Why

- `silver_monitor.py` currently has no duplicate-instance guard, so ad hoc
  launches can multiply and spam overlapping monitoring/analysis work.
- GoldDigger has already been moved under Task Scheduler, but a stray dry-run
  process can still start alongside the live signal-only instance and write to
  the same logs/state surfaces.
- `data/metals_loop.py` already protects itself with a singleton lock; the other
  long-running monitors should follow the same pattern.

### Risks

- A too-broad lock could block useful manual one-shot diagnostics.
- A duplicate exit path paired with an auto-restart wrapper can create a tight
  restart loop if the wrapper is not taught to stop on the duplicate exit code.
- Live-process restarts must preserve current behavior:
  - GoldDigger stays `trade_enabled=false`
  - Telegram notifications stay enabled
  - silver monitoring continues with a single canonical process

### Execution Order

1. Add a small shared lock helper with non-blocking file locking.
2. Add singleton wrappers to `silver_monitor.py` and GoldDigger runner.
3. Update the silver auto-restart wrapper to stop when a duplicate is detected.
4. Add targeted regression tests for duplicate-start behavior.
5. Run targeted tests.
6. Clean up duplicate live processes and restart only the canonical processes.

## Addendum: Main Loop Backbone Convergence

Updated: 2026-03-10
Worktree: `Q:\finance-analyzer`

### Goal

Converge the main loop, metals loop, GoldDigger, and silver monitoring onto the
same reusable backbone libraries without forcing them into one giant process or
one giant file.

### Findings

- The main loop already provides the strongest generic backbone:
  - cadence, crash handling, heartbeat, and scheduling in `portfolio/main.py`
  - market data collection and rate limiting in `portfolio/data_collector.py`
    and `portfolio/shared_state.py`
  - cross-system signal summaries in `portfolio/reporting.py`
  - durable JSON helpers in `portfolio/file_utils.py`
  - centralized Telegram/message storage in `portfolio/telegram_notifications.py`
    and `portfolio/message_store.py`
- The main loop is already the signal brain for metals:
  - `portfolio/tickers.py` defines `XAU-USD` and `XAG-USD` from Binance FAPI
  - `data/metals_loop.py` reads `data/agent_summary*.json` for XAU/XAG context
- GoldDigger is partially integrated already:
  - it reads compact summary context for XAU consensus, macro, Chronos, and
    event risk
  - it already reuses the shared yfinance limiter from `portfolio/shared_state.py`
- The biggest remaining duplication is infrastructure, not strategy logic:
  - direct Telegram senders in metals, silver monitor, and GoldDigger
  - separate config loading and runtime loop wrappers
  - duplicated Binance/FAPI and yfinance helper patterns
  - duplicated singleton lock / process supervision patterns
  - broker/session logic living outside the reusable `portfolio/` backbone
- Full physical merge into `portfolio/main.py` is the wrong move:
  - main loop cadence is 60s and multi-timeframe heavy
  - GoldDigger needs ~5s polling and live Avanza quote handling
  - silver monitor has a different job shape (10s checks + 5m external analysis)
  - broker failures should not take down the main research/signal loop

### Architecture Decision

Do not merge the systems into one loop.

Instead, create one shared backbone with three layers:

1. Research backbone
   - ticker definitions
   - market data collectors
   - macro context
   - signal summaries / compact contexts

2. Execution backbone
   - Avanza session lifecycle
   - quote fetching
   - holdings reconciliation
   - order placement / trade queue / stop helpers

3. Runtime backbone
   - singleton locks
   - loop cadence / heartbeat / crash backoff
   - config loading
   - notifications / message storage
   - state/log file helpers

Strategies then stay separate on top of that backbone:

- main loop = multi-asset research and signal generation
- metals loop = brokered metals execution / position management
- GoldDigger = fast gold intraday strategy
- silver monitor = analysis-only silver watcher

### Recommended Implementation Order

#### Phase 1: Shared Infrastructure

- Move all strategy Telegram sending to shared notification/message-store helpers.
- Move all singleton locking to `portfolio/process_lock.py`.
- Standardize config loading across metals, silver, and GoldDigger.
- Standardize JSON/JSONL state helpers across all loops.

#### Phase 2: Shared Market Context

- Extract a small summary-reader module for:
  - XAU/XAG consensus
  - macro context
  - Chronos/forecast reads
  - event windows
- Make metals loop and GoldDigger consume the same summary-reader contract.
- Reuse `portfolio/tickers.py` and `portfolio/data_collector.py` instead of
  local Binance/yfinance fetch paths where cadence permits.

#### Phase 3: Shared Avanza Execution Layer

- Extract `portfolio`-level Avanza runtime/service modules for:
  - session bootstrap and health checks
  - quote lookup by orderbook and api type
  - holdings discovery
  - order placement
  - stop-loss placement
- Repoint metals loop and GoldDigger to that shared execution layer.

#### Phase 4: Shared Runtime Kernel

- Extract a small reusable loop runner with:
  - target cadence
  - crash backoff
  - heartbeat
  - startup/shutdown hooks
  - singleton enforcement
- Keep separate loop cadences:
  - main loop: 60s market-aware
  - metals loop: 60s broker-aware
  - GoldDigger: 5s
  - silver monitor: 10s + 300s analysis sidecar

#### Phase 5: Optional Supervisor

- Only after the shared libraries stabilize, consider one supervisor process
  that owns:
  - one market-data cache
  - one Avanza session
  - multiple strategy tasks at different cadences

This is optional and should come last because it increases coupling and failure
blast radius.

### Immediate Low-Risk Wins

1. Replace custom Telegram senders in `data/metals_loop.py`,
   `data/silver_monitor.py`, and `portfolio/golddigger/runner.py`.
2. Replace the local metals-loop lock implementation with `portfolio/process_lock.py`.
3. Introduce a shared summary-reader module instead of bespoke
   `read_signal_data()`, `read_xau_consensus()`, and `read_macro_context()`.
4. Move Avanza helper imports behind a `portfolio` execution service boundary.

### Risks

- If we centralize Telegram naively, strategy messages may get muted by
  `layer1_messages=false`; execution/runtime alerts need a separate path from
  optional Layer 1 chatter.
- If we centralize broker state too early, one bad Playwright/Avanza failure can
  impact all strategy runtimes at once.
- If we move fast-price polling into the main loop, we risk making the signal
  loop slower and noisier.

Updated: 2026-03-10
Worktree: `Q:\wt\metals-state-store`
Branch: `metals-state-store`

## Goal

Harden the metals subsystem state files without disturbing the live environment.
The immediate target is the shared state used by `data/metals_loop.py`,
`data/metals_risk.py`, supporting scripts, and the Layer 2 trade-queue prompt.

## Findings

- The metals loop still persists critical shared state via raw `open(..., "w")`
  JSON overwrites:
  - `data/metals_positions_state.json`
  - `data/metals_trade_queue.json`
  - `data/metals_stop_orders.json`
  - `data/metals_spike_state.json`
- `data/metals_risk.py` persists `data/metals_guard_state.json` the same way and
  silently resets on read failure.
- The Layer 2 prompt in `data/metals_agent_prompt.txt` explicitly instructs the
  agent to overwrite `data/metals_trade_queue.json` directly.
- Several consumers outside the main loop still read the JSON files directly,
  including `data/silver_monitor.py` and docs such as `docs/STOP_LOSS_SETUP.md`.
- The repo already has two durability primitives that should be reused first:
  - `portfolio.file_utils.atomic_write_json`
  - `portfolio.signal_db.SignalDB` (SQLite WAL example)

## Decision

Use the smallest safe change first:

1. Keep the JSON file contract for now so existing readers and the dashboard are
   not broken.
2. Replace raw overwrites with atomic writes using shared file utilities.
3. Replace silent read-reset behavior with explicit logging and safe defaults.
4. Update the Layer 2 queue-writing guidance to use an atomic write pattern.

SQLite/WAL remains a follow-up option, but it is not the first batch because it
would require a wider contract migration for scripts, docs, and operator habits.

## Risks

- If malformed JSON is currently being tolerated silently, making failures more
  visible could surface operator issues that were previously hidden.
- Updating the prompt contract may require small test adjustments where file I/O
  is mocked broadly.
- The metals loop is a live-trading path, so changes must stay narrowly scoped to
  persistence helpers and must not alter trading decisions or thresholds.

## Batch Plan

### Batch 1: Tests First

- Add regression tests for:
  - atomic round-trips for positions, trade queue, stop orders, and spike state
  - malformed/corrupt JSON falling back with explicit logging
  - guard-state reads no longer failing silently

Files expected:
- `tests/test_metals_loop_functions.py`
- `tests/test_metals_risk.py`

### Batch 2: Shared State Hardening

- Introduce small shared helpers for metals JSON state reads/writes, or reuse
  `portfolio.file_utils` directly where that keeps the diff smaller.
- Update:
  - `data/metals_loop.py`
  - `data/metals_risk.py`

Goals:
- all shared JSON writes are atomic
- all read failures are logged explicitly
- no behavior change to trading logic

### Batch 3: Prompt and Docs

- Update `data/metals_agent_prompt.txt` so the trade queue is written atomically.
- Update affected docs to match the new state-handling contract and note the
  rationale.

Files expected:
- `data/metals_agent_prompt.txt`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/STOP_LOSS_SETUP.md`

## Verification

Targeted first:

- `tests/test_metals_loop_functions.py`
- `tests/test_metals_risk.py`
- `tests/test_metals_loop_autonomous.py`
- `tests/test_unified_loop.py`

Broader follow-up if the targeted slice stays green:

- `pytest -n auto`
- `ruff check data/metals_loop.py data/metals_risk.py portfolio/file_utils.py`

## Rollback

1. Revert the batch commit in the worktree branch.
2. Leave live processes untouched until the branch is reviewed and explicitly
   merged.
3. If prompt/docs changes prove noisy, revert those separately from the runtime
   persistence helper changes.

## Addendum: 4-Area System Improvement Plan

Updated: 2026-03-11
Branch: `improve/4areas-2026-03-11`

### Area 1: Layer 2 Agent Completion Rate

**Problem:** Agent completion is not tracked post-exit. Exit codes are lost. No rolling
metrics on success/failure rate by tier.

**Batch 1A — Completion tracking (HIGH)**
Files: `portfolio/agent_invocation.py`
- Add `check_agent_completion()` called from main loop each cycle
- Poll `_agent_proc.poll()`, log exit_code/duration/tier to `data/invocations.jsonl`
- Track rolling completion rate (last 24h) in health state
- Surface in agent_summary_compact.json

**Batch 1B — Incomplete session detection (MEDIUM)**
Files: `portfolio/agent_invocation.py`, `portfolio/health.py`
- After agent exits, check if journal + Telegram were written
- Log as "incomplete" vs "failed" vs "success"

### Area 2: Signal Accuracy

**Problem:** Consensus 48.1%. F&G 31.4%, MACD 31.6%.

**Batch 2A — F&G regime gating (HIGH)**
Files: `portfolio/signal_engine.py`
- F&G abstains in trending regimes (trending-up/trending-down)
- Only votes in ranging/high-vol where contrarian logic works

**Batch 2B — Per-ticker signal weighting (MEDIUM)**
Files: `portfolio/signal_engine.py`, `portfolio/accuracy_stats.py`
- Use per-ticker accuracy where available (50+ samples)
- Fall back to global weights when insufficient data

### Area 3: Avanza Execution Reliability

**Batch 3A — Pre-trade validation (HIGH)**
Files: new `portfolio/trade_validation.py`
- Bid/ask spread check (reject if >2%)
- Position size limit (max 50% cash)
- Cash verification
- Price sanity (reject if >5% from last known)

### Area 4: Better Data Sources

**Batch 4A — VIX via yfinance (HIGH)**
Files: `portfolio/data_collector.py`, `portfolio/reporting.py`
- Fetch `^VIX` via yfinance, cache 15min TTL
- Surface in agent_summary_compact.json → macro.vix
- Use for regime detection: >30=high-vol, <15=complacent

**Batch 4B — Market breadth from existing data (LOW)**
Files: `portfolio/reporting.py`
- % above 200-SMA / 50-SMA from existing 19 tickers
- Add to macro section

### Priority: 1A+2A+3A+4A (parallel) → 1B+2B → 3B+4B

## Addendum: Unify API Boilerplate (DRY Refactor)

Updated: 2026-03-11

### Problem

9 API modules reinvent the same patterns instead of using existing shared infrastructure:
- `http_retry.py` has `fetch_with_retry()` but callers duplicate JSON parsing + error handling (~12 blocks)
- `shared_state.py` has `_cached()` but 3 modules roll their own TTL cache (~60 lines)
- `circuit_breaker.py` exists but `alpha_vantage.py` reimplements it manually (~30 lines)
- `api_utils.py` has `load_config()` but 3 modules load config.json themselves
- Rate limiters exist in `shared_state.py` but `macro_context.py` uses `time.sleep()`

### Batch 1: `fetch_json()` helper in `http_retry.py`

Add `fetch_json()` that wraps `fetch_with_retry()` + `raise_for_status()` + `.json()`.

Migrate callers (12 blocks across 9 files):
- `data_collector.py` (2), `alpha_vantage.py` (1), `sentiment.py` (2),
  `onchain_data.py` (1), `futures_data.py` (1), `fx_rates.py` (1),
  `fear_greed.py` (1), `funding_rate.py` (1), `macro_context.py` (2)

Risk: `data_collector.py` raises `ConnectionError` on None — `fetch_json()` returns
`default` instead, so adjust callers that catch `ConnectionError`.

### Batch 2: Delegate local caches to `shared_state._cached()`

- `funding_rate.py`: `_cache` dict → `_cached("funding_rate_{ticker}", FUNDING_RATE_TTL, ...)`
- `fx_rates.py`: `_fx_cache` dict → `_cached("fx_usd_sek", 900, ...)` (keep 10.85 fallback)
- `macro_context.py`: 3 local caches → `_cached("macro_{metric}", 300, ...)`

Risk: `fx_rates.py` has stale-data Telegram alerting — keep alert, delegate caching.

### Batch 3: CircuitBreaker + config + rate limiters

- `alpha_vantage.py`: manual CB → `CircuitBreaker("alpha_vantage", 3, 300)`
- `onchain_data.py`: `json.loads(config.read_text())` → `api_utils.load_config()`
- `macro_context.py`: `time.sleep(0.1/0.4)` → `_binance_limiter.wait()` / `_alpaca_limiter.wait()`

Risk: `alpha_vantage.py` CB has daily budget tracking interleaved — keep budget separate.

### Not Doing

- Persistent file cache unification (unique per module, leave as-is)
- Sentiment model paths → config (separate task)
- Data collector source dispatch (correct design, not duplicated)
