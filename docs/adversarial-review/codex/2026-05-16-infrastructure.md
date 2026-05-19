## [P1] Failed Telegram commands are consumed in-process
**File:** portfolio/telegram_poller.py:161  
**Bug:** `_handle_update()` advances `self.offset` before dispatch. If `on_command()` raises, the persisted offset is left old, but the running poller now requests updates after the failed command.  
**Why it matters:** A transient Avanza/session error on `sold MSTR ...` drops the command until a process restart; portfolio state stays wrong while the bot keeps running.  
**Fix:** Only advance the in-memory offset after a settled command, or restore `self.offset = prev_offset` on raised dispatch and retry with backoff/idempotency.

## [P1] Successful Telegram commands can replay after offset write failure
**File:** portfolio/telegram_poller.py:105  
**Bug:** `_save_offset()` swallows persist failures after a command has already been processed.  
**Why it matters:** If `bought 10 MSTR ...` updates state but `telegram_poller_state.json` cannot be written, the next restart re-fetches the same Telegram update and applies the buy again.  
**Fix:** Store processed `update_id`s transactionally with the command side effect, or fail-stop/alert when offset persistence fails after a state-mutating command.

## [P1] JSONL rewrites can discard concurrent appends
**File:** portfolio/file_utils.py:287; portfolio/file_utils.py:349  
**Bug:** `atomic_write_jsonl()` and `prune_jsonl()` replace JSONL files without taking `jsonl_sidecar_lock()`, while appenders use that lock.  
**Why it matters:** If a signal/trade append lands between prune/rewrite read and `os.replace()`, the append is silently lost.  
**Fix:** Hold the same sidecar lock across read, filter, temp write, fsync, and replace for every JSONL rewrite.

## [P2] GPU stale-lock breaker can delete a live lock
**File:** portfolio/gpu_gate.py:129  
**Bug:** `_try_break_stale_lock()` reads stale lock metadata, then unconditionally unlinks the path. Another waiter can delete the stale file, acquire a new lock, and then this process unlinks the new live lock.  
**Why it matters:** Two LLM workers can enter the GPU critical section concurrently, causing OOM, failed inference, or corrupt timing around trading signals.  
**Fix:** Delete only if the lock contents/mtime still match what was read, or replace the PID-file scheme with an OS-level interprocess lock.

## [P2] Queued cache keys can get stuck forever
**File:** portfolio/shared_state.py:201  
**Bug:** `_cached_or_enqueue()` checks `key not in _loading_keys` but never evicts expired `_loading_keys`; the stuck-key eviction exists only in `_cached()`.  
**Why it matters:** If an LLM batch flush crashes, future calls stop enqueueing that key and eventually return `None` forever after stale data expires.  
**Fix:** Run the same `_loading_timestamps` timeout eviction inside `_cached_or_enqueue()` before deciding whether to enqueue.

## [P2] FTD rally days are counted per refresh, not per trading day
**File:** portfolio/market_health.py:255; portfolio/market_health.py:244  
**Bug:** `detect_ftd_state()` processes only the latest bar and increments `rally_day` every time the function refreshes on an up day; it also persists `ftd_day_offset` as an index in a rolling window, so the failure window does not age correctly.  
**Why it matters:** Hourly refreshes can turn one up day into day 4 and falsely confirm FTD; confirmed states can also stick longer than intended.  
**Fix:** Persist and compare the last processed trading date, and store FTD date/timestamp rather than a rolling list index.

## [P2] 200-SMA status is always unavailable/false
**File:** portfolio/market_health.py:383; portfolio/market_health.py:420  
**Bug:** `_compute_market_health()` fetches only `90d`, but later requires `len(closes) >= 200` to compute `spy_above_200sma`.  
**Why it matters:** The digest/state can report SPY not above the 200-SMA solely because only 90 bars were fetched, not because price is below the real 200-SMA.  
**Fix:** Fetch at least 220 trading days for market health, or remove/rename the 200-SMA field and scoring component.

## [P2] Failed loop heartbeats are still classified healthy
**File:** portfolio/loop_health.py:124; portfolio/loop_health.py:214  
**Bug:** `read_loop_status()` classifies freshness only by timestamp, and `write_heartbeat()` writes `"status": "ok"` even when `ok=False`.  
**Why it matters:** A loop can report a fresh heartbeat for a failed cycle and still be omitted from `unhealthy`, hiding a broken trading loop from the watchdog/dashboard.  
**Fix:** Encode status from `ok`, and classify fresh-but-failed payloads as unhealthy or degraded.

## [P2] Weekly P&L ignores holdings
**File:** portfolio/weekly_digest.py:52  
**Bug:** `_portfolio_summary()` computes `pnl_sek = cash - initial`, explicitly ignoring active holdings.  
**Why it matters:** A 500k SEK portfolio with 50k cash and 600k SEK of holdings is reported near `-90%` instead of positive performance.  
**Fix:** Use the same portfolio valuation path as other digests, including current prices, FX, cash, holdings, and fees.

## [P2] Last JSONL entry reader fails on long final records
**File:** portfolio/file_utils.py:328  
**Bug:** `last_jsonl_entry()` reads only the last 4096 bytes and parses complete lines from that tail.  
**Why it matters:** A Layer 2 journal or invocation entry longer than 4KB is skipped, so health checks can use an older timestamp or report false agent silence.  
**Fix:** Scan backward until a newline before the final record is found, or use a tail loader that can expand until one complete last line is available.

## [P3] Journal readers bypass safe JSONL helpers
**File:** portfolio/journal.py:28; portfolio/journal_index.py:373  
**Bug:** Both journal loaders use raw `open()` and `json.loads()` instead of the project file utilities.  
**Why it matters:** Rotation/rewrite races, malformed lines, and read errors are handled inconsistently; smart context can silently lose recent Layer 2 memory.  
**Fix:** Use `load_jsonl()`/`load_jsonl_tail()` or add a locked JSONL read helper with explicit logging behavior.

## [P3] Placeholder required config values pass validation
**File:** portfolio/config_validator.py:54  
**Bug:** Required config validation only rejects empty strings, despite the comment saying placeholders are checked.  
**Why it matters:** Values like `"YOUR_ALPACA_KEY"` or `"changeme"` pass startup validation and fail later inside broker/data calls.  
**Fix:** Reject known placeholder patterns and validate basic shape for tokens/secrets before startup succeeds.

## SUMMARY P1=3 P2=7 P3=2