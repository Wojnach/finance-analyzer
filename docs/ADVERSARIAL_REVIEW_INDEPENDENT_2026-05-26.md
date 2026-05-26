# Adversarial Review — Independent Pass (2026-05-26)

Reviewer: main thread (Claude Opus 4.7, 1M ctx), in parallel with 8 subagents.
Methodology: targeted Grep + Read, cross-cutting concerns, anti-patterns
the per-subsystem reviewers are likely to miss. NOT a substitute for the
subsystem files — companion to them.

Format: `path:line: <severity>: <problem>. <fix>.`

---

## P0 — Loop-killing / silent-failure / data-loss

data/metals_loop.py:6765: P0: `subprocess.Popen(cmd, stdout=log_fh, stderr=STDOUT)` bypasses `claude_gate.invoke_claude` AND has no `detect_auth_failure(...)` scan of the log slice after completion. Mirror of the Mar-Apr 2026 Layer 2 silent auth outage — main loop now defends via `claude_gate.detect_auth_failure`; metals loop does not. If `claude -p` prints `Not logged in` and exits 0, the metals loop logs "Claude T<N> invoked" and treats it as success forever. Fix: mirror `agent_invocation._scan_agent_log_for_auth_failure` pattern at the metals-loop completion branch, or route through `invoke_claude` (which auto-scans). [NEW]

portfolio/agent_invocation.py:619 / claude_gate.py:295: P0: With `--output-format json` (claude_gate.py:588), CLI prints a single `{...}` envelope. `detect_auth_failure` rejects lines starting with `{` via `_AUTH_MARKER_PREFIX_REJECT`. If the CLI failure mode shifts to "JSON envelope with `result: 'Not logged in'`" the scan misses entirely. Fix: when first non-blank line begins with `{`, also try `json.loads` and run the marker scan against `result_text` extracted via `_parse_claude_json_stdout`. [NEW]

data/metals_loop.py / portfolio/main.py / data/mstr_loop / data/oil_loop.py / data/crypto_loop.py: P0: 5 independent loops each call Binance directly (spot + FAPI) with no shared rate limiter. Single sync window can burst 30-50 requests/min into Binance. With `weight=2400/min/IP` limit, a coincident wake-up after a system sleep can trigger HTTP 429 → IP ban. `portfolio/http_retry.py` exists but is per-call; no central token bucket across loops. Fix: introduce a shared SQLite-backed token bucket in `portfolio/shared_state.py`, or move all Binance calls behind a single in-process queue (the 5 loops should share one process). [NEW]

## P1 — Likely-incident

portfolio/signal_engine.py:3795: P1: When a single enhanced signal raises, `votes[sig_name] = "HOLD"` and `_signal_failures.append(...)` happen, but the WARNING is suppressed unless `len(_signal_failures) > 3`. A persistently-broken signal (e.g. yfinance 401 after key rotation) silently force-HOLDs one slot forever. Fix: always log at INFO on per-signal failure; reserve the >3 WARNING for the surfacing-pile. [NEW]

portfolio/signal_engine.py:3713: P1: `_TICKER_DISABLED_SIGNALS.get(ticker, ())` returns empty tuple for unknown tickers — no validation that `ticker` is in `TIER1_TICKERS`. If a config drift introduces `XYZ-USD` to a per-ticker override map but not to the trade universe, the signal silently force-HOLDs for the ghost ticker with zero warning. Fix: assertion at module import time that all `_TICKER_DISABLED_SIGNALS` keys are present in `tickers.TIER1_SYMBOLS`. [NEW]

portfolio/portfolio_mgr.py:44-62: P1: `_rotate_backups` performs `shutil.copy2` chain (read + write across 3 files) but the per-file lock is acquired only in `_save_state_to:111`. Rotation runs INSIDE the lock — OK. But `load_state()` → `_load_state_from` does NOT take the lock, so a reader hitting `path.json.bak2` mid-rotation (between the `.bak → .bak2` and `.bak2 → .bak3` copy steps) sees a moving target. `load_json` swallows JSONDecodeError → silent default state → 500K fresh cash if PMS path corrupts the live `.json`. Fix: acquire the per-file lock in `_load_state_from` too. [NEW]

portfolio/claude_gate.py:155-161: P1: `_load_config_layer2_enabled()` uses raw `open(CONFIG_FILE).read()` → `json.load(f)`. Violates project rule (`Q:\finance-analyzer\.claude\rules\infrastructure.md`: "Atomic I/O only — Never `json.loads(open(...).read())`"). Fail-open with True if exception is fine for safety, but consistency matters; if someone copies this pattern they'll skip atomic_write. Fix: use `file_utils.load_json(CONFIG_FILE, default={})`. [NEW]

dashboard/auth.py:164: P1: `?token=<dashboard_token>` flow logs the literal token in Cloudflare/nginx access logs, browser history, Referer headers to outbound links. While `hmac.compare_digest` defends against timing attacks at the endpoint, the token leak vector is the LOG layer. Cookie-only refresh after first hit mitigates but the first request is forever-leaked. Fix: deprecate query-param entry — accept it once to set the cookie then 302-redirect to a clean URL (already done? verify), and warn in docs that tokens-in-URL are not recommended for shared/CDN-fronted deployments. [NEW]

portfolio/file_utils.py:240-247: P1: Sidecar lock pre-seed has a TOCTOU window — two callers passing `if not lock_path.exists():` simultaneously can both enter the branch and race the `open(lock_path, "ab")`. On Windows append-mode this is benign (both succeed, single byte appended twice or once depending on race), but technically the existence check is non-atomic. Fix: `open(lock_path, "ab")` unconditionally; the if-check provides nothing the open doesn't. [NEW]

## P2 — Latent risk

portfolio/agent_invocation.py:1117: P2: `_agent_log_start_offset` is captured against the LIVE `agent.log`. If log rotation fires during a long T3 (900s) invocation, the next `_scan_agent_log_for_auth_failure` seeks to an offset that's now post-EOF on the rotated-fresh file, scans empty bytes, returns False. Auth-error markers that lived in the rotated-away segment are silently missed. Same class as the Mar-Apr outage. Fix: record (path, inode/stat, offset) tuple at start; on scan, re-validate that the inode matches; if rotated, scan the rotated file too (or skip detection and log WARN about the rotation race). [NEW — partial-overlap with orchestration reviewer]

portfolio/risk_management.py:373: P2: `atr_pct = min(atr_pct, 15.0)` silently caps without logging. For underlying with ATR > 15% (BTC pump-vol regimes), the stop tightens unexpectedly. Add a `logger.warning("atr_pct cap triggered for %s: raw=%.2f%% capped=15%%", ticker, raw)` so the operator notices when this gates real volatility. [NEW]

dashboard/app.py:2255: P2: `int(request.args.get("days", "7"))` is NOT inside a try/except — bad input raises ValueError → 500. The surrounding `_parse_limit_arg` exists for exactly this reason but isn't used here. Fix: switch to `_parse_limit_arg("days", 7, 365)`. [NEW]

portfolio/main.py + portfolio/signal_engine.py: P2: There's no end-to-end smoke test that exercises the full 60s cycle in CI — only unit tests per signal. The Mar-Apr Layer 2 outage class (subprocess exits 0 + wrong-but-shaped output) would not be caught by current tests because they all mock the subprocess. Recommend a nightly integration test that runs a real `claude -p "say HELLO"` subprocess and asserts (a) exit_code == 0 AND (b) result_text == "HELLO". This single test would have caught Mar-Apr in <24h.

## Cross-cutting observations

1. **Per-subsystem reviewer pattern shows recurring [REPEAT] findings** — Avanza-API 5/5 repeats, Portfolio-Risk most repeats, Orchestration 10 repeats. The fgl process is generating findings; the implementation pipeline is the bottleneck. Recommend dedicating one session per week to ONLY closing [REPEAT]-tagged P0/P1 items.

2. **Five parallel loops (main, metals, mstr, oil, crypto) increasingly converge on Binance**. Without a shared rate limiter, the system is one schtasks-restart-storm away from a 429 ban. This is architectural debt — the loops should be one process with a scheduler, or there must be a shared rate-budget enforced via `shared_state`.

3. **Auth-failure detection is the highest-leverage safety mechanism** in the system (per March outage cost). Currently lives ONLY in `claude_gate.detect_auth_failure` consumed by `agent_invocation._scan_agent_log_for_auth_failure`. Metals loop, mstr loop, multi_agent_layer2 specialist children, and any future direct `subprocess.Popen([claude...])` are blind spots. Either route ALL claude invocations through `claude_gate.invoke_claude` (forbidden direct Popen — already documented but not enforced), or extract the scanner into a re-usable post-Popen helper that EVERY caller MUST invoke.

4. **The accuracy gate (47% / 50% tiered) and the directional gate (40%) are the heart of signal voting.** Multiple reviewers flagged divergence between cache-only and SQLite-only accuracy paths (signals-core P0 #1). Until the dual-write/dual-read divergence is fixed, every gate downstream reads inconsistent data — making backtests un-reproducible.

## Top 5 highest-impact items

1. metals_loop direct Popen with no auth scan (P0, NEW) — same outage class as Mar-Apr just on a different loop
2. JSON-envelope auth scan blind spot (P0, NEW) — current `--output-format json` makes the existing scanner less effective than before
3. SQLite/JSONL accuracy divergence (P0, REPEAT 3x from signals-core reviewer) — every gate downstream is reading stale data
4. portfolio_mgr backup rotation without read-side lock (P1, NEW) — silent fresh-defaults on rare race wins the worst-case for trading
5. 5-loop unrate-limited Binance fan-out (P0, NEW) — architectural risk waiting for the next schtasks-restart storm

---

## How this companion review differs from the subsystem ones

- **Cross-cutting** — connects findings across subsystems (auth detection is signals-core's reviewer's
  blind spot, orchestration's reviewer's blind spot, AND metals-core's reviewer's blind spot until they
  cross-reference each other).
- **Architectural** — calls out the 5-loop fan-out and the dual-write divergence as system-level
  concerns. Per-subsystem reviewers focus on file-level bugs.
- **History-aware** — explicitly maps findings to historical incidents (Mar-Apr auth outage, Mar 3
  stop-loss API instant-fill, BUG-178 silent hangs).

The synthesis doc deduplicates findings from all 9 inputs (8 subsystems + this independent pass)
into a prioritized actionable list.
