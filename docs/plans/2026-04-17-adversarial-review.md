ultrathink

# Adversarial Review Plan — 2026-04-17

Branch: `research/adversarial-2026-04-17`
Worktree: `Q:/finance-analyzer-adv`
Source: 6 parallel Explore agents + direct code verification of every finding.

## Context

- Started 2026-04-17. 2 unresolved critical-errors journal entries (Layer 2 timeout-cascade) already surfaced to user.
- Last session shipped per-ticker per-horizon blacklists (99206ffa). System healthy otherwise.
- 26 pre-existing test failures are documented in `docs/TESTING.md`; treating as noise.

## Verified P1 findings (fix this session)

### B1 — `atomic_append_jsonl` torn-line bug under Windows thread contention
- `portfolio/file_utils.py:155-167` — text-mode `open("a", encoding="utf-8")` with buffered writes can produce torn JSONL lines (head bytes lost, tail bytes survive) when two threads hammer the same file.
- Xfail test exists at `tests/test_fix_agent_dispatcher.py:409-499` (strict=False, 2026-04-13). Affects ~20 JSONL writers system-wide (signal_log, claude_invocations, critical_errors, telegram_messages, accuracy_snapshots, etc.).
- **Fix**: switch to binary-append mode (`open(path, "ab")`), encode line as UTF-8 bytes, single atomic `write()` call. OS-level O_APPEND is atomic for ≤PIPE_BUF on POSIX and always on Windows; binary mode avoids text-mode buffering that caused the torn-line symptom.

### B2 — `place_stop_loss` missing min-1000 SEK guard (3 locations)
All three Avanza trading layers lack the minimum-order-size check that `avanza_session.place_order` has at lines 590-592:
- `portfolio/avanza_session.py:706-774` — `place_stop_loss`
- `portfolio/avanza/trading.py:38-92` — `place_order` (unified package)
- `portfolio/avanza/trading.py:203+` — `place_stop_loss` (unified package)
- `data/metals_avanza_helpers.py:253-307` — `place_order`
- `data/metals_avanza_helpers.py:310-373` — `place_stop_loss`

**Fix policy** (minimizes blast radius):
- `place_order` in all 3 modules: RAISE `ValueError` on `vol*price < 1000` (match `avanza_session.py` policy).
- `place_stop_loss` in all 3 modules: log WARNING on `vol*price < 1000` (metals_loop cascades stops into ≤3 legs; per-leg can be <1000 SEK; surfacing via log is safer than breaking live cascading logic).

### B3 — Drawdown circuit breaker optimistic cash fallback
- `portfolio/risk_management.py:114-122` — when `agent_summary` is missing but holdings exist, `current_value` silently falls back to `cash_sek` (ignoring unrealized P&L on holdings). If holdings are underwater while the price feed is stale, drawdown looks tiny and circuit never trips. Current comment calls this a "conservative estimate" but it's the opposite — it's optimistic (cash ignores unrealized losses).

**Fix**: log `WARNING` when this fallback fires, so the Layer 2 journal and dashboard can see "feed stale" state rather than silently compute a false 0% drawdown. Don't change the fallback value (preserves behavior in true-offline scenarios) — just make the blind spot visible.

### B4 — `fetch_price` None → silent skip of stop-distance guard in metals loop
- `data/metals_loop.py:2066-2073` — `cur_bid = (cur_price_data or {}).get("bid", 0)` then `if cur_bid > 0:` silently skips the "stop too close to market" distance check when `fetch_price()` returns None (auth/network failure). Orders are then placed at stale `stop_base` without the safety guard.

**Fix**: when `cur_price_data is None`, log a WARNING and RETURN WITHOUT PLACING STOPS. Better to skip a stop update (covered by existing stops) than to place orders with no distance guard.

## Verified P2 findings (fix if time permits)

### B5 — `scripts/fin_fish_monitor.py:393-403` subprocess exit code not checked
- `result.returncode` never inspected after `subprocess.run(...)`. A crashed `fin_fish.py` would be invisible.
- Fix: log WARNING on non-zero exit code with stderr content.

### B6 — `portfolio/claude_gate.py:_kill_process_tree` swallowed exception detail
- On kill failure, logs a generic error and falls back silently; orphan Claude processes can accumulate.
- Fix: include exception details in the log so audits can find leaks.

## Deferred (in this review's scope but fix another session)

### D1 — Layer 2 overnight timeout cascades
- Recurring pattern (~daily). Agent investigation misread the code: the respawn after `_agent_proc = None` IS already in place (falls through to normal spawn at line 216+). Real cause appears to be T3 timeout (900s) + tight 18m health grace window interacting with multi-trigger queuing. Fix requires loop-contract redesign — out of scope.
- **Action**: add a note to `data/critical_errors.jsonl` resolving the 2 open entries as "recurring-pattern, documented, out-of-scope for this session".

### D2 — Fish engine 6 integration bugs
- Engine disabled at `data/metals_loop.py:766` since 2026-04-15 after a -12,257 SEK session. Already gated; no code executes. Safe.

### D3 — `agent_invocation.py:261` async specialists TODO
- Real architectural improvement; scope too large for this session.

### D4 — Metals loop 12 bare-except blocks
- All log exceptions. Cosmetic cleanup.

### D5 — `equity_curve.py` FIFO float drift
- ~1 bp per round trip. Immaterial.

### D6 — CRITICAL-2 (ticker="" dispatch)
- Production-safe: `main.py:486` always passes `ticker=name`. Defended by scattered `if ticker` checks. Tightening to a single early-reject would be cleaner but is a risk/reward tradeoff — leaving as-is.

### D7 — `backup.py` and `migrate_signal_log.py` unused modules
- One-time utilities. Not catastrophic; leave for future cleanup.

## Execution plan

### Batch 1 — Atomic JSONL + money guards
Files (5):
- `portfolio/file_utils.py` (B1)
- `portfolio/avanza_session.py` (B2)
- `portfolio/avanza/trading.py` (B2)
- `data/metals_avanza_helpers.py` (B2)
- `tests/test_file_utils.py` or `tests/test_fix_agent_dispatcher.py` (verify B1 fix unxfails the concurrent-append test)

### Batch 2 — Observability / fail-closed
Files (4):
- `portfolio/risk_management.py` (B3)
- `data/metals_loop.py` (B4)
- `scripts/fin_fish_monitor.py` (B5)
- `portfolio/claude_gate.py` (B6)

### Batch 3 — Journal resolution + tests
- Append resolutions to `data/critical_errors.jsonl` for the 2 open Layer 2 timeout-cascade entries (documenting as recurring pattern).
- Run `pytest tests/ -n auto` and fix any broken tests.

## Verification

After all batches:
1. `.venv/Scripts/python.exe -m pytest tests/ -n auto --timeout=60`
2. Check that `test_concurrent_append_does_not_corrupt_jsonl` now passes (xfail should become XPASS).
3. Codex adversarial review on branch.
4. Merge to main; push via Windows git; clean worktree.
