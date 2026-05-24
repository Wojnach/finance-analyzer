# Main-Thread Independent Adversarial Review — 2026-05-24

**Style:** Cross-cutting pass from main thread, written independently of the 8 subsystem agents. Worktree: `Q:\finance-analyzer\finance-analyzer-reviews\2026-05-24` (branch `review/fgl-2026-05-24`), empty-baseline reference branch `review-baseline-empty`.

**Method:** Grep across the entire codebase for known anti-patterns + verify status of prior-review P0s + flag freshly-introduced commit risk.

---

## P0 — Verified-still-broken (re-flagged from 2026-05-19, **not fixed despite the 5/24 batch-fix sweep**)

### IND-P0-1 Bias-penalty applied twice in weighted consensus
- `portfolio/signal_engine.py:2677` (first multiplication via `norm_weight = act_data["normalized_weight"]` which is set to `rarity_weight * bias_penalty` at `portfolio/accuracy_stats.py:849`).
- `portfolio/signal_engine.py:2698` (second multiplication via `_resolve_bias_penalty(signal_bias)`).
- Fix shipped in 2026-05-19 review made the second multiplier *conditional on bias-direction vote* but never removed the unconditional first multiplier. End state: BUY-direction vote on a calendar-bias signal gets `bias_penalty²`; contrarian SELL gets `bias_penalty¹`. Net effect = 10× over-penalty plus broken contract documented in the docstring at 2685.
- **Fix:** Remove `bias_penalty` from `normalized_weight` computation in `accuracy_stats.py:849` (keep `normalized_weight = rarity_weight` only) and let `_resolve_bias_penalty` be the sole tiered application. Re-run signal weight tests.

### IND-P0-2 Cross-process race on portfolio_state.json
- `portfolio/portfolio_mgr.py:30,110-159` — uses `threading.Lock()` (per-process only). Layer 2 subprocess, dashboard Flask process, and main loop are all separate OS processes. `update_state(mutate_fn)` reads→mutates→writes under a Lock that nothing else honors.
- Same regression that caused the Mar 3 stop-loss double-fill incident (coordination assumption that doesn't hold cross-process).
- **Fix:** wrap read+write in `file_utils.jsonl_sidecar_lock`-style cross-process advisory lock (the sidecar `.lock` pattern already used by `atomic_append_jsonl`), or migrate state to SQLite WAL with `BEGIN IMMEDIATE`.

### IND-P0-3 warrant_portfolio has ZERO concurrency protection
- `portfolio/warrant_portfolio.py:42-49` — `save_warrant_state()` just calls `atomic_write_json` (the write itself is atomic, but the read-modify-write window is not). metals_loop + grid_fisher + dashboard all race on `data/portfolio_state_warrants.json`. Worse than portfolio_mgr — not even a threading.Lock present.
- **Fix:** same cross-process pattern as IND-P0-2; or move to SQLite.

### IND-P0-4 BULL vs BEAR warrant direction not honored in P&L math
- `portfolio/warrant_portfolio.py:96` — `implied_pnl_pct = underlying_change * leverage` treats `leverage` as positive scalar. BEAR cert with positive underlying must produce NEGATIVE P&L; this code returns positive.
- `portfolio/risk_management.py:374,382` — ATR stop computed as `entry * (1 - 2*atr/100)` (LONG-only); trigger check `current_price < stop_price` (LONG-only). Any SHORT/BEAR holding mis-stopped.
- **Fix:** read `direction = holding.get("direction", "LONG")` and sign-adjust both leverage and stop direction.

### IND-P0-5 Account-id leak in get_positions() — pension positions visible to ISK trader
- `portfolio/avanza_session.py:676-717` — `ALLOWED_ACCOUNT_IDS` defined at line 43 but **not consulted** in `get_positions()`. Callers (metals_loop, dashboard `/api/portfolio`) assume returned list is ISK-only.
- `memory/feedback_isk_only.md` is unambiguous: pension account 2674244 must never leak.
- **Fix:** filter response by `entry["account"]["id"] in ALLOWED_ACCOUNT_IDS` inside the for-loop.

### IND-P0-6 Grid Fisher stop-sell-price still 0.5% buffer on 5x certs
- `portfolio/grid_fisher.py:1496` — `stop_sell_price = round(stop_price * 0.995, 2)`. Same as 2026-05-19. The recent commit `289e5030 (P1) grid_fisher stop_needs_rearm flag + tick retry` did *not* widen this buffer.
- A 0.5% gap past trigger leaves a 5x cert with an unfilled limit; market reopens and the stop is now dead while the position bleeds.
- **Fix:** widen to ≥3% buffer (`stop_price * 0.97`) for 5x MINIs; or use a market-sell on triggered stops.

### IND-P0-7 Binance error responses still pass through as garbage candles
- `portfolio/data_collector.py:88-93` — `data = r.json(); if not data: ... ; df = pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)`.
- Binance returns 200 OK with `{"code":-1121, "msg":"Invalid symbol"}` on bad symbol or `10m` interval. `if not data` is False for a non-empty dict. `pd.DataFrame(dict, columns=...)` raises ValueError or silently produces a 1-row frame of garbage. Even though the resulting `cb.record_failure()` traps it via the broad except, the failure is logged as a network failure instead of an API-error and the circuit breaker counts it wrong.
- **Fix:** insert before line 93: `if isinstance(data, dict) and "code" in data: raise ConnectionError(data.get("msg", "binance error"))`.

### IND-P0-8 Layer 2 off-hours skip path still has no autonomous fallback
- `portfolio/main.py:972-979` — when `escalation_router` path is **not** taken (the `elif layer2_cfg.get("enabled", True):` branch) and `_is_agent_window()` is False, the code logs `skipped_offhours` and **does not** call `autonomous_decision`. Compare with line 947-966 where the router-path *does* call autonomous in the same window-closed case.
- Weekend XAG/BTC/ETH F&G crossings → no journal, no Telegram, no recommendation.
- **Fix:** mirror lines 960-966 here: call `autonomous_decision(...)` before the `skipped_offhours` log.

---

## P1 — High confidence, lower money-at-risk

### IND-P1-1 send_telegram.py / data/silver_monitor.py raw json.load(open()) on config
- `send_telegram.py:27` and `data/silver_monitor.py:606`. CLAUDE.md Critical Rule #4 forbids this pattern. Standalone utilities, not in the loop hot path, but corrupt config produces cryptic `JSONDecodeError` rather than `load_json`'s graceful default.
- **Fix:** `from portfolio.file_utils import load_json` then `config = load_json("config.json")`.

### IND-P1-2 _build_regime_context silently swallows all exceptions
- `portfolio/agent_invocation.py:239-240` — `except Exception: return ""`. A malformed `agent_summary.json` or KeyError mid-loop kills the regime line and the calibration warning without any log. Could mask the very condition the calibration warning was meant to surface.
- **Fix:** `except Exception as e: logger.warning("regime_context: %s", e); return ""`.

### IND-P1-3 _build_regime_context depends on undocumented _buy_count/_sell_count keys
- `portfolio/agent_invocation.py:231-234` reads `tdata["extra"]["_buy_count"]` and `_sell_count`. If `reporting.py` ever stops populating those (rename, schema change), the calibration warning silently never triggers. No assert, no test on the contract.
- **Fix:** assert presence at first invocation; add unit test in `tests/test_agent_invocation.py` for both populated and missing branches.

### IND-P1-4 Utility-boost magnitude still over-amplifies (gate-bypass fixed, math still wrong)
- `portfolio/signal_engine.py:4202` — `boost = min(1.0 + u_score, 1.5)` where `u_score = avg_return` (percent units, e.g. 0.5 = +0.5%). A modest +0.5% per-vote avg-return produces 1.5× boost ⇒ 0.50 raw accuracy → 0.75 boosted. The IND-P0-2 from 2026-05-19 has been narrowed (gate check now precedes boost) but the magnitude is still bug-shaped: a 0.5pp directional edge does not justify a 50% accuracy weight bump.
- **Fix:** `boost = min(1.0 + u_score/2, 1.10)` cap at +10%.

### IND-P1-5 broad `except Exception: pass`/`return HOLD` patterns hide failure
- 16 files (`grep` count above). Hot signals: `portfolio/signals/btc_etf_flow.py`, `news_event.py`, `gold_overnight_bias.py`, `intraday_seasonality.py`. Each silent return masks a math/data bug as a HOLD vote — invisible until accuracy-stats accumulates 30+ samples.
- **Fix:** require `signal_registry` enforce structured `SignalResult(verdict, confidence, error=None)`; downgrade plain `Exception` to `(KeyError, ValueError, TypeError)` only.

### IND-P1-6 Threading.Lock-only on cross-process accuracy & cache writes
Many caches use `threading.Lock` only:
- `accuracy_stats.py:16,31,71,95,1127,1434` — signal_utility cache, accuracy compute, regime accuracy cache.
- `forecast_accuracy.py:31`, `fx_rates.py:23`, `earnings_calendar.py:28`.

When the dashboard process recomputes accuracy concurrently with the main loop, the disk-backed L2 cache can be overwritten with mid-computation state. `signal_utility_cache.json` showed exactly this race in past incidents.
- **Fix:** for any cache whose disk file is written, swap to `file_utils.jsonl_sidecar_lock` around read+write.

---

## P2 — Code quality / observability

### IND-P2-1 file_utils tempfile leaves orphan .tmp on PowerShell kill
- `portfolio/file_utils.py:40-50` — `tempfile.mkstemp(dir=path.parent, suffix=".tmp")` followed by `fdopen + write + replace`. On Ctrl-C kill (SIGTERM equivalent on Windows = `TerminateProcess`), the `BaseException` handler cleans up; but a `os.kill` mid-fsync leaves `.tmp` files. Repeated kills accumulate `.tmp*` files in `data/`.
- **Fix:** Add a startup sweep in `main.py` for `data/*.tmp*` older than 1h.

### IND-P2-2 Multiple `Q:/finance-analyzer/...` path strings — Windows-only
- Throughout the codebase, hardcoded `Q:\` or `Q:/` paths in non-test files. Limits portability and complicates CI.
- **Fix:** centralize in `portfolio/paths.py` (single source); use `BASE_DIR` resolution.

### IND-P2-3 ScheduleWakeup, .skip files, and TODO-debt have no expiry sweep
- Multiple `# TODO: MANUAL REVIEW` comments scattered (grep `TODO: MANUAL`). No sweep job.
- **Fix:** weekly cron to grep TODO age and Telegram-alert anything > 30d.

---

## P3 — Style / dead code / observed but low priority

- `portfolio/signal_weights.py` — entire module is dead (confirmed by signals-core agent referencing `outcome_tracker.py:497-500` comment).
- `portfolio/signal_decay_alert.py:12` — unused `import json`.
- 67 files in `portfolio/signals/`, 49 of which are force-HOLD via `DISABLED_SIGNALS`. Still imported, still loaded by registry, still occupy memory. Could be moved to `portfolio/signals/_disabled/` and explicitly unimported.

---

## Cross-cutting observations

1. **The 2026-05-24 batch-fix sweep** (commits `be4273d3`, `c7d60b72`, `81e0c78e`, `4adeec2d`, `289e5030`) closed 7-8 of the 22 P0s from the 2026-05-19 review but **did not touch the foundational concurrency bugs** (IND-P0-2, IND-P0-3, IND-P0-4) or the bias-double-application (IND-P0-1). These are higher engineering effort (lock semantics change, P&L sign change) and were deferred.
2. **The infrastructure for atomic I/O is correct** (`file_utils.jsonl_sidecar_lock` with msvcrt+fcntl, fsync before replace). Callers are inconsistent — half use it (`atomic_append_jsonl`), half use raw `threading.Lock` (`portfolio_mgr`).
3. **Empty-baseline technique surfaces issues diff-mode reviewers skip.** Several P0s above (IND-P0-6, IND-P0-8) are pre-existing code that no recent diff touched; only re-reading the whole file with adversarial intent catches them.
4. **Silent-fail patterns are systemic.** The `except Exception: return HOLD` idiom in signals (and `except: return ""` in regime context) masks bugs as a vote that gets averaged into consensus. Need a structured error contract.
5. **Cross-process lock confusion is the most common bug class** — 5 of the 8 self-review P0s touch it. Suggests one shared utility (`from portfolio.file_utils import update_json_atomic`) that wraps read+lock+modify+write+release would close most of them in one sweep.

---

## Recommendation for the synthesis doc

Prioritize a *single batch* that fixes the cross-process write race (IND-P0-2/3/4) by introducing one shared `update_json_atomic` helper in `file_utils.py`. That single change removes the foundation of 3 P0s.

Next batch: bias-penalty deduplication (IND-P0-1) — a single line removal in `accuracy_stats.py:849` plus tests.

Third batch: Binance error-response guard (IND-P0-7) + autonomous fallback in main.py:972-979 (IND-P0-8) + account-id filter in get_positions (IND-P0-5) — three small, isolated edits.
