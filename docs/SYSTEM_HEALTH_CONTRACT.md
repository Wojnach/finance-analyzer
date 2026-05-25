# System Health Contract

> **Purpose**: Define invariants the finance-analyzer system must maintain at all times.
> A scheduled Claude Code session checks these daily and reports violations via Telegram.
>
> **Last reviewed**: 2026-03-31
> **Owner**: Automated daily health check (PF-HealthCheck scheduled task)

---

## 1. Process Liveness

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **Main data loop** | PF-DataLoop scheduled task state = "Running" | `Get-ScheduledTask -TaskName PF-DataLoop` |
| **Main loop PID** | At least one python process with `main.py` running | `Get-Process python` — match PID from `data/trigger_state.json:last_loop_pid` |
| **Metals loop** | PF-MetalsLoop scheduled task state = "Running" | `Get-ScheduledTask -TaskName PF-MetalsLoop` |
| **Metals loop output** | `data/metals_loop_out.txt` has entries within last 5 minutes | `tail` + check timestamp |

**Failure response**: Restart the stopped task. If it fails to start, check `data/loop_out.txt` / `data/metals_loop_out.txt` for crash reason.

---

## 2. Heartbeat & Cycle Health

| Check | Invariant | Source |
|-------|-----------|--------|
| **Heartbeat age** | `data/health_state.json:last_heartbeat` < 5 minutes old | Compare to current UTC |
| **Error count** | `data/health_state.json:error_count` = 0 (or unchanged since last check) | Read file |
| **Cycle count** | `data/health_state.json:cycle_count` increases between checks | Compare to previous value |
| **Signal modules** | All entries in `signal_health` have `total_failures` = 0 in recent window | Check `recent_results` for any `false` |
| **Module failures** | `last_module_failures.modules` list is empty or contains only known-acceptable items | Read file |

**Known-acceptable module failures** (non-critical, analytics-only):
- `monte_carlo` — risk simulation, not used for signals
- `price_targets` — structural levels, informational only
- `equity_curve` — performance tracking, not used for decisions

---

## 3. Signal Coverage

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **All 5 instruments scanned** | `data/agent_summary.json` contains entries for all 5 Tier-1 tickers | Count keys |
| **Signal freshness** | Each ticker's signal snapshot < 15 minutes old (crypto/metals) or < 1 hour (stocks, outside market hours) | Check timestamps in summary |
| **No zero-voter consensus** | No ticker has `applicable_signals = 0` | Check signal counts |
| **Accuracy gating active** | `data/accuracy_cache.json` exists and was updated within 24h | File mtime |

**Expected tickers** (5 Tier-1):
```
Crypto:  BTC-USD, ETH-USD
Metals:  XAU-USD, XAG-USD
Stocks:  MSTR
```

---

## 4. LLM Inference (Metals Loop)

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **Chronos coverage** | All 4 tickers (XAG-USD, XAU-USD, BTC-USD, ETH-USD) have Chronos predictions within last 5 minutes | Grep `metals_loop_out.txt` for `[LLM] Chronos {ticker}` |
| **Ministral coverage** | All 4 tickers have Ministral predictions within last 10 minutes | Grep for `[LLM] Ministral {ticker}` |
| **No ticker dropout** | `data/metals_llm_predictions.jsonl` contains recent entries for all 4 tickers | Check last 20 JSONL lines |
| **Accuracy tracking** | Accuracy log line shows sample counts increasing | Compare `chronos_1h` count to previous check |

**Bug history**: XAG/XAU LLM inference silently dropped when no warrant positions were held (fixed 2026-03-31). Monitor for regression.

---

## 5. Layer 2 Agent (Decision Engine)

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **Auth valid** | `data/agent.log` does NOT contain "Not logged in" in last 10 lines | Grep last lines |
| **Invocation success** | When triggers fire, at least some agent invocations succeed | `data/health_state.json:last_invocation_ts` + grep agent.log for success |
| **No nested sessions** | `data/agent.log` does NOT contain "nested session" errors | Grep |
| **Reasonable latency** | Agent invocations complete within their tier timeout (T1: 120s, T2: 600s, T3: 900s) | Check agent.log for timing |

**When Layer 2 is disabled** (`config.layer2.enabled = false`): Skip these checks. Layer 3 autonomous fallback handles decisions instead — verify autonomous log lines appear in `metals_loop_out.txt`.

**Failure response**: If auth expired, alert user to run `/login` in Claude CLI. This cannot be auto-fixed.

---

## 6. Telegram Delivery

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **Delivery rate** | No messages in `data/telegram_messages.jsonl` with `"sent": false` older than 30 minutes | Parse last 20 JSONL entries |
| **Digest cadence** | `data/trigger_state.json:last_digest_time` < 5 hours old | Compare to current UTC |
| **Daily digest** | `data/trigger_state.json:last_daily_digest_time` < 25 hours old | Compare to current UTC |

**Note**: Some messages are intentionally `sent: false` (e.g. crypto reports during quiet hours). Only flag if multiple consecutive messages fail delivery.

---

## 7. Data Freshness

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **Portfolio state** | `data/portfolio_state.json` readable and valid JSON | `load_json()` |
| **Fundamentals cache** | `data/fundamentals_cache.json` updated within 48h | File mtime |
| **Prophecy beliefs** | `data/prophecy.json` exists and has recent review | Check `last_reviewed` field |
| **Signal log** | `data/signal_log.db` (SQLite) has entries from today | Query last entry timestamp |
| **Forecast health** | `data/forecast_health.jsonl` has entries from today | Check last line timestamp |

---

## 8. Singleton Locks

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **No stale locks** | Every `.singleton.lock` file in `data/` belongs to a running PID | Read lock file → check if PID exists |
| **Lock age** | No lock file older than its owning process start time | Compare file mtime to process start |

**Known lock files**:
- `metals_loop.singleton.lock` — should match PF-MetalsLoop PID
- `golddigger.singleton.lock` — should match PF-GoldDigger PID (or be absent if not running)
- `fin_snipe_manager.singleton.lock` — often stale (0 bytes = definitely stale)

**Failure response**: Delete stale lock files. The next process start will create a fresh one.

---

## 9. External API Connectivity

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **Binance spot** | BTC-USD price fetchable | Quick HTTP to `api.binance.com/api/v3/ticker/price?symbol=BTCUSDT` |
| **Binance FAPI** | XAG-USD price fetchable | Quick HTTP to `fapi.binance.com/fapi/v1/ticker/price?symbol=XAGUSDT` |
| **Alpaca** | No auth errors in loop output | Grep for "alpaca" errors |
| **Avanza session** | Session valid (if warrants are active) | `api_get("/_api/trading/rest/orders")` returns non-error |

**Note**: API checks should be lightweight (single ticker price, not full data pull). Don't burn rate limits.

---

## 10. Dashboard

| Check | Invariant | How to verify |
|-------|-----------|---------------|
| **Process running** | PF-Dashboard task running OR port 5055 responsive | `Get-ScheduledTask` or HTTP GET `localhost:5055/api/health` |
| **Data served** | `/api/portfolio` returns valid JSON | HTTP GET |

**Note**: Dashboard is currently Disabled in task scheduler. This check is optional until re-enabled.

---

## Check Execution Schedule

| Time (CET) | Check scope | Rationale |
|-------------|-------------|-----------|
| **11:00** | Full contract (all 10 sections) | Mid-morning full health check |
| **15:25** | Sections 1-6 (process, signals, agent) | Pre-US-market-open verification |
| **22:05** | Sections 1, 4, 5 (process, LLM, agent) | Post-US-close, verify overnight readiness |

---

## Reporting Format

The health check session should send a single Telegram message:

**All clear**:
```
✅ Health Check — {date} {time} CET
All 10 checks passed. Loop cycle #{n}, {uptime}d uptime.
LLM: 4/4 tickers, Agent: {ok/disabled/auth-expired}
```

**Violations found**:
```
⚠️ Health Check — {date} {time} CET

FAIL:
• [Section N] {description of violation}
• [Section N] {description of violation}

OK: {count}/10 sections passed
Action needed: {yes/no — yes if any FAIL requires human intervention}
```

---

## Revision History

| Date | Change |
|------|--------|
| 2026-03-31 | Initial contract. Created after discovering silent XAG/XAU LLM dropout bug and Layer 2 auth expiry going unnoticed. |
