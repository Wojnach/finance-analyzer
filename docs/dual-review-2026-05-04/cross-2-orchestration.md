# Cross-critique — orchestration

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `agent_invocation.py:147-149` — `_extract_ticker()` regex only matches stock tickers followed by `flipped|crossed|broke`. For triggers like `"MSTR consensus BUY (...)"` or `"NVDA moved 2.1% up"` it falls back to **XAG-USD** as default. Trade-guard gate, decision-feedback injection, and multi-agent specialist all use silver as primary ticker for stock triggers. | Claude reviewed `_extract_ticker` for crash safety but didn't trace what the *fallback* default value is. The default-to-silver is the kind of bug that slips past per-line review and only shows up via end-to-end trace. |
| `trigger.py:183-190` — startup-grace path rewrites `last` and `triggered_consensus` but leaves `sustained_counts`/`sustained_sentiment`/`stable_sentiment` from previous process intact. Stale `_mono_start` values can immediately fire a "sentiment bullish→bearish (sustained)" trigger from one sample. | Partial overlap with Claude's P2-1 (`_mono_start` serialized to disk) but Codex traced the operational consequence (spurious sustained trigger fires). Claude noted the persistence but flagged it as "safe direction by accident" — Codex showed that's wrong: when `mono_now` is large after long uptime, a restart can land in the unsafe direction. |
| `market_timing.py:334-336` — `is_swedish_market_holiday()` defined but `get_market_state()` only checks `is_us_market_holiday()`. On Swedish-only closures (Ascension Day 2026-05-14) the system thinks markets are open and runs stock/warrant processing. | Claude reviewed `market_timing.py` for DST and minute-precision but didn't check whether both holiday calendars are consulted. |
| `agent_invocation.py:724-731` — `pf-agent.bat` fallback explicitly logs "always Tier 3" but `_agent_timeout = timeout` keeps the originally-requested tier's budget (T1=120s/T2=600s instead of T3=900s). Recovery path unreliable when most needed. | Claude reviewed the timeout logic for the happy path but didn't audit the fallback path's timeout consistency with its log message. |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `journal.py:28-40` — `load_recent` reads JSONL with bare `open()` while `atomic_append_jsonl` renames over the same file → Windows `PermissionError` blocks the appender. | Codex reviewed orchestration but didn't check the journal read/write path for Windows-specific atomic-rename interaction. This is OS-specific behavior that requires knowing how `os.replace` interacts with open file handles on Windows. |
| `multi_agent_layer2.py:166-178` — file handle leak when `Popen()` raises between `open()` and `proc._log_fh = log_fh`. | Codex didn't audit the multi-agent error path. The leak is subtle because the `try/finally` is one frame above. |
| `trigger.py:130-148` — `_check_recent_trade()` only catches `KeyError, AttributeError`; `JSONDecodeError`/`OSError` propagates and crashes `check_triggers()`. | Codex reviewed `trigger.py` but for a different concern (state reset). Both bugs co-exist. |
| `agent_invocation.py:739-741` — auth-scan offset captured BEFORE `open()` — log rotation between stat+open makes scan cover zero bytes. **Recurrence risk for the March-April outage class of bug.** | Codex didn't scrutinize the order of stat() vs open(). Highly project-specific concern (the March-April outage history). |
| `multi_agent_layer2.py:193-210` — sequential `proc.wait()` means first specialist's timeout consumes all budget. **Multi-agent path effectively never produces specialist output.** | Codex didn't check the wait_for_specialists arithmetic against the per-specialist timeout. |
| `bigbet.py:173-181` — `CLAUDECODE` env var not popped before subprocess — silent nested-session failure in dev mode (since main loop in production unsets it). | Codex didn't compare bigbet's env handling to `agent_invocation.py:745` and `multi_agent_layer2.py:143` which do strip it. |

## Disagreements

None. Both reviewers identified independent bugs; no conflict over the same code.

## What both missed (likely)

- **`pf-agent.bat` fallback path's full lifecycle** — Codex caught the timeout bug; neither asked whether the bat script even respects PF_HEADLESS_AGENT or other env contracts.
- **`message_throttle.py` and `alert_budget.py`** — neither reviewer flagged anything in these. They're small but interact with crash recovery / Telegram suppression. Worth a focused later pass.
- **`reflection.py` and `analyze.py`** — neither reviewer flagged anything; possibly because they're report-generation modules with low blast radius. Still, a wrong reflection feeds back into Layer 2 prompts.

## Reconciled verdict

**P0 (must fix):**
1. (Claude) `journal.py:28-40` bare `open()` racing with atomic-rename appender — production failure mode (Windows).
2. (Claude) `multi_agent_layer2.py:166-178` file handle leak on `Popen` failure.
3. (Claude) `trigger.py:130-148` narrow exception clause crashes trigger system on JSONDecodeError.
4. (Claude) `agent_invocation.py:739-741` auth-scan offset rotation race — same class as the March-April outage.

**P1:**
5. (Codex) `_extract_ticker()` defaults to XAG-USD for stock triggers — silver-bias on every stock-only trigger.
6. (Codex) `is_swedish_market_holiday()` not consulted — system runs on Swedish closures.
7. (Codex) `pf-agent.bat` fallback uses wrong tier timeout.
8. (Claude) `multi_agent_layer2.py:193-210` sequential `proc.wait()` — multi-agent path silently produces no specialist output.
9. (Claude) `bigbet.py:173-181` CLAUDECODE not stripped.

**P2:**
10. (Codex + Claude jointly) `trigger.py` startup-grace + `_mono_start` persisted from prior process — both reviewers flagged related but different aspects. Fix together.
11. (Claude) `agent_invocation.py:147-149` hour-only market_open misses NYSE :30 open.
12. (Claude) `journal.py:70-74` timezone-naive entries crash `_entry_age_hours`.
13. (Claude) heartbeat keepalive blind spot for bigbet/iskbets (loop contract doesn't measure post-cycle work).
