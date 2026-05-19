# Adversarial Review — orchestration (Claude-independent)

## Executive Summary
Reviewed 12 orchestration modules (60s main loop, ThreadPoolExecutor(8), Layer 2 subprocess orchestration, market timing, process locks, GPU gates). **Found 12 actionable findings** across subprocess handling, loop resilience, and edge cases.

## Findings

### P0 · agent_invocation.py:620–644 — ThreadPool cancel_futures leak
Line 644 calls `pool.shutdown(wait=False, cancel_futures=True)`. `cancel_futures=True` silently cancels pending queue items — but futures already *executing* in worker threads keep running, and the "canceled" future is never waited for. Over 10+ hours, hundreds of orphaned ticker threads accumulate, holding pandas DataFrames/indicator caches/locks. ThreadPool becomes exhausted; new cycles hang waiting for a free worker. **Fix:** `with ThreadPoolExecutor(...) as pool:` context manager OR explicitly `pool.shutdown(wait=True, cancel_futures=False)` to drain in-flight work.

### P0 · claude_gate.py:496–512 — Auth detector short-circuit ambiguity
Lines 502-512 scan stdout and stderr separately. But line 509 short-circuits: `if not stdout_hit: scan stderr`. If stdout has a DIFFERENT auth marker than stderr, stderr is never scanned. Works today (single marker type) but becomes a hidden bug if new markers are added. Comment at 496 overstates what the code does. **Fix:** remove short-circuit, always scan both, change line 510 to `if stdout_hit or stderr_hit:`.

### P0 · agent_invocation.py:288–294 — --bare comment booby-trap
Comment at lines 283-289 says "2026-04-13: DO NOT add --bare" as if defending against active code adding it. But the flag isn't there. Risk: a future maintainer cargo-cults `--bare` into another invocation site (see claude_gate.py invoke_claude which correctly omits it). **Fix:** rewrite comment to explain historical context clearly.

### P1 · main.py:547–644 — BUG-178 stale phase_log
Timeout handler (lines 613-643) logs timed-out tickers but does NOT clear their in-flight state from signal_engine's `get_last_signal()` / `get_phase_log()`. Next cycle reads stale tracker data → false "still stuck" diagnostics → alert fatigue. **Fix:** after line 642, add `clear_ticker_state(name)` for each timed-out ticker.

### P1 · loop_contract.py:126–223 — Grace window too short
`LAYER2_JOURNAL_GRACE_S = 18m` is too tight. Real end-to-end time: trigger → queue → prior agent wait → subprocess startup (30s) → T3 (900s) → journal write = ~40m. Contract fires at 36m-after-trigger → false CRITICAL violations every ~4 cycles → ViolationTracker escalates → self-heal spawns Claude Code session every 90m to confirm "no, Layer 2 is fine." **Fix:** raise to 25m = 1500s; track agent-invocation `_agent_start` and compare journal `ts` to that, not to trigger_time.

### P1 · gpu_gate.py:123–154 — Non-atomic Windows lock + PID reuse
`os.O_CREAT | os.O_EXCL` is atomic on POSIX but only best-effort on Windows (especially FAT32/network shares). Multiple threads can both create file, `_pid_alive()` returns False, both break lock → dual VRAM load → OOM GPU → silent CUDA failure. Additionally, if PID 12345 exits and another process reuses it, re-entry check at line 141 is a false positive. **Fix:** on Windows use `msvcrt.sopen()` with O_EXCL; include `threading.get_ident()` in lock metadata for PID-reuse defense.

### P1 · market_timing.py:241–243 — Missing Swedish holiday check
`_is_agent_window()` checks weekend + US holidays, but NOT Swedish market holidays. Swedish holidays (Epiphany Jan 6, Midsummer Eve, etc.) are MORE than NYSE. On Jan 6 2027, NYSE opens but Avanza-tracked warrants are frozen. Layer 2 invokes on stale warrant data → execution slippage or rejections. **Fix:** add `is_swedish_market_holiday(now)` check and return False if either market closed.

### P2 · autonomous.py:310–326 — Divergent vote thresholds
Autonomous hardcodes `_MIN_BUY_VOTES = 3` and `_BUY_MUST_DOMINATE = True`, but signal_engine's MIN_VOTERS is config-driven. If fallback (layer2.enabled=false) kicks in, autonomous is more conservative than Layer 2 → identical triggers produce different recommendations → debugging confusion. **Fix:** import MIN_VOTERS from signal_engine; drop hardcoded constant.

### P2 · main.py:1070–1108 — Pathological exponential backoff
Crash recovery: `_crash_alert()` → `_crash_sleep()` exponential backoff. After crash #5, alerts are suppressed but backoff keeps growing (10→20→40→80→160→320s). If each attempt takes <1s and keeps crashing, after 10 crashes loop sleeps 5120s (85min) between tries → silent 2-hour stall → no alerts since #5. **Fix:** cap backoff at 5min; reset counter if last crash >1h ago (transient burst ≠ persistent failure).

### P2 · multi_agent_layer2.py:146–176 — Specialist log truncation + no exec-check
Specialists spawned with `"w"` (truncate). Popen doesn't validate subprocess actually started (e.g., `claude` not on PATH → Popen succeeds, process dies at exec, exit 127). Line 172 optimistically logs "launched." Lost logs + synthesis agent proceeds with empty specialist reports. **Fix:** use `"a"` (append); after Popen, check `proc.poll() is None` immediately; treat exit 127 as launch failure.

### P2 · loop_contract.py:866–906 — Type mismatch crash
`verify_and_act()` falls back to `verify_contract()` which requires `CycleReport.signals`. If metals_loop passes a `MetalsCycleReport` without a custom `verify_fn`, it crashes inside the contract framework — the very thing meant to catch loop failures. **Fix:** type-check: `if verify_fn is None and not hasattr(report, "signals"): warn & return`.

### P3 · crypto_scheduler.py:72–97 — Silent timezone config error
Invalid `timezone` in config (typo like "Europe/Stockholme") → ZoneInfo KeyError → silent fallback to DEFAULT_TZ. Operator never knows config is broken. **Fix:** log a WARNING on fallback.

## Things That Looked OK

1. **Singleton lock (main.py:56–91)** — cross-platform msvcrt + fcntl, atomic write, proper cleanup.
2. **Auth-failure detector (claude_gate.py:194–247)** — separate stdout/stderr, fenced-code tracking, line-1 limit (BUG-ECHO fix). Only short-circuit ambiguity noted.
3. **Process tree kill (claude_gate.py:293–327)** — `taskkill /T` on Windows, `killpg` on Unix, fallback to `proc.kill()`. No leaks observed.
4. **Market timing DST (market_timing.py:29–102)** — Anonymous Gregorian Easter, correct DST start/end dates, NYSE holidays correct.
5. **Loop contract framework (loop_contract.py:654–906)** — violation tracking, escalation, self-heal trigger sound.
6. **Perception gate (perception_gate.py:29–96)** — simple rule-based, no LLM, no false positives in history.
7. **GPU gate layer 1 (gpu_gate.py:112–118)** — threading.Lock usage is correct.

## Summary Table

| P | File | Line(s) | Issue |
|---|------|---------|-------|
| P0 | agent_invocation.py | 644 | cancel_futures leak |
| P0 | claude_gate.py | 509 | Auth scanner short-circuit |
| P0 | agent_invocation.py | 288 | --bare comment booby-trap |
| P1 | main.py | 613–642 | Stale phase_log after timeout |
| P1 | loop_contract.py | 49 | 18m grace too tight |
| P1 | gpu_gate.py | 126 | Non-atomic Windows lock |
| P1 | market_timing.py | 241 | Missing Swedish holiday check |
| P2 | autonomous.py | 323 | Divergent min BUY votes |
| P2 | main.py | 1071 | Pathological backoff |
| P2 | multi_agent_layer2.py | 161 | Specialist log + exec-check |
| P2 | loop_contract.py | 878 | Type mismatch crash |
| P3 | crypto_scheduler.py | 73 | Silent tz fallback |

## Reviewer confidence

0.80
