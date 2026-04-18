# Adversarial Code Review: Infrastructure Subsystem (Claude Reviewer)

## Executive Summary

**5 P1 findings, 10 P2 findings, 10 P3 findings.**

Most severe: timing-attack-vulnerable token comparison in dashboard (single-line fix), non-atomic config.json mutation from Telegram commands, TOCTOU race in GPU lock, non-atomic journal context file write, and 68MB+ signal_log slurped into memory by weekly digest.

---

## P1 Findings

### P1-1: Dashboard auth token compared with `==` (timing attack)
**File:** `dashboard/app.py:675-682`
Use `hmac.compare_digest()` instead. Combined with `Access-Control-Allow-Origin: *`, any website can brute-force the token.

### P1-2: Telegram poller writes config.json non-atomically and without lock
**File:** `portfolio/telegram_poller.py:197-208`
Read-modify-write race with main loop. Config contains ALL API keys -- corruption is catastrophic.

### P1-3: GPU lock TOCTOU race -- two processes can both acquire
**File:** `portfolio/gpu_gate.py:134-146`
Between `_release_lock()` stale cleanup and `os.open(O_CREAT|O_EXCL)` retry, two processes race.

### P1-4: Journal context file written non-atomically
**File:** `portfolio/journal.py:568,580`
`Path.write_text()` violates Critical Rule #4 (Atomic I/O only). Layer 2 can read truncated context.

### P1-5: Weekly digest reads entire signal_log.jsonl (68MB+) into memory
**File:** `portfolio/weekly_digest.py:28-41,154`
Causes OOM risk. Every other consumer uses `load_jsonl_tail()`.

## P2 Findings (10)

P2-1: Stale `now` timestamp in `_cached()` -- `shared_state.py:48,95`
P2-2: Sidecar lockfile TOCTOU race -- `file_utils.py:198-204`
P2-3: Rate limiter serializes parallel requests -- `shared_state.py:247-262`
P2-4: Indefinite block on Windows file lock -- `file_utils.py:212`
P2-5: Wildcard CORS with auth tokens -- `dashboard/app.py:45`
P2-6: Health reads without lock see inconsistent state -- `health.py:44-49`
P2-7: `_loading_timestamps` memory leak -- `shared_state.py:96`
P2-8: Non-idempotent POST retries (duplicate Telegram sends) -- `http_retry.py:27-53`
P2-9: Mutable config cache shared across threads -- `api_utils.py:36`
P2-10: `prune_jsonl` races with concurrent appends -- `file_utils.py:292`

## P3 Findings (10)

P3-1: NewsAPI quota resets on restart -- `shared_state.py:307`
P3-2: Import inside function body -- `health.py:327`
P3-3: Missing Binance key validation -- `config_validator.py:15`
P3-4: Weekly digest bypasses mute gates -- `weekly_digest.py:262`
P3-5: Bot token logged in URLs -- `telegram_notifications.py:54`
P3-6: Held-ticker cache not thread-safe -- `reporting.py:744`
P3-7: Chat ID as sole command auth -- `telegram_poller.py:82`
P3-8: Accuracy computed on every request -- `dashboard/app.py:788`
P3-9: `load_jsonl_tail` edge case on boundary -- `file_utils.py:145`
P3-10: WMIC command injection potential -- `subprocess_utils.py:211`
