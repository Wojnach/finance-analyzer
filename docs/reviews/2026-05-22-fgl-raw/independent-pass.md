# FGL Adversarial Review — Lead Independent Pass (2026-05-22)

Independent fresh-eyes review by the lead agent, conducted in parallel with the 8
subsystem subagents and BEFORE reading their results (except where noted). Files read
in full: `file_utils.py`, `portfolio_mgr.py`, `agent_invocation.py`, `trigger.py`,
`risk_management.py`, `avanza_orders.py`; targeted reads of `signal_engine.py`,
`metals_swing_trader.py`.

Severity: P0 production incident · P1 wrong behavior on a realistic path · P2 latent
bug / fragility · P3 minor.

| ID | Sev | Location | Finding | Fix |
|----|-----|----------|---------|-----|
| IND-1 | P1 | file_utils.py:295-313 | `atomic_write_jsonl` does a full-file rewrite + `os.replace` WITHOUT taking `jsonl_sidecar_lock`. `atomic_append_jsonl` and `prune_jsonl` both take the lock; `atomic_write_jsonl` does not. Any caller using it to rewrite a file that also receives `atomic_append_jsonl` appends loses any append that lands between its read and its `os.replace` — the exact lost-append race the sidecar lock was built to prevent. | Take `jsonl_sidecar_lock(path)` around the write, or document that callers must hold it. |
| IND-2 | P1 | portfolio_mgr.py:35-41,136-159 | `update_state` / `_get_lock` use `threading.Lock` — intra-process only. Portfolio state is also written by the Layer 2 `claude -p` subprocess (separate process). Two processes interleaving read-modify-write corrupt portfolio state / drop a transaction; `atomic_write_json` makes each write atomic but not the RMW. | Use an OS file lock (sidecar-lock pattern) for the RMW, or guarantee a single writer process. |
| IND-3 | P2 | file_utils.py:248-265 | `jsonl_sidecar_lock` docstring claims `msvcrt.locking(LK_LOCK)` "blocks on contention". It does not — Windows `LK_LOCK` retries for ~10s then raises `OSError`. Under sustained contention (e.g. `prune_jsonl` parsing a large file under the lock), a concurrent `atomic_append_jsonl` raises `OSError` and the entry is lost / the caller may crash. | Catch the `OSError`, retry with backoff, and surface a warning; don't let an append silently vanish. |
| IND-4 | P2 | portfolio_mgr.py:116-133 | `load_state()` + `save_state()` used as a read-modify-write is NOT atomic even within one process — only `update_state()` holds the lock across read+write. Any caller doing `s=load_state(); mutate; save_state(s)` races. | Audit callers; route all mutations through `update_state`. |
| IND-5 | P2 | portfolio_mgr.py:65-75 | `_validated_state` validates `holdings`/`transactions` types but not `cash_sek` / `initial_value_sek`. A corrupt string `cash_sek` flows into `portfolio_value` math and propagates a string total downstream. | Coerce/validate numeric fields; reset to default on bad type. |
| IND-6 | P3 | file_utils.py:32-71 | Atomic writers fsync the temp file but never fsync the parent directory, so the `os.replace` rename itself is not durable on power loss — contradicts the "durability on power loss (H34)" docstring claim. | fsync the parent dir fd after `os.replace`, or soften the docstring. |
| IND-7 | P2 | agent_invocation.py:582-681 | If `taskkill` genuinely fails or `wait()` times out, `_kill_overrun_agent` keeps `_agent_proc` set to block respawn. Every subsequent `invoke_agent` + watchdog tick then re-attempts the kill, fails, returns False → Layer 2 is permanently wedged with only `logger.error`, NO Telegram / `critical_errors.jsonl` escalation. Same silent-outage class as the Mar–Apr auth incident. | Escalate to `send_or_store` + `critical_errors.jsonl` after N consecutive failed kills. |
| IND-8 | P2 | agent_invocation.py:952-979 | Multi-agent path calls `wait_for_specialists(procs, timeout=30)` synchronously — blocks the main-loop thread for up to 30s of a 60s cycle. Specialists still running at timeout are not killed → orphaned `claude` subprocesses leak quota/GPU. (config-gated: `layer2.multi_agent`.) | Run specialists on a background thread; kill leftovers at timeout. |
| IND-9 | P3 | agent_invocation.py:1486-1505 | `check_agent_completion` sends a Telegram alert for `failed` and `incomplete` but NOT for `auth_error` — the single most critical Layer 2 failure mode. Relies entirely on `detect_auth_failure`'s own path. | Add an explicit alert for `auth_error`. |
| IND-10 | P2 | trigger.py:506,615 | First-of-day T3 is unreachable. `check_triggers` sets `state["last_trigger_date"]=today` and `_save_state` BEFORE `main.py:849` calls `classify_tier(reasons_list)` (state=None → reloads from disk). `classify_tier` then sees `last_trigger_date == today`, so the `last_trigger_date != today → return 3` branch never fires. The intended "first real trigger of the day = full review" (comment C4/NEW-2) is dead. Confirmed via `main.py` call ordering. | Pass the pre-save state into `classify_tier`, or track `last_trigger_date` as the *previous* trigger's date. |
| IND-11 | P3 | trigger.py:130-161 | `_update_sustained` docstring says monotonic origin "resets on process restart" — false; `time.monotonic()` is boot-relative and stable across process restarts on Windows/Linux. Persisted `_mono_start` stays comparable across a restart (and is conservatively negative across a reboot). Behavior is safe; the stated rationale is wrong. | Correct the docstring. |
| IND-12 | P3 | trigger.py:3 | Module docstring says "Layer 1 runs on a 10-minute cadence" — contradicts the 60s loop in CLAUDE.md / `market_timing.py`. Stale. | Update docstring. |
| IND-13 | P1 | avanza_orders.py:82-146,154-217 | `request_order` (called by the Layer 2 subprocess) and `check_pending_orders` (main loop) both do `_load_pending()` → mutate → `_save_pending()` on `avanza_pending_orders.json` with no lock. Cross-process interleaving silently drops a pending order — it is never executed, never expired, and the user is never notified. | OS file lock around the pending-file RMW. |
| IND-14 | P2 | avanza_orders.py:220-347 | `_check_telegram_confirm` polls Telegram `getUpdates` with its own offset file, but `telegram_poller.py` also polls `getUpdates`. Telegram deletes confirmed updates server-side; an update consumed by one poller is invisible to the other. A `CONFIRM <token>` reply can be eaten by `telegram_poller` → the order never confirms and expires. | Single getUpdates consumer that fans out, or a shared offset + dispatch. |
| IND-15 | P3 | avanza_orders.py:77-79,216 | `avanza_pending_orders.json` is never pruned — executed/expired/error orders accumulate forever; every cycle re-iterates the full list. | Drop terminal-state orders older than N hours on save. |
| IND-16 | P2 | metals_swing_trader.py:2714-2786 | `_set_stop_loss` computes the stop purely as a fixed warrant-% drop (`sl_warrant_pct`, BASE×leverage) with NO check that the resulting trigger price stays on the safe side of the knockout barrier. Barrier safety relies entirely on the selection-time `MIN_BARRIER_DISTANCE_PCT` gate; for an adopted orphan position or post-entry underlying drift, a 30% warrant stop can sit beyond the knockout level → the warrant knocks out before the stop fills. | Clamp the stop trigger to a safe margin inside the live knockout barrier; recompute on orphan ingest. |
| IND-17 | P3 | risk_management.py:462 | `compute_probabilistic_stops` annualizes vol with `sqrt(252.0/14)` hardcoded for ALL instruments — crypto and metals trade 365 days/yr. Same 252-vs-365 class the recent batches fixed in monte_carlo/exit_optimizer; this call site was missed. Low impact (vol floored at 0.05, intraday sim). | Use 365 for crypto/metals. |

## Notes on what is solid

- `file_utils` atomic JSON writers (`atomic_write_json/_text/_jsonl`) correctly use
  `mkstemp` + fsync + `os.replace` and resolve symlinks (`config.json`). The only gap
  is IND-1 (jsonl rewrite lock) and IND-6 (dir fsync).
- `agent_invocation` is heavily defended: monotonic-clock timeout with wall-clock
  fallback, completion watchdog, auth-error scan on both happy and timeout paths,
  stack-overflow auto-disable, count-delta journal/telegram detection. The residual
  risks are the permanent-kill-failure wedge (IND-7) and the `auth_error` alert gap
  (IND-9).
- `avanza_orders` per-order `confirm_token` design correctly closes the stale-CONFIRM
  / wrong-order / no-pending-yet races and adds sender authentication. The residual
  risks are the cross-process file race (IND-13) and the shared getUpdates stream
  (IND-14).
- `trigger.py` startup-grace, empty-ticker prune guard, and flip cooldown are sound.
