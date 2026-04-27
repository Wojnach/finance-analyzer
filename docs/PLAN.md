# PLAN — Contract Alert Spam + Telegram Poller Hygiene (2026-04-28)

## Context

Two diagnostic threads pulled tonight, both rooted in observability/state plumbing rather than core trading logic:

1. **`accuracy_degradation` Telegram spam** — at ~01:15 UTC on 2026-04-28 the contract had fired
   192 consecutive cycles (~32 h, ≈10 min cadence) of an identical CRITICAL alert listing 12
   degraded signals, 5 of which are MSTR signals collapsing in lockstep (sentiment 90→39,
   volume_flow 82→34, fibonacci 78→35, heikin_ashi 79→35, momentum_factors 89→30 — orthogonal
   signal families, can't all decay at once unless the **outcome labels** moved). The
   detector is doing its job; the *replay-on-throttle* design path
   (`accuracy_degradation.py:348`) keeps `ViolationTracker.consecutive` alive across cycles
   by re-emitting the cached Violation list, and `loop_contract._alert_violations` ships
   a Telegram message for every CRITICAL violation it sees — so cached replays alert just
   like fresh detections. The 2026-04-16 memory entry already named this risk class
   (regime-transition baselines firing phantom drops); we now have it landing on MSTR.

2. **Telegram inbound poller staleness** — `data/telegram_inbound.jsonl` has 5 entries, all
   from 2026-04-17, all `update_id: 1, message_id: null, from: {id: null, username: null}`.
   The metadata pattern (literally never produced by real Telegram updates) confirms these
   are synthetic test-harness injections, not real DMs. Real status: poller is alive
   (heartbeat fresh), but (a) zero real user traffic since Apr 17, and (b) the `offset`
   field is in-memory only — on every loop restart it resets to 0, fetches all pending
   updates, then drops messages older than `startup_time - 60s` via the stale filter.
   Survivable today because nobody DMs the bot, but it's the kind of latent bug that
   bites the moment you actually rely on inbound commands.

## Goal

Stop the Telegram spam without losing detection signal, and harden the
inbound poller so a future restart doesn't silently drop legitimate commands.
**Do not change signal weights, gating thresholds, or anything that affects
trading decisions.** This is a plumbing/observability change.

## Non-goals

* Fixing the underlying MSTR accuracy reading. The 2026-04-16 memory entry
  already prescribes the structural fix (per-ticker per-horizon blacklist,
  fresher baseline). That's a config change requiring user judgment on each
  un-blacklist; out of scope for this autonomous session.
* Migrating off the Telegram Bot API to a user-account stack (`tdl`/telethon).
  Today's "we use Telegram heavily" is bot-only by design.
* Deeper investigation of the `sentiment` aggregate 75→40 drop. Worth a separate
  research session — sentiment_in_process landed Apr 09, FinGPT shadow accuracy
  backfill landed Apr 22 (`ab616e18`) — possible regression window but we'd need
  a ground-truth re-tagging run to confirm, and that crosses into trading-logic
  territory.

## Changes

### 1. Per-invariant Telegram cooldown (`portfolio/loop_contract.py`)

Add a deduplicated alert state in `contract_state.json`:

```json
"telegram_alert_state": {
  "<invariant_name>": {
    "last_sent_ts": <epoch>,
    "last_message_hash": "<sha1-of-message-text>"
  }
}
```

In `_alert_violations`, before constructing/sending the Telegram message, drop any
CRITICAL violation whose `(invariant, sha1(message))` matches a recent alert (default
cooldown: **4 h**, configurable via `notification.contract_alert_cooldown_s`). Re-fire
immediately if the message text changes (so the moment a *new* signal joins the alert
list, it goes out — we suppress only "exactly the same complaint, again").

Failure mode: if the cooldown logic raises, fail-open (alert anyway). Telegram noise is
worse than a missed alert here.

### 2. Wire `accuracy_degradation` CRITICAL violations into `critical_errors.jsonl`

Add a `record_critical_error` call in `loop_contract.check_signal_accuracy_degradation_safe`
for any CRITICAL violation produced by the accuracy stack — same pattern
`check_layer2_journal_activity` uses today (loop_contract.py:341–351). Keys on
`(invariant, message_hash)` so we don't append the identical row every cycle. This
makes the auto-fix-agent dispatcher see degradation alerts (CLAUDE.md startup check
already reads this file).

### 3. Persist Telegram poller offset (`portfolio/telegram_poller.py`)

Add `data/telegram_poller_state.json` with `{"offset": <int>, "updated_ts": "..."}`.
Load on `__init__`, save inside `_handle_update` after the offset advances.
Atomic via `file_utils.atomic_write_json`.

Stop-gap: when the persisted offset is loaded, we skip the stale-at-startup filter
for messages whose `update_id > persisted_offset` — those are by definition
post-restart pending updates that arrived during downtime, and the user expects
them to execute (e.g. a `bought MSTR …` confirmation sent while the loop was
restarting).

### 4. Operational fix-up (one-shot, not committed)

After merging, run:
* `save_full_accuracy_snapshot()` — fresh baseline so age delta is full 7 d
* Clear `last_full_check_time` / `last_full_check_violations` in `data/degradation_alert_state.json`
* Remove `accuracy_degradation` key from `consecutive` in `data/contract_state.json`

Then restart `PF-DataLoop` + `PF-MetalsLoop` so the new cooldown logic loads.

## Risk

* **False quiet on the alert path.** A 4 h cooldown on identical messages means
  if the same 12 signals stay degraded for 4 h, we emit one alert per 4 h instead
  of one per 10 min. That's the goal, but a regression that genuinely needs more
  attention could be ignored. Mitigation: text-hash sensitivity — any new signal
  joining the alert list immediately re-fires the alert.
* **`record_critical_error` noise.** If we wrote per-cycle, the 192-fire streak
  would have written 192 critical_errors entries. We keep this in check via the
  same `(invariant, message_hash)` dedup as the Telegram cooldown — write once
  per text change.
* **Poller offset persistence + stale filter interaction.** The stale filter
  was added defensively to prevent re-execution on restart. We loosen it only
  when a persisted offset exists AND `update_id > offset` (i.e. truly new), so
  the original protection still kicks in for the cold-start case.

## Execution

1. Commit this plan on `main`.
2. `git worktree add .worktrees/fix-contract-spam-20260428 -b fix/contract-spam-20260428`.
3. **Batch 1** — failing tests (pytest RED), commit.
4. **Batch 2** — per-invariant Telegram cooldown impl, tests pass, commit.
5. **Batch 3** — accuracy_degradation → critical_errors.jsonl wire, tests pass, commit.
6. **Batch 4** — poller offset persistence, tests pass, commit.
7. Full test suite (`pytest -n auto --timeout=60`); fix non-pre-existing reds.
8. Codex adversarial review on the branch; address P1/P2.
9. Merge to main, push via `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`.
10. Operational fix-up on main repo, restart loops, verify quiet.
11. Clean up worktree + branch, append `docs/SESSION_PROGRESS.md`, send Telegram summary.
