# Session Progress — Contract Alert Spam + Poller Hygiene (2026-04-28 night)

**Session start:** 2026-04-28 ~01:00 UTC
**Status:** COMPLETE — Merged + Pushed (7d507748). Loops restarted with new code.

## Contract alert spam + telegram poller hygiene (2026-04-28)

User pulled the last 10 telegram messages, surfaced two latent issues. Both fixed.

### What was wrong

1. **Accuracy_degradation Telegram spam** — 192 consecutive identical CRITICAL alerts in ~32 h.
   The detector's throttled-replay design (re-emits cached Violation list every cycle to keep
   ViolationTracker.consecutive alive) collided with `_alert_violations` shipping a Telegram
   per CRITICAL it sees. Result: same 12-signal alert went out every ~10 min.
2. **Telegram poller offset reset on every restart**. Real user commands (`bought`, `sold`)
   sent during a restart window would silently get stale-dropped. Latent bug — no real
   inbound traffic since Apr 17, so it hadn't bitten yet.

### What shipped (merge 7d507748)

3 plumbing changes + 1 .gitignore:
- **`portfolio/loop_contract.py`**: per-invariant Telegram cooldown (4h default, configurable)
  with multi-hash dedup, mute-aware persist, per-loop state file isolation, fail-open on bad
  config. Routes accuracy_degradation CRITICAL to `critical_errors.jsonl` (post-tracker, with
  6h TTL + journal-aware resolved-row check) so PF-FixAgentDispatcher engages.
- **`portfolio/telegram_poller.py`**: persisted offset across restarts in
  `data/telegram_poller_state.json`, bounded restart-bypass (1h max), settled-only persist
  (don't ack on dispatch raise so a transient crash doesn't lose the user's command).
- **`portfolio/claude_gate.py`**: `record_critical_error` returns bool now so the dispatcher
  can refuse to claim a dedup slot when the underlying append failed.

### Codex review

Ran `codex review --base main` 8 iterations. Each round surfaced new corner cases — all P1/P2
addressed in fix-up commits before merge. Hit codex daily usage limit on round 9 — stopped
iterating, all substantive findings already addressed.

### Tests

- 217 tests across `test_loop_contract*`, `test_accuracy_degradation`, `test_telegram_*`,
  `test_claude_gate`, `test_message_store` all green.
- Full suite has 50 pre-existing failures (15 freqtrade integration ModuleNotFound + ~35
  rotating xdist isolation flakes per `docs/TESTING.md`). Verified those reproduce on main
  with my changes reverted — none mine.

### Operational fixup (one-shot, not committed)

After merge:
- Forced fresh accuracy snapshot via `save_full_accuracy_snapshot()` so age delta is full 7d
- Cleared `last_full_check_*` in `data/degradation_alert_state.json` so the next contract
  cycle re-evaluates against the fresh snapshot (was 6.7d, now 0d)
- Removed `accuracy_degradation` key from `consecutive` in `data/contract_state.json`
  (was at 203 consecutive cycles by the time we reset it)
- Killed orphan python processes + restarted PF-DataLoop and PF-MetalsLoop
- Verified heartbeat fresh (last write within 30s of expected)

### What's not addressed (deliberately)

- **Underlying MSTR-cluster accuracy collapse**: 5 of 12 alerted signals are MSTR-specific
  signals dropping in lockstep (sentiment 90→39, volume_flow 82→34, fibonacci 78→35,
  heikin_ashi 79→35, momentum_factors 89→30). Five orthogonal signal families don't decay
  simultaneously — almost certainly outcome relabeling from the MSTR move. Refresh of the
  baseline (operational step above) probably resolves it; otherwise it's a config-fix
  decision per `memory/project_accuracy_degradation_20260416.md`, requires user judgment
  per signal.
- **`sentiment` aggregate 75→40 drop**: more credible than the MSTR cluster. Worth a
  separate research session — possible regression around Apr 22 when the FinGPT shadow
  accuracy backfill landed (`ab616e18`). Not blocking trading today.

---

# Session Progress — After-Hours Research (2026-04-27/28)

**Session start:** 2026-04-27 ~21:30 UTC
**Status:** COMPLETE — Merged + Pushed (125d12a8)

## What was done

### Phase 0-5: Research & Analysis
- Daily review: system healthy (488 cycles, 0 errors). Patient SELL 50% XAG (ATR stop). Bold BTC +3.1%.
- Macro research: densest risk week of 2026 — FOMC (Powell's LAST), 4 central banks, Mag 7 earnings, GDP+PCE. Hormuz 9th week closed. Inflation expectations spiked 3.8%→4.7%.
- Quant research: Regime-Aware LightGBM (HMM regime + meta_learner), ADWIN drift detection (River library), BUY-bias correction, timeframe-optimal signal allocation
- Signal audit: 6 signals below 40% at 1d_recent, claude_fundamental crash -18.5pp (40.5%), BUY-bias cluster identified
- Ticker deep-dives: XAG (COT weak, G/S widening), XAU (Goldman $5,400), BTC (ETF inflows, exchange reserves 7yr low)

### Batch 1: Signal Gating + Weight Update (commit ce173f46)
- **claude_fundamental gated in ranging _default**: was only gated at 3h/4h — 40.5% at 1d_recent (1178 sam), 78-83% BUY bias was poisoning consensus at longer horizons
- **sentiment gated in ranging + trending-down _default**: 40.1% at 1d_recent, 33.8% at 3h — BUY-only bias harmful at all horizons
- **HORIZON_SIGNAL_WEIGHTS updated**: stale since Mar 29. Key changes:
  - news_event 0.5x → 1.4x at 1d (70.0% recent — SELL-focused edge in declining market)
  - bb 1.2x → 1.3x at 1d (62.5% recent)
  - claude_fundamental 0.5x penalty, sentiment 0.4x
  - structure/smart_money/ema/trend penalties tightened to 0.5
  - Added rsi/credit_spread_risk/volume 1.1x at 1d
- **Per-ticker**: claude_fundamental disabled for XAG-USD, XAU-USD at 1d (metals have no earnings/guidance)
- 5 new tests, 1 test updated. All 85 signal engine tests pass.

### Deliverables
- `data/daily_research_review.json` — Phase 0 review
- `data/daily_research_macro.json` — Phase 1 macro research
- `data/daily_research_quant.json` — Phase 2 quant research (10 topics, actionable items)
- `data/daily_research_signal_audit.json` — Phase 3 signal audit
- `data/daily_research_ticker_deep_dive.json` — Ticker deep-dives (XAG, XAU, BTC)
- `docs/RESEARCH_PLAN.md` — Implementation plan with 4 bugs (P1-P4)
- `data/morning_briefing.json` — Morning briefing for 2026-04-28
- Telegram briefing sent

## What's next
1. **ADWIN drift detection** (easy, high impact) — pip install river, create portfolio/signal_drift.py
2. **HMM regime detection** (medium, high impact) — probabilistic regime via hmmlearn, feed to meta_learner
3. **Per-signal-per-horizon IC auto-disable** (medium, high impact) — eliminate 30-40% noise votes
4. **Regime-conditional BUY-bias penalty** (easy, high impact) — penalize BUY bias harder in bear regimes
5. **Rolling Sharpe per signal** (medium, medium-high impact) — dynamic weight multiplier

## Previous session (2026-04-26/27)
- BB cluster reclassification, ranging gate expansion (volume_flow, credit_spread_risk, ministral)
- Ministral ranging boost removal (collapsed 58.4% → 41.5%)
- EPU + TIPS real yield sub-signals added to metals cross-asset composite

### 2026-04-27 23:25 UTC | main
92628226 plan: contract alert spam + telegram poller hygiene (2026-04-28)
docs/PLAN.md

### 2026-04-27 23:28 UTC | fix/contract-spam-20260428
74951793 test(contract,poller): RED tests for alert cooldown, critical_errors wire, and poller offset
tests/test_loop_contract_accuracy_dispatcher.py
tests/test_loop_contract_alert_cooldown.py
tests/test_telegram_poller_offset.py

### 2026-04-27 23:30 UTC | fix/contract-spam-20260428
c549773e fix(contract): per-invariant Telegram cooldown in _alert_violations
portfolio/loop_contract.py

### 2026-04-27 23:31 UTC | fix/contract-spam-20260428
7ded0f41 fix(contract): wire accuracy_degradation CRITICAL into critical_errors.jsonl
portfolio/loop_contract.py

### 2026-04-27 23:33 UTC | fix/contract-spam-20260428
08dee0a4 fix(poller): persist Telegram getUpdates offset across loop restarts
portfolio/telegram_poller.py

### 2026-04-27 23:38 UTC | fix/contract-spam-20260428
6dedaf7e chore(gitignore): exclude telegram poller runtime files
.gitignore

### 2026-04-27 23:50 UTC | fix/contract-spam-20260428
e10dd84d fix(contract,poller): address Codex review findings
portfolio/loop_contract.py
portfolio/telegram_poller.py
tests/test_loop_contract_accuracy_dispatcher.py
tests/test_telegram_poller_offset.py

### 2026-04-28 00:10 UTC | fix/contract-spam-20260428
4a6d91fe fix(contract): codex round 2/3 — mute-aware cooldown + multi-hash dedup + return-bool
portfolio/claude_gate.py
portfolio/loop_contract.py
tests/test_loop_contract_accuracy_dispatcher.py
tests/test_loop_contract_alert_cooldown.py

### 2026-04-28 00:20 UTC | fix/contract-spam-20260428
cda77a0a fix(contract,poller): codex round 4 — stable hash + offset clamp
portfolio/loop_contract.py
portfolio/telegram_poller.py
tests/test_loop_contract_accuracy_dispatcher.py
tests/test_loop_contract_alert_cooldown.py
tests/test_telegram_poller_offset.py

### 2026-04-28 00:30 UTC | fix/contract-spam-20260428
04e3cfc4 fix(contract,poller): codex round 5 — bounded restart bypass + per-incident identity + journal-aware dedup
portfolio/loop_contract.py
portfolio/telegram_poller.py
tests/test_loop_contract_accuracy_dispatcher.py
tests/test_loop_contract_alert_cooldown.py
tests/test_telegram_poller_offset.py

### 2026-04-28 00:45 UTC | fix/contract-spam-20260428
858723d8 fix(contract): codex round 6 — per-loop state, fail-open cooldown_s parse, NO_TELEGRAM honor
portfolio/loop_contract.py
tests/test_loop_contract_alert_cooldown.py

### 2026-04-28 00:57 UTC | fix/contract-spam-20260428
7ef03690 fix(poller): codex round 7 — persist offset only after message settles
portfolio/telegram_poller.py
tests/test_telegram_poller_offset.py

### 2026-04-28 01:08 UTC | fix/contract-spam-20260428
8fdcb935 fix(contract): codex round 8 — clear cooldown when invariant clears
portfolio/loop_contract.py
tests/test_loop_contract_alert_cooldown.py
