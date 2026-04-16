# Signal Accuracy Degradation Tracker

**Status:** active
**Branch:** feat/accuracy-degradation
**Worktree:** ../finance-analyzer-degradation
**Approved plan:** /root/.claude/plans/zazzy-strolling-candy.md

## Why

This morning's investigation (`memory/project_accuracy_degradation_20260416.md`)
found Tier-1 1d consensus collapsed from 52-56% to 36-41% across W15/W16 because
of a horizon-mismatched per-ticker blacklist. **No alert ever fired** — the
existing 11 main-loop runtime contracts (`portfolio/loop_contract.py`) check
execution health, not decision quality. Dormant infrastructure for snapshot +
change-detection has lived in `accuracy_stats.py` since the original signal
research push but was never wired up.

Goal: detect signal/forecast/consensus accuracy drops within hours, not weeks,
and notify via the same plumbing as every other contract violation.

## Scope (user-confirmed)

- **All four scopes**: per-signal global, per-ticker per-signal, forecast
  models (chronos/kronos/ministral/qwen3), aggregate consensus.
- **Severity gate**: drop >15pp vs 7d baseline AND current accuracy <50%.
- **Cadence**: daily snapshot at fixed UTC hour, hourly comparison (gated
  inside the running loop; no new schtask).
- **Notification**: BOTH a `loop_contract` violation (auto-escalates via
  `ViolationTracker`, journals to `data/critical_errors.jsonl`) AND a
  dedicated daily Telegram summary.

## Architecture

| Piece | What |
|---|---|
| `portfolio/accuracy_degradation.py` (new) | Snapshot save, hourly check, severity classifier, daily summary builder |
| `portfolio/accuracy_stats.py` | Optional `extras` kwarg on `save_accuracy_snapshot()`, new `consensus_accuracy()` helper |
| `portfolio/forecast_accuracy.py` | New `cached_forecast_accuracy()` 1h-TTL wrapper |
| `portfolio/loop_contract.py` | New `check_signal_accuracy_degradation()` invariant slotted at end of `verify_contract()` |
| `portfolio/main.py` | Two `_track()` calls in post-cycle path (snapshot save + summary send), gated by hour-of-day |
| `data/accuracy_snapshots.jsonl` | Extended schema (signals/per_ticker/forecast/consensus blocks); back-compat with old single-block |
| `data/degradation_alert_state.json` | Hourly throttle, per-signal 24h cooldown, last summary send |
| `data/accuracy_snapshot_state.json` | Once-per-day snapshot guard |
| `data/forecast_accuracy_cache.json` | 1h-TTL cache for forecast compute |
| `tests/test_accuracy_degradation.py` (new) | ~15 unit tests |
| `tests/test_loop_contract_accuracy.py` (new) | 3 integration tests |

## Reuse — do not reimplement

- `signal_accuracy("1d")`, `accuracy_by_ticker_signal_cached("1d")`,
  `_find_snapshot_near()`, `_load_accuracy_snapshots()` already exist in
  `accuracy_stats.py`.
- `Violation`, `ViolationTracker`, `record_critical_error()`,
  `_alert_violations()`, `verify_and_act()` already in `loop_contract.py`.
- `message_store.send_or_store(category="daily_digest")` already routed.
- `econ_calendar.events_within_hours()` for FOMC/CPI blackout already exists.
- `file_utils.atomic_append_jsonl()` and `atomic_write_json()` per
  project I/O rules.

## Severity rules

- **WARNING**: 1-2 single-signal drops meeting the 15pp + <50% gate.
  `ViolationTracker` auto-escalates to CRITICAL after 3 consecutive hourly
  fires (~3 hours sustained).
- **CRITICAL** immediately on:
  - 3+ signals dropping in the same check, OR
  - Aggregate consensus drop (single check, single condition).

## Anti-noise gates (all mandatory)

- Min historical samples ≥ 100 AND min current samples ≥ 100 per signal.
- Snapshot age ≥ 6 days (don't alert until we have a real baseline).
- Per-signal 24h cooldown via state file.
- Hourly throttle: short-circuit `[]` if < 55 min since last full check
  (per-cycle cost ~0.1ms).
- Skip during FOMC/CPI ±24h windows.

## Telegram formats

**Contract path** (matches existing `*LOOP CONTRACT (main)*` style):

```
*LOOP CONTRACT (main)* — 1 critical violation(s)
• accuracy_degradation: 3 signals dropped >15pp vs 7d
  (rsi 62%→44%, macd 58%→41%, bb 55%→39%); consensus 56%→47%
```

**Daily summary** (category `daily_digest`):

```
*ACCURACY DAILY* · 2026-04-16
`Consensus: 56% (Δ -2.1pp 7d) · 880 sam`
`Forecast:  chronos 51% · kronos 49% · ministral 53% · qwen3 47%`

*Degraded (>15pp drop, <50% abs)*
`rsi       62% -> 44% (-18pp, 1240 sam)`
`macd_btc  61% -> 42% (-19pp, 210  sam)`

*Improved (>10pp gain)*
`obv       48% -> 61% (+13pp, 980  sam)`

`Snapshot age: 7.0d · 27 signals tracked`
```

## Codex integration

1. **Pre-implementation review** — adversarial-review the plan doc with
   `--effort xhigh` to challenge thresholds, anti-noise rules, integration
   choice. Address findings before code.
2. **Post-implementation review** — adversarial-review the full branch.
   Fix every P1/P2 finding. Decide P3 case-by-case.
3. **Codex rescue** for stuck implementation problems (delegate after 2
   failed attempts on the same problem).

## Execution batches

- **Batch 1**: snapshot infra (`accuracy_stats.py` + `forecast_accuracy.py`).
- **Batch 2**: `accuracy_degradation.py` + tests.
- **Batch 3**: loop integration (`loop_contract.py` + `main.py` + integration tests).
- **Batch 4**: e2e verification + docs.

Then: post-impl Codex review → fix → full test suite → merge → push →
restart loop via `pf-restart.bat loop` → cleanup worktree.

## Test list

`tests/test_accuracy_degradation.py` (all `tmp_path` + monkeypatch):

1. `test_snapshot_writes_all_four_scopes`
2. `test_snapshot_back_compat_reads_single_block_old_snapshots`
3. `test_degradation_below_threshold_no_alert` (10pp drop)
4. `test_degradation_above_threshold_emits_warning` (18pp + 44%)
5. `test_three_signals_escalates_to_critical`
6. `test_consensus_drop_emits_critical`
7. `test_min_samples_gate_skips_low_n` (50 historical / 120 current)
8. `test_snapshot_age_under_6d_gate_returns_empty`
9. `test_fomc_blackout_skips_check`
10. `test_24h_cooldown_per_signal_blocks_repeat`
11. `test_24h_cooldown_expires_at_25h_re_alerts`
12. `test_hourly_throttle_short_circuits_when_recent`
13. `test_per_ticker_alert_format_uses_ticker_signal_key`
14. `test_forecast_alert_format_uses_forecast_prefix_key`
15. `test_summary_contains_consensus_top_drops_top_gains`

`tests/test_loop_contract_accuracy.py`:

1. `test_invariant_wired_into_verify_contract`
2. `test_check_runs_in_under_one_second_when_throttled`
3. `test_snapshot_and_summary_only_fire_at_configured_hour`

## End-to-end verification

1. Backup: `cp data/accuracy_snapshots.jsonl /tmp/snapshot_backup.jsonl`
2. Hand-craft a snapshot dated `now - 7d` with `rsi.accuracy=0.65,
   total=500` and consensus=0.55; append.
3. Force the cache hot via `get_or_compute_accuracy('1d')`.
4. Mutate the cache file in place to set `rsi.accuracy=0.42, total=600`,
   consensus to `0.43`.
5. Run check: expect 1 WARNING (rsi) + 1 CRITICAL (consensus) → CRITICAL overall.
6. Run again immediately → expect `[]` from hourly throttle.
7. Restore snapshot.

## Deferred (follow-up PRs)

- Per-regime degradation tracking (RSI in ranging vs trending).
- Auto-tuning thresholds via false-alarm-rate Bayesian estimate.
- Snapshot file rotation / compaction (append-only forever; not a
  problem yet at <1 MB after a year).
- Dashboard widget for `degradation_alert_state.json`.
- Per-horizon (3h/4h/12h/3d/5d/10d) tracking — currently 1d-only.
- Bot loop (GoldDigger/Elongir) accuracy contracts.
- Auto-disable degraded signals — out of scope; the existing 47% accuracy
  gate already silences them. This work is purely about NOTIFYING.
