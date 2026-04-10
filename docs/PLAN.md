# PLAN — Fix BUG-178 silent signal hangs (4×/day, 49 events total)

## Context

While investigating the (closed) `project_chronos_vram_contention` follow-up on 2026-04-10, found a much bigger ongoing problem: **49 `BUG-178: Ticker pool timeout` events** since 2026-04-09 (45 yesterday, 4 today). Original VRAM-contention hypothesis was wrong; Chronos is fast in steady state.

## Root cause (high-confidence)

`portfolio/signal_engine.py:1564` — the per-ticker enhanced-signal dispatch loop iterates `_enhanced_entries.items()` and calls every signal's `compute_fn()`, but **does NOT check `DISABLED_SIGNALS`**. The check is only present in `count_active_signals()` (line 468) and in dynamic correlation grouping (line 558). Reporting and accuracy code also filters disabled, but the actual compute path runs them.

Three "disabled" signals are still running on every ticker every cycle:
- `crypto_macro` — fetches Deribit options + BGeometrics netflow
- `cot_positioning` — fetches CFTC data on cache miss
- `credit_spread_risk` — fetches FRED HY OAS data; runs for **all 5 trading tickers**

These all do network I/O. They were added 2026-04-08 (commits `bcd71bb`, `3d47626`). The first BUG-178 events fired 2026-04-09, exactly tracking the deployment. CLAUDE.md says they are "registered but force-HOLD via DISABLED_SIGNALS pending live validation" — the *intent* is they don't run.

## Evidence

Last `[SLOW]` log line in **all 4 of today's BUG-178 events** (05:15 / 06:25 / 11:56 / 14:06) is always `metals_cross_asset` or `futures_flow` — never any signal further down the iteration order. The next signals to run after metals_cross_asset are: `cot_positioning`, `credit_spread_risk`. Then 150+ seconds of complete log silence, then the 180s pool timeout.

## Scope (single small change + diagnostic)

### Lever 1 — Skip DISABLED_SIGNALS in dispatch loop (the actual fix)

Edit `portfolio/signal_engine.py:1564-1568`:

```python
for sig_name, entry in _enhanced_entries.items():
    if sig_name in DISABLED_SIGNALS:
        votes[sig_name] = "HOLD"
        continue
    if skip_gpu and sig_name in GPU_SIGNALS:
        votes[sig_name] = "HOLD"
        continue
    ...
```

Same pattern as the `skip_gpu` check immediately below it. Restores the documented behavior. Eliminates 5 tickers × ~3 disabled-signal compute calls per cycle.

### Lever 2 — Per-ticker last-signal diagnostic (so we can verify)

Add a thread-safe last-signal tracker in `signal_engine.py`. On `BUG-178` fire in `main.py`, log per-ticker last signal name + how long it has been hung. If the dispatch-loop fix doesn't fully eliminate the hangs, the next BUG-178 event will surface the actual culprit.

```python
# signal_engine.py (module-level)
_last_signal_per_ticker: dict[str, tuple[str, float]] = {}
_last_signal_lock = threading.Lock()

def _set_last_signal(ticker: str, sig_name: str):
    with _last_signal_lock:
        _last_signal_per_ticker[ticker] = (sig_name, time.monotonic())

def get_last_signal(ticker: str) -> tuple[str, float] | None:
    with _last_signal_lock:
        entry = _last_signal_per_ticker.get(ticker)
        if entry is None:
            return None
        sig_name, started = entry
        return sig_name, time.monotonic() - started
```

In the dispatch loop: call `_set_last_signal(ticker, sig_name)` right before `compute_fn()`.

In `main.py:577` (BUG-178 handler), enrich the log line:
```python
from portfolio.signal_engine import get_last_signal
last_sigs = {n: get_last_signal(n) for n in timed_out}
logger.error("BUG-178: Ticker pool timeout after %ds. Stuck: %s. Last signals: %s",
             _TICKER_POOL_TIMEOUT, timed_out, last_sigs)
```

## Out of scope

- Do NOT change which signals are in `DISABLED_SIGNALS` itself — that's a separate decision.
- Do NOT add timeouts to credit_spread / cot_positioning / crypto_macro fetches — they should not be running at all.
- Do NOT touch Chronos / Kronos / forecast — premise was wrong, see `project_chronos_vram_contention.md`.

## Risks

| Risk | Mitigation |
|---|---|
| Skipping disabled signals breaks consensus calculations | Their votes are already discarded by accuracy/regime gating in most cases. Verify by running pytest. |
| Skipping breaks shadow accuracy tracking (we want to track them for potential re-enable) | They were already tracked via `outcome_tracker` which reads from `data/signal_log.jsonl`. With `vote=HOLD`, they just won't get accuracy updates. This matches "pending live validation" — we already weren't using the votes anyway. |
| Diagnostic adds dict-write overhead per signal | Negligible — single dict write under a lock, ~22 signals × 5 tickers = ~110 writes/cycle. |
| The fix doesn't actually eliminate BUG-178 (hang is elsewhere) | The diagnostic lever 2 will surface the real culprit on the next event. Two-layered defense. |

## Verification

### Unit / integration
- `pytest tests/test_signal_engine*.py -v` — must still pass
- `pytest tests/ -n auto --ignore=tests/integration` — full suite

### Live
After merge + restart, monitor `data/portfolio.log` for:
- Zero new `BUG-178` events for ≥6 hours (current rate: 4/day)
- Zero new `Signal X failed: ...` warnings on the disabled three (they should not be called)
- Per-cycle `Signal loop done` should drop to <120s consistently

If a NEW `BUG-178` fires, the per-ticker last-signal diagnostic will name the real culprit and we open a follow-up.

## Execution order

1. Worktree `/mnt/q/finance-analyzer-bug178` ✅ (already created)
2. Edit `signal_engine.py` (Lever 1 + Lever 2 helpers)
3. Edit `main.py` BUG-178 handler (Lever 2 logging)
4. Run targeted tests
5. Run full suite via pytest
6. Commit
7. Pull main, rebase if needed
8. Merge to main, push via cmd.exe
9. Restart PF-DataLoop
10. Tail log for 30+ min, verify zero BUG-178

## Rollback

```bash
git revert <merge-sha>
cmd.exe /c "cd /d Q:\finance-analyzer && git push"
schtasks /end /tn PF-DataLoop && schtasks /run /tn PF-DataLoop
```

Single commit, isolated to two files. Full revert is one merge revert.
