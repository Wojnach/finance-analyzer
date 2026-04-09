# PLAN — Fingpt warm daemon + main loop cadence 60s → 600s

**Date:** 2026-04-09
**Branch:** `chore/fingpt-daemon-and-cadence`

## Problem

Main loop currently overruns its 60s target. Post-reduction to 5 tickers this morning
cut the overhead but not the root cause. `data/portfolio.log` shows:

```
[WARNING] portfolio.gpu_gate: GPU file-lock timeout (90s) — held by fingpt
[WARNING] portfolio.signals.forecast: GPU gate timeout for Kronos XAU-USD
```

fingpt cold-loads a 1.2 GB GGUF model from disk on **every subprocess invocation**, holds
the shared GPU file-lock for the entire 70-90s load+inference window, then exits. With
10-15 fingpt calls per cycle this serializes Kronos/Chronos forecasting behind it.

## Change

1. **Warm daemon**: new long-lived subprocess that loads the fingpt model once and
   services per-request NDJSON inference over stdin/stdout. GPU lock held only during
   inference (1-3s), not model load. Eliminates the 70-90s lock hold pattern.

2. **Cadence**: main loop sleep interval 60→600 (market open) and 120→600 (market
   closed). Weekend already 600. This is a simple "give it room" measure; the reduced
   ticker universe + warm fingpt should fit comfortably inside 600s with huge margin.

Metals loop is explicitly **out of scope** — it runs on its own independent 60s cycle,
does not call fingpt, and has had no reported problems.

## Files touched

| File | Change |
|---|---|
| `scripts/fingpt_daemon.py` (new) | Long-lived inference daemon, ~130 LOC |
| `portfolio/sentiment.py` | `_run_fingpt()` body replaced with thread-safe daemon client; module-level singleton + lock + atexit cleanup |
| `portfolio/market_timing.py` | `INTERVAL_MARKET_OPEN` 60→600, `INTERVAL_MARKET_CLOSED` 120→600 |
| `portfolio/main.py` | Log rotation at line ~372-374 switched from cycle-count to wall-clock (else rotates every 10h instead of 1h) |
| `tests/test_fingpt_daemon.py` (new) | Daemon client unit tests (lazy init, thread safety, crash recovery) |

## Risks

- **Daemon crashes** → `_run_fingpt` catches BrokenPipeError, marks proc dead, retries once; second failure falls through to existing FinBERT fallback in `sentiment.py:527-575`
- **Persistent VRAM** → ~1.5 GB for Q4 1.2B model, leaves ~7 GB headroom on RTX 3080 with llama-server + Kronos + Chronos
- **10-min cadence breaking time assumptions** → verified: triggers, L2 timeouts, health, digest, accuracy gates are all wall-clock based. Only cycle-count dependency found is log rotation (fixed in this PR).
- **Merge conflict with parallel worktree** `fix/metals-loop-reliability-apr09` → confirmed no file overlap, but will pull main before merge regardless.

## Rollback

```
git -C /mnt/q/finance-analyzer revert <merge-sha>
cmd.exe /c "cd /d Q:\finance-analyzer && git push"
powershell.exe -c "Stop-Process -Id <main.py pid> -Force"
cmd.exe /c "schtasks /run /tn PF-DataLoop"
```

The daemon client's failure modes all degrade to the existing FinBERT fallback rather
than crashing the loop, so worst case is a lost fingpt signal component until revert.

## Execution

Batches (each a commit):
1. Plan doc (this file)
2. Daemon script + sentiment client + targeted tests
3. Cadence + log rotation fix
4. Full suite + code review + merge + push
5. Restart main loop only, measure fingpt lock drops + cycle durations

## Success criteria

- `grep "GPU file-lock timeout.*fingpt" data/portfolio.log` drops to near zero/hour (from ~4+/hour baseline)
- `grep "Schedule:" data/portfolio.log` shows `600s interval` during market open
- Cycle durations all well under 600s (trivially, given 10x headroom)
- Sentiment signal returns `fingpt:*` model tag, not `fingpt:*:error`
- VRAM baseline rises by ~1.5 GB and stays flat between cycles
