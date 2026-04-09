# PLAN — Reduce main loop ticker universe to crypto + metals (+ MSTR)

**Date:** 2026-04-09
**Branch:** `chore/reduce-tickers-apr09`

## Motivation

Main loop cycles are 2.4x the 60s target (p50=143s over 48h of logs, p90=310s, max=1057s).
71% of cycles miss the cadence target. Per-day trend is improving (Apr-7 mean 208s →
Apr-9 mean 105s) but still not at target. Signal work for stocks is a significant part
of that cost, and the user does not actively trade them — only crypto and metals are
traded via Avanza warrants. Shrinking the universe to what's actually traded is the
cheapest, lowest-risk way to bring the loop back to cadence.

## What

Reduce `portfolio/tickers.py` `SYMBOLS` to 5 tickers:

| Ticker   | Source       | Rationale                                    |
|----------|--------------|----------------------------------------------|
| BTC-USD  | Binance spot | actively traded                              |
| ETH-USD  | Binance spot | actively traded                              |
| XAU-USD  | Binance FAPI | actively traded via gold warrants            |
| XAG-USD  | Binance FAPI | actively traded via silver warrants          |
| MSTR     | Alpaca       | BTC NAV-premium reference used by metals_loop |

Removed: `PLTR`, `NVDA`, `MU`, `SMCI`, `TSM`, `TTWO`, `VRT`.

`STOCK_SYMBOLS` shrinks to `{"MSTR"}`.

## Why this shape

- **MSTR kept**: `data/metals_loop.py` docstring line 4 says "Tracks: XAG/XAU (Binance
  FAPI), BTC/ETH (Binance SPOT), MSTR (Yahoo)" and lists "MSTR-BTC NAV premium tracking"
  as a feature. Dropping MSTR would require touching metals_loop.py separately.
- **Oil NOT added**: `CL=F` is already pulled as a metals cross-asset context feature
  by `portfolio/metals_cross_assets.py:get_oil_data()` and precomputed every 2h by
  `portfolio/oil_precompute.py`. Adding oil as a standalone Tier-1 ticker would need a
  new data source (yfinance is not currently in `data_collector.py` for primary fetch),
  new signal module, ~500 LOC of plumbing — opposite of the simplification goal.
- **Hard edit, not config flag**: `tickers.py` advertises itself as "single source of
  truth" in the module docstring. A config flag would add conditional logic to that
  module. Scope reduction is not a toggle we expect to flip frequently; git is the
  rollback mechanism. Soft flags are appropriate for feature flags, A/B tests, or kill
  switches — none of which this is.

## What could break (verified)

Stage A read-only checks done 2026-04-09 before writing this plan:

1. **`portfolio/main.py:438`** filters the active ticker set:
   `active_items = [(name, source) for name, source in SYMBOLS.items() if name in active]`.
   Dropped tickers are never fetched, never signaled, never reported. Confirmed.
2. **Portfolio state is empty of positions**: `portfolio_state.json` (Patient, 497k SEK)
   and `portfolio_state_bold.json` (Bold, 457k SEK) both have `holdings: {}`. No positions
   would be orphaned.
3. **No warrant state file**: `data/portfolio_state_warrants.json` does not exist. No
   MINI-TSMC warrant holdings to break when TSM is dropped.
4. **TSM references outside `tickers.py`** are all static lookup data — no runtime fetches
   or signal computation is gated on TSM being in SYMBOLS:
   - `portfolio/news_keywords.py:90` — semiconductor keyword set
   - `portfolio/monte_carlo_risk.py:118-121` — correlation pairs dict
   - `portfolio/sentiment.py:66` — sector map
   - `portfolio/risk_management.py:532-535` — correlation peer list
5. **`metals_cross_asset` signal** (XAU/XAG only) fetches HG=F, ^GVZ, GC=F, SI=F, SPY,
   CL=F directly via yfinance ad-hoc calls. No dependency on `tickers.py` stock entries.
6. **`macro_regime` signal** uses DXY, FRED yields, FOMC calendar. No stock dependency.
7. **`autonomous.py` Layer 3 fallback** iterates triggered and held tickers dynamically,
   no hardcoded stock logic.
8. **Dashboard endpoints** read portfolio_state.json / agent_summary.json / trades journal
   — they surface whatever tickers are present and don't require any specific ticker.

## Expected side effects (acceptable)

- `data/accuracy_cache.json` will have orphaned entries for the dropped stocks —
  auto-rebuilds over time, safe to ignore.
- `data/signal_log.jsonl` / `signal_log.db` will contain historical data for the dropped
  tickers — useful for backtests, no runtime issue.
- `CLAUDE.md`'s Tier-1 instrument list is now doubly stale (Mar 15 removed 8, this change
  removes 7 more, leaving 5). Leave for a separate docs cleanup PR to keep this change
  minimal-surface.
- `signal_engine.py:110-111` has a cleanup path that drops sentiment cache entries for
  tickers no longer in `ALL_TICKERS`. This will run once on first cycle after deployment
  and clean up stale state. No error.

## Residual risk (does NOT resolve)

This change does not fix the **llama_server qwen3↔ministral3 swap thrashing** seen in
the logs this morning. With 5 tickers instead of 12, the swap cadence drops
proportionally (fewer per-ticker invocations) so the loop should get closer to target.
If post-deployment measurement shows the loop still misses target, next steps are:

- (a) persist both LLMs on separate ports (requires `llama_server.py` refactor)
- (b) consolidate or gate one of the two LLM-based signals (ministral vs qwen3)
- (c) fix the `_kill_server_by_pid` pid-recycling leak pathway in `llama_server.py:131-155`

These are out of scope for this PR — driven by post-reduction timing data.

## Execution order

1. Edit `portfolio/tickers.py` (one file) — shrink `SYMBOLS` and `STOCK_SYMBOLS`, add
   Apr 09 removal comment with rationale.
2. Import smoke test (`python -c "from portfolio.tickers import ..."`).
3. Targeted tests in the modules that import from `tickers.py`:
   - `tests/test_consensus.py`
   - `tests/test_signal_pipeline.py`
   - `tests/test_market_timing.py`
   - `tests/test_metals.py`
   - `tests/test_alpha_vantage.py`
   - `tests/test_telegram_formatting.py`
   - `tests/test_trigger_edge_cases.py`
4. Full parallel suite: `.venv/Scripts/python.exe -m pytest tests/ -n auto` (in
   background while doing follow-up work).
5. Codex adversarial review on the branch.
6. Fix any valid findings; document false positives.
7. Merge to main.
8. Push via Windows git (`cmd.exe /c "cd /d Q:\finance-analyzer && git push"`).
9. Clean up worktree and branch (`git worktree remove` + `git branch -d`).
10. Restart `PF-DataLoop` scheduled task (per memory `feedback_restart_loops.md`).
11. Monitor cycle timing for 15 minutes; compare to the 158s mean baseline.

## Rollback

If the reduced loop breaks:

```
git -C /mnt/q/finance-analyzer revert <commit-sha>
cmd.exe /c "cd /d Q:\finance-analyzer && git push"
schtasks /run /tn "PF-DataLoop"
```

## Tests to update (expected failures)

- `tests/test_consensus.py:58-60` asserts `PLTR`, `NVDA`, `MSTR` are in `STOCK_SYMBOLS`.
  Update to assert only `MSTR`, or drop the `PLTR`/`NVDA` assertions.
- `tests/test_signal_pipeline.py:141` iterates `list(STOCK_SYMBOLS)[:5]` for a sample.
  Will still work (just tests 1 ticker) but worth noting.
- `tests/test_alpha_vantage.py:368-369` iterates `STOCK_SYMBOLS`. Will test only MSTR.
- `tests/test_telegram_formatting.py` uses a locally-defined `ALL_TICKERS` list — no
  change needed.
- `tests/test_trigger_edge_cases.py` uses a locally-defined `ALL_TICKERS` — no change
  needed.

Any test that hard-codes removed tickers in its expectations will need to be updated.
