# PLAN — BUG-178 Instrumentation + Timeout Bump

**Branch:** `fix/bug178-instrumentation-and-timeout`
**Date started:** 2026-04-15
**Author:** Claude (Opus 4.6, 1M)

## Context

Telegram alert at 10:34: `LOOP ERRORS (884s cycle) 5 ticker(s) failed entirely` +
`LOOP CONTRACT (main) — 1 critical violation: min_success_rate 0%`.

Investigation showed the 180s `_TICKER_POOL_TIMEOUT` (set in `3ac7b81` on 2026-04-09 after
fingpt moved to post-cycle batch) is now regularly firing on legitimate work. Cycle history
across Apr 14-15 shows dozens of BUG-178 timeouts with zombie threads completing 330-525s
into the cycle, all 5 within ~10s of each other — signature of a shared-resource wait rather
than genuinely stuck work.

Since the 2026-04-09 baseline (226s cycle, 45s/ticker) the ticker path has grown:

| Commit | Date | Effect |
|--------|------|--------|
| `3062c01` | 04-10 | LLM rotation scheduling (ministral/qwen3/fingpt one at a time) |
| `7264889` | 04-10 | llama_server active VRAM poll + KV cache reuse |
| `489d4a3` | 04-12 | Intraday (60m) DXY cross-asset for metals 1-3h |
| `d86c435` | 04-12 | New `vix_term_structure` signal (ticker-path) |
| `711ff94` | 04-13 | Per-ticker signal gating |
| `c8556b3` | 04-13 | Fundamental correlation cluster + ETH qwen3 gate |
| `6ec4be9` | 04-11 | Per-ticker directional accuracy gate |

The commit message on `3ac7b81` said:

> "If cycles start creeping above ~180s again, the first place to look is an added signal/LLM
> in the ticker path — do NOT bump this timeout without understanding why, it's meant to
> fire on hangs not on legitimate slow processing."

That warning was correct but we have no per-phase visibility into WHERE the 500s goes. The
`__post_dispatch__` marker is the last tracked point before a ~330-525s zombie tail, which
is too coarse. We need finer instrumentation before we can say confidently whether the extra
work is (a) legitimate growth, (b) a shared-resource lock, or (c) a Windows-specific hang.

## Hypotheses (to be validated by the instrumentation)

1. **accuracy_stats cold-cache contention.** `signal_utility()` is uncached and does a full
   scan of ~6320 snapshot entries (measured 3.6s cold, <50ms hot). 5 threads simultaneously
   cold-missing could amplify through OS file cache pressure and `_accuracy_write_lock`.
2. **`_weighted_consensus` compute growth.** Added signals increase the iteration cost of
   this function; per-ticker directional gating adds a disk lookup per signal.
3. **Windows `tasklist` subprocess timeouts.** `portfolio/llama_server.py:141` recently
   started logging `subprocess.TimeoutExpired` on PID verification with 5s timeouts —
   symptomatic of Windows stress but unclear whether it costs real ticker-phase time.

## Changes

### Batch 1 — Phase-level instrumentation
Files: `portfolio/signal_engine.py`, `portfolio/main.py`

Add named phase markers inside `generate_signal()` after `__post_dispatch__`:

- `__acc_load__` after the `accuracy_stats` blocks complete (lines ~1866-1920)
- `__ticker_gate__` after BUG-158 per-ticker override (line ~1977)
- `__utility_overlay__` after signal_utility + best-horizon blocks (~line 2020)
- `__weighted__` after `_weighted_consensus` (~line 2029)
- `__penalties__` after apply_confidence_penalties + market_health + earnings (~line 2103)
- `__linear_factor__` after linear factor block (~line 2135)
- `__consensus_gate__` after per-ticker consensus gate (~line 2164)

Add a per-phase timer that logs WARNING when any single phase exceeds 2s for a ticker.

Update the `main.py` slow-cycle diagnostic to dump the complete phase-sequence snapshot for
the cycle that tripped (use a new thread-local phase log, not just last-seen).

### Batch 2 — signal_utility in-memory cache
Files: `portfolio/accuracy_stats.py`

Wrap `signal_utility()` in a dogpile-protected cache (5-minute TTL, matching
ACCURACY_CACHE_TTL/12 effective window). Current cold time is ~3.6s; if we get unlucky with
OS cache eviction and 5 threads cold-miss simultaneously on the 108MB DB, this serializes
behind the file cache paging in.

### Batch 3 — Timeout bump with justification
Files: `portfolio/main.py`

Raise `_TICKER_POOL_TIMEOUT` from 180 → 360. Rationale (in the comment):

- Observed p50 cycle: 50-130s (still within old budget)
- Observed p95 cycle: 330-525s (triggers BUG-178 under old budget, zombie threads finish
  130-345s AFTER pool timeout fires — we lose their results for no reason)
- Cycle cadence: 600s — 360s leaves 240s margin for post-cycle LLM batch + trigger +
  journal + telegram work
- 2× p50 slow cycle (~180s) was the old rule; at current signal count the p50-slow is ~60s
  but p95 spikes legitimately reach 400s+ under llama-server contention bursts
- Loop contract's own `cycle_dur` check at 600s + Layer 2 timeout remain the catch-all for
  genuine hangs — 360s in pool, 600s cadence, 180s Layer 2 retry

Update the timeline comment block to include the 2026-04-15 bump + signal growth list so
the next cold-reader understands *why* the number moved.

## Tests

- Add `tests/test_signal_utility_cache.py` — verify TTL behavior, dogpile protection, cache
  invalidation path.
- Run full suite: `.venv/Scripts/python.exe -m pytest tests/ -n auto`
- Tests that touch `signal_engine.generate_signal()` must continue to pass unchanged.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Instrumentation adds overhead to every ticker | One `time.monotonic()` per phase, <1μs; no I/O |
| Cached `signal_utility` masks real accuracy changes | 5-min TTL; upstream writes already invalidate via accuracy_cache file mtime |
| 360s timeout delays hang detection by 3 min | Loop contract cycle_dur check (600s) is unaffected; Layer 2 invoker has its own timeouts |
| Log volume increases from phase warnings | Gated at 2s threshold per phase — normal cycles emit zero warnings |

## Execution order

1. Commit this plan.
2. Batch 1: instrumentation. Test. Commit.
3. Batch 2: signal_utility cache. Test. Commit.
4. Batch 3: timeout bump + comment rewrite. Test. Commit.
5. Full pytest run.
6. Codex adversarial review.
7. Fix findings. Commit.
8. Merge to main. Push via `cmd.exe`.
9. Clean up worktree.
10. Restart loops: `PF-DataLoop` + `PF-MetalsLoop`.
