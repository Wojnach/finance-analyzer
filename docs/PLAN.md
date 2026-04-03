# Plan: Integrate Fish Engine with Full Signal Intelligence

## Date: 2026-04-03

## Findings from Exploration

### Signal Duplication — NOT what we thought
The metals loop does NOT compute its own signal consensus. It reads the main
loop's signals via `read_signal_data()` (from agent_summary.json). The
`metals_signal_log.jsonl` is just LOGGING the main loop's signals plus LLM
predictions. The "metals disagree" problem from Apr 2 was a TIMING LAG.

**Fix:** Remove `metals_action` as separate field. Use `signal_action` only.

### Layer 2 Journal — Rich context being wasted
Each journal entry has per-ticker: outlook, thesis, conviction, levels,
watchlist. Fish engine reads NONE of this. Layer 2 runs 2-5x per day.

### Other wasted intelligence
- Monte Carlo bands (price_bands_1d/3d) — not used for TP/SL
- Chronos forecast (chronos_1h/24h_pct) — not read
- Prophecy ($120 target, 0.8 conviction) — not read

## Batches

### Batch 1: Fish engine reads Layer 2 journal + all wasted signals
Files: data/fish_engine.py, data/metals_loop.py

- Add to state dict: layer2_outlook, layer2_conviction, layer2_levels,
  mc_bands_1d, chronos_1h_pct, chronos_24h_pct, prophecy_target,
  prophecy_conviction
- Read from: layer2_journal.jsonl (last XAG-USD entry),
  agent_summary_compact.json (MC, Chronos, prophecy)
- Remove metals_action — eliminate false "metals disagree" exits

### Batch 2: Layer 2 as 9th tactic vote + MC for dynamic TP/SL
Files: data/fish_engine.py

- _vote_layer2(): outlook+conviction → LONG/SHORT vote (weight 2)
- MC bands for TP/SL: replace fixed +2%/-3% with data-driven levels
- Chronos as confidence modifier (not separate vote)
- Stale check: ignore Layer 2 data older than 4h

### Batch 3: Fishing context from Layer 2 journal post-processing
Files: portfolio/agent_invocation.py

- After Layer 2 writes journal, extract XAG-USD entry
- Write data/fishing_context.json with direction_bias, rules, context
- Fish engine reads as additional strategic context

### Batch 4: Auto-disable + tests
Files: data/fish_engine.py, data/metals_loop.py, tests/test_fish_engine.py

- Auto-disable after 21:55 CET
- Tests for voting, Layer 2 vote weight, MC TP/SL, stale data, auto-disable
