# Signals-Core Review — subagent result (caveman:cavecrew-reviewer)

Totals: 6 P1 (🔴), 9 P2 (🟡)

## P1 / 🔴

1. **accuracy_stats.py:930** — Sample-count inflation: `total = max(at_samples, rc_samples)` but directional stats sum (line 967): `total_buy = at_v + rc_v`. Overall gate sees max(100,30)=100, directional gate sees 130. Asymmetric → signal passes overall, fails directional.
2. **signal_engine.py:3933** — `buy_conf = buy / active_voters` no ZeroDivisionError guard. Today gated by line 3929, but a refactor removing the gate ZeroDivisionErrors. Weighted consensus (2644) guards properly; simple consensus does not.
3. **ic_computation.py:128** — IC sign inversion asymmetry: `ic_sell = -sum(sell_returns)/n`, `ic_buy = sum(buy_returns)/n`. Both +1%/-1% give ic_buy=+1.0, ic_sell=+1.0. Sign convention broken if used directionally.
4. **accuracy_stats.py:951-967** — Directional merge picks max-sample source for buy/sell_accuracy, but `total_buy = at_v + rc_v` sums. Accuracy from one source, sample count from another — pairing violated.
5. **signal_engine.py:1475-2471** — `DISABLED_SIGNAL_OVERRIDES` applied at compute-time only; not re-checked at consensus-time accuracy gate. Rescued signal computed but gated → inconsistent "what voted" contract.
6. **accuracy_stats.py:1001** — Cache TTL uses wall `time.time()`. NTP/DST backward jump → stale cache appears fresh (elapsed=0 or negative).
7. **accuracy_stats.py:1000-1006** — Truncated JSON (partial write) may return partial dict that passes `cache.get("time_{horizon}")` TTL check with stale data. No completeness validation.

## P2 / 🟡

- signal_engine.py:1478 — `_TICKER_DISABLED_SIGNALS.get(ticker, ())` returns tuple default but value is frozenset. Works by accident; set-operations break.
- outcome_tracker.py:449-450 — `base_price <= 0` entries skip but still create `outcomes[ticker] = {h: None}` initialization → phantom entries with no usable data.
- signal_engine.py:2430 — `avg_trend_acc` divisor guard on truthiness, not length. Cosmetic clarity issue.
- signal_engine.py:927 — `_get_horizon_disabled_signals()` `.get(horizon, {})` silently returns empty on typo; hides config bugs.
- signal_engine.py:1593 — Dynamic correlation groups drop DISABLED_SIGNALS but not per-ticker disabled signals → wasted CPU.
- ic_computation.py:73 — `compute_signal_ic(horizon)` no validation; invalid horizons pollute per-horizon IC cache.
- signal_engine.py:2469 — `_get_ic_data(None)` guards correctly but `compute_and_cache_ic("unknown")` would persist bad horizon key, fail `load_cached_ic("1d")` match.
- signal_engine.py:4152-4156 — `post_persistence_voters` correct in MIN_VOTERS check, but `active_voters` at 3916 is pre-persistence → diagnostic inconsistent.
- signal_engine.py:2550-2551 — Cache safety pattern inconsistent (`.get()` vs subscript defaults); minor.
