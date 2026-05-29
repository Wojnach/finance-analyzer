# Signals-Modules Adversarial Review — 2026-05-29 (fgl)

Subsystem: individual signal modules + LLM signals feeding the voting engine.
Reviewed at HEAD. 25 modules read in full (3 LLM wrappers live in `portfolio/`,
not `portfolio/signals/`: `ml_signal.py`, `ministral_signal.py`, `qwen3_signal.py`).

## Counts
- P0: 0
- P1: 3
- P2: 8
- P3: 5

No finding rises to P0 (no module silently injects a *wrong-direction* vote into
live trading or crashes the loop). The P1s are correctness/robustness issues that
can degrade or distort votes; all error paths I traced default to HOLD/conf=0,
which is the safe abstention.

---

## P1 — incorrect signal math / fake-vote risk

- portfolio/signals/amihud_illiquidity_regime.py:107-112: P1: the "volume confirmation"
  sub-signal votes **directionally** (RVOL>1.3 → BUY, RVOL<0.6 → SELL) inside what
  is supposed to be a *liquidity regime* signal. High relative volume is not
  inherently bullish (it accompanies both rallies and crashes); this injects a
  directional vote with no directional basis and is 1 of only 3 voters, so it can
  flip the composite. → Make volume a gate/confirmation multiplier (suppress to
  HOLD when RVOL<0.5) or drop the directional mapping; do not vote BUY/SELL on raw
  RVOL.

- portfolio/ml_signal.py:124-126: P1: `if r is None: return None` then
  `r.raise_for_status()` — when Binance returns a non-None error response (429/5xx),
  `raise_for_status()` raises an uncaught `HTTPError` out of `get_ml_signal` instead
  of returning the safe `None`. Whether this becomes a fake vote depends on the
  caller's try/except; within this module the contract ("return None on failure") is
  violated. → Wrap the fetch+`raise_for_status` in try/except and return None on any
  HTTP error. Also no NaN check before `model.predict(last_row)` (line 154): on a
  short/gappy klines payload a NaN feature row would feed sklearn and produce a
  garbage BUY/SELL — add `if np.isnan(last_row).any(): return None`.

- portfolio/signals/finance_llama.py:204-214 (and cryptotrader_lm.py:150-158): P1:
  `ministral_trader._parse_response` falls back to `re.search(r"\b(BUY|SELL|HOLD)\b", text)`
  on the *whole* completion and takes the **first** match. For a malformed/echoed
  completion the first directional token in the text becomes the vote, and confidence
  is then set to 0.50 (not 0.0) — a non-HOLD directional vote derived from noise rather
  than a real classification. The few-shot prompt itself begins with "BUY" example
  text, so any prompt-echo regression would systematically bias to BUY. → On regex
  fallback (no JSON action recovered) return HOLD/conf=0 rather than the first stray
  token; only trust a decision parsed from the structured JSON / explicit
  `decision:` field.

---

## P2 — robustness / regime-gate hygiene

- portfolio/signals/choppiness_regime_gate.py:126-129: P2: a *gate* emits directional
  votes in the "neutral" CHOP band (38.2–61.8) at 0.7× confidence. The module's stated
  job is to suppress in choppy regimes; voting BUY/SELL in the undefined middle band
  is scope creep. → In NEUTRAL regime, return HOLD (suppress) rather than a discounted
  directional vote.

- portfolio/signals/choppiness_regime_gate.py:123-125: P2: when TREND regime but
  composite vote is HOLD, it forces `action = direction` (raw price-vs-SMA) at a
  hardcoded 0.35. This overrides the abstention with a single-indicator directional
  call. → Keep HOLD when the composite abstains; do not synthesize a directional vote.

- portfolio/signals/econ_calendar.py:128,136,141: P2: `_post_event_relief` calls
  `recent_high_impact_events(24)` / `next_event(...)` using real `datetime.now()` for
  the hours computation while the rest of the module derives `ref_date` from the
  DataFrame's last bar. In live mode this is fine, but in any replay/backtest it is a
  lookahead/clock mismatch (relief window computed against wall clock, not bar time).
  → Thread `ref_time`/`ref_date` consistently into all event lookups.

- portfolio/signals/crypto_macro.py:226-231 & 281: P2: `OPTIONS_TTL` is referenced
  inside `compute_crypto_macro_signal` (line 227) but defined at module bottom
  (line 281). It resolves at call time so it works today, but it is fragile — any
  import-time call or refactor that moves the call above the def, or a tooling check,
  trips a `NameError`. → Move the `OPTIONS_TTL = 900` constant to the top with the
  other thresholds.

- portfolio/signals/metals_cross_asset.py:91-102: P2: `_get_fred_key` has a convoluted
  ternary on the non-dict config branch (`getattr(...) if hasattr(...) else ...`) that
  is hard to verify and can return a bound-but-empty attribute. The dict branch is the
  live path; the object branch is effectively dead/untested. → Simplify to explicit
  `if/elif` and unit-test the object-config path or remove it.

- portfolio/signals/crypto_evrp.py:204-242: P2: `_evrp_percentile_signal` is documented
  as ranking the eVRP percentile but actually ranks **DVOL** alone ("Use just the DVOL
  percentile as proxy"). The `current_evrp` and `rv_series` args are computed and then
  largely discarded (rv_hist is computed, length-checked, never used in the final
  percentile). The vote is therefore an IV-level percentile, not an eVRP percentile —
  semantics diverge from the name/docstring and from sub-signal 1. → Either rank the
  actual eVRP series or rename the sub-signal to `dvol_percentile` so the composite
  isn't double-counting DVOL level (sub-1 already keys off the eVRP level).

- portfolio/signals/news_event.py:610-612: P2: thesis_alignment vote is appended to the
  ballot only when enabled, but the six base sub-signals share heavy keyword overlap
  (headline_velocity, keyword_severity, sentiment_shift, dissemination all derive from
  the same `keyword_severity`/positive-keyword sets). A single critical headline can
  drive 4 correlated SELL votes → inflated consensus from one underlying fact. The 0.7
  cap mitigates magnitude but not the false-consensus count. → Consider collapsing the
  keyword-derived sub-signals into one weighted vote, or document the intended
  correlation.

- portfolio/signals/statistical_jump_regime.py:53-69: P2: `_classify_vol_regime` builds
  a rolling percentile via `.rolling(...).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])`
  which is O(n·window) python-callback per cycle on every ticker/timeframe — a known
  perf footgun in the 60s loop. Functionally OK, but on long series it adds latency. →
  Vectorize with `vol.rolling(window).rank(pct=True)` (pandas-native) or precompute.

---

## P3 — nits

- portfolio/signals/connors_rsi2.py:111-113: P3: ticker guard uses
  `ticker.startswith(t.split("-")[0])`, so "BTCFOO" or "ETHX" would pass as crypto.
  Harmless given the fixed Tier-1 universe, but prefer exact membership in
  `_APPLICABLE_TICKERS`.

- portfolio/signals/momentum.py:356-359 (and throughout): P3: per-sub-indicator
  `except Exception: HOLD` swallows everything silently with no debug log (unlike
  mean_reversion.py which logs `exc_info=True`). A persistently broken sub-indicator
  is invisible. → Add `logger.debug(..., exc_info=True)` per branch as mean_reversion
  does.

- portfolio/signals/crypto_macro.py:156: P3: `_exchange_netflow_signal` returns SELL on
  `distribution` only when `consecutive_neg == 0`; a `distribution` trend with a stray
  1-day negative netflow (`consecutive_neg == 1`) silently becomes HOLD. Likely
  intended, but the asymmetry vs the BUY branch (which accepts `>=3`) is undocumented.

- portfolio/signals/vol_ratio_regime.py:256-258: P3: the reported `*_regime` sub-signal
  labels recompute thresholds inline (`"ranging" if gk_cc>2.0 ...`) duplicating
  `_classify_regime`; risk of drift if one set of thresholds changes. → Derive labels
  from the same constants.

- portfolio/signals/realized_skewness.py:55-61: P3: `_sub_skew_zscore` computes a single
  `stats.skew` over the lookback AND a separate `_compute_rolling_skewness` over the
  same lookback for the mean/std, doubling the heaviest computation. Cache/share the
  rolling series across sub-signals 1, 2, 4 (all call `_compute_rolling_skewness`).

---

## Verified-correct (checked, not bugs)

- btc_gold_correlation_regime.py inversion math for XAU (invert→ BUY when z>1.5,
  SELL when z<-2.0) matches docstring — correct.
- Williams VIX Fix direction (spike→BUY capitulation, complacency+RSI>70→SELL) — correct.
- COT / credit_spread contrarian + safe-haven sign conventions — correct.
- All `compute_*` top-level error/empty-data paths return HOLD/conf=0 (no default
  directional vote on missing data). LLM wrappers (`_abstain`, GPU-gate timeout,
  Plex-VRAM skip, batch failure) all return HOLD — safe.
- mean_reversion gap-fill negative-fill_pct guard (line 229) correctly blocks
  gap-widening false BUYs.
- Regime gates that should suppress (choppiness HOLD-on-choppy, drift, vol_ratio
  uncertain→HOLD, sentiment_extremity extreme→HOLD) do return HOLD in their
  suppression branches.
