# Cross-Critique — 6 signals-modules

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/signals/mahalanobis_turbulence.py:99` + `complexity_gap_regime.py:92` — `_cached` argument-order bug, P0.** Both reviewers identify the exact same TypeError-on-re-enable for both modules. The signature mismatch is `_cached(key, ttl, func, *args)` vs the call `_cached(key, func, ttl=...)`. Both note CLAUDE.md "DISABLED_SIGNALS" status protects production today but the moment ops re-enables, the ticker cycle explodes. **Independent rediscovery, very high confidence.** Action: fix call sites to `_cached("...", _CACHE_TTL, _do_fetch)`.

- **`portfolio/signals/copper_gold_ratio.py:248-265` — sub-signals dict recorded with PRE-inversion votes while action is POST-inversion, P0.** Both reviewers identify the exact lines. Codex extends with the second consequence: composite confidence inherited from un-inverted majority, so a 0.75-conviction BUY becomes a 0.75-conviction SELL on metals — semantically OK but the audit trail is misleading. **Strong cross-validation.** Action: also invert the sub_signals dict, or record both raw and effective.

- **`portfolio/signals/hurst_regime.py:283-302` — same vote double-counted, P1.** Both reviewers identify the exact lines. Codex extends to line 333 `votes = list(sub_signals.values())` — confirms majority pool counts the duplicate. **Both right.** Action: dedupe or rename one of the twins to a regime classifier voting HOLD.

- **`portfolio/signals/vwap_zscore_mr.py:124-125` — bare except returns HOLD with no logging, P1.** Both identify same lines. Codex explicitly ties to the 3-week Layer 2 outage shape. **Independent rediscovery.** Action: `logger.warning(..., exc_info=True)` + health metric.

- **`portfolio/signals/futures_flow.py` direct `[-1]["longShortRatio"]` access, P1.** Both flag. Codex lists exact line numbers (118, 135, 162, 287, 293-300). **Confidence high.** Action: `.get(..., 0.0)` or wrap compute in try/except with logging.

- **`portfolio/signals/williams_vix_fix.py` — 3 of 4 sub-indicators can only emit BUY/HOLD (P1).** Both flag the structural BUY bias. Codex extends with exact line range (~67-167) and SELL conditions (`low_count >= 8 AND rsi > 70` double-confirm). **Both right.** Action: document the asymmetry in per-direction accuracy, OR allow SELL on bear regimes.

- **`portfolio/signals/cot_positioning.py:213-217` — `commercial_change` is actually `-noncomm_net_change`, P1.** Both flag. Codex extends with `indicators["comm_net_change"] = -change` line. Same audit-trail mismatch. **Both right.** Action: rename to `speculator_change_inverse` or use real `comm_net_change` upstream if it exists.

- **`portfolio/signals/realized_skewness.py` — rolling window equals data length, std underestimated, z-scores inflated (P1).** Claude says "per subagent finding"; Codex computes the math (`window=80, min_periods=40, NORM_WINDOW=60`, overlapping adjacent rolling values share 79/80 underlying returns → std artificially small). **Codex's math sharper, both right.**

- **`portfolio/signals/gold_overnight_bias.py:118-140` — `_fix_proximity_vote` only votes BUY (P1).** Both flag both branches return BUY. Codex extends with a concrete scenario: London PM session SELL + proximity BUY → structural SELL weakened at fix boundaries. **Both right.**

## Codex found, Claude missed

- **`portfolio/signals/metals_cross_asset.py:220, 224` — "≥3 of 4 sources healthy" gate but the gated source still injects HOLD into 8-vote tally.** Repeated single-source outages won't trigger any alert. **Real P1.**

- **`portfolio/signals/structure.py:79-82` — `_highlow_breakout` votes BUY when current_close is within 2% of 52-week HIGH.** This is proximity, not breakout. At a 52w peak it flashes BUY perpetually until price drops 2%. Feeds Layer 2 near tops. **Real P2 — Claude missed.**

- **`portfolio/signals/forecast.py:914, 921` — `kronos_ok` and `chronos_ok` health booleans use different definitions.** Chronos passes empty-dict, missing-horizon-key silently into HOLD. Real P2.

- **`portfolio/signals/intraday_seasonality.py:110-118` — `_hour_alpha_vote` returns BUY for `mult >= 1.2` regardless of trend direction.** Time-of-day classifier mistakenly treated as directional vote in sub_signals dict. Real P2.

- **`portfolio/signals/calendar_seasonal.py` — 6/8 sub-signals BUY-only.** Same asymmetry as williams_vix_fix. `_MAX_CONFIDENCE = 0.6` partially mitigates. Real P2 — structural long-bias.

- **`portfolio/signals/copper_gold_ratio.py:43-44` — module-level `_CACHE` dict used without lock under ThreadPool with 8 workers.** Claude said "no race conditions found" — Codex catches one. Last-write-wins on dict set is atomic in CPython for one key, but the in-flight computation can race. Real P2.

- **`portfolio/signals/credit_spread.py:154` — z-score includes `current` in its own rolling mean/std.** 1/252 self-bias. Real P2.

- **`portfolio/signals/cubic_trend_persistence.py:142` — `phi_threshold` fires ~50% of time on trending asset, exhaustion vote dominates `trend_direction` → contrarian bias contradicting module's "weak trends persist" intent.** Math-grounded P2.

- **`portfolio/signals/metals_vrp.py:153-176` — `current_rv` and `current_gvz` come from independent sources, not date-aligned.** Off-by-one-day VRP flips sign near regime boundaries. Real P2.

- **`portfolio/signals/breakeven_inflation_momentum.py:60-63` + duplicates in `metals_vrp`, `metals_cross_asset`, `credit_spread` — convoluted config lookup pattern.** Code quality smell. Real P2.

- **`portfolio/signals/treasury_risk_rotation.py:187-188` — off-by-one-safe but obscure index math.** P2.

- **`portfolio/signals/futures_flow.py:54, 89, 196-197` — `if recent_oi[0]` treats 0 and None identically.** Real P2.

## Claude found, Codex missed

- **`portfolio/signals/news_event.py:46-49` — `_HEADLINES_PATH` derived via 3 `dirname` calls.** Path silently shifts if module moves. Real P2.

- **`portfolio/signals/news_event.py:65-77` — keyword substring matching: "raise" matches "raise prices" AND "raise concerns".** False positives. Real P2.

- **`portfolio/signals/copper_gold_ratio.py:254-255` — `confidence = min(confidence, 0.7)` arbitrary cap.** Real but P3-level.

- **`portfolio/signals/__init__.py` — no `__all__` allowlist on signal discovery.** A non-signal helper file gets auto-registered. Real P2.

## Disagreements

None substantive. Both reviewers concur on the P0s. Codex covers more disabled-signal modules in depth (mahalanobis, complexity_gap_regime are explicitly named) where Claude relies on "subagent findings" — Codex is the firmer audit trail.

## What BOTH missed (third pass)

- **`portfolio/signal_registry.py` discovery of new modules.** Neither reviewer audited what happens if `_cached` argument-order bug triggers at signal-discovery time vs at signal-compute time. If discovery imports the module and the bug fires there, the registry could be partially populated; downstream `signal_engine` then iterates a mismatched set.

- **Confidence calibration across modules.** Each module returns confidence ∈ [0, 1] but the engine `_weighted_consensus` weights them by accuracy not by confidence magnitude. Codex flags `confidence=0.7 cap` on copper_gold_ratio. Neither reviewer audited whether ALL the new modules respect the same scale — e.g. one module returning 0.95 means much more than another's 0.5, but engine treats them symmetrically.

- **DST handling consistency across the new signal modules.** Claude flagged `gold_overnight_bias`. Codex flagged `intraday_seasonality` UTC defaults. Neither reviewer cross-checked `metals_vrp`, `credit_spread`, `treasury_risk_rotation` for the same pattern — they probably have it too.

- **Memory growth of module-level caches.** Codex flagged unlocked `_CACHE` in copper_gold_ratio. Neither reviewer asked whether ANY module-level dict cache has eviction. With 5 tickers × 7 timeframes × N signals, a per-(ticker, tf) cache key accumulates indefinitely.

- **Disabled signals**: CLAUDE.md lists 19 force-HOLD signals "pending live validation". Neither reviewer ran a re-enable smoke test. With the `_cached` argument-order bug found, the priority of re-enablement is now uncertain — every disabled signal needs a smoke test before flip-on.

## Verdict

P0 list after cross: **3 confirmed** (mahalanobis `_cached`, complexity_gap `_cached`, copper_gold_ratio sub_signal/action mismatch).
P1 list after cross: **~9 confirmed** (hurst double-count, vwap bare except, futures_flow direct-key, williams_vix_fix BUY-only, cot mislabeled, realized_skewness window, gold_overnight_bias BUY-only proximity, metals_cross_asset gate-but-vote-HOLD, gold_overnight_bias DST).
P2 list after cross: ~15 (mostly Codex-found structural biases, asymmetries, and ordering bugs).

The signals-modules subsystem is the largest by file count (50 modules) and the **highest concentration of structural design issues**. Most don't bite production yet because of the DISABLED_SIGNALS gate, but every re-enablement is a latent failure surface.
