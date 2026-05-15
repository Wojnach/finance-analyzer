# Adversarial Review — 6 signals-modules (main-thread Claude, independent)

> 50 modules. Sampled 10 in depth (newer additions + selection from systemic
> grep patterns). Coverage notes: did not deeply read momentum, trend,
> structure, fibonacci, candlestick (older, well-tested core modules).

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/signals/mahalanobis_turbulence.py:99` and `portfolio/signals/complexity_gap_regime.py:92` — both invoke `_cached(key, fn, ttl=...)` but the function signature is `_cached(key, ttl, func, *args)`.
  ```python
  # mahalanobis_turbulence.py:99
  return _cached("mahalanobis_turb_closes", _do_fetch, ttl=_CACHE_TTL)
  # shared_state.py:37
  def _cached(key, ttl, func, *args):
  ```
  Calling with `(key, func, ttl=value)` passes `_do_fetch` as the positional `ttl` parameter, then passes ttl=... as kwarg — `TypeError: got multiple values` (or `func` value used as ttl). Per CLAUDE.md these signals are in DISABLED_SIGNALS so production is unaffected, but the moment ops re-enables them (post-validation), the function crashes the entire ticker cycle (no per-signal try/except in this module, exception propagates to the dispatch wrapper). Fix the call sites: `_cached("mahalanobis_turb_closes", _CACHE_TTL, _do_fetch)`.

- `portfolio/signals/copper_gold_ratio.py:248-252` — sub-signal direction recorded BEFORE the metals-inversion is applied.
  ```python
  action, confidence = majority_vote(votes, count_hold=False)
  # For metals tickers: INVERT the signal direction.
  if is_metals and action != "HOLD":
      action = "SELL" if action == "BUY" else "BUY"
  ...
  return {"action": action, "sub_signals": {"ratio_zscore": zscore_vote, ...}}
  ```
  The dict at line 260-265 stores `zscore_vote`, `trend_vote`, etc — which are the PRE-inversion votes. The aggregated `action` is the post-inversion vote. So on XAU/XAG, the recorded sub-signals contradict the recorded action: action=BUY but sub_signals all say SELL. Downstream consumers (forecast_accuracy backfill, audit dashboards, accuracy_degradation tracker) compute the wrong per-sub-signal accuracy because they think this signal voted SELL when it actually voted BUY. Either invert the sub_signals dict too when is_metals, or record both raw and metals-inverted forms.

## P1 — high-confidence bugs (should fix)

- `portfolio/signals/hurst_regime.py:284-285, 301-302` — double-counted vote in majority math.
  ```python
  sub_signals["hurst_regime"] = trend_vote
  sub_signals["trend_direction"] = trend_vote
  ...
  sub_signals["hurst_regime"] = mr_vote
  sub_signals["mr_extreme"] = mr_vote
  ```
  Two keys carry the same vote. If a downstream majority counts each key as one vote, hurst_regime has effectively double-weight against the other sub-signals (momentum, mr_extreme, trend_direction). For TRENDING regime: trend_vote × 2 + momentum + mr_extreme(HOLD) → 2/3 weight to trend. For MR regime: mr_vote × 2 → 2/3 weight to mean-reversion. This is a structural bias, not noise.

- `portfolio/signals/vwap_zscore_mr.py:124-125` — bare except returning HOLD with NO logging.
  ```python
  except Exception:
      return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
  ```
  Every failure (DataFrame schema change, integer overflow, missing volume column) is silently swallowed. This is one of the seven new signals added recently — if a single bug in the implementation makes it always fail, accuracy reports show 0 votes / 0 hits, but the signal counts as "passed gate" in registry health checks. Add `logger.warning("vwap_zscore_mr failed", exc_info=True)`.

- `portfolio/signals/gold_overnight_bias.py:35-41, 161` — DST blindness on fix times.
  ```python
  _AM_FIX_HOUR = 10
  _AM_FIX_MIN = 30
  ...
  _AM_FIX_MINUTES = _AM_FIX_HOUR * 60 + _AM_FIX_MIN   # 630
  ...
  utc_hour, utc_minute = _get_utc_time(df)
  minutes_of_day = utc_hour * 60 + utc_minute
  ```
  `minutes_of_day` is in UTC (line 51 explicitly converts via `astimezone(UTC)`). `_AM_FIX_MINUTES=630` is meant to be the London AM fix (10:30 *London* time). In British Summer Time, 10:30 London = 09:30 UTC = 570 minutes. In GMT, 10:30 London = 10:30 UTC = 630 minutes. The constant assumes GMT year-round, so the fix-proximity gate fires 60 min late half the year. P1 because the signal is sized only for XAU/XAG and is new (2026-05-x); silent miscalibration. Fix: use zoneinfo("Europe/London") and the fix time hour/min in that zone.

- `portfolio/signals/gold_overnight_bias.py:118-140` — `_fix_proximity_vote` is BUY-only.
  ```python
  if min_dist == dist_pm:
      return "BUY", 0.3 * proximity_strength
  else:
      return "BUY", 0.2 * proximity_strength
  ```
  Both branches return BUY. If aggregated with `majority_vote(..., count_hold=False)` and the other sub-signals are split, the proximity vote can break ties toward BUY systematically. Either also emit SELL when overnight bias is negative (e.g., during a documented bear regime), or rename to reflect "boost only".

- `portfolio/signals/futures_flow.py` — per subagent finding, direct `[-1]["longShortRatio"]` access on Binance payload without KeyError handling. Same module previously gated as crypto-only; if Binance changes the field name or returns `[]`, this crashes during a high-volatility window.

- `portfolio/signals/williams_vix_fix.py` — per subagent, 3 of 4 sub-indicators can only emit BUY/HOLD — structural BUY bias. Williams VIX Fix is fundamentally a contrarian-bottom signal; the asymmetry may be by design, but it should be documented in the per-signal accuracy gate (else SELL accuracy stays at NaN forever and the directional gate is effectively bypassed).

- `portfolio/signals/cot_positioning.py:213-217` — per subagent, sub-signal labeled `commercial_change` is actually `-noncomm_net_change`. Audit-trail mismatch. P1.

- `portfolio/signals/realized_skewness.py` — per subagent, window equals data length, std underestimated → z-scores inflated.

## P2 — concerns / smells (worth addressing)

- `portfolio/signals/news_event.py:46-49` — `_HEADLINES_PATH` derived via `os.path.dirname(...)` traversal. Three `dirname` calls = goes up three dirs from this module to reach `data/`. If the file is ever moved (e.g., into a deeper signals subfolder), the path silently shifts. Use `Path(__file__).resolve().parent.parent.parent / "data" / "headlines_latest.json"` pattern consistent with other modules.

- `portfolio/signals/news_event.py:65-77` — sentiment determined by keyword presence in `lower(title)`. The list at line 69-72 has "raise" which matches both "raise prices" (good for sellers) and "raise concerns" (bad). Substring matching of single tokens produces false positives. Use word-boundary regex.

- `portfolio/signals/gold_overnight_bias.py:108-109` — `if fast is None or slow is None or np.isnan(fast) or np.isnan(slow) or slow == 0: return "HOLD", 0.0`. Correct guard, but the function is called per-tick and constructs the entire EMA series each call — O(n) per call vs O(1) incremental update. Performance only.

- `portfolio/signals/copper_gold_ratio.py:254-255` — `confidence = min(confidence, 0.7)`. The 0.7 cap is arbitrary; per `accuracy_gate` at the engine level, signals below 47% are HOLD'd. A copper_gold signal at 49% accuracy returning 0.7 confidence after the engine cap looks "strong" but is statistically marginal. Document the cap rationale.

- `portfolio/signals/intraday_seasonality.py`, `portfolio/signals/treasury_risk_rotation.py`, `portfolio/signals/breakeven_inflation_momentum.py`, `portfolio/signals/cubic_trend_persistence.py`, `portfolio/signals/metals_vrp.py` — newly added, not deeply read this pass. Subagent flagged `cubic_trend_persistence` exhaustion threshold and `metals_vrp` date misalignment. Trust subagent on these.

- `portfolio/signals/__init__.py:1-4` — module docstring only; no `__all__`. Signal discovery in `signal_registry.py` likely uses `pkgutil.iter_modules`. Anyone adding a non-signal file under `portfolio/signals/` (e.g., a `_utils.py` helper) gets it auto-registered. Add explicit allowlist or naming convention check.

- `portfolio/signals/futures_basis.py`, `portfolio/signals/network_momentum.py`, `portfolio/signals/realized_skewness.py`, etc — all currently in DISABLED_SIGNALS pending live validation per CLAUDE.md. Per the `_cached` ttl P0 above, verify each disabled signal compiles and runs in a smoke test before any re-enable.

## Did NOT find

1. **Silent failures**: see P1 (vwap_zscore_mr bare except).
2. **Race conditions**: signal modules are stateless functions; no shared mutable state observed.
3. **Money-losing bugs**: the metals-inversion mismatch (P0) feeds wrong accuracy data into Kelly sizing, indirect $.
4. **State corruption**: news_event uses atomic_write_json for headlines_latest.json.
5. **Logic errors that pass tests**: hurst double-count (P1) — tests likely check sub_signals dict keys but not vote-weight invariance.
6. **Resource leaks**: no subprocess / file handles in signal modules I read.
7. **Time/timezone bugs**: P1 — gold_overnight_bias DST.
8. **API misuse**: news_event uses internal NewsAPI wrapper; Binance interval not hit in sampled modules.
9. **Trust boundary violations**: ticker strings flow into log statements only; no eval/exec.
10. **Incorrect partial-state assumptions**: copper_gold_ratio (P0) inverted-vote mismatch — assumes downstream reads sub_signals dict as raw, but they're consumed as effective votes.
