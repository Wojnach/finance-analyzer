# Signals-Modules Adversarial Review

Scope: 16 signal plugins in `portfolio/signals/` (diff `fgl-baseline..HEAD`).
Contract verified against `portfolio/signal_registry.py` (lazy load, 300s import
cooldown) and `portfolio/signal_engine.py` (`_validate_signal_result` normalizes
bad dicts to HOLD; a *raised exception* is caught → force-HOLD + warning + counts
toward `_signal_failures`; a *hang* blows the per-ticker pool timeout / BUG-178).

## Counts
- P0 (crash-the-voter / look-ahead bias / inverted direction): 0
- P1 (wrong under realistic conditions): 3
- P2 (latent): 5
- P3 (minor): 4

All 16 modules return the correct canonical HOLD shape on empty/missing/short
data; none invert the oversold=BUY / overbought=SELL convention; none normalize
against a window that includes future bars. The serious findings are about
silent dead-signaling (data never aligns / wrong asset class) and a known
LLM-hang-vs-pool-timeout interaction.

---

## P1 — wrong under realistic conditions

portfolio/signals/sentiment_extremity_gate.py:144 P1 wrong-asset-class: module
is documented and registered as crypto-only (alt.me F&G is crypto-specific) but
`compute_sentiment_extremity_gate_signal` has NO ticker guard. It will run on
XAU/XAG/MSTR, fetch the crypto F&G index, and emit BUY/SELL on non-crypto assets
using a sentiment series that does not apply to them. fix: at top of compute,
`ticker = (context or {}).get("ticker"); if ticker not in {"BTC-USD","ETH-USD"}: return HOLD-shape`.

portfolio/signals/cryptotrader_lm.py:139 P1 hang-vs-pool-timeout:
`query_llama_server` blocks on `_acquire_file_lock(timeout=300)` + HTTP
`timeout=240` (llama_server.py:560,605). On the cycle this shadow LLM actually
runs (cycle_modulo throttle), a cold model swap can occupy a worker for up to
~240s, exceeding the T1 180s ticker-pool budget and tripping the BUG-178 silent-
hang handler. It does NOT hang forever (timeouts exist) and degrades to HOLD on
timeout, but it can starve the cycle. This module is still under an OVERDUE 72h
shadow verification. fix: pass a short, signal-local timeout into
`query_llama_server` (e.g. 45-60s) so an inference miss abstains well within the
pool budget rather than relying on the 240s server default; verify the shadow
throttle is actually gating it (confirm `shadow_registry` status=='shadow' +
cycle_modulo set, else the fail-closed list `_KNOWN_SHADOW_LLMS` must contain
`cryptotrader_lm`).

portfolio/signals/btc_gold_correlation_regime.py:93 P1 dead-signal-on-intraday:
`merged = pd.DataFrame({"target": df.close, "counter": counter_close}).dropna()`
index-joins the engine's per-ticker `df` (frequently intraday: 5m/1h) against
counterpart klines fetched at `interval="1d"` (line 53). Daily and intraday
indices share almost no timestamps → `dropna()` empties the frame →
`len(merged) < _MIN_ROWS` (282) → permanent HOLD on every non-daily timeframe.
The signal silently never fires except on the daily TF. fix: resample `df.close`
to daily before merging, or fetch counterpart at the same interval as `df`;
assert overlap count and log when it is near zero.

---

## P2 — latent

portfolio/signals/crypto_evrp.py:204 P2 mislabeled-subsignal:
`_evrp_percentile_signal` is documented to rank *eVRP* percentile and even
computes `rv_hist` (line 216) but then ranks pure *DVOL* percentile (line 236,
`dvol_vals`/`current_dvol`); `rv_hist` is dead code. The vote (low IV → BUY) is
defensible but the indicator name `evrp_percentile` misrepresents what is
measured, and it partially contradicts `_evrp_momentum_signal` (falling DVOL →
BUY) at the extremes. fix: either rank an actual eVRP series or rename to
`dvol_percentile`; delete dead `rv_hist`.

portfolio/signals/crypto_evrp.py:134 P2 slow-path-blocking:
`_fetch_dvol_history` loops up to 5 chunks, each `fetch_with_retry(timeout=15,
retries=2)` + `time.sleep(0.2)` → worst case tens of seconds inside the voter on
a cold cache. Caught/cached, but on the first call after a restart it can
contribute to a slow cycle. fix: cap total wall time / reduce chunk count /
lower per-chunk timeout.

portfolio/signals/cryptotrader_lm.py:150 P2 unvalidated-decision-passthrough:
`decision` from `_parse_response` is returned as `action` without local
normalization; relies entirely on `_validate_signal_result` to coerce a bad
string to HOLD. Works today, but if the engine ever calls this fn outside the
validated path the raw value leaks. fix: clamp `decision` to
{"BUY","SELL","HOLD"} before return (cheap defense in depth).

portfolio/signals/metals_cross_asset.py:309 P2 zero-vs-missing-conflation: oil
fallback triggers only when `ctx["oil_change_pct"] == 0.0`, but a genuine flat
oil move (0.0%) is indistinguishable from a missing fetch, so a real 0% reading
spuriously pulls in stale macro daily data. fix: have `_get_cross_asset_context`
return None (not 0.0) for missing sources and branch on `is None`.

portfolio/signals/statistical_jump_regime.py:240 P2 confidence-floor-on-HOLD:
when `action=="HOLD"`, `raw_confidence` from `majority_vote` is already 0.0 so
the `0.5 + ...` multiplier yields 0 — fine. But when action is directional with a
single voter, `confidence = 1.0 * (0.5 + ...)` can reach ~0.95 off one jump-based
voter agreeing, richer than the evidence warrants. Latent over-confidence, capped
by the 0.7 max_confidence default? No — this signal has NO max_confidence cap in
the registry (line 158-159), so it can emit up to 1.0. fix: register with
`max_confidence=0.7` like its peers, or damp the directional confidence.

---

## P3 — minor

portfolio/signals/credit_spread.py:285 P3 relative-path-fallback:
`load_json("config.json", ...)` uses a relative path — the exact SM-P1-4 CWD bug
fixed elsewhere (cot_positioning.py). Only a fallback after context config, and
degrades to HOLD, so low impact. fix: resolve repo-root absolute path.

portfolio/signals/sentiment_extremity_gate.py:40 P3 cross-ticker-cache-bleed:
`_get_fg_value` returns the module-level cached F&G before considering the
`ticker` arg, so the first ticker's value is reused for all tickers. Harmless
for the crypto-wide F&G index, but the signature implies per-ticker behavior it
does not honor. fix: key the cache by ticker, or drop the ticker param.

portfolio/signals/crypto_evrp.py:264 P3 missing-asset-class-strictness: ticker
inference via substring (`"BTC" in ticker.upper()`) would match e.g. a future
"BTC-PERP" or odd symbol and route to BTC DVOL. Low risk given Tier-1 set. fix:
restrict to the explicit `_TICKER_TO_CURRENCY` map only.

portfolio/signals/realized_skewness.py:56 P3 scipy-empty-warning: several
`stats.skew(...)`/`stats.kurtosis(...)` calls on potentially tiny dropna'd arrays
emit RuntimeWarnings (not exceptions) and return nan, handled downstream. Noise,
not a bug. fix: guard `len(window) >= N` before the scipy call (already done in
most sub-signals; `_sub_skew_zscore` line 56 lacks the pre-check but isnan-guards
the result).

---

## Verified clean (no findings at >=P3 threshold worth listing)
mean_reversion.py (detrend compounding already fixed P1-6; gap-fill inversion
explicitly guarded), momentum.py (all 8 sub-indicators guarded, denom
zero-guarded, directions correct), oscillators.py (Aroon recency math correct,
twin-peaks uses dropna()[-2], no leakage), williams_vix_fix.py (WVF spike=BUY /
complacency=SELL correct), econ_calendar.py (every sub-signal try-wrapped,
post-event-relief restores BUY capability), cot_positioning.py (newest-first
ordering correct, percentile self-inclusion standard, SOCRATA URL uses fixed
internal commodity names), news_event.py (keyword-only, no price math, all
sub-signals wrapped, "cut" phrase polarity correctly disambiguated),
credit_spread.py (asset-class direction split correct, FRED newest-first).
