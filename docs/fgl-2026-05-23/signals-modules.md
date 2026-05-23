# SIGNALS-MODULES adversarial review — 2026-05-23

Scope: `portfolio/signals/*.py` (63 plugin modules) + `portfolio/signal_registry.py`
(loader contract). Active 17 (per CLAUDE.md / `signal_engine.DISABLED_SIGNALS`)
ship every loop and got the deepest read; disabled (49) got contract / IO checks.

The 17 ACTIVE signals shipping in consensus right now (after intersecting
`signal_registry._register_defaults` with `tickers.DISABLED_SIGNALS`):

* In `signals/`: momentum, mean_reversion, momentum_factors, news_event,
  econ_calendar, crypto_macro, metals_cross_asset, cot_positioning,
  statistical_jump_regime, drift_regime_gate, crypto_evrp.
* Outside `signals/` (not in this scope): RSI, BB, Fear&Greed, Ministral-8B,
  Qwen3, On-Chain BTC, BTC Proxy.

Note: CLAUDE.md still lists `credit_spread_risk` as active, but
`tickers.DISABLED_SIGNALS` has it disabled since 2026-05-21 — CLAUDE.md is
stale.

---

## Per-module triage

| Module | Tag | Status | Notes |
|---|---|---|---|
| momentum | PURE | ACTIVE | OK. 8 sub-signals, all NaN-guarded, copy-on-input. |
| mean_reversion | PURE | ACTIVE | OK. Detrending fix from 2026-05-02 still correct (uses `original_close`). |
| momentum_factors | PURE | ACTIVE | Inherent BUY-bias in `_high_proximity` (BUY at 52w high, SELL only when 20% off); documented design choice. |
| news_event | IO+STATE | ACTIVE | Headlines via `_cached`+ `_fetch_*_headlines`; writes `headlines_latest.json` under lock. Negativity-biased (see P1.2). |
| econ_calendar | PURE | ACTIVE | OK. Confidence dampener for pause periods. |
| crypto_macro | IO | ACTIVE | OK. `fetch_json` w/ timeout+retries. Module-level `OPTIONS_TTL` defined AFTER function but resolved at call time — fine. |
| metals_cross_asset | IO | ACTIVE | OK. FRED + yfinance cross-asset. Pre-locked cache, `fetch_with_retry`. See P2.1 on relative paths. |
| cot_positioning | IO+FS | ACTIVE | OK. Loads precomputed `_deep_context`; CFTC fallback w/ `requests.get(timeout=15)`. Absolute paths fixed 2026-05-02. |
| statistical_jump_regime | PURE | ACTIVE | **P1 — regime transition bug** (P1.1). |
| drift_regime_gate | PURE | ACTIVE | Misattributes paper Sharpe (P3.1); math is OK. |
| crypto_evrp | IO | ACTIVE | Deribit DVOL fetch via http_retry. |
| credit_spread | IO | DISABLED 2026-05-21 | Latent relative-path bug L285 (P2.2). |
| finance_llama | ML | DISABLED+shadow | OK. Plex VRAM guard, `_abstain` on every failure path. |
| cryptotrader_lm | ML | DISABLED+shadow | OK. Crypto-only guard, same abstention pattern. |
| meta_trader | ML | DISABLED+shadow | Pure scaffold (`_FEATURE_AVAILABLE=False`). |
| connors_rsi2 | PURE | DISABLED+shadow | **P0 — broken ticker guard** (P0.1). |
| sentiment_extremity_gate | IO | DISABLED+shadow | **P1 — non-per-ticker F&G cache** (P1.3). |
| absorption_ratio_regime | IO | DISABLED+shadow | yfinance multi-ticker fetch, 1h cache. OK. |
| choppiness_regime_gate | PURE | DISABLED+shadow | OK. |
| adx_regime_switch | PURE | DISABLED+shadow | OK. |
| amihud_illiquidity_regime | PURE | DISABLED+shadow | OK. |
| ttm_squeeze, tsi_chop_mr, trend_slope_momentum, vwap_zscore_mr, gold_overnight_bias, cubic_trend_persistence, intraday_seasonality, williams_vix_fix, vol_ratio_regime, realized_skewness, hurst_regime, shannon_entropy | PURE | DISABLED | All OHLCV math, no IO, NaN-guarded. |
| metals_vrp, breakeven_inflation_momentum, treasury_risk_rotation, copper_gold_ratio, cross_asset_tsmom, gold_real_yield_paradox, vix_term_structure, mahalanobis_turbulence, complexity_gap_regime, xtrend_equity_spillover, ovx_metals_spillover, network_momentum, residual_pair_reversion, futures_basis | IO | DISABLED | FRED/yfinance through `http_retry`. Most have a `_Shim` fallback with bare `requests.get` if `http_retry` missing — timeouts still passed via kwargs, no retry. Acceptable. |
| hash_ribbons | IO | DISABLED | blockchain.info; bare `requests.get` fallback with timeout. |
| trend, volume_flow, volatility, candlestick, structure, fibonacci, smart_money, oscillators, heikin_ashi, calendar_seasonal, macro_regime | PURE | DISABLED | All historical signal failures captured in `tickers.DISABLED_SIGNALS` comments. |
| futures_flow, orderbook_flow, dxy_cross_asset, claude_fundamental, forecast | IO/ML | DISABLED | Disabled for accuracy reasons; not re-read for this review. |

---

## P0 — Critical

### P0.1 — `connors_rsi2` ticker guard is dead code; signal computes on every ticker

**File**: `portfolio/signals/connors_rsi2.py`
**Line**: 103-118 (function signature) + dispatcher contract.

`signal_registry.py` registers connors_rsi2 with `requires_context=True`,
so `signal_engine.py:3617-3618` calls
`compute_fn(df, context=context_data)`. But the function signature is:

```python
def compute_connors_rsi2_signal(df: pd.DataFrame, ticker: str = "", **kwargs)
```

There is no `context` parameter. The `context=context_data` kwarg gets
silently absorbed by `**kwargs`, and `ticker` keeps its default `""`.
Therefore the guard at line 110

```python
if ticker and not any(ticker.startswith(t.split("-")[0]) for t in _APPLICABLE_TICKERS):
```

short-circuits to False on every call (because `ticker == ""`), and the
RSI(2) crypto signal computes on XAU-USD, XAG-USD, MSTR, etc.

It's currently in `DISABLED_SIGNALS` (shadow mode, added 2026-05-19), so
it doesn't vote in consensus — but it DOES emit shadow predictions for
non-crypto tickers, which will pollute the per-ticker accuracy stats used
by the promotion pipeline. When the promotion review fires, it'll see
RSI(2) accuracy on XAU-USD (a meaningless 24/7 mean-reversion signal on
gold) and use that to decide whether to promote.

**Fix**: change signature to
`def compute_connors_rsi2_signal(df, context=None)` and pull `ticker =
(context or {}).get("ticker", "")`. Match the pattern in
`sentiment_extremity_gate.py:159`.

Confidence: **95**.

---

## P1 — Important

### P1.1 — `statistical_jump_regime._compute_regime_with_persistence` transition bug

**File**: `portfolio/signals/statistical_jump_regime.py`
**Lines**: 91-110.

In the `neutral` branch, `current_count` is incremented on EVERY non-zero
jump regardless of direction, then the regime is locked in based on the
direction of the LAST jump that hit `persistence_min`. So a sequence
like (+, –, –) locks in BEAR after 3 bars even though only 2 of those
3 jumps were down.

```python
elif current_regime == "neutral":
    if j > 0:
        opposing_count = 0
        current_count += 1
        if current_count >= persistence_min:
            current_regime = "bull"
            current_count = persistence_min
    elif j < 0:
        opposing_count = 0
        current_count += 1   # NO reset when direction flipped
        if current_count >= persistence_min:
            current_regime = "bear"
            current_count = persistence_min
```

The intended logic — "3 consecutive same-direction jumps locks in the
regime" — needs a sign tracker. Fix:

```python
elif current_regime == "neutral":
    if j > 0:
        if last_neutral_dir != 1:
            current_count = 1
        else:
            current_count += 1
        last_neutral_dir = 1
        if current_count >= persistence_min:
            current_regime = "bull"; current_count = persistence_min
    elif j < 0:
        if last_neutral_dir != -1:
            current_count = 1
        else:
            current_count += 1
        last_neutral_dir = -1
        ...
```

Confidence: **88**. Impact is bounded — once a regime locks in, the
opposing_count tracker handles direction changes correctly. But the
initial regime call is wrong, and the signal is the system's only
active regime detector after macro_regime was disabled
(`tickers.DISABLED_SIGNALS:128`).

### P1.2 — `news_event` is structurally negative-biased

**File**: `portfolio/signals/news_event.py`
**Lines**: 250-318 (`_sentiment_shift`), 222-247 (`_keyword_severity_vote`).

* `_keyword_severity_vote` only ever returns SELL or HOLD; there is no
  branch that returns BUY for positive keyword severity.
* `_sentiment_shift` defaults UNMATCHED moderate-severity headlines to
  `neg += 1` (line 303). Anything tagged as moderate severity that
  doesn't contain one of 7 positive keywords or a "rate cut"/"tax cut"
  phrase is counted as bearish — even neutral phrasing.

In combination, two of the six sub-signals are structurally tilted SELL.
Result: news_event will systematically over-fire SELL during news-heavy
periods (which already correlate with higher volatility, where price
direction is hardest to predict).

This may be intentional ("bad news drives markets" thesis) but it isn't
documented as such, and the existing accuracy tracking can't distinguish
"the signal is biased" from "the signal is wrong" without a parallel
BUY-capable detector.

Confidence: **80**. The bias is real; the question of whether to fix
depends on observed BUY/SELL hit-rate split (check
`accuracy_cache.json` for news_event direction-split accuracy).

### P1.3 — `sentiment_extremity_gate` F&G cache is not per-ticker

**File**: `portfolio/signals/sentiment_extremity_gate.py`
**Lines**: 34-51.

Module-level `_fg_cache = {"value": None, "ts": 0.0}` is shared across
all tickers. `get_fear_greed(ticker)` returns different values for crypto
vs stock (`fear_greed.py:174-177`), but the cache key doesn't include
ticker. First caller wins for 60s.

The signal is currently scoped to crypto in the docstring ("Crypto-only
... stock F&G is VIX-derived and does not exhibit the same extremity
premium"), but `compute_sentiment_extremity_gate_signal` has NO ticker
gate — it'll run on XAU/XAG/MSTR if called. When it does, the first
call this minute sets the cache and the others get a stale crypto F&G.

Currently DISABLED (shadow mode, 2026-05-20), so impact is shadow
predictions only — but accuracy tracking is corrupted by mismatched
ticker / F&G value pairs.

**Fix**: key the cache by ticker, or add an explicit ticker filter
matching the docstring intent. Pattern: copy
`crypto_macro.compute_crypto_macro_signal` lines 207-217 (explicit
non-crypto → HOLD).

Confidence: **82**.

### P1.4 — `news_event._persist_headlines` is a per-cycle file write on every active signal call

**File**: `portfolio/signals/news_event.py`
**Lines**: 55-106 (`_persist_headlines`) + 552 (call site).

Every active news_event signal call writes `data/headlines_latest.json`
atomically. With 5 tickers × N timeframes per ticker, that's many writes
per cycle, all serialized through a single lock. The headlines are
fetched and persisted regardless of whether they materially changed.
The file is consumed by the "fish monitor" but doesn't need cycle-rate
updates.

This is in the active signal hot path. With timeframe variants, this
likely runs 20-30 times per 60s cycle. Each call holds the lock for the
duration of the atomic-write. Other threads block on `_headlines_lock`.

**Fix**: persist only when the top-headline set actually changes
(content-hash gate), or gate on a 5-min TTL. Move out of the hot
compute path.

Confidence: **80**.

---

## P2 — Tracking

### P2.1 — `metals_cross_asset._get_fred_key` config traversal is fragile

**File**: `portfolio/signals/metals_cross_asset.py`
**Lines**: 91-102.

```python
return getattr(cfg, "fred_api_key", "") or getattr(
    getattr(cfg, "golddigger", None), "fred_api_key", ""
) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
```

The expression precedence here is `(A or B) if cond else ""`, which is
correct, but unreadable. Also `getattr(None, "fred_api_key", "")` works
(returns the default) so no crash, but the intent is obscured. The
identical `_get_fred_key` in `credit_spread.py:125` has the same shape.

Refactor for readability; same behavior. Confidence: **82**.

### P2.2 — `credit_spread` falls back to relative path `config.json`

**File**: `portfolio/signals/credit_spread.py`
**Line**: 285.

```python
cfg = load_json("config.json", default={}) or {}
```

If CWD is not the repo root (which is the symptom that broke
cot_positioning before the 2026-05-02 absolute-path migration — see
`cot_positioning.py:27-33`), this returns `{}` and the FRED key path
fails silently. The signal is currently disabled, so latent.

**Fix**: resolve via `Path(__file__).resolve().parent.parent.parent /
"config.json"` like `_DATA_DIR` in `cot_positioning.py`.

Confidence: **85**, but P2 since the signal is disabled.

---

## P3 — Notes

### P3.1 — `drift_regime_gate` repurposes the paper signal direction

The cited paper (arXiv:2511.12490) uses drift fraction as a GATE that
activates value/reversal signals during high-drift regimes, then claims
OOS Sharpe > 13 on the gated portfolio. This module uses drift fraction
DIRECTLY as a contrarian signal (high drift → SELL). That's a plausible
mean-reversion signal but the Sharpe number in the docstring does not
apply — it was measured on a different strategy.

Update the docstring to describe what the module actually does. The
current claim sets unrealistic expectations during accuracy review.

Confidence: **80**, docs-only.

### P3.2 — `momentum_factors._high_proximity` baked-in trend-following bias

`_high_proximity` returns BUY when price is within 5% of the 52w high
and SELL only when 20%+ below the high. There is no mirror branch on
the low side (that's `_low_reversal`). So at a 52w LOW with steady
movement up to within 5% of it, this sub-signal would still be SELL
(because ratio = close/high is below 0.80).

This is intentional momentum-factor design — the paper basis (Asness,
Moskowitz, Pedersen 2013) treats 52w high proximity as a directional
factor — but pair with a true symmetric signal for completeness. Not a
bug.

Confidence: **70**, design decision.

---

## Summary

| Severity | Count | Modules |
|---|---|---|
| P0 | 1 | connors_rsi2 |
| P1 | 4 | statistical_jump_regime, news_event ×2, sentiment_extremity_gate |
| P2 | 2 | metals_cross_asset, credit_spread |
| P3 | 2 | drift_regime_gate, momentum_factors |

The biggest functional risk in the active 17 is **P1.1
(statistical_jump_regime)** — it's the system's only active regime
detector after macro_regime was killed, and its regime-transition logic
allows mixed-direction jumps to lock in a regime based on the last jump
alone. **P1.2 (news_event negative bias)** is the second-biggest active
risk because news_event ships at `max_confidence=0.7` and could
systematically drag consensus toward SELL during noisy news cycles. The
P0 (connors_rsi2 ticker guard) is currently shadow-only but it corrupts
the accuracy data that the promotion pipeline reads — fix before next
promote cycle.
