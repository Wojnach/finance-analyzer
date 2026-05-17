# Adversarial Review — Signals Modules Subsystem

**Scope:** `portfolio/signals/` (58 plugin modules; 10 active prioritized, ~48 disabled spot-checked) + cross-reference to `portfolio/signal_registry.py` and `portfolio/tickers.py` (DISABLED_SIGNALS gate).
**Reviewer focus:** active-signal correctness (look-ahead, divide-by-zero, side-effects, sticky state) and disabled-signal leakage.
**Convention assumed:** signals return a dict with `action`/`confidence`/`sub_signals`/`indicators`; never raise; never None; never perform I/O side-effects.

Files materially inspected:
- Q:/finance-analyzer/portfolio/signals/momentum.py
- Q:/finance-analyzer/portfolio/signals/mean_reversion.py
- Q:/finance-analyzer/portfolio/signals/momentum_factors.py
- Q:/finance-analyzer/portfolio/signals/news_event.py
- Q:/finance-analyzer/portfolio/signals/econ_calendar.py
- Q:/finance-analyzer/portfolio/signals/crypto_macro.py
- Q:/finance-analyzer/portfolio/signals/metals_cross_asset.py
- Q:/finance-analyzer/portfolio/signals/cot_positioning.py
- Q:/finance-analyzer/portfolio/signals/credit_spread.py
- Q:/finance-analyzer/portfolio/signals/statistical_jump_regime.py
- Q:/finance-analyzer/portfolio/signal_registry.py
- Q:/finance-analyzer/portfolio/tickers.py (DISABLED_SIGNALS, SIGNAL_NAMES)
- Spot-checked: forecast.py, claude_fundamental.py, trend.py, dxy_cross_asset.py,
  realized_skewness.py, orderbook_flow.py, calendar_seasonal.py, ovx_metals_spillover.py,
  finance_llama.py, macro_regime.py, volatility.py

---

## P0 — Critical (active signal, materially affects votes)

### P0-1 — `momentum_factors._apply_seasonality` compounds detrending geometrically (un-fixed copy of the 2026-05-02 mean_reversion bug)
File: `Q:/finance-analyzer/portfolio/signals/momentum_factors.py:350-358`

```python
for i in range(1, len(detrended)):
    hour = detrended.index[i].hour
    raw_ret = returns.iloc[i]
    if np.isfinite(raw_ret):
        adj_ret = detrend_return(raw_ret, hour, profile)
        # Reconstruct close from detrended returns
        detrended.iloc[i, detrended.columns.get_loc("close")] = (
            detrended["close"].iloc[i - 1] * (1 + adj_ret)
        )
```

This is the exact pattern fixed in `mean_reversion.py` on 2026-05-02 (P1-6 follow-up). The sister
fix in `mean_reversion.py:462-487` (commented "P1-6 (2026-05-02 adversarial follow-ups): detrending
must NOT compound across iterations. Capture the ORIGINAL close column BEFORE the loop and
reconstruct each detrended bar from its ORIGINAL i-1 value, not from the just-modified df.") was
NEVER propagated to `momentum_factors._apply_seasonality`. The loop above reads `detrended["close"].iloc[i-1]`,
which is the just-detrended close from the previous iteration — exactly the bug the sister comment
calls out. On a 100-bar flat metals series with a uniform -0.001 hour-of-day detrend, this drifts
close ~10% instead of staying at -0.10%. Every momentum sub-indicator (`_time_series_momentum`,
`_roc_20`, `_high_proximity`, `_low_reversal`, `_price_acceleration`) consumes this drifted price
and will vote on the artificial trend rather than the genuine residual.

Affects only XAU-USD / XAG-USD (the only tickers with `seasonality_profile` set in
`signal_engine.py:3500-3505`) but those are the user's stated primary focus.

Fix: mirror the mean_reversion pattern — `original_close = df["close"].astype(float).copy()` before
the loop, compute `returns = original_close.pct_change()`, and inside the loop write
`original_close.iloc[i-1] * (1 + adj_ret)` instead of `detrended["close"].iloc[i-1] * (1 + adj_ret)`.

---

## P1 — Important

### P1-1 — `news_event` performs unconditional disk I/O (`atomic_write_json`) on every call from every ticker, into a single shared filename — last-write-wins
File: `Q:/finance-analyzer/portfolio/signals/news_event.py:46-49, 91-96, 544`

```python
_HEADLINES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "headlines_latest.json",
)
...
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "ticker": ticker,
            "headlines": top,
        }
        atomic_write_json(_HEADLINES_PATH, payload)
...
    # Persist top scored headlines for fish monitor
    _persist_headlines(ticker, headlines)
```

Two problems:

1. **Signal purity violation.** Stated convention: "signals should be PURE, no I/O." `compute_news_event_signal`
   unconditionally writes a file on every invocation, on every cycle, for every ticker.
2. **Concurrent overwrite race.** `signal_engine` runs tickers through an 8-worker
   `ThreadPoolExecutor`. The filename has no ticker key, so the ticker that happens to write last
   wins. Each cycle, the file flips between e.g. BTC / ETH / XAU / XAG / MSTR headlines depending
   on thread scheduling. The downstream consumer (fish monitor) sees ~1 in 5 cycles of useful data
   for any given ticker, and the `ticker` field inside the JSON is the only way to know what you
   got. `atomic_write_json` is atomic for the single replace, but the last replace clobbers prior
   ones.

Fix: either (a) move persistence to the caller in `signal_engine`, or (b) use a per-ticker file
`headlines_latest_{ticker}.json`. Also gate persistence behind a feature flag so the signal can be
called purely (e.g. for backtests) without side effects.

### P1-2 — `credit_spread._oas_cache` is read/written from worker threads without a lock
File: `Q:/finance-analyzer/portfolio/signals/credit_spread.py:53, 62-71, 112-115`

```python
_oas_cache: dict = {}

def _fetch_hy_oas(fred_api_key: str) -> list[float] | None:
    now = time.time()
    if (
        _oas_cache.get("key") == fred_api_key
        and _oas_cache.get("data")
        and now - _oas_cache.get("time", 0) < _CACHE_TTL
    ):
        return _oas_cache["data"]
    ...
        if values:
            _oas_cache["key"] = fred_api_key
            _oas_cache["data"] = values
            _oas_cache["time"] = now
```

Module-level dict accessed from the `ThreadPoolExecutor` (8 workers × N tickers). The
check-then-update pattern is not atomic. The sister module `metals_cross_asset.py` uses
`_fred_cache_lock` (`portfolio/signals/metals_cross_asset.py:88`) around the equivalent block —
`credit_spread.py` does not. Likely outcome on first cold-call: 2–4 redundant FRED fetches
in the same second (mild API quota burn, not a correctness issue at steady state given the
4 h TTL). On Windows with the GIL the dict mutations don't corrupt the dict, but the check
in lines 62–66 can race against the writes in lines 113–115 to return a stale `[]` or `None`
intermittently. Add a `threading.Lock()` around both blocks — match the metals_cross_asset
pattern.

### P1-3 — `credit_spread` uses relative `"config.json"` for fallback FRED key — silently fails when loop CWD is not the repo root (same root-cause as the SM-P1-4 cot_positioning fix)
File: `Q:/finance-analyzer/portfolio/signals/credit_spread.py:283-288`

```python
    if not fred_key:
        try:
            cfg = load_json("config.json", default={}) or {}
            fred_key = cfg.get("golddigger", {}).get("fred_api_key", "") or ""
        except Exception:
            logger.debug("config.json fallback read failed", exc_info=True)
```

`cot_positioning.py` already documents and fixes this class of bug (lines 27–32):
"SM-P1-4 (2026-05-02 adversarial follow-ups): absolute path resolution. The previous code used
relative `Path("data")` / `data/...` which silently broke when the scheduled task CWD differed
from the repo root (e.g. PF-DataLoop launched from C:\\Windows)." `credit_spread` was not
updated. Effect: when `_get_fred_key(context)` returns "" (config came in as something other
than the `golddigger`-wrapped dict the helper expects) the fallback reads `config.json` from
whatever CWD the loop started in — silently returns `{}` if missing — and the signal degrades
to permanent `empty` HOLD. Fix: use the `_DATA_DIR`/`Path(__file__).resolve().parent...`
absolute-path pattern as in `cot_positioning.py:33`.

### P1-4 — `_get_fred_key` fallback path in both `credit_spread.py:134-136` and `metals_cross_asset.py:100-102` has a subtle precedence bug
File: `Q:/finance-analyzer/portfolio/signals/credit_spread.py:134-136`
       `Q:/finance-analyzer/portfolio/signals/metals_cross_asset.py:100-102`

```python
    return getattr(cfg, "fred_api_key", "") or getattr(
        getattr(cfg, "golddigger", None), "fred_api_key", ""
    ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""
```

Python parses this as:
`(getattr(cfg, "fred_api_key", "") or getattr(getattr(cfg, "golddigger", None), "fred_api_key", "")) if (...) else ""`

When `cfg` is an object without `fred_api_key` or `golddigger` attributes, returns `""` — OK.
But when `cfg` has only `golddigger` (the documented attribute path), the first `getattr(cfg,
"fred_api_key", "")` returns `""` (falsy), then `getattr(getattr(cfg, "golddigger", None),
"fred_api_key", "")` runs `getattr(None_or_obj, "fred_api_key", "")`. If `cfg.golddigger` is
`None` (placeholder), this returns `""` silently — fine. If `cfg.golddigger` is an object
without the attribute — also `""` silently. Net effect: signal becomes a no-op without any
log. The dict-path on lines 132-133 already handles this case correctly. The object-path is
duplicated, hard to read, and likely undertested. Reduce to one path or add an explicit `else`
branch logging when no key is reachable.

### P1-5 — `cot_positioning._compute_cot_index` includes the current value in the historical min/max, so the percentile is biased upward by construction
File: `Q:/finance-analyzer/portfolio/signals/cot_positioning.py:138-155, 173-179`

```python
    # _sub_cot_index
    nc_net_history = [nc_net]
    for h in historical:
        val = h.get("nc_net")
        if val is not None:
            nc_net_history.append(val)

    cot_index = _compute_cot_index(nc_net_history)
    ...
def _compute_cot_index(nc_net_history: list[int]) -> float | None:
    current = nc_net_history[0]
    hist_min = min(nc_net_history)
    hist_max = max(nc_net_history)
```

`current` is `nc_net_history[0]` which is also the value compared against in `min/max`. If the
current week is a new 156-week high, `cot_index = 100` exactly. Standard CFTC COT-index
construction compares CURRENT to PAST history only. The bias is small for a 156-element series
but biases the contrarian SELL trigger (>80) high. Fix: `_compute_cot_index(history_past)` where
the caller passes only the historical tail without the current value, then computes
`(current - min(past)) / (max(past) - min(past))`. P1 because it's a contrarian signal where
percentile thresholds drive the vote.

### P1-6 — `econ_calendar._post_event_relief` calls `next_event` twice on the same `ref_date` and treats them as independent
File: `Q:/finance-analyzer/portfolio/signals/econ_calendar.py:114-147`

```python
    relief_events = [e for e in recent if e.get("hours_since", 0) >= 4]

    if relief_events:
        indicators["post_event_relief"] = True
        ...
        evt = next_event(ref_date.date() if ...)
        if evt is None or evt["hours_until"] > 24:
            return "BUY", indicators

    # Event-free calm window: next event >72h away
    evt = next_event(ref_date.date() if ...)
    if evt is not None and evt["hours_until"] > 72:
        ...
        return "BUY", indicators
```

Two `next_event(...)` calls with the same input — second call is reached only when the first
branch returned without `return "BUY"`. Functionally correct because the function is referentially
transparent for a fixed `ref_date`, but the second call shadows `evt` from the first branch even
when relief_events was empty. The pattern reads as if a stale `evt` could be reused. Memoize
`evt = next_event(...)` once at the top of the function and reuse — current shape is a P3 in
isolation, but worth flagging at P1 because it sits next to the BUG-218 fix this module
specifically exists to support and an accidental swap would re-introduce the SELL-only bias.

---

## P2 — Worth fixing

### P2-1 — `crypto_macro.compute_crypto_macro_signal` references `OPTIONS_TTL` before its module-level definition; works only by virtue of late binding
File: `Q:/finance-analyzer/portfolio/signals/crypto_macro.py:226-231, 280-281`

```python
    macro_data = _cached(
        f"crypto_macro_{ticker}",
        OPTIONS_TTL,
        get_crypto_macro_data,
        ticker,
    )
...
# Cache TTL imported from data module
OPTIONS_TTL = 900
```

The comment "Cache TTL imported from data module" is false — there is no import; `OPTIONS_TTL`
is just defined at the bottom of the module. Python resolves the name at call time (the function
body doesn't execute at import), so this works. But it's brittle: anyone moving the function or
adding an early `__main__` smoke test will trip a `NameError`. Move the constant to the top of
the module above the function, or import it from `crypto_macro_data` as the comment claims.

### P2-2 — `metals_cross_asset` daily fallback path conditional on `gs_daily` truthiness can produce zero G/S velocity silently
File: `Q:/finance-analyzer/portfolio/signals/metals_cross_asset.py:254-261`

```python
        daily = get_all_cross_asset_data()
        copper = daily.get("copper")
        # Daily G/S ratio already pre-fetched above — reuse for both the
        # velocity field (5d change) and the z-score.
        spy = daily.get("spy")
        oil = daily.get("oil")
        result["copper_change_pct"] = copper["change_5d_pct"] if copper else 0.0
        result["gs_velocity_pct"] = gs_daily["change_5d_pct"] if gs_daily else 0.0
```

When `gs_daily` is None (the pre-fetch on line 210 returned None), `gs_velocity_pct` silently
becomes 0.0 and the `gs_velocity` sub-signal goes HOLD without a log. The intraday path already
WARNS about degraded sources (lines 232-237); the daily-fallback path does not. Add a parallel
warning when any of `gvz`, `gs_daily`, `copper`, `spy`, `oil` is None on the daily path.

### P2-3 — `news_event` keyword "raise" treated as bullish without a context-of-use whitelist (mirror of the bare-"cut" bug that the SM-P1-1 patch fixed)
File: `Q:/finance-analyzer/portfolio/signals/news_event.py:199, 269-271, 472-473`

The same SM-P1-1 reasoning that turned bare "cut" from default-positive to context-aware also
applies to "raise": "raises capital" / "capital raise" / "raises debt" are bearish dilution
events; "rate hike" / "raises rates" is hawkish (bearish equities); only "raises guidance" /
"raises dividend" / "raises forecast" are unambiguously bullish. Current logic counts ANY
appearance of "raise" in the title as positive (`_sentiment_shift`, `_source_weight_vote`,
`_thesis_alignment_vote`, `_persist_headlines`). Same fix pattern as cut: whitelist the bullish
phrases, blacklist the bearish ones, default the bare token to neutral or bearish based on a
small empirical sample.

### P2-4 — `metals_cross_asset` and `credit_spread` write to module-level `_epu_cache` / `_tips_cache` / `_oas_cache` that persist across signal computations — sticky state risk if a test or backtest re-imports with different config
File: `Q:/finance-analyzer/portfolio/signals/metals_cross_asset.py:86-88`
       `Q:/finance-analyzer/portfolio/signals/credit_spread.py:53`

Production loop is fine (single process, single fred_key, intentional 4 h cache). Cross-cycle
state means tests must clear these between runs or risk cached values leaking across test cases.
`metals_cross_asset` keys cache by `fred_api_key` (line 112) so a test that swaps keys is safe.
`credit_spread` also keys by `fred_api_key` (line 63). But neither exposes a `_reset_cache()`
helper for `tests/_state_reset.py`. Add reset hooks and register in `tests/_state_reset.py`
(the recent grid_fisher commit `fa6973e8` followed this exact pattern).

### P2-5 — `statistical_jump_regime._classify_vol_regime` returns "low_vol"/"normal"/"high_vol" strings keyed off rank but does not validate `rank` is not all NaN before assignment
File: `Q:/finance-analyzer/portfolio/signals/statistical_jump_regime.py:53-70`

```python
    rank = vol.rolling(window=effective_window, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    result = pd.Series("normal", index=vol.index)
    result[rank < VOL_LOW_PCTILE / 100] = "low_vol"
    result[rank > VOL_HIGH_PCTILE / 100] = "high_vol"
    return result
```

The boolean indexing with a NaN-containing series silently keeps NaN positions as "normal" (NaN
comparisons return False). That's the desired fallback behavior — annotate as intentional or
add explicit `notna()` guard for readability. Lower-confidence finding; flagged for completeness.

---

## P3 — Cosmetic / hygiene

- `crypto_macro.py:280-281` — `OPTIONS_TTL` defined below its only call site with misleading
  comment "Cache TTL imported from data module" — see P2-1.
- `metals_cross_asset.py:100-102` and `credit_spread.py:134-136` — duplicated `_get_fred_key`
  helper, opaque ternary, would benefit from extraction to a shared helper.
- `news_event.py:108-110` — `short = ticker.upper().replace("-USD", "")` repeated logic; ticker
  normalization should be a shared utility.
- `news_event.py` — 7 sub-signals all share repeated `keyword_severity / score_headline /
  positive-keyword check` boilerplate. Extracting a shared scorer would reduce drift between
  `_sentiment_shift` / `_source_weight_vote` / `_persist_headlines` / `_thesis_alignment_vote`
  (each currently has slightly different positive-keyword tuples).
- `forecast.py`, `claude_fundamental.py` — disk I/O side effects exist but are intentional
  (prediction backfill logs). Out of scope for purity — they are gated by DISABLED_SIGNALS plus
  shadow mode. No fix needed; noted for visibility.

---

## Disabled-signal leakage check (negative results — none found)

Cross-referenced `_register_defaults()` in `portfolio/signal_registry.py:97-345` against
`DISABLED_SIGNALS` in `portfolio/tickers.py:65-209`. Every registered signal that should be
inactive per the project documentation is either:
- in `DISABLED_SIGNALS` (49 entries), or
- a shadow-tracked LLM (`finance_llama`, `cryptotrader_lm`, `meta_trader`) that is
  intentionally **not** in `SIGNAL_NAMES` and is gated through `shadow_registry` /
  `_KNOWN_SHADOW_LLMS` (`portfolio/signal_engine.py:658-664`).

The 17 active signals named in the project CLAUDE.md align with `SIGNAL_NAMES − DISABLED_SIGNALS`
modulo the in-core RSI / BB / Fear&Greed / Ministral / Qwen3 / OnChain / BTC Proxy that aren't
in `portfolio/signals/`. No accidental-active leakage found in the plugin registry.

---

## Active-signal contract conformance (negative results — none found)

All ten priority-active `compute_*_signal` entry points return the required dict shape on
every code path I traced (no `None` returns, no uncaught exceptions reach the caller — outer
try/excepts collapse sub-signal failures to "HOLD"). Look-ahead bias check: none observed; all
sub-indicators use `iloc[-1]` over historical series and don't predict bar t from bar t.
`majority_vote(votes, count_hold=...)` is used consistently. No bare `except:` masking — the
broad `except Exception` blocks all log via `logger.exception` / `logger.debug`.

---

## Summary

| Severity | Count | Notable |
|----------|-------|---------|
| P0       | 1     | momentum_factors seasonality compounding bug — un-propagated fix from mean_reversion |
| P1       | 6     | news_event I/O race; credit_spread relative-path + cache-lock + key-helper; cot_positioning percentile bias; econ_calendar stale-evt risk |
| P2       | 5     | OPTIONS_TTL ordering; daily-path missing warn; "raise" keyword polysemy; test-reset hooks; rank-NaN annotation |
| P3       | ~5    | duplicated helper, docstring drift, shared scorer extraction |

Highest-impact action: fix P0-1 (one-file edit, mirror the existing mean_reversion fix) and
P1-1 (news_event I/O race — either move persistence out, or key file by ticker). Both directly
affect XAU/XAG accuracy which the user identifies as the primary focus.
