# 3-Hour Signal Accuracy Optimization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve 3h prediction accuracy from ~49.4% to 55-60% by fixing calibration, weighting, and parameter tuning without changing the 1d prediction path.

**Architecture:** Add a `horizon` parameter threading through `generate_signal()` → `_weighted_consensus()` → accuracy loading. When `horizon="3h"`, the system loads 3h accuracy data, caps confidence, applies time-of-day scaling, disables slow signals, and uses short-term indicator parameters. The 1d path (default) is unchanged.

**Tech Stack:** Python, pandas, numpy, pytest

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `portfolio/signal_engine.py` | Core signal voting, consensus, confidence | Modify |
| `portfolio/indicators.py` | RSI/MACD/BB computation | Modify |
| `portfolio/short_horizon.py` | 3h-specific config (slow signals, ToD scaling, confidence cap) | Create |
| `tests/test_short_horizon.py` | Tests for `short_horizon.py` | Create |
| `tests/test_signal_engine_core.py` | Existing tests — add 3h-aware weighted consensus tests | Modify |
| `tests/test_indicators_short.py` | Tests for parameterized indicator computation | Create |

---

### Task 1: Create `short_horizon.py` with 3h configuration constants

**Files:**
- Create: `portfolio/short_horizon.py`
- Create: `tests/test_short_horizon.py`

This module holds all 3h-specific configuration: which signals to disable, time-of-day scaling factors, and the confidence cap. Keeping it separate avoids cluttering `signal_engine.py` and makes it easy to tune.

- [ ] **Step 1: Write tests for short_horizon module**

```python
# tests/test_short_horizon.py
"""Tests for portfolio/short_horizon.py — 3h horizon configuration."""

import pytest
from portfolio.short_horizon import (
    CONFIDENCE_CAP_3H,
    SLOW_SIGNALS_3H,
    time_of_day_scale_3h,
    is_slow_signal_3h,
)


class TestSlowSignals3H:
    def test_trend_is_slow(self):
        assert is_slow_signal_3h("trend")

    def test_fibonacci_is_slow(self):
        assert is_slow_signal_3h("fibonacci")

    def test_macro_regime_is_slow(self):
        assert is_slow_signal_3h("macro_regime")

    def test_rsi_is_not_slow(self):
        assert not is_slow_signal_3h("rsi")

    def test_news_event_is_not_slow(self):
        assert not is_slow_signal_3h("news_event")

    def test_qwen3_is_not_slow(self):
        assert not is_slow_signal_3h("qwen3")


class TestTimeOfDayScale3H:
    def test_peak_noise_hours_get_dampened(self):
        # 10-17 UTC should be dampened (< 1.0)
        for hour in [10, 12, 14, 16, 17]:
            factor = time_of_day_scale_3h(hour)
            assert factor < 1.0, f"Hour {hour} should be dampened, got {factor}"

    def test_quiet_hours_get_boosted(self):
        # 20-01 UTC should be boosted (> 1.0)
        for hour in [20, 21, 22, 23, 0]:
            factor = time_of_day_scale_3h(hour)
            assert factor > 1.0, f"Hour {hour} should be boosted, got {factor}"

    def test_neutral_hours_near_one(self):
        # 7-9, 18-19 should be near 1.0
        for hour in [7, 8, 9, 18, 19]:
            factor = time_of_day_scale_3h(hour)
            assert 0.95 <= factor <= 1.05, f"Hour {hour} should be neutral, got {factor}"

    def test_returns_float(self):
        assert isinstance(time_of_day_scale_3h(12), float)

    def test_all_hours_valid(self):
        for hour in range(24):
            factor = time_of_day_scale_3h(hour)
            assert 0.5 <= factor <= 1.5, f"Hour {hour} factor {factor} out of range"


class TestConfidenceCap3H:
    def test_cap_is_below_one(self):
        assert CONFIDENCE_CAP_3H < 1.0

    def test_cap_is_reasonable(self):
        assert 0.6 <= CONFIDENCE_CAP_3H <= 0.85
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_short_horizon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio.short_horizon'`

- [ ] **Step 3: Implement `short_horizon.py`**

```python
# portfolio/short_horizon.py
"""Configuration for 3-hour prediction horizon.

Research-backed constants for optimizing signal accuracy at 3h:
- Slow signals (trend, fibonacci, macro_regime) are noise at 3h (38-45% accuracy)
- Peak trading hours (10-17 UTC) have 43-45% accuracy; quiet hours (20-01 UTC) have 52-55%
- Confidence above 75% at 3h is anti-correlated with accuracy (90%+ bucket = 28% actual)
"""

# Signals that need daily+ data and degrade at 3h.
# trend: 38.8% at 3h (needs SMA200 = 33+ days context)
# fibonacci: 43.3% at 3h (swing detection needs broad history)
# macro_regime: 45.1% at 3h (200-SMA regime is daily-scale)
SLOW_SIGNALS_3H = frozenset({"trend", "fibonacci", "macro_regime"})

# 3h confidence cap. The 90%+ confidence bucket has 28.1% actual accuracy.
# The 70-80% band is the best-performing at 58.9%. Cap at 75%.
CONFIDENCE_CAP_3H = 0.75

# Time-of-day scaling for 3h predictions (UTC hours).
# Based on measured consensus accuracy by hour.
_TOD_SCALE = {
    # Quiet hours (20:00-01:00 UTC) — 52-55% accuracy
    20: 1.10, 21: 1.10, 22: 1.08, 23: 1.08, 0: 1.10,
    # Asian session (1-6 UTC) — 48-50%, slightly below baseline
    1: 0.97, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.97,
    # Pre-EU (7-9 UTC) — neutral
    7: 1.0, 8: 1.0, 9: 1.0,
    # Peak noise (10-17 UTC) — 43-45% accuracy
    10: 0.88, 11: 0.88, 12: 0.85, 13: 0.88, 14: 0.88,
    15: 0.88, 16: 0.85, 17: 0.85,
    # Transition (18-19 UTC) — near neutral
    18: 1.0, 19: 1.0,
}


def is_slow_signal_3h(signal_name: str) -> bool:
    """Check if a signal should be disabled for 3h horizon predictions."""
    return signal_name in SLOW_SIGNALS_3H


def time_of_day_scale_3h(hour: int) -> float:
    """Return confidence scaling factor for 3h predictions at given UTC hour."""
    return float(_TOD_SCALE.get(hour % 24, 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_short_horizon.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/short_horizon.py tests/test_short_horizon.py
git commit -m "feat: add short_horizon module with 3h-specific configuration"
```

---

### Task 2: Add parameterized indicator computation (RSI-7, MACD 8/17/9)

**Files:**
- Modify: `portfolio/indicators.py:13-110`
- Create: `tests/test_indicators_short.py`

Add a `horizon` parameter to `compute_indicators()`. When `horizon="3h"`, use RSI(7) with thresholds 25/75 and MACD(8,17,9). Default behavior (no horizon or `horizon="1d"`) is unchanged.

- [ ] **Step 1: Write tests for parameterized indicators**

```python
# tests/test_indicators_short.py
"""Tests for horizon-parameterized indicator computation."""

import numpy as np
import pandas as pd
import pytest

from portfolio.indicators import compute_indicators


def _make_df(n=100, close_start=100.0):
    """Build minimal OHLCV DataFrame."""
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    np.random.seed(42)
    closes = close_start + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    volumes = np.random.randint(100, 10000, n).astype(float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


class TestComputeIndicatorsDefault:
    """Default (1d) behavior is unchanged."""

    def test_default_rsi_period_14(self):
        df = _make_df(100)
        ind = compute_indicators(df)
        assert ind is not None
        assert "rsi" in ind
        # Default thresholds
        assert ind["rsi_p20"] >= 15  # min clamp in signal_engine, but raw is from rolling

    def test_default_macd_12_26_9(self):
        df = _make_df(100)
        ind = compute_indicators(df)
        assert ind is not None
        assert "macd_hist" in ind
        assert "macd_hist_prev" in ind

    def test_no_horizon_same_as_1d(self):
        df = _make_df(100)
        ind_default = compute_indicators(df)
        ind_1d = compute_indicators(df, horizon="1d")
        assert ind_default["rsi"] == ind_1d["rsi"]
        assert ind_default["macd_hist"] == ind_1d["macd_hist"]


class TestComputeIndicators3H:
    """3h horizon uses RSI(7) and MACD(8,17,9)."""

    def test_3h_indicators_returned(self):
        df = _make_df(100)
        ind = compute_indicators(df, horizon="3h")
        assert ind is not None
        assert "rsi" in ind
        assert "macd_hist" in ind

    def test_3h_rsi_differs_from_default(self):
        df = _make_df(100)
        ind_1d = compute_indicators(df)
        ind_3h = compute_indicators(df, horizon="3h")
        # RSI(7) vs RSI(14) will produce different values on same data
        assert ind_1d["rsi"] != ind_3h["rsi"]

    def test_3h_macd_differs_from_default(self):
        df = _make_df(100)
        ind_1d = compute_indicators(df)
        ind_3h = compute_indicators(df, horizon="3h")
        # MACD(8,17,9) vs MACD(12,26,9) will produce different values
        assert ind_1d["macd_hist"] != ind_3h["macd_hist"]

    def test_3h_rsi_thresholds(self):
        df = _make_df(100)
        ind_3h = compute_indicators(df, horizon="3h")
        # 3h uses wider rolling window for adaptive thresholds
        assert "rsi_p20" in ind_3h
        assert "rsi_p80" in ind_3h

    def test_3h_min_rows_lower(self):
        """3h needs only 17 rows min (MACD slow=17), not 26."""
        df = _make_df(20)
        ind_3h = compute_indicators(df, horizon="3h")
        assert ind_3h is not None

    def test_1d_still_needs_26_rows(self):
        """Default still needs 26 rows."""
        df = _make_df(20)
        ind_1d = compute_indicators(df)
        assert ind_1d is None  # 20 < 26
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_indicators_short.py -v`
Expected: FAIL — `compute_indicators() got an unexpected keyword argument 'horizon'`

- [ ] **Step 3: Implement parameterized `compute_indicators`**

Modify `portfolio/indicators.py` — add `horizon` parameter, parameterize RSI and MACD:

The key changes to `compute_indicators(df, horizon=None)`:

```python
def compute_indicators(df, horizon=None):
    # Parameter selection based on horizon
    if horizon == "3h":
        rsi_period = 7
        macd_fast, macd_slow, macd_signal_period = 8, 17, 9
        min_rows = macd_slow  # 17
    else:
        rsi_period = 14
        macd_fast, macd_slow, macd_signal_period = 12, 26, 9
        min_rows = macd_slow  # 26

    if len(df) < min_rows:
        logger.debug("compute_indicators: insufficient data (%d rows, need %d)", len(df), min_rows)
        return None

    close = df["close"].copy()
    # ... existing NaN guards unchanged ...

    # RSI(rsi_period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss_safe = avg_loss.replace(0, np.finfo(float).eps)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))

    # MACD(macd_fast, macd_slow, macd_signal_period)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=macd_signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal

    # ... rest unchanged (EMA9/21, BB20/2, ATR14 stay the same) ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_indicators_short.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing indicator tests to verify no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -k "indicator" -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add portfolio/indicators.py tests/test_indicators_short.py
git commit -m "feat: parameterize indicators for 3h horizon (RSI-7, MACD 8/17/9)"
```

---

### Task 3: Thread `horizon` through `generate_signal()` — accuracy loading

**Files:**
- Modify: `portfolio/signal_engine.py:489-1057` (the `generate_signal` function)
- Modify: `tests/test_signal_engine_core.py`

Add `horizon` parameter to `generate_signal()`. When `horizon="3h"`:
1. Load 3h accuracy data instead of 1d for `_weighted_consensus`
2. Call `compute_indicators(df, horizon="3h")` upstream (callers handle this)

This is the single biggest fix — funding rate goes from 29.9% (demoted) to 74.2% (promoted).

- [ ] **Step 1: Write tests for horizon-aware accuracy loading**

Add to `tests/test_signal_engine_core.py`:

```python
class TestWeightedConsensus3H:
    """3h horizon uses 3h accuracy data for weighting."""

    def test_3h_accuracy_weights_applied(self):
        """When horizon=3h, funding (74.2% at 3h) should dominate over sentiment (45.4%)."""
        votes = {"funding": "BUY", "sentiment": "SELL"}
        accuracy_3h = {
            "funding": {"accuracy": 0.742, "total": 535},
            "sentiment": {"accuracy": 0.454, "total": 31131},
        }
        action, conf = _weighted_consensus(votes, accuracy_3h, "breakout")
        assert action == "BUY"
        expected = 0.742 / (0.742 + 0.454)
        assert conf == pytest.approx(expected, abs=0.01)

    def test_1d_accuracy_default_unchanged(self):
        """Without 3h accuracy, funding (29.9% at 1d) gets gated."""
        votes = {"funding": "BUY"}
        accuracy_1d = {
            "funding": {"accuracy": 0.299, "total": 536},
        }
        action, conf = _weighted_consensus(votes, accuracy_1d, "breakout")
        # 0.299 < 0.45 gate with 536 samples => gated => HOLD
        assert action == "HOLD"
```

- [ ] **Step 2: Run tests to verify they pass** (these test `_weighted_consensus` directly, which already accepts accuracy_data — no code change needed for this part)

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py::TestWeightedConsensus3H -v`
Expected: PASS (the function already works with any accuracy dict — the fix is in *which* data gets loaded)

- [ ] **Step 3: Add `horizon` parameter to `generate_signal()`**

In `portfolio/signal_engine.py`, modify the `generate_signal` function signature and accuracy loading block:

Change line 489:
```python
def generate_signal(ind, ticker=None, config=None, timeframes=None, df=None, horizon=None):
```

Change lines 940-989 (the accuracy loading block). Replace:
```python
    accuracy_data = {}
    activation_rates = {}
    try:
        from portfolio.accuracy_stats import (
            load_cached_accuracy,
            load_cached_activation_rates,
            signal_accuracy,
            signal_accuracy_recent,
            write_accuracy_cache,
        )

        # Load all-time accuracy
        alltime = load_cached_accuracy("1d")
        if not alltime:
            alltime = signal_accuracy("1d")
            if alltime:
                write_accuracy_cache("1d", alltime)

        # Load recent accuracy (7d window) — more responsive to regime changes
        recent = load_cached_accuracy("1d_recent")
        if not recent:
            recent = signal_accuracy_recent("1d", days=7)
            if recent:
                write_accuracy_cache("1d_recent", recent)
```

With:
```python
    accuracy_data = {}
    activation_rates = {}
    try:
        from portfolio.accuracy_stats import (
            load_cached_accuracy,
            load_cached_activation_rates,
            signal_accuracy,
            signal_accuracy_recent,
            write_accuracy_cache,
        )

        # Select accuracy horizon — use 3h accuracy when predicting 3h moves
        acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"

        # Load all-time accuracy
        alltime = load_cached_accuracy(acc_horizon)
        if not alltime:
            alltime = signal_accuracy(acc_horizon)
            if alltime:
                write_accuracy_cache(acc_horizon, alltime)

        # Load recent accuracy (7d window) — more responsive to regime changes
        recent_key = f"{acc_horizon}_recent"
        recent = load_cached_accuracy(recent_key)
        if not recent:
            recent = signal_accuracy_recent(acc_horizon, days=7)
            if recent:
                write_accuracy_cache(recent_key, recent)
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py -v`
Expected: All PASS (default behavior unchanged — `horizon=None` → `acc_horizon="1d"`)

- [ ] **Step 5: Commit**

```bash
git add portfolio/signal_engine.py tests/test_signal_engine_core.py
git commit -m "feat: thread horizon parameter through generate_signal for accuracy loading"
```

---

### Task 4: Apply 3h confidence cap and slow signal gating in `generate_signal()`

**Files:**
- Modify: `portfolio/signal_engine.py:489-1057`
- Modify: `tests/test_signal_engine_core.py`

When `horizon="3h"`:
1. Skip slow signals (trend, fibonacci, macro_regime) — force HOLD
2. Cap final confidence at `CONFIDENCE_CAP_3H` (0.75)

- [ ] **Step 1: Write tests for slow signal gating and confidence cap**

Add to `tests/test_signal_engine_core.py`:

```python
class TestSlowSignalGating3H:
    """Slow signals are force-HOLD when horizon=3h."""

    def test_trend_gated_at_3h(self):
        from portfolio.short_horizon import SLOW_SIGNALS_3H, is_slow_signal_3h
        assert is_slow_signal_3h("trend")
        # The gating is applied inside generate_signal() — tested via integration

    def test_fibonacci_gated_at_3h(self):
        from portfolio.short_horizon import is_slow_signal_3h
        assert is_slow_signal_3h("fibonacci")

    def test_macro_regime_gated_at_3h(self):
        from portfolio.short_horizon import is_slow_signal_3h
        assert is_slow_signal_3h("macro_regime")

    def test_rsi_not_gated_at_3h(self):
        from portfolio.short_horizon import is_slow_signal_3h
        assert not is_slow_signal_3h("rsi")


class TestConfidenceCap3H:
    """Confidence is capped at CONFIDENCE_CAP_3H for 3h horizon."""

    def test_cap_value(self):
        from portfolio.short_horizon import CONFIDENCE_CAP_3H
        assert CONFIDENCE_CAP_3H == 0.75
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py::TestSlowSignalGating3H tests/test_signal_engine_core.py::TestConfidenceCap3H -v`
Expected: PASS

- [ ] **Step 3: Add slow signal gating to `generate_signal()`**

In `portfolio/signal_engine.py`, after the enhanced signals loop (around line 897), add:

```python
    # 3h horizon: gate slow signals that are noise at short timeframes
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import is_slow_signal_3h
        for sig_name in list(votes.keys()):
            if is_slow_signal_3h(sig_name) and votes[sig_name] != "HOLD":
                votes[sig_name] = "HOLD"
```

- [ ] **Step 4: Add confidence cap at end of `generate_signal()`**

In `portfolio/signal_engine.py`, after the confidence penalty cascade (line 1052), before the return, add:

```python
    # 3h horizon: cap confidence to prevent overconfident short-term predictions
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import CONFIDENCE_CAP_3H
        conf = min(conf, CONFIDENCE_CAP_3H)
```

- [ ] **Step 5: Store horizon in extra_info for downstream debugging**

After line 1040 (`extra_info["_regime"] = regime`), add:

```python
    if horizon:
        extra_info["_horizon"] = horizon
```

- [ ] **Step 6: Run all signal engine tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add portfolio/signal_engine.py
git commit -m "feat: gate slow signals and cap confidence for 3h horizon"
```

---

### Task 5: Expand `_time_of_day_factor()` to be horizon-aware

**Files:**
- Modify: `portfolio/signal_engine.py:266-270`
- Modify: `tests/test_signal_engine_core.py`

Replace the current `_time_of_day_factor()` with a version that uses `short_horizon.time_of_day_scale_3h()` when horizon is 3h. The existing 1d behavior (0.8x during 2-6 UTC, 1.0 otherwise) is preserved.

- [ ] **Step 1: Write tests for horizon-aware time-of-day factor**

Add to `tests/test_signal_engine_core.py`:

```python
class TestTimeOfDayFactor3H:
    """Time-of-day factor with horizon awareness."""

    def test_default_behavior_unchanged(self):
        """Without horizon, existing behavior: 0.8 during 2-6 UTC, 1.0 otherwise."""
        with mock.patch("portfolio.signal_engine.datetime") as m:
            m.now.return_value = datetime(2026, 3, 15, 3, 0, tzinfo=UTC)
            m.side_effect = lambda *a, **k: datetime(*a, **k)
            assert _time_of_day_factor() == pytest.approx(0.8)

        with mock.patch("portfolio.signal_engine.datetime") as m:
            m.now.return_value = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
            m.side_effect = lambda *a, **k: datetime(*a, **k)
            assert _time_of_day_factor() == pytest.approx(1.0)

    def test_3h_peak_hours_dampened(self):
        with mock.patch("portfolio.signal_engine.datetime") as m:
            m.now.return_value = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
            m.side_effect = lambda *a, **k: datetime(*a, **k)
            factor = _time_of_day_factor(horizon="3h")
            assert factor < 1.0  # 12 UTC is peak noise hour

    def test_3h_quiet_hours_boosted(self):
        with mock.patch("portfolio.signal_engine.datetime") as m:
            m.now.return_value = datetime(2026, 3, 15, 20, 0, tzinfo=UTC)
            m.side_effect = lambda *a, **k: datetime(*a, **k)
            factor = _time_of_day_factor(horizon="3h")
            assert factor > 1.0  # 20 UTC is quiet/predictable
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py::TestTimeOfDayFactor3H -v`
Expected: FAIL — `_time_of_day_factor() got an unexpected keyword argument 'horizon'`

- [ ] **Step 3: Implement horizon-aware `_time_of_day_factor()`**

Replace `_time_of_day_factor()` in `portfolio/signal_engine.py`:

```python
def _time_of_day_factor(horizon=None):
    hour = datetime.now(UTC).hour
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import time_of_day_scale_3h
        return time_of_day_scale_3h(hour)
    # Default 1d behavior
    if 2 <= hour <= 6:
        return 0.8
    return 1.0
```

- [ ] **Step 4: Update the call site in `generate_signal()`**

Change line 1025 from:
```python
    tod_factor = _time_of_day_factor()
```
to:
```python
    tod_factor = _time_of_day_factor(horizon=horizon)
```

- [ ] **Step 5: Run all signal engine tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add portfolio/signal_engine.py tests/test_signal_engine_core.py
git commit -m "feat: horizon-aware time-of-day confidence scaling for 3h"
```

---

### Task 6: Wire RSI/MACD thresholds for 3h in signal voting

**Files:**
- Modify: `portfolio/signal_engine.py:502-520`

When `horizon="3h"`, use wider RSI thresholds (25/75 instead of 30/70 adaptive) since RSI(7) is more sensitive.

- [ ] **Step 1: Modify RSI voting thresholds for 3h**

In `generate_signal()`, replace lines 502-512:

```python
    # RSI — only votes at extremes (adaptive thresholds from rolling percentiles)
    rsi_lower = ind.get("rsi_p20", 30)
    rsi_upper = ind.get("rsi_p80", 70)
    rsi_lower = max(rsi_lower, 15)
    rsi_upper = min(rsi_upper, 85)
    if ind["rsi"] < rsi_lower:
        votes["rsi"] = "BUY"
    elif ind["rsi"] > rsi_upper:
        votes["rsi"] = "SELL"
    else:
        votes["rsi"] = "HOLD"
```

With:

```python
    # RSI — only votes at extremes (adaptive thresholds from rolling percentiles)
    if horizon in ("3h", "4h"):
        # 3h: RSI(7) is more sensitive — use fixed 25/75 thresholds
        rsi_lower = 25
        rsi_upper = 75
    else:
        rsi_lower = ind.get("rsi_p20", 30)
        rsi_upper = ind.get("rsi_p80", 70)
        rsi_lower = max(rsi_lower, 15)
        rsi_upper = min(rsi_upper, 85)
    if ind["rsi"] < rsi_lower:
        votes["rsi"] = "BUY"
    elif ind["rsi"] > rsi_upper:
        votes["rsi"] = "SELL"
    else:
        votes["rsi"] = "HOLD"
```

- [ ] **Step 2: Run existing tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add portfolio/signal_engine.py
git commit -m "feat: use RSI(7) 25/75 thresholds for 3h horizon"
```

---

### Task 7: Full integration test and final verification

**Files:**
- Create: `tests/test_3h_integration.py`

End-to-end test that verifies all 3h optimizations work together.

- [ ] **Step 1: Write integration test**

```python
# tests/test_3h_integration.py
"""Integration test: 3h signal optimizations work end-to-end."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest


def _make_df(n=100, close_start=100.0):
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    np.random.seed(42)
    closes = close_start + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    volumes = np.random.randint(100, 10000, n).astype(float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


class TestGenerate3HSignal:
    """Test generate_signal with horizon='3h'."""

    @mock.patch("portfolio.signal_engine.should_skip_gpu", return_value=True)
    @mock.patch("portfolio.signal_engine._cached", return_value=None)
    def test_3h_returns_valid_signal(self, mock_cached, mock_skip):
        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import generate_signal

        df = _make_df(100)
        ind = compute_indicators(df, horizon="3h")
        assert ind is not None

        action, conf, extra = generate_signal(
            ind, ticker="BTC-USD", df=df, horizon="3h"
        )
        assert action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= conf <= 0.75  # confidence cap

    @mock.patch("portfolio.signal_engine.should_skip_gpu", return_value=True)
    @mock.patch("portfolio.signal_engine._cached", return_value=None)
    def test_3h_stores_horizon_in_extra(self, mock_cached, mock_skip):
        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import generate_signal

        df = _make_df(100)
        ind = compute_indicators(df, horizon="3h")
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df, horizon="3h")
        assert extra.get("_horizon") == "3h"

    @mock.patch("portfolio.signal_engine.should_skip_gpu", return_value=True)
    @mock.patch("portfolio.signal_engine._cached", return_value=None)
    def test_1d_not_capped(self, mock_cached, mock_skip):
        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import generate_signal

        df = _make_df(100)
        ind = compute_indicators(df)
        action, conf, extra = generate_signal(ind, ticker="BTC-USD", df=df)
        # 1d path: no horizon stored, no 0.75 cap enforced
        assert "_horizon" not in extra
        # conf can be > 0.75 in 1d mode (if signal is strong enough)
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_3h_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine_core.py tests/test_short_horizon.py tests/test_indicators_short.py tests/test_3h_integration.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_3h_integration.py
git commit -m "test: add 3h signal optimization integration tests"
```

---

## Notes for Callers

After these changes, callers of `generate_signal()` can opt into 3h mode:

```python
# Existing 1d path — unchanged
ind = compute_indicators(df)
action, conf, extra = generate_signal(ind, ticker=ticker, df=df)

# New 3h path
ind_3h = compute_indicators(df, horizon="3h")
action, conf, extra = generate_signal(ind_3h, ticker=ticker, df=df, horizon="3h")
```

The `horizon` parameter controls:
- Which accuracy data is loaded (3h vs 1d)
- Which indicator parameters are used (RSI-7 vs RSI-14, MACD 8/17/9 vs 12/26/9)
- RSI thresholds (fixed 25/75 vs adaptive percentile)
- Whether slow signals (trend, fibonacci, macro_regime) are gated
- Time-of-day confidence scaling (data-driven 3h pattern vs simple 2-6 UTC dampening)
- Confidence cap (0.75 vs 1.0)
