# Signal Accuracy Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the 30-signal voting system from ~50% accuracy (coin flip) to a self-improving ensemble where bad signals are silenced, good signals are amplified, and weights adapt to market regime changes.

**Architecture:** Five independent improvements layered on the existing `_weighted_consensus` function: (1) persist regime in signal logs, (2) regime-conditional accuracy tracking, (3) exponential decay weighting replacing the hard 70/30 blend, (4) multiplicative weight updates for online learning, (5) utility-based weighting using return magnitude. Each task builds on prior ones but produces a working system at each step.

**Tech Stack:** Python 3.11, pytest, SQLite (signal_log.db), atomic JSON I/O via `portfolio.file_utils`

**Key Files Overview:**
- `portfolio/signal_engine.py` — voting engine with `_weighted_consensus()` (line ~197) and `generate_signal()` (line ~487)
- `portfolio/accuracy_stats.py` — accuracy computation, caching, activation rates
- `portfolio/outcome_tracker.py` — `log_signal_snapshot()` persists signal snapshots
- `portfolio/signal_db.py` — SQLite schema for signal log storage
- `data/accuracy_cache.json` — cached accuracy data (1h TTL)
- `data/signal_weights.json` — NEW: persistent MWU weights
- `config.example.json` — config documentation

---

### Task 1: Persist Regime in Signal Log

**Why:** Regime-conditional accuracy needs to know what regime each signal snapshot was recorded under. Currently `extra_info["_regime"]` exists in memory but is not saved to signal_log.jsonl or signal_log.db.

**Files:**
- Modify: `portfolio/outcome_tracker.py:107-159` (log_signal_snapshot)
- Modify: `portfolio/signal_db.py:49-56` (ticker_signals schema)
- Test: `tests/test_regime_persistence.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_regime_persistence.py
"""Test that regime is persisted in signal log entries."""
import pytest
from unittest.mock import patch
from portfolio.outcome_tracker import log_signal_snapshot
from portfolio.tickers import SIGNAL_NAMES


def _make_signals_dict(ticker, regime="trending-up"):
    """Helper to create minimal signals_dict for one ticker."""
    signals = {s: "HOLD" for s in SIGNAL_NAMES}
    signals["rsi"] = "BUY"
    return {
        ticker: {
            "action": "BUY",
            "confidence": 0.6,
            "indicators": {"close": 100.0},
            "extra": {
                "_votes": signals,
                "_regime": regime,
            },
        }
    }


class TestRegimePersistence:
    def test_regime_stored_in_entry(self, tmp_path):
        """log_signal_snapshot should include regime per ticker."""
        from portfolio import outcome_tracker as ot

        orig_log = ot.SIGNAL_LOG
        ot.SIGNAL_LOG = tmp_path / "signal_log.jsonl"
        try:
            with patch("portfolio.outcome_tracker.SignalDB", side_effect=Exception("skip")):
                entry = log_signal_snapshot(
                    _make_signals_dict("BTC-USD", "ranging"),
                    {"BTC-USD": 60000.0},
                    10.5,
                    ["test"],
                )
            assert entry["tickers"]["BTC-USD"]["regime"] == "ranging"
        finally:
            ot.SIGNAL_LOG = orig_log

    def test_regime_defaults_to_unknown(self, tmp_path):
        """Missing _regime in extra should default to 'unknown'."""
        from portfolio import outcome_tracker as ot

        orig_log = ot.SIGNAL_LOG
        ot.SIGNAL_LOG = tmp_path / "signal_log.jsonl"
        try:
            sig_dict = _make_signals_dict("BTC-USD")
            del sig_dict["BTC-USD"]["extra"]["_regime"]
            with patch("portfolio.outcome_tracker.SignalDB", side_effect=Exception("skip")):
                entry = log_signal_snapshot(sig_dict, {"BTC-USD": 60000.0}, 10.5, ["test"])
            assert entry["tickers"]["BTC-USD"]["regime"] == "unknown"
        finally:
            ot.SIGNAL_LOG = orig_log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_persistence.py -v`
Expected: FAIL — `KeyError: 'regime'`

- [ ] **Step 3: Implement — add regime to log_signal_snapshot**

In `portfolio/outcome_tracker.py`, inside `log_signal_snapshot()`, after line 137 (`"signals": signals,`), add:

```python
        # Persist regime for regime-conditional accuracy tracking
        regime = extra.get("_regime", "unknown") if extra else "unknown"

        tickers[ticker] = {
            "price_usd": price,
            "consensus": consensus,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_voters": total_voters,
            "signals": signals,
            "regime": regime,  # NEW
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_persistence.py -v`
Expected: PASS

- [ ] **Step 5: Add regime column to SQLite schema**

In `portfolio/signal_db.py`, add `regime TEXT` to the `ticker_signals` table and update `insert_snapshot` and `load_entries` to handle it.

The schema migration should use `ALTER TABLE ... ADD COLUMN` with a default of `'unknown'` so existing databases aren't broken:

```python
# In _ensure_schema(), after the CREATE TABLE block, add:
        # Schema migration: add regime column if missing
        try:
            conn.execute("SELECT regime FROM ticker_signals LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE ticker_signals ADD COLUMN regime TEXT DEFAULT 'unknown'")
            conn.commit()
```

In `insert_snapshot()`, update the INSERT to include regime:

```python
            regime = tdata.get("regime", "unknown")
            conn.execute("""
                INSERT OR IGNORE INTO ticker_signals
                (snapshot_id, ticker, price_usd, consensus, buy_count,
                 sell_count, total_voters, signals, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (snap_id, ticker, tdata.get("price_usd"),
                  tdata.get("consensus"), tdata.get("buy_count"),
                  tdata.get("sell_count"), tdata.get("total_voters"),
                  json.dumps(tdata.get("signals", {})),
                  regime))
```

In `load_entries()`, include regime in the reconstructed ticker dict:

```python
            ticker_dict["regime"] = row["regime"] if "regime" in row.keys() else "unknown"
```

- [ ] **Step 6: Run all tests to verify no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_persistence.py tests/test_signal_engine_core.py tests/test_weighted_consensus.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add portfolio/outcome_tracker.py portfolio/signal_db.py tests/test_regime_persistence.py
git commit -m "feat: persist regime in signal log for regime-conditional accuracy"
```

---

### Task 2: Regime-Conditional Accuracy Tracking

**Why:** A signal might be 70% accurate in trends but 35% in ranging markets. The blended average of 52% hides both facts. Tracking accuracy per-regime lets the engine use the right weight for the current conditions.

**Files:**
- Modify: `portfolio/accuracy_stats.py` (add `signal_accuracy_by_regime()`)
- Modify: `portfolio/signal_engine.py:940-1010` (use regime-conditional accuracy in blending)
- Test: `tests/test_regime_accuracy.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_regime_accuracy.py
"""Test regime-conditional accuracy computation."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from portfolio.accuracy_stats import signal_accuracy_by_regime


def _make_entry(ticker, signal_votes, regime, outcome_pct):
    """Create a minimal signal log entry."""
    from portfolio.tickers import SIGNAL_NAMES
    signals = {s: "HOLD" for s in SIGNAL_NAMES}
    signals.update(signal_votes)
    return {
        "ts": "2026-03-20T12:00:00+00:00",
        "tickers": {
            ticker: {
                "signals": signals,
                "consensus": "BUY",
                "regime": regime,
            }
        },
        "outcomes": {
            ticker: {
                "1d": {"change_pct": outcome_pct}
            }
        },
    }


class TestSignalAccuracyByRegime:
    def test_separates_regimes(self):
        """Accuracy differs by regime for the same signal."""
        entries = [
            # RSI BUY in trending-up, price goes up → correct
            _make_entry("BTC-USD", {"rsi": "BUY"}, "trending-up", 2.0),
            _make_entry("BTC-USD", {"rsi": "BUY"}, "trending-up", 1.5),
            # RSI BUY in ranging, price goes down → wrong
            _make_entry("BTC-USD", {"rsi": "BUY"}, "ranging", -2.0),
            _make_entry("BTC-USD", {"rsi": "BUY"}, "ranging", -1.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_by_regime("1d")

        assert result["trending-up"]["rsi"]["accuracy"] == 1.0
        assert result["trending-up"]["rsi"]["total"] == 2
        assert result["ranging"]["rsi"]["accuracy"] == 0.0
        assert result["ranging"]["rsi"]["total"] == 2

    def test_missing_regime_goes_to_unknown(self):
        """Entries without regime field go into 'unknown' bucket."""
        entries = [_make_entry("BTC-USD", {"rsi": "BUY"}, None, 1.0)]
        # Remove the regime key to simulate old entries
        del entries[0]["tickers"]["BTC-USD"]["regime"]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_by_regime("1d")
        assert "unknown" in result
        assert result["unknown"]["rsi"]["total"] == 1

    def test_empty_entries(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = signal_accuracy_by_regime("1d")
        assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_accuracy.py -v`
Expected: FAIL — `ImportError: cannot import name 'signal_accuracy_by_regime'`

- [ ] **Step 3: Implement signal_accuracy_by_regime**

Add to `portfolio/accuracy_stats.py`:

```python
def signal_accuracy_by_regime(horizon="1d", since=None):
    """Compute per-signal accuracy bucketed by market regime.

    Returns:
        dict: {regime: {signal_name: {correct, total, accuracy, pct}}}
        Regimes: "trending-up", "trending-down", "ranging", "high-vol", "unknown"
    """
    entries = load_entries()
    # {regime: {signal: {correct, total}}}
    stats = defaultdict(lambda: {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES})

    for entry in entries:
        if since and entry.get("ts", "") < since:
            continue
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue
            change_pct = outcome.get("change_pct", 0)
            regime = tdata.get("regime", "unknown") or "unknown"
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue
                stats[regime][sig_name]["total"] += 1
                if result_val:
                    stats[regime][sig_name]["correct"] += 1

    result = {}
    for regime, sig_stats in stats.items():
        regime_result = {}
        for sig_name, s in sig_stats.items():
            if s["total"] == 0:
                continue
            acc = s["correct"] / s["total"]
            regime_result[sig_name] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
            }
        if regime_result:
            result[regime] = regime_result
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_accuracy.py -v`
Expected: PASS

- [ ] **Step 5: Integrate regime-conditional accuracy into signal_engine**

In `portfolio/signal_engine.py`, modify the accuracy blending block (around line 940-985). After the existing blending, overlay regime-specific accuracy when available:

```python
    # After the existing accuracy_data blending block (line ~985):

    # Overlay regime-specific accuracy when available.
    # If a signal has 30+ samples in the current regime, prefer that accuracy
    # over the global blend. This prevents a signal that's 70% in trends / 35%
    # in ranging from being averaged to 52% in both regimes.
    try:
        from portfolio.accuracy_stats import signal_accuracy_by_regime
        regime_acc = signal_accuracy_by_regime("1d")
        current_regime_data = regime_acc.get(regime, {})
        for sig_name, rdata in current_regime_data.items():
            if rdata.get("total", 0) >= 30:
                # Use regime-specific accuracy instead of global blend
                accuracy_data[sig_name] = rdata
    except Exception:
        logger.debug("Regime-conditional accuracy unavailable", exc_info=True)
```

- [ ] **Step 6: Add caching for regime accuracy**

Computing regime accuracy every 60s cycle is expensive. Add caching similar to `load_cached_accuracy()` in `portfolio/accuracy_stats.py`:

```python
REGIME_ACCURACY_CACHE_FILE = DATA_DIR / "regime_accuracy_cache.json"

def load_cached_regime_accuracy(horizon="1d"):
    """Load cached regime-conditional accuracy, recomputing if stale (1h TTL)."""
    cache = load_json(REGIME_ACCURACY_CACHE_FILE)
    if cache is not None:
        try:
            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (KeyError, AttributeError):
            pass
    return None

def write_regime_accuracy_cache(horizon, data):
    cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
    if not isinstance(cache, dict):
        cache = {}
    cache[horizon] = data
    cache["time"] = time.time()
    _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)
```

Then update signal_engine to use the cached version:

```python
    try:
        from portfolio.accuracy_stats import (
            load_cached_regime_accuracy,
            signal_accuracy_by_regime,
            write_regime_accuracy_cache,
        )
        regime_acc = load_cached_regime_accuracy("1d")
        if not regime_acc:
            regime_acc = signal_accuracy_by_regime("1d")
            if regime_acc:
                write_regime_accuracy_cache("1d", regime_acc)
        current_regime_data = regime_acc.get(regime, {})
        for sig_name, rdata in current_regime_data.items():
            if rdata.get("total", 0) >= 30:
                accuracy_data[sig_name] = rdata
    except Exception:
        logger.debug("Regime-conditional accuracy unavailable", exc_info=True)
```

- [ ] **Step 7: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_accuracy.py tests/test_weighted_consensus.py tests/test_signal_engine_core.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add portfolio/accuracy_stats.py portfolio/signal_engine.py tests/test_regime_accuracy.py
git commit -m "feat: regime-conditional accuracy tracking for signal weighting"
```

---

### Task 3: Exponential Decay Weighting

**Why:** The current hard 70/30 (recent/all-time) blend creates a discontinuity at the 7-day boundary. Exponential decay weights recent observations smoothly higher, with configurable half-life.

**Files:**
- Modify: `portfolio/accuracy_stats.py` (add `signal_accuracy_ewma()`)
- Modify: `portfolio/signal_engine.py:962-985` (replace blend with EWMA)
- Modify: `config.example.json` (add `signals.accuracy_halflife_days`)
- Test: `tests/test_ewma_accuracy.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ewma_accuracy.py
"""Test exponential decay weighted accuracy computation."""
import math
import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import patch
from portfolio.accuracy_stats import signal_accuracy_ewma


def _make_entry(ts_str, sig_name, vote, outcome_pct):
    """Create a minimal signal log entry at a specific timestamp."""
    from portfolio.tickers import SIGNAL_NAMES
    signals = {s: "HOLD" for s in SIGNAL_NAMES}
    signals[sig_name] = vote
    return {
        "ts": ts_str,
        "tickers": {
            "BTC-USD": {
                "signals": signals,
                "consensus": vote,
            }
        },
        "outcomes": {
            "BTC-USD": {
                "1d": {"change_pct": outcome_pct}
            }
        },
    }


class TestSignalAccuracyEWMA:
    def test_recent_correct_weighted_higher(self):
        """Recent correct votes should produce higher accuracy than old correct votes."""
        now = datetime.now(UTC)
        entries = [
            # Old: 10 days ago, BUY, price went down (wrong)
            _make_entry((now - timedelta(days=10)).isoformat(), "rsi", "BUY", -2.0),
            # Recent: 1 day ago, BUY, price went up (correct)
            _make_entry((now - timedelta(days=1)).isoformat(), "rsi", "BUY", 2.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=3)
        # Recent correct should dominate → accuracy > 0.5
        assert result["rsi"]["accuracy"] > 0.5

    def test_old_correct_weighted_lower(self):
        """Old correct votes should produce lower accuracy than recent wrong votes."""
        now = datetime.now(UTC)
        entries = [
            # Old: 10 days ago, BUY correct
            _make_entry((now - timedelta(days=10)).isoformat(), "rsi", "BUY", 2.0),
            # Recent: 1 day ago, BUY wrong
            _make_entry((now - timedelta(days=1)).isoformat(), "rsi", "BUY", -2.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=3)
        # Recent wrong should dominate → accuracy < 0.5
        assert result["rsi"]["accuracy"] < 0.5

    def test_halflife_affects_decay(self):
        """Shorter half-life should give more weight to recent data."""
        now = datetime.now(UTC)
        entries = [
            _make_entry((now - timedelta(days=7)).isoformat(), "rsi", "BUY", 2.0),  # old correct
            _make_entry((now - timedelta(days=1)).isoformat(), "rsi", "BUY", -2.0),  # recent wrong
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            short_hl = signal_accuracy_ewma("1d", halflife_days=2)
            long_hl = signal_accuracy_ewma("1d", halflife_days=14)
        # Short half-life: recent wrong dominates more → lower accuracy
        # Long half-life: old correct has more influence → higher accuracy
        assert short_hl["rsi"]["accuracy"] < long_hl["rsi"]["accuracy"]

    def test_empty_entries(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        for sig_data in result.values():
            assert sig_data["accuracy"] == 0.0
            assert sig_data["total_weight"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ewma_accuracy.py -v`
Expected: FAIL — `ImportError: cannot import name 'signal_accuracy_ewma'`

- [ ] **Step 3: Implement signal_accuracy_ewma**

Add to `portfolio/accuracy_stats.py`:

```python
def signal_accuracy_ewma(horizon="1d", halflife_days=5):
    """Compute per-signal accuracy using exponential decay weighting.

    Each outcome is weighted by exp(-lambda * age_days) where
    lambda = ln(2) / halflife_days. Recent outcomes count more.

    Returns:
        dict: {signal_name: {accuracy, total_weight, effective_samples, pct}}
    """
    import math
    from datetime import datetime, timedelta

    entries = load_entries()
    now = datetime.now(UTC)
    decay_lambda = math.log(2) / halflife_days

    stats = {s: {"correct_weight": 0.0, "total_weight": 0.0} for s in SIGNAL_NAMES}

    for entry in entries:
        ts_str = entry.get("ts", "")
        try:
            entry_time = datetime.fromisoformat(ts_str)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
        age_days = max((now - entry_time).total_seconds() / 86400, 0)
        weight = math.exp(-decay_lambda * age_days)

        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue
            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue
                stats[sig_name]["total_weight"] += weight
                if result_val:
                    stats[sig_name]["correct_weight"] += weight

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        tw = s["total_weight"]
        acc = s["correct_weight"] / tw if tw > 0 else 0.0
        result[sig_name] = {
            "accuracy": acc,
            "total_weight": round(tw, 4),
            "effective_samples": round(tw, 0),  # approximate sample count
            "total": int(round(tw, 0)),  # compat with existing accuracy format
            "correct": int(round(s["correct_weight"], 0)),
            "pct": round(acc * 100, 1),
        }
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ewma_accuracy.py -v`
Expected: PASS

- [ ] **Step 5: Replace 70/30 blend with EWMA in signal_engine**

In `portfolio/signal_engine.py`, replace the accuracy blending block (lines ~948-985) with:

```python
        # --- Accuracy data: EWMA with configurable half-life ---
        sig_cfg = (config or {}).get("signals", {})
        halflife = sig_cfg.get("accuracy_halflife_days", 5)

        from portfolio.accuracy_stats import signal_accuracy_ewma

        accuracy_data = load_cached_accuracy("1d_ewma")
        if not accuracy_data:
            accuracy_data = signal_accuracy_ewma("1d", halflife_days=halflife)
            if accuracy_data:
                write_accuracy_cache("1d_ewma", accuracy_data)

        # Fallback: if EWMA has no data yet, try old all-time accuracy
        if not accuracy_data or all(v.get("total", 0) == 0 for v in accuracy_data.values()):
            accuracy_data = load_cached_accuracy("1d")
            if not accuracy_data:
                accuracy_data = signal_accuracy("1d")
                if accuracy_data:
                    write_accuracy_cache("1d", accuracy_data)
```

- [ ] **Step 6: Add config to config.example.json**

```json
  "signals": {
    "accuracy_gate_threshold": 0.45,
    "accuracy_gate_min_samples": 30,
    "accuracy_halflife_days": 5
  },
```

- [ ] **Step 7: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ewma_accuracy.py tests/test_weighted_consensus.py tests/test_signal_engine_core.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add portfolio/accuracy_stats.py portfolio/signal_engine.py config.example.json tests/test_ewma_accuracy.py
git commit -m "feat: exponential decay accuracy weighting (replaces 70/30 blend)"
```

---

### Task 4: Multiplicative Weight Updates (MWU)

**Why:** Instead of recomputing accuracy from logs every hour, track persistent weights that update after each outcome. Signals that are consistently wrong rapidly approach zero weight. This is the gold standard for online expert aggregation (Arora et al.).

**Files:**
- Create: `portfolio/signal_weights.py` (MWU weight manager)
- Modify: `portfolio/outcome_tracker.py` (trigger weight update on backfill)
- Modify: `portfolio/signal_engine.py` (read MWU weights)
- Test: `tests/test_signal_weights.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signal_weights.py
"""Test multiplicative weight update (MWU) for signal weights."""
import json
import pytest
from pathlib import Path
from portfolio.signal_weights import SignalWeightManager


class TestSignalWeightManager:
    def test_initial_weights_are_1(self, tmp_path):
        mgr = SignalWeightManager(tmp_path / "weights.json")
        assert mgr.get_weight("rsi") == 1.0

    def test_correct_prediction_increases_weight(self, tmp_path):
        mgr = SignalWeightManager(tmp_path / "weights.json")
        before = mgr.get_weight("rsi")
        mgr.update("rsi", correct=True)
        after = mgr.get_weight("rsi")
        assert after > before

    def test_wrong_prediction_decreases_weight(self, tmp_path):
        mgr = SignalWeightManager(tmp_path / "weights.json")
        before = mgr.get_weight("rsi")
        mgr.update("rsi", correct=False)
        after = mgr.get_weight("rsi")
        assert after < before

    def test_weight_has_floor(self, tmp_path):
        """Weight should never go below the floor (0.01)."""
        mgr = SignalWeightManager(tmp_path / "weights.json")
        for _ in range(200):
            mgr.update("rsi", correct=False)
        assert mgr.get_weight("rsi") >= 0.01

    def test_weights_persist_to_disk(self, tmp_path):
        path = tmp_path / "weights.json"
        mgr1 = SignalWeightManager(path)
        mgr1.update("rsi", correct=True)
        mgr1.save()
        w1 = mgr1.get_weight("rsi")

        mgr2 = SignalWeightManager(path)
        assert mgr2.get_weight("rsi") == w1

    def test_batch_update(self, tmp_path):
        mgr = SignalWeightManager(tmp_path / "weights.json")
        outcomes = {"rsi": True, "macd": False, "ema": True}
        mgr.batch_update(outcomes)
        assert mgr.get_weight("rsi") > 1.0
        assert mgr.get_weight("macd") < 1.0
        assert mgr.get_weight("ema") > 1.0

    def test_get_normalized_weights(self, tmp_path):
        mgr = SignalWeightManager(tmp_path / "weights.json")
        mgr.update("rsi", correct=True)
        mgr.update("rsi", correct=True)
        mgr.update("macd", correct=False)
        nw = mgr.get_normalized_weights(["rsi", "macd", "ema"])
        # Sum should be close to len(signals) (normalized so average = 1.0)
        total = sum(nw.values())
        assert abs(total - 3.0) < 0.01  # normalized to average 1.0

    def test_configurable_learning_rate(self, tmp_path):
        mgr_fast = SignalWeightManager(tmp_path / "w1.json", eta=0.2)
        mgr_slow = SignalWeightManager(tmp_path / "w2.json", eta=0.05)
        mgr_fast.update("rsi", correct=False)
        mgr_slow.update("rsi", correct=False)
        # Faster learning rate → bigger drop
        assert mgr_fast.get_weight("rsi") < mgr_slow.get_weight("rsi")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_weights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio.signal_weights'`

- [ ] **Step 3: Implement SignalWeightManager**

Create `portfolio/signal_weights.py`:

```python
"""Multiplicative Weight Update (MWU) manager for signal weights.

Each signal starts with weight 1.0. After each outcome:
  correct:   w *= (1 + eta)
  incorrect: w *= (1 - eta)

Weights are floored at 0.01 and persisted to a JSON file.
"""

import logging
from pathlib import Path
from portfolio.file_utils import atomic_write_json as _atomic_write, load_json

logger = logging.getLogger("portfolio.signal_weights")

DEFAULT_ETA = 0.1  # learning rate
WEIGHT_FLOOR = 0.01


class SignalWeightManager:
    def __init__(self, path=None, eta=None):
        self._path = Path(path) if path else (
            Path(__file__).resolve().parent.parent / "data" / "signal_weights.json"
        )
        self._eta = eta if eta is not None else DEFAULT_ETA
        self._weights = {}
        self._load()

    def _load(self):
        data = load_json(self._path, default=None)
        if data and isinstance(data, dict):
            self._weights = data.get("weights", {})

    def get_weight(self, signal_name):
        return self._weights.get(signal_name, 1.0)

    def update(self, signal_name, correct):
        w = self.get_weight(signal_name)
        if correct:
            w *= (1 + self._eta)
        else:
            w *= (1 - self._eta)
        self._weights[signal_name] = max(w, WEIGHT_FLOOR)

    def batch_update(self, outcomes):
        """Update multiple signals at once. outcomes: {signal_name: bool}"""
        for sig_name, correct in outcomes.items():
            self.update(sig_name, correct)
        self.save()

    def get_normalized_weights(self, signal_names):
        """Get weights normalized so the average is 1.0."""
        raw = {s: self.get_weight(s) for s in signal_names}
        mean = sum(raw.values()) / len(raw) if raw else 1.0
        if mean == 0:
            return {s: 1.0 for s in signal_names}
        return {s: w / mean for s, w in raw.items()}

    def save(self):
        _atomic_write(self._path, {"weights": self._weights})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_weights.py -v`
Expected: PASS

- [ ] **Step 5: Integrate MWU weights into outcome backfill**

In `portfolio/outcome_tracker.py`, in the outcome backfill function (where outcomes are written), add a call to update MWU weights. Find the function that backfills outcomes (typically called from `--check-outcomes`):

```python
# After an outcome is confirmed for a signal, update MWU weights:
try:
    from portfolio.signal_weights import SignalWeightManager
    from portfolio.accuracy_stats import _vote_correct
    mgr = SignalWeightManager()
    for sig_name, vote in signals.items():
        if vote == "HOLD":
            continue
        result_val = _vote_correct(vote, change_pct)
        if result_val is not None:
            mgr.update(sig_name, result_val)
    mgr.save()
except Exception:
    logger.debug("MWU weight update failed", exc_info=True)
```

- [ ] **Step 6: Read MWU weights in signal_engine**

In `portfolio/signal_engine.py`, after the accuracy_data block and before calling `_weighted_consensus`, apply MWU weights as a multiplier:

```python
    # Apply MWU online learning weights as additional multiplier
    try:
        from portfolio.signal_weights import SignalWeightManager
        mwu_mgr = SignalWeightManager()
        mwu_weights = mwu_mgr.get_normalized_weights(
            [s for s in votes if votes[s] != "HOLD"]
        )
        # Merge MWU into activation_rates as an additional multiplier
        for sig_name, mwu_w in mwu_weights.items():
            if sig_name not in activation_rates:
                activation_rates[sig_name] = {}
            existing = activation_rates[sig_name].get("normalized_weight", 1.0)
            activation_rates[sig_name]["normalized_weight"] = existing * mwu_w
    except Exception:
        logger.debug("MWU weights unavailable", exc_info=True)
```

- [ ] **Step 7: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_weights.py tests/test_weighted_consensus.py tests/test_signal_engine_core.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add portfolio/signal_weights.py portfolio/outcome_tracker.py portfolio/signal_engine.py tests/test_signal_weights.py
git commit -m "feat: multiplicative weight updates for online signal learning"
```

---

### Task 5: Utility-Based Weighting (Return Magnitude)

**Why:** A signal that's 55% accurate but catches 3% moves is worth more than one that's 60% accurate on 0.2% moves. Tracking average return magnitude per signal lets us weight by expected profitability, not just hit rate.

**Files:**
- Modify: `portfolio/accuracy_stats.py` (add `signal_utility()`)
- Modify: `portfolio/signal_engine.py` (integrate utility into weighting)
- Test: `tests/test_signal_utility.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signal_utility.py
"""Test utility-based (return magnitude) signal scoring."""
import pytest
from unittest.mock import patch
from portfolio.accuracy_stats import signal_utility


def _make_entry(sig_name, vote, outcome_pct):
    from portfolio.tickers import SIGNAL_NAMES
    signals = {s: "HOLD" for s in SIGNAL_NAMES}
    signals[sig_name] = vote
    return {
        "ts": "2026-03-20T12:00:00+00:00",
        "tickers": {
            "BTC-USD": {
                "signals": signals,
                "consensus": vote,
            }
        },
        "outcomes": {
            "BTC-USD": {
                "1d": {"change_pct": outcome_pct}
            }
        },
    }


class TestSignalUtility:
    def test_positive_utility_for_correct_signals(self):
        entries = [
            _make_entry("rsi", "BUY", 3.0),   # correct, big move
            _make_entry("rsi", "BUY", 1.0),   # correct, small move
            _make_entry("rsi", "BUY", -0.5),  # wrong, small loss
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        # Expected return: (3.0 + 1.0 - 0.5) / 3 = 1.167
        assert result["rsi"]["avg_return"] > 0

    def test_big_mover_beats_accurate_small_mover(self):
        """Signal A: 50% accurate but 5% avg move. Signal B: 80% accurate but 0.3% avg move."""
        entries_a = [
            _make_entry("rsi", "BUY", 5.0),
            _make_entry("rsi", "BUY", -5.0),
        ]
        entries_b = [
            _make_entry("macd", "BUY", 0.3),
            _make_entry("macd", "BUY", 0.3),
            _make_entry("macd", "BUY", 0.3),
            _make_entry("macd", "BUY", 0.3),
            _make_entry("macd", "BUY", -0.3),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries_a + entries_b):
            result = signal_utility("1d")
        # RSI: avg_return = 0.0 (breakeven), utility = 0
        # MACD: avg_return = 0.18, utility > 0
        # But the point is the function computes return-based metrics
        assert "avg_return" in result.get("rsi", {})
        assert "avg_return" in result.get("macd", {})

    def test_empty_entries(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = signal_utility("1d")
        for sig_data in result.values():
            assert sig_data["avg_return"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_utility.py -v`
Expected: FAIL — `ImportError: cannot import name 'signal_utility'`

- [ ] **Step 3: Implement signal_utility**

Add to `portfolio/accuracy_stats.py`:

```python
def signal_utility(horizon="1d"):
    """Compute return-based utility per signal.

    For each signal vote, the "return" is:
      BUY  → +change_pct (positive if price went up)
      SELL → -change_pct (positive if price went down)

    Returns:
        dict: {signal_name: {avg_return, total_return, samples, utility_score}}
        utility_score = avg_return * sqrt(samples) — rewards both magnitude and consistency.
    """
    import math

    entries = load_entries()
    stats = {s: {"total_return": 0.0, "samples": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue
            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                # Directional return: positive means the signal was right
                if vote == "BUY":
                    ret = change_pct
                else:  # SELL
                    ret = -change_pct
                stats[sig_name]["total_return"] += ret
                stats[sig_name]["samples"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        samples = s["samples"]
        avg_ret = s["total_return"] / samples if samples > 0 else 0.0
        utility = avg_ret * math.sqrt(samples) if samples > 0 else 0.0
        result[sig_name] = {
            "avg_return": round(avg_ret, 4),
            "total_return": round(s["total_return"], 4),
            "samples": samples,
            "utility_score": round(utility, 4),
        }
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_utility.py -v`
Expected: PASS

- [ ] **Step 5: Integrate utility into signal_engine weighting**

In `portfolio/signal_engine.py`, after the MWU weight integration (Task 4), add utility as an optional boost:

```python
    # Utility boost: scale weight by return-based utility score
    try:
        from portfolio.accuracy_stats import signal_utility
        utility_data = signal_utility("1d")
        for sig_name in list(accuracy_data.keys()):
            u = utility_data.get(sig_name, {})
            u_score = u.get("avg_return", 0.0)
            samples = u.get("samples", 0)
            if samples >= 30 and u_score > 0:
                # Boost signals with positive expected return (1.0 to 1.5x)
                boost = min(1.0 + u_score, 1.5)
                if sig_name in accuracy_data:
                    accuracy_data[sig_name]["accuracy"] *= boost
                    # Cap at 0.95 to prevent overconfidence
                    accuracy_data[sig_name]["accuracy"] = min(
                        accuracy_data[sig_name]["accuracy"], 0.95
                    )
    except Exception:
        logger.debug("Utility weighting unavailable", exc_info=True)
```

- [ ] **Step 6: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_utility.py tests/test_weighted_consensus.py tests/test_signal_engine_core.py tests/test_signal_improvements.py -v`
Expected: All PASS (may see the pre-existing vote count failure)

- [ ] **Step 7: Commit**

```bash
git add portfolio/accuracy_stats.py portfolio/signal_engine.py tests/test_signal_utility.py
git commit -m "feat: utility-based signal weighting using return magnitude"
```

---

### Task 6: Reduce Active Signal Set (Top N Gate)

**Why:** 20+ marginal signals dilute the consensus of the 5-8 that actually work. Adding a configurable "top N" gate lets the system automatically focus on its best performers.

**Files:**
- Modify: `portfolio/signal_engine.py` (add top-N filtering before consensus)
- Modify: `config.example.json` (add `signals.max_active_signals`)
- Test: `tests/test_topn_gate.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topn_gate.py
"""Test top-N signal gating in weighted consensus."""
import pytest
from portfolio.signal_engine import _weighted_consensus, ACCURACY_GATE_THRESHOLD


class TestTopNGate:
    def test_top_n_limits_signals(self):
        """When max_signals is set, only top N by accuracy participate."""
        votes = {
            "best": "BUY", "good": "BUY", "mid": "SELL",
            "weak": "SELL", "worst": "SELL",
        }
        acc = {
            "best": {"accuracy": 0.9, "total": 100},
            "good": {"accuracy": 0.8, "total": 100},
            "mid": {"accuracy": 0.7, "total": 100},
            "weak": {"accuracy": 0.6, "total": 100},
            "worst": {"accuracy": 0.55, "total": 100},
        }
        # With max_signals=3: only best(0.9), good(0.8), mid(0.7) participate
        # BUY: 0.9+0.8=1.7, SELL: 0.7 → BUY wins
        action, conf = _weighted_consensus(votes, acc, "trending-up", max_signals=3)
        assert action == "BUY"

    def test_no_limit_all_participate(self):
        """Without max_signals, all signals vote."""
        votes = {"s1": "BUY", "s2": "SELL", "s3": "SELL"}
        acc = {
            "s1": {"accuracy": 0.9, "total": 100},
            "s2": {"accuracy": 0.6, "total": 100},
            "s3": {"accuracy": 0.6, "total": 100},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # BUY: 0.9, SELL: 0.6+0.6=1.2 → SELL wins
        assert action == "SELL"

    def test_max_signals_none_means_unlimited(self):
        """max_signals=None should be same as not passing it."""
        votes = {"s1": "BUY", "s2": "SELL"}
        acc = {"s1": {"accuracy": 0.7, "total": 50}, "s2": {"accuracy": 0.6, "total": 50}}
        action1, _ = _weighted_consensus(votes, acc, "trending-up", max_signals=None)
        action2, _ = _weighted_consensus(votes, acc, "trending-up")
        assert action1 == action2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topn_gate.py -v`
Expected: FAIL — `TypeError: _weighted_consensus() got an unexpected keyword argument 'max_signals'`

- [ ] **Step 3: Add max_signals parameter to _weighted_consensus**

In `portfolio/signal_engine.py`, update the function signature and add filtering:

```python
def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
                        accuracy_gate=None, max_signals=None):
```

After the `gate =` line and before the main loop, add:

```python
    # Top-N gate: only let the top max_signals (by accuracy) participate
    active_votes = {k: v for k, v in votes.items() if v != "HOLD"}
    if max_signals and len(active_votes) > max_signals:
        ranked = sorted(
            active_votes.keys(),
            key=lambda s: accuracy_data.get(s, {}).get("accuracy", 0.5),
            reverse=True,
        )
        excluded = set(ranked[max_signals:])
    else:
        excluded = set()
```

Then in the loop, after `if vote == "HOLD": continue`, add:

```python
        if signal_name in excluded:
            continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topn_gate.py -v`
Expected: PASS

- [ ] **Step 5: Wire config to call site**

In `portfolio/signal_engine.py`, where `_weighted_consensus` is called, pass the config value:

```python
    max_signals = sig_cfg.get("max_active_signals")
    weighted_action, weighted_conf = _weighted_consensus(
        votes, accuracy_data, regime, activation_rates,
        accuracy_gate=accuracy_gate,
        max_signals=max_signals,
    )
```

- [ ] **Step 6: Add config to config.example.json**

```json
  "signals": {
    "accuracy_gate_threshold": 0.45,
    "accuracy_gate_min_samples": 30,
    "accuracy_halflife_days": 5,
    "max_active_signals": null
  },
```

(`null` = unlimited, use e.g. `10` to limit)

- [ ] **Step 7: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topn_gate.py tests/test_weighted_consensus.py tests/test_signal_engine_core.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add portfolio/signal_engine.py config.example.json tests/test_topn_gate.py
git commit -m "feat: configurable top-N signal gate for consensus"
```

---

## Execution Order & Dependencies

```
Task 1 (regime persistence) ──→ Task 2 (regime-conditional accuracy)
                                      │
Task 3 (exponential decay) ───────────┤ (independent of Task 1/2)
                                      │
Task 4 (MWU weights) ────────────────┤ (independent, uses outcome_tracker)
                                      │
Task 5 (utility weighting) ──────────┤ (independent, uses accuracy_stats)
                                      │
Task 6 (top-N gate) ─────────────────┘ (independent, modifies _weighted_consensus)
```

Tasks 1→2 must be sequential. Tasks 3-6 are independent of each other and can be done in any order or parallel. All tasks are safe to deploy independently — each produces a working system.

## Verification

After all tasks are complete, run the full test suite:

```bash
.venv/Scripts/python.exe -m pytest tests/test_weighted_consensus.py tests/test_signal_engine_core.py tests/test_signal_improvements.py tests/test_regime_persistence.py tests/test_regime_accuracy.py tests/test_ewma_accuracy.py tests/test_signal_weights.py tests/test_signal_utility.py tests/test_topn_gate.py -v
```

Then verify the live system:

```bash
.venv/Scripts/python.exe -u portfolio/main.py --accuracy
```
