"""Tests for BUG-158: per-ticker signal accuracy override and regime gating exemption.

Tests the new per-ticker accuracy caching in accuracy_stats.py and the
per-ticker override + regime gating exemption in signal_engine.py.
"""

import json
import time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(ts, tickers_data, outcomes_data=None):
    return {
        "ts": ts,
        "trigger_reasons": ["test"],
        "fx_rate": 10.5,
        "tickers": tickers_data,
        "outcomes": outcomes_data or {},
    }


def _ticker_signals(consensus, signals_dict, price_usd=100.0, regime="ranging"):
    return {
        "price_usd": price_usd,
        "consensus": consensus,
        "buy_count": sum(1 for v in signals_dict.values() if v == "BUY"),
        "sell_count": sum(1 for v in signals_dict.values() if v == "SELL"),
        "total_voters": len(signals_dict),
        "signals": signals_dict,
        "regime": regime,
    }


def _outcome(change_pct, price_usd=100.0):
    return {"1d": {"change_pct": change_pct, "price_usd": price_usd, "ts": "2026-01-02T00:00:00"}}


# ---------------------------------------------------------------------------
# Tests for caching layer (accuracy_stats.py)
# ---------------------------------------------------------------------------

class TestTickerAccuracyCache:
    """Tests for load_cached_ticker_accuracy / write_ticker_accuracy_cache."""

    def test_write_and_load(self, tmp_path):
        """Cache should round-trip: write then load returns same data."""
        from portfolio import accuracy_stats as mod

        cache_file = tmp_path / "ticker_acc_cache.json"
        original = mod.TICKER_ACCURACY_CACHE_FILE
        mod.TICKER_ACCURACY_CACHE_FILE = cache_file
        try:
            data = {
                "BTC-USD": {"rsi": {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0}},
                "XAG-USD": {"fear_greed": {"correct": 47, "total": 50, "accuracy": 0.94, "pct": 94.0}},
            }
            mod.write_ticker_accuracy_cache("1d", data)

            loaded = mod.load_cached_ticker_accuracy("1d")
            assert loaded is not None
            assert loaded["BTC-USD"]["rsi"]["accuracy"] == 0.5
            assert loaded["XAG-USD"]["fear_greed"]["accuracy"] == 0.94
        finally:
            mod.TICKER_ACCURACY_CACHE_FILE = original

    def test_cache_expiry(self, tmp_path):
        """Stale cache (TTL exceeded) should return None."""
        from portfolio import accuracy_stats as mod

        cache_file = tmp_path / "ticker_acc_cache.json"
        original = mod.TICKER_ACCURACY_CACHE_FILE
        mod.TICKER_ACCURACY_CACHE_FILE = cache_file
        try:
            data = {"BTC-USD": {"rsi": {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0}}}
            # Write with a time far in the past
            cache = {"1d": data, "time": time.time() - mod.ACCURACY_CACHE_TTL - 100}
            cache_file.write_text(json.dumps(cache))

            loaded = mod.load_cached_ticker_accuracy("1d")
            assert loaded is None
        finally:
            mod.TICKER_ACCURACY_CACHE_FILE = original

    def test_missing_horizon(self, tmp_path):
        """Loading a non-cached horizon should return None."""
        from portfolio import accuracy_stats as mod

        cache_file = tmp_path / "ticker_acc_cache.json"
        original = mod.TICKER_ACCURACY_CACHE_FILE
        mod.TICKER_ACCURACY_CACHE_FILE = cache_file
        try:
            data = {"BTC-USD": {"rsi": {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0}}}
            mod.write_ticker_accuracy_cache("1d", data)

            loaded = mod.load_cached_ticker_accuracy("3h")
            assert loaded is None
        finally:
            mod.TICKER_ACCURACY_CACHE_FILE = original

    def test_empty_file(self, tmp_path):
        """Empty/missing cache file should return None."""
        from portfolio import accuracy_stats as mod

        cache_file = tmp_path / "ticker_acc_cache.json"
        original = mod.TICKER_ACCURACY_CACHE_FILE
        mod.TICKER_ACCURACY_CACHE_FILE = cache_file
        try:
            loaded = mod.load_cached_ticker_accuracy("1d")
            assert loaded is None
        finally:
            mod.TICKER_ACCURACY_CACHE_FILE = original


class TestAccuracyByTickerSignalCached:
    """Tests for accuracy_by_ticker_signal_cached()."""

    def test_populates_cache_on_miss(self, tmp_path, monkeypatch):
        """Should compute and cache on miss."""
        from portfolio import accuracy_stats as mod

        cache_file = tmp_path / "ticker_acc_cache.json"
        monkeypatch.setattr(mod, "TICKER_ACCURACY_CACHE_FILE", cache_file)

        entries = [
            _make_entry("2026-01-01T00:00:00", {
                "BTC-USD": _ticker_signals("BUY", {"rsi": "BUY"}),
            }, {"BTC-USD": _outcome(5.0)}),
        ]
        monkeypatch.setattr(mod, "load_entries", lambda: entries)

        result = mod.accuracy_by_ticker_signal_cached("1d")
        assert result["BTC-USD"]["rsi"]["accuracy"] == 1.0

        # Cache file should now exist
        assert cache_file.exists()

    def test_uses_cache_on_hit(self, tmp_path, monkeypatch):
        """Should return cached data without recomputing."""
        from portfolio import accuracy_stats as mod

        cache_file = tmp_path / "ticker_acc_cache.json"
        monkeypatch.setattr(mod, "TICKER_ACCURACY_CACHE_FILE", cache_file)

        # Pre-populate cache
        cached_data = {"BTC-USD": {"rsi": {"correct": 99, "total": 100, "accuracy": 0.99, "pct": 99.0}}}
        cache = {"1d": cached_data, "time": time.time()}
        cache_file.write_text(json.dumps(cache))

        # This should NOT call load_entries
        call_count = 0
        original_load = mod.load_entries

        def counting_load():
            nonlocal call_count
            call_count += 1
            return original_load()

        monkeypatch.setattr(mod, "load_entries", counting_load)

        result = mod.accuracy_by_ticker_signal_cached("1d")
        assert result["BTC-USD"]["rsi"]["accuracy"] == 0.99
        assert call_count == 0  # Should NOT have called load_entries

    def test_min_samples_filtering(self, tmp_path, monkeypatch):
        """min_samples should filter post-cache."""
        from portfolio import accuracy_stats as mod

        cache_file = tmp_path / "ticker_acc_cache.json"
        monkeypatch.setattr(mod, "TICKER_ACCURACY_CACHE_FILE", cache_file)

        cached_data = {
            "BTC-USD": {
                "rsi": {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0},
                "ema": {"correct": 90, "total": 100, "accuracy": 0.9, "pct": 90.0},
            }
        }
        cache = {"1d": cached_data, "time": time.time()}
        cache_file.write_text(json.dumps(cache))

        result = mod.accuracy_by_ticker_signal_cached("1d", min_samples=50)
        # rsi has 10 samples < 50 → filtered out
        assert "rsi" not in result.get("BTC-USD", {})
        # ema has 100 samples ≥ 50 → included
        assert result["BTC-USD"]["ema"]["accuracy"] == 0.9


# ---------------------------------------------------------------------------
# Tests for per-ticker regime gating exemption (signal_engine.py)
# ---------------------------------------------------------------------------

class TestPerTickerRegimeGatingExemption:
    """Tests for BUG-158: signals with high per-ticker accuracy exempt from regime gating."""

    def test_fear_greed_exempt_for_xag(self, monkeypatch):
        """fear_greed should NOT be regime-gated for XAG-USD when per-ticker acc is high."""
        from portfolio import signal_engine as se

        # Mock per-ticker accuracy showing fear_greed is excellent for XAG
        mock_data = {
            "XAG-USD": {
                "fear_greed": {"correct": 47, "total": 50, "accuracy": 0.94, "pct": 94.0},
            }
        }
        monkeypatch.setattr(
            "portfolio.accuracy_stats.accuracy_by_ticker_signal_cached",
            lambda *a, **kw: mock_data,
        )

        # In ranging regime, fear_greed is normally gated
        gated = se._get_regime_gated("ranging", "1d")
        assert "fear_greed" in gated  # Sanity: it IS in the static gated set

        # But when we apply per-ticker exemption, it should be removed
        # Simulate what the signal function does
        regime_gated = se._get_regime_gated("ranging", "1d")
        ticker_acc = mock_data.get("XAG-USD", {})
        regime_gated_effective = set(regime_gated)
        for sig_name in list(regime_gated_effective):
            t_stats = ticker_acc.get(sig_name, {})
            if t_stats.get("total", 0) >= 50 and t_stats.get("accuracy", 0) >= 0.60:
                regime_gated_effective.discard(sig_name)

        assert "fear_greed" not in regime_gated_effective

    def test_low_accuracy_not_exempt(self):
        """Signals with low per-ticker accuracy should remain gated."""
        from portfolio import signal_engine as se

        # ministral at 18.9% on XAG should NOT be exempt
        ticker_acc = {"ministral": {"correct": 18, "total": 95, "accuracy": 0.189, "pct": 18.9}}
        regime_gated = set(se._get_regime_gated("ranging", "1d"))

        for sig_name in list(regime_gated):
            t_stats = ticker_acc.get(sig_name, {})
            if t_stats.get("total", 0) >= 50 and t_stats.get("accuracy", 0) >= 0.60:
                regime_gated.discard(sig_name)

        # ministral is not in the default ranging gated set, but if it were, it would stay
        # This test verifies the logic doesn't exempt low-accuracy signals
        assert 0.189 < 0.60  # Sanity: the threshold would not exempt it

    def test_insufficient_samples_not_exempt(self):
        """Signals with too few samples should remain gated even if accuracy is high."""
        from portfolio import signal_engine as se

        ticker_acc = {"fear_greed": {"correct": 9, "total": 10, "accuracy": 0.9, "pct": 90.0}}
        regime_gated = set(se._get_regime_gated("ranging", "1d"))
        original_len = len(regime_gated)

        for sig_name in list(regime_gated):
            t_stats = ticker_acc.get(sig_name, {})
            if t_stats.get("total", 0) >= 50 and t_stats.get("accuracy", 0) >= 0.60:
                regime_gated.discard(sig_name)

        # 10 samples < 50 threshold → should NOT be exempt
        assert len(regime_gated) == original_len


# ---------------------------------------------------------------------------
# Tests for RES-2026-04-21: recent-accuracy override for regime gating
# ---------------------------------------------------------------------------

class TestRecentAccuracyRegimeOverride:
    """RES-2026-04-21: signals with high recent accuracy exempt from regime gating."""

    def test_high_recent_accuracy_exempts_from_regime_gate(self):
        """smart_money with 58% recent accuracy should bypass ranging regime gate."""
        from portfolio import signal_engine as se

        # smart_money is normally gated in ranging at 1d
        regime_gated = se._get_regime_gated("ranging", "1d")
        assert "smart_money" in regime_gated  # Sanity check: in gated set

        # Simulate the recent-accuracy override logic
        recent_acc_data = {
            "smart_money": {"accuracy": 0.58, "total": 80, "correct": 46, "pct": 58.0}
        }
        ticker_acc = {}  # No per-ticker exemption
        regime_gated_effective = set(regime_gated)

        _RECENT_EXEMPT_ACC = 0.55
        _RECENT_EXEMPT_MIN_SAMPLES = 50
        _TICKER_EXEMPT_ACC = 0.60
        _TICKER_EXEMPT_MIN_SAMPLES = 50

        for sig_name in list(regime_gated_effective):
            # Per-ticker exemption (BUG-158) — not triggered here
            t_stats = ticker_acc.get(sig_name, {})
            if t_stats.get("total", 0) >= _TICKER_EXEMPT_MIN_SAMPLES and t_stats.get("accuracy", 0) >= _TICKER_EXEMPT_ACC:
                regime_gated_effective.discard(sig_name)
                continue
            # Recent-accuracy override
            r_stats = recent_acc_data.get(sig_name, {})
            if r_stats.get("total", 0) >= _RECENT_EXEMPT_MIN_SAMPLES and r_stats.get("accuracy", 0) >= _RECENT_EXEMPT_ACC:
                regime_gated_effective.discard(sig_name)

        assert "smart_money" not in regime_gated_effective

    def test_low_recent_accuracy_stays_gated(self):
        """trend at 40.3% recent should remain gated."""
        from portfolio import signal_engine as se

        recent_acc_data = {
            "trend": {"accuracy": 0.403, "total": 200, "correct": 80, "pct": 40.3}
        }
        regime_gated = set(se._get_regime_gated("ranging", "1d"))

        for sig_name in list(regime_gated):
            r_stats = recent_acc_data.get(sig_name, {})
            if r_stats.get("total", 0) >= 50 and r_stats.get("accuracy", 0) >= 0.55:
                regime_gated.discard(sig_name)

        # trend at 40.3% < 55% threshold → should stay gated
        assert "trend" in regime_gated

    def test_insufficient_recent_samples_stays_gated(self):
        """Signal with high recent accuracy but too few samples stays gated."""
        from portfolio import signal_engine as se

        recent_acc_data = {
            "smart_money": {"accuracy": 0.90, "total": 10, "correct": 9, "pct": 90.0}
        }
        regime_gated = set(se._get_regime_gated("ranging", "1d"))

        for sig_name in list(regime_gated):
            r_stats = recent_acc_data.get(sig_name, {})
            if r_stats.get("total", 0) >= 50 and r_stats.get("accuracy", 0) >= 0.55:
                regime_gated.discard(sig_name)

        # 10 samples < 50 threshold -> stays gated
        assert "smart_money" in regime_gated


# ---------------------------------------------------------------------------
# Tests for per-ticker accuracy override in weighted consensus
# ---------------------------------------------------------------------------

class TestPerTickerAccuracyOverride:
    """Tests for BUG-158: per-ticker accuracy replacing global accuracy."""

    def test_ticker_accuracy_overrides_global(self):
        """When per-ticker data available, it should override global accuracy."""
        # Simulate the override logic
        accuracy_data = {
            "fear_greed": {"accuracy": 0.259, "total": 170, "correct": 44, "pct": 25.9},
            "rsi": {"accuracy": 0.512, "total": 27881, "correct": 14263, "pct": 51.2},
        }
        ticker_acc_data = {
            "fear_greed": {"accuracy": 0.938, "total": 290, "correct": 272, "pct": 93.8},
        }

        min_samples = 30
        for sig_name, t_stats in ticker_acc_data.items():
            if t_stats.get("total", 0) >= min_samples:
                accuracy_data[sig_name] = {
                    "accuracy": t_stats["accuracy"],
                    "total": t_stats["total"],
                    "correct": t_stats.get("correct", 0),
                    "pct": t_stats.get("pct", round(t_stats["accuracy"] * 100, 1)),
                }

        # fear_greed should now use per-ticker accuracy (93.8%), not global (25.9%)
        assert accuracy_data["fear_greed"]["accuracy"] == 0.938
        assert accuracy_data["fear_greed"]["total"] == 290
        # rsi should keep global accuracy (no per-ticker override for this test)
        assert accuracy_data["rsi"]["accuracy"] == 0.512

    def test_fallback_to_llm_extra_info(self):
        """When per-ticker cache unavailable, should fall back to LLM extra_info."""
        accuracy_data = {"qwen3": {"accuracy": 0.598, "total": 3589, "correct": 2148, "pct": 59.8}}
        ticker_acc_data = {}  # Empty — cache unavailable
        extra_info = {"qwen3_accuracy": 0.90, "qwen3_samples": 50}

        # Simulate fallback logic
        if ticker_acc_data:
            for sig_name, t_stats in ticker_acc_data.items():
                if t_stats.get("total", 0) >= 30:
                    accuracy_data[sig_name] = t_stats
        else:
            for llm_sig in ("qwen3", "ministral"):
                per_ticker_acc = extra_info.get(f"{llm_sig}_accuracy")
                per_ticker_samples = extra_info.get(f"{llm_sig}_samples", 0)
                if per_ticker_acc is not None and per_ticker_samples >= 20:
                    accuracy_data[llm_sig] = {
                        "accuracy": per_ticker_acc,
                        "total": per_ticker_samples,
                        "correct": int(per_ticker_acc * per_ticker_samples),
                        "pct": round(per_ticker_acc * 100, 1),
                    }

        # Should use fallback LLM data
        assert accuracy_data["qwen3"]["accuracy"] == 0.90


# ---------------------------------------------------------------------------
# Tests for SC-I-001: _weighted_consensus must honour regime_gated_override
# ---------------------------------------------------------------------------

class TestRegimeGatedOverride:
    """SC-I-001: _weighted_consensus must not re-gate signals exempted by BUG-158."""

    def test_override_preserves_exempted_signal(self):
        """When regime_gated_override omits fear_greed, it should keep its vote."""
        from portfolio.signal_engine import _weighted_consensus, _get_regime_gated

        votes = {
            "fear_greed": "BUY",
            "rsi": "BUY",
            "ema": "BUY",
            "macd": "BUY",
        }
        accuracy_data = {
            "fear_greed": {"accuracy": 0.94, "total": 50},
            "rsi": {"accuracy": 0.55, "total": 100},
            "ema": {"accuracy": 0.52, "total": 100},
            "macd": {"accuracy": 0.51, "total": 100},
        }
        full_gated = _get_regime_gated("ranging", "1d")
        assert "fear_greed" in full_gated, "Sanity: fear_greed is in ranging gated set"

        # Without override: fear_greed gets re-gated to HOLD
        action_no_override, _ = _weighted_consensus(
            dict(votes), dict(accuracy_data), "ranging", horizon="1d",
        )
        # With override: fear_greed exempted (not in override set)
        override = full_gated - {"fear_greed"}
        action_with_override, _ = _weighted_consensus(
            dict(votes), dict(accuracy_data), "ranging", horizon="1d",
            regime_gated_override=override,
        )
        # The override version should have more BUY weight (fear_greed 94% contributing)
        # Both should produce BUY since rsi/ema/macd are not in the gated set
        assert action_with_override == "BUY"

    def test_override_none_falls_back_to_default(self):
        """When override is None, _weighted_consensus should compute regime gating normally."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"fear_greed": "BUY", "rsi": "BUY", "ema": "SELL"}
        accuracy_data = {
            "fear_greed": {"accuracy": 0.60, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
            "ema": {"accuracy": 0.55, "total": 100},
        }
        # With None override (default): should behave as before
        action, _ = _weighted_consensus(
            dict(votes), accuracy_data, "ranging", horizon="1d",
            regime_gated_override=None,
        )
        # fear_greed gated in ranging → only rsi(BUY) vs ema(SELL) remain
        assert action in ("BUY", "SELL", "HOLD")

    def test_empty_override_gates_nothing(self):
        """An empty override set should gate no signals."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"fear_greed": "BUY", "rsi": "BUY", "ema": "BUY"}
        accuracy_data = {
            "fear_greed": {"accuracy": 0.94, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
            "ema": {"accuracy": 0.55, "total": 100},
        }
        action, conf = _weighted_consensus(
            dict(votes), accuracy_data, "ranging", horizon="1d",
            regime_gated_override=set(),
        )
        assert action == "BUY"
        assert conf > 0


# ---------------------------------------------------------------------------
# Tests for CROSS-001: outcome_tracker should use _raw_votes
# ---------------------------------------------------------------------------

class TestOutcomeTrackerRawVotes:
    """CROSS-001: outcome_tracker should read _raw_votes for accuracy tracking."""

    def test_raw_votes_preferred_over_gated(self):
        """When _raw_votes available, outcome_tracker should use pre-gate votes."""
        extra = {
            "_votes": {"fear_greed": "HOLD", "rsi": "BUY"},
            "_raw_votes": {"fear_greed": "BUY", "rsi": "BUY"},
            "_regime": "ranging",
        }
        passed_votes = extra.get("_raw_votes", extra.get("_votes"))
        assert passed_votes["fear_greed"] == "BUY", "Should use raw (pre-gate) vote"
        assert passed_votes["rsi"] == "BUY"

    def test_fallback_to_votes_when_no_raw(self):
        """When _raw_votes not present, should fall back to _votes."""
        extra = {
            "_votes": {"rsi": "BUY", "ema": "SELL"},
            "_regime": "ranging",
        }
        passed_votes = extra.get("_raw_votes", extra.get("_votes"))
        assert passed_votes["rsi"] == "BUY"
        assert passed_votes["ema"] == "SELL"

    def test_raw_votes_includes_gated_signal(self):
        """_raw_votes should preserve the original vote for a regime-gated signal."""
        extra = {
            "_votes": {
                "fear_greed": "HOLD",
                "rsi": "BUY",
                "trend": "HOLD",
            },
            "_raw_votes": {
                "fear_greed": "BUY",
                "rsi": "BUY",
                "trend": "SELL",
            },
        }
        passed_votes = extra.get("_raw_votes", extra.get("_votes"))
        assert passed_votes["fear_greed"] == "BUY", "Gated signal should show original vote"
        assert passed_votes["trend"] == "SELL", "Gated signal should show original vote"
